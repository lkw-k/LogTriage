"""예측 채점. 대표 지표는 macro F1 (절대규칙 4). accuracy 는 보고하지 않는다.

입력은 runs/<exp_id>/preds_<split>.parquet 이고 컬럼은
  y_true, y_pred, msg_norm  (+ 선택: p_normal, p_kernel_mem, p_kernel_ops, p_app)
확률 컬럼이 val/test 양쪽에 있으면 클래스 가중치를 val 에서 맞춰 test 에 적용한다.
val 에서만 맞춘다 — test 를 보며 맞추면 그 숫자는 일반화 추정치가 아니다.

전체 macro F1 과 함께 **기등장/미등장 템플릿 부분집합**을 따로 채점한다.
모델 비교 판정은 미등장 쪽으로 한다 (docs/RISKS.md 9, SPEC 4-3).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

CLASSES = ["normal", "kernel_mem", "kernel_ops", "app"]
PROB_COLS = [f"p_{c}" for c in CLASSES]
WEIGHT_GRID = np.round(np.exp(np.linspace(np.log(0.02), np.log(50), 25)), 4)


def prf(y_true, y_pred, cls):
    tp = int(((y_pred == cls) & (y_true == cls)).sum())
    fp = int(((y_pred == cls) & (y_true != cls)).sum())
    fn = int(((y_pred != cls) & (y_true == cls)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    # support 와 fp 를 같이 남긴다. kernel_mem 은 test support 가 100 이라
    # F1 만 보면 그 숫자가 무엇으로 결정됐는지 놓친다 (docs/RISKS.md B).
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn, "tp": tp, "fp": fp}


def score(y_true, y_pred):
    per = {c: prf(y_true, y_pred, c) for c in CLASSES}
    return {"macro_f1": sum(per[c]["f1"] for c in CLASSES) / len(CLASSES), "per_class": per}


def weighted_pred(probs, w):
    return np.asarray(CLASSES)[np.argmax(probs * w, axis=1)]


def tune_weights(y_true, probs, grid=WEIGHT_GRID, passes=3):
    """클래스별 가중치 좌표상승으로 macro F1 최대화. val 에서만 호출할 것.

    argmax 는 오분류율용 결정 규칙이라 macro F1 과 어긋난다. 정상이 92.7% 인
    데이터에서 argmax 는 희귀 클래스를 체계적으로 과소 예측한다.
    """
    w = np.ones(len(CLASSES))
    best = score(y_true, weighted_pred(probs, w))["macro_f1"]
    for _ in range(passes):
        for i in range(len(CLASSES)):
            for g in grid:
                trial = w.copy()
                trial[i] = g
                m = score(y_true, weighted_pred(probs, trial))["macro_f1"]
                if m > best:
                    best, w = m, trial
    return w, best


def seen_mask(msgs, train_ref):
    """학습에 등장했던 템플릿인가. 아니면 미등장이다.

    전체 macro F1 은 기등장 구간의 암기분이 섞여 있어 모델 비교에 쓸 수 없다
    (docs/RISKS.md 9). 판정은 미등장 부분집합으로 한다.
    """
    ref = set(pd.read_parquet(train_ref, columns=["msg_norm"])["msg_norm"])
    return msgs.isin(ref).to_numpy()


def confusion(y_true, y_pred):
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def save_cm(cm, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_majority_preds(processed, split):
    """E0. 학습 없이 train 최빈 클래스로 고정 예측한다."""
    major = pd.read_parquet(processed / "train.parquet", columns=["label4"])["label4"].mode()[0]
    df = pd.read_parquet(processed / f"{split}.parquet", columns=["label4", "msg_norm"])
    return pd.DataFrame({"y_true": df["label4"], "y_pred": major, "msg_norm": df["msg_norm"]})


def report(name, s):
    print(f"\n[{name}]  macro F1 = {s['macro_f1']:.4f}")
    print(f"  {'클래스':<11}{'precision':>10}{'recall':>9}{'F1':>8}{'support':>10}{'오탐':>9}")
    for c in CLASSES:
        m = s["per_class"][c]
        print(
            f"  {c:<11}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}"
            f"{m['support']:>10,}{m['fp']:>9,}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--baseline", choices=["majority"], help="예측 파일 없이 E0 를 만든다")
    ap.add_argument("--train-ref", help="미등장 판정 기준. 기본 processed/train.parquet")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    processed = Path(cfg["paths"]["processed"])
    rundir = Path(cfg["paths"]["runs"]) / args.exp_id
    rundir.mkdir(parents=True, exist_ok=True)

    if args.baseline == "majority":
        for sp in ["val", "test"]:
            build_majority_preds(processed, sp).to_parquet(
                rundir / f"preds_{sp}.parquet", index=False
            )
        print(f"E0 고정 예측 생성 -> {rundir}/preds_{{val,test}}.parquet")

    pred_path = rundir / f"preds_{args.split}.parquet"
    if not pred_path.exists():
        raise SystemExit(
            f"{pred_path} 가 없다. 학습이 이 파일을 남기거나, --baseline majority 로 E0 를 만들 것."
        )

    df = pd.read_parquet(pred_path)
    y_true, y_pred = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    print(f"exp {args.exp_id} / {args.split}  {len(df):,}행")

    out = {"exp_id": args.exp_id, "split": args.split, "n_rows": len(df)}
    s = score(y_true, y_pred)
    report("argmax", s)
    out["argmax"] = s

    train_ref = Path(args.train_ref) if args.train_ref else processed / "train.parquet"
    seen = seen_mask(df["msg_norm"], train_ref)
    for key, name, mask in [("seen", "기등장", seen), ("unseen", "미등장", ~seen)]:
        share = float(mask.mean() * 100)
        sub = score(y_true[mask], y_pred[mask])
        report(f"{name} 템플릿 {mask.sum():,}행 ({share:.1f}%)", sub)
        out[key] = sub | {"n_rows": int(mask.sum()), "share": share,
                          "train_ref": train_ref.name}

    # 임계값(클래스 가중치) 보정 — val 에서 맞춰 test 에 그대로 적용
    val_path = rundir / "preds_val.parquet"
    if set(PROB_COLS) <= set(df.columns) and val_path.exists():
        val = pd.read_parquet(val_path)
        if set(PROB_COLS) <= set(val.columns):
            w, val_best = tune_weights(val["y_true"].to_numpy(), val[PROB_COLS].to_numpy())
            s2 = score(y_true, weighted_pred(df[PROB_COLS].to_numpy(), w))
            report(f"val 보정 후 (가중치 {np.round(w, 3).tolist()})", s2)
            print(f"  val 에서의 macro F1 {val_best:.4f} -> {args.split} {s2['macro_f1']:.4f}"
                  f"  (감쇠 {val_best - s2['macro_f1']:+.4f})")
            out["calibrated"] = s2 | {"weights": w.tolist(), "tuned_on": "val",
                                      "val_macro_f1": val_best}
    else:
        print("\n확률 컬럼이 없어 가중치 보정은 건너뛴다 (argmax 만 보고).")

    cm = confusion(y_true, y_pred)
    save_cm(cm, rundir / "cm.png", f"{args.exp_id} / {args.split} (argmax)")
    out["confusion_matrix"] = {"classes": CLASSES, "counts": cm.tolist()}

    wrong = df[y_true != y_pred]
    n = cfg["evaluate"]["n_error_samples"]
    wrong.head(n).to_csv(rundir / "errors.csv", index=False, encoding="utf-8")

    (rundir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n오분류 {len(wrong):,}행 중 상위 {min(n, len(wrong))}건 -> {rundir}/errors.csv")
    print(f"출력: {rundir}/{{metrics.json, cm.png, errors.csv}}")


if __name__ == "__main__":
    main()
