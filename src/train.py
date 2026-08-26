"""E2. BERT 파인튜닝. 하이퍼파라미터의 정본은 configs/base.yaml 의 train 절이다 (SPEC 4-1).

**학습 시작은 사용자가 한다.** 이 모듈은 명령을 직접 받았을 때만 돈다.

- 에폭이 끝날 때마다 last.pt 에 모델·옵티마이저·스케줄러·RNG·조기종료 상태를 전부 남긴다.
  컴퓨터가 꺼져도 같은 명령을 다시 실행하면 그 지점부터 이어서 학습한다.
- runs/<exp_id>/progress.html 을 브라우저로 열어두면 5초마다 스스로 갱신된다.
- 베스트 체크포인트는 early_stopping.metric 기준. 기본값 val_macro_f1 은 test 의 대리
  지표가 아니다 — E2c 에서 val 0.9941 이 test 0.6829 로 무너졌다 (docs/RISKS.md A).
- --inner-val-frac 을 주면 train 뒤쪽을 시간순으로 떼어 미등장 템플릿 검증셋을 만들고
  inner_val_unseen_macro_f1 로 체크포인트를 고를 수 있다. val 로는 이 지표를 만들 수
  없다 (val 미등장에 kernel_mem 3행, app 0행).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src import config
from src.dataset import CLASSES, LogDataset, build_text, encode_unique
from src.evaluate import score

PLOT_EVERY = 100
METRICS = ["val_macro_f1", "inner_val_macro_f1", "inner_val_unseen_macro_f1"]
E1C_TEST = 0.9316    # 넘어야 할 선. 그래프에 기준선으로 그린다.
E1C_UNSEEN = 0.7451  # 미등장 템플릿만 골라낸 E1c. 진짜 비교 대상이다 (docs/RISKS.md 9).


def class_weights(labels, device):
    """w_c = N / (K * n_c). SPEC 4-1."""
    counts = np.array([(labels == c).sum() for c in CLASSES], dtype=float)
    w = len(labels) / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def inner_split(df, ref_ts, val_frac):
    """train 안에서 시간순으로 검증 구간을 떼어낸다. 경계는 ref_ts 가 정한다.

    df 의 행 위치로 자르면 안 된다. sample 이 시간축을 따라 불균등하게 행을 지우기 때문에
    표집된 파일의 80% 지점은 시간의 80% 지점이 아니다. train_cap 을 위치로 자르면
    inner-val 미등장에서 알럿 3클래스가 전부 0행이 된다. ref_ts 는 지우기 전
    train.parquet 의 unix_ts 다.
    """
    t = int(ref_ts[int(len(ref_ts) * (1 - val_frac))])
    return df[df["unix_ts"] < t], df[df["unix_ts"] >= t], t


def fingerprint(tcfg, dcfg, train_path, n_rows, inner_val_frac):
    """이어서 학습해도 되는 판인지 확인하는 지문. 하나라도 다르면 재개하지 않는다."""
    return {
        "train_file": Path(train_path).name,
        "n_rows": int(n_rows),
        "model_name": tcfg["model_name"],
        "batch_size": tcfg["batch_size"],
        "max_epochs": tcfg["max_epochs"],
        "learning_rate": float(tcfg["learning_rate"]),
        "max_length": dcfg["max_length"],
        "input_mode": dcfg["input_mode"],
        "inner_val_frac": float(inner_val_frac),
        "select_metric": tcfg["early_stopping"]["metric"],
    }


def save_state(path, model, opt, sched, epoch, best, best_epoch, hist, steps, fp):
    """저장 도중 전원이 나가도 이전 체크포인트가 살아남도록 임시 파일에 쓰고 교체한다."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best": best,
            "best_epoch": best_epoch,
            "history": hist,
            "steps": steps,
            "fingerprint": fp,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
        },
        tmp,
    )
    tmp.replace(path)


def save_plot(rundir, hist, steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if steps:
        ax1.plot([s[0] for s in steps], [s[1] for s in steps], lw=1, color="#3b6ea5")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss")
    ax1.set_title("training loss")
    ax1.grid(alpha=0.3)

    if hist:
        ep = [h["epoch"] for h in hist]
        ax2.plot(ep, [h["val_macro_f1"] for h in hist], "o-", lw=2, label="val macro F1")
        if "inner_val_unseen_macro_f1" in hist[0]:
            ax2.plot(ep, [h["inner_val_unseen_macro_f1"] for h in hist], "s-", lw=2,
                     color="darkorange", label="inner-val unseen")
            ax2.axhline(E1C_UNSEEN, color="darkorange", ls=":", lw=1, label="E1c unseen 0.7451")
        for c in CLASSES:
            ax2.plot(ep, [h["per_class_f1"][c] for h in hist], "--", lw=1, label=c)
        ax2.set_xticks(ep)
    ax2.axhline(E1C_TEST, color="crimson", ls=":", lw=1, label="E1c test 0.9316")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("validation")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rundir / "progress.png", dpi=110)
    plt.close(fig)


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>{exp} 학습 진행</title>
<style>body{{font-family:system-ui,'Malgun Gothic',sans-serif;margin:24px;background:#fafafa}}
h1{{font-size:20px}} .s{{display:inline-block;margin-right:22px;font-size:14px}}
.s b{{font-size:20px}} table{{border-collapse:collapse;margin-top:14px;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px 10px;text-align:right}} th{{background:#eee}}
img{{max-width:100%;margin-top:14px;border:1px solid #ddd;background:#fff}}
p{{color:#888;font-size:12px}}</style></head><body>
<h1>{exp} &mdash; {state}</h1>
<div class="s">에폭 <b>{epoch}/{max_epochs}</b></div>
<div class="s">스텝 <b>{step:,}</b>/{total_steps:,}</div>
<div class="s">loss <b>{loss:.4f}</b></div>
<div class="s">베스트 {metric} <b>{best:.4f}</b> (에폭 {best_epoch})</div>
<div class="s">경과 <b>{elapsed:.1f}분</b></div>
<div class="s">에폭 남은 시간 <b>{eta:.1f}분</b></div>
<img src="progress.png?t={stamp}">
<table><tr><th>에폭</th><th>train loss</th><th>{metric}</th><th>시간</th>{heads}</tr>
{rows}</table>
<p>5초마다 자동 갱신. 학습이 끝나면 갱신이 멈춥니다.</p></body></html>"""


def render(rundir, exp_id, hist, steps, status):
    save_plot(rundir, hist, steps)
    rows = ""
    for h in hist:
        cells = "".join("<td>{:.3f}</td>".format(h["per_class_f1"][c]) for c in CLASSES)
        rows += (
            "<tr><td>{}</td><td>{:.4f}</td><td><b>{:.4f}</b></td><td>{:.1f}분</td>{}</tr>".format(
                h["epoch"], h["train_loss"], h[status["metric"]], h["minutes"], cells
            )
        )
    heads = "".join(f"<th>{c}</th>" for c in CLASSES)
    (rundir / "progress.html").write_text(
        HTML.format(exp=exp_id, heads=heads, rows=rows, stamp=int(time.time()), **status),
        encoding="utf-8",
    )


@torch.no_grad()
def predict(model, enc, codes, device, batch_size):
    """고유 텍스트만 통과시키고 행으로 되돌린다. 예측은 텍스트에만 의존한다."""
    model.eval()
    ids, mask = enc["input_ids"], enc["attention_mask"]
    out = []
    for i in range(0, len(ids), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(
                input_ids=ids[i : i + batch_size].to(device),
                attention_mask=mask[i : i + batch_size].to(device),
            ).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu())
    model.train()
    return torch.cat(out).numpy()[codes]


def write_preds(probs, df, path):
    pred = np.asarray(CLASSES)[probs.argmax(axis=1)]
    out = pd.DataFrame(probs, columns=["p_" + c for c in CLASSES])
    out["y_true"] = df["label4"].to_numpy()
    out["y_pred"] = pred
    out["msg_norm"] = df["msg_norm"].to_numpy()
    out[["y_true", "y_pred", "msg_norm"] + ["p_" + c for c in CLASSES]].to_parquet(
        path, index=False
    )
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--input", help="기본값은 processed/train_sampled.parquet")
    ap.add_argument("--fresh", action="store_true", help="last.pt 를 무시하고 처음부터")
    ap.add_argument("--inner-val-frac", type=float, default=0.0,
                    help="train 뒤쪽을 시간순으로 떼어 미등장 템플릿 검증셋으로 쓴다 (0 이면 끔)")
    ap.add_argument("--select-metric", choices=METRICS,
                    help="early_stopping.metric 을 이번 실행에만 덮어쓴다")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    tcfg, dcfg = cfg["train"], cfg["dataset"]
    if args.select_metric:
        tcfg["early_stopping"]["metric"] = args.select_metric
    metric = tcfg["early_stopping"]["metric"]
    if metric not in METRICS:
        raise SystemExit(f"early_stopping.metric 이 {metric} 이다. {METRICS} 중 하나여야 한다.")
    if metric.startswith("inner_val") and not args.inner_val_frac:
        raise SystemExit(f"early_stopping.metric 이 {metric} 인데 --inner-val-frac 이 0 이다.")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = Path(cfg["paths"]["processed"])
    rundir = Path(cfg["paths"]["runs"]) / args.exp_id
    rundir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.input) if args.input else processed / "train_sampled.parquet"

    cols = ["msg_norm", "label4"]
    if dcfg["input_mode"] == "with_meta":
        cols += ["component", "level"]
    tr_cols = cols + ["unix_ts"] if args.inner_val_frac else cols
    tr = pd.read_parquet(train_path, columns=tr_cols)
    val = pd.read_parquet(processed / "val.parquet", columns=cols)
    fp = fingerprint(tcfg, dcfg, train_path, len(tr), args.inner_val_frac)
    print(f"train {len(tr):,}줄 ({train_path.name}) | val {len(val):,}줄 | device {device}")

    iv, iv_unseen = None, None
    if args.inner_val_frac:
        ref_ts = pd.read_parquet(processed / "train.parquet", columns=["unix_ts"])["unix_ts"]
        tr, iv, boundary = inner_split(tr, ref_ts.to_numpy(), args.inner_val_frac)
        iv_unseen = ~iv["msg_norm"].isin(set(tr["msg_norm"])).to_numpy()
        dist = dict(iv.loc[iv_unseen, "label4"].value_counts())
        print(f"inner split: 경계 unix_ts {boundary} | "
              f"inner-train {len(tr):,}줄 / inner-val {len(iv):,}줄")
        print(f"  inner-val 미등장 {iv_unseen.sum():,}줄 "
              f"({iv_unseen.mean() * 100:.1f}%)  클래스별 {dist}")

    tok = AutoTokenizer.from_pretrained(tcfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg["model_name"], num_labels=tcfg["num_labels"]
    ).to(device)

    ds = LogDataset(tr, tok, dcfg["max_length"], dcfg["input_mode"])
    bs = tcfg["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)
    val_enc, val_codes = encode_unique(build_text(val, dcfg["input_mode"]), tok, dcfg["max_length"])
    print("고유 텍스트 train {:,}종 / val {:,}종".format(
        len(ds.input_ids), len(val_enc["input_ids"])))
    if iv is not None:
        iv_enc, iv_codes = encode_unique(
            build_text(iv, dcfg["input_mode"]), tok, dcfg["max_length"])

    w = class_weights(tr["label4"].to_numpy(), device) if tcfg["class_weight"] else None
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=tcfg["weight_decay"]
    )
    total = len(loader) * tcfg["max_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(total * tcfg["warmup_ratio"]), total)

    start, best, best_epoch, hist, steps = 1, -1.0, 0, [], []
    last_path = rundir / "last.pt"
    if last_path.exists() and not args.fresh:
        st = torch.load(last_path, map_location=device, weights_only=False)
        if st["fingerprint"] != fp:
            raise SystemExit(
                "{} 의 설정이 지금과 다르다. 이어서 학습할 수 없다.\n  저장됨: {}\n  현재:   {}\n"
                "처음부터 돌리려면 --fresh 를 붙일 것.".format(last_path, st["fingerprint"], fp)
            )
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        best, best_epoch = st["best"], st["best_epoch"]
        hist, steps, start = st["history"], st["steps"], st["epoch"] + 1
        torch.set_rng_state(st["rng_torch"].cpu())
        if st["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in st["rng_cuda"]])
        np.random.set_state(st["rng_numpy"])
        print("이어서 학습: 에폭 {} 까지 완료, 베스트 {:.4f} (에폭 {})".format(
            st["epoch"], best, best_epoch))
        if start > tcfg["max_epochs"]:
            raise SystemExit(
                f"이미 {tcfg['max_epochs']} 에폭을 끝냈다. "
                "다시 돌리려면 --fresh 를 붙일 것."
            )

    print("배치 {} | 에폭당 {:,} 스텝 | 에폭 {}~{}".format(
        bs, len(loader), start, tcfg["max_epochs"]))
    print("진행 상황: {} 을 브라우저로 열어둘 것 (5초마다 갱신)".format(rundir / "progress.html"))
    patience = tcfg["early_stopping"]["patience"]
    log_path = rundir / "train_log.jsonl"
    t_all = time.perf_counter()

    for epoch in range(start, tcfg["max_epochs"] + 1):
        model.train()
        t0, run = time.perf_counter(), 0.0
        for step, batch in enumerate(loader, 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
            loss = lossf(logits.float(), batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()

            if step % PLOT_EVERY == 0 or step == len(loader):
                done = step / len(loader)
                el = time.perf_counter() - t0
                eta = el / done * (1 - done)
                steps.append([(epoch - 1) * len(loader) + step, run / step])
                filled = int(done * 28)
                bar = "█" * filled + "·" * (28 - filled)
                head = f"  에폭 {epoch}/{tcfg['max_epochs']} |{bar}| {done * 100:5.1f}%"
                tail = f"loss {run / step:.4f}  경과 {el / 60:4.1f}분  남은 {eta / 60:4.1f}분"
                print(f"{head}  {tail}", end="\r")
                render(rundir, args.exp_id, hist, steps, {
                    "state": f"에폭 {epoch} 학습 중", "epoch": epoch, "metric": metric,
                    "max_epochs": tcfg["max_epochs"], "step": step, "total_steps": len(loader),
                    "loss": run / step, "best": max(best, 0.0), "best_epoch": best_epoch,
                    "elapsed": (time.perf_counter() - t_all) / 60, "eta": eta / 60})

        probs = predict(model, val_enc, val_codes, device, bs * 4)
        s = score(val["label4"].to_numpy(), np.asarray(CLASSES)[probs.argmax(axis=1)])
        rec = {
            "epoch": epoch,
            "train_loss": run / len(loader),
            "val_macro_f1": s["macro_f1"],
            "minutes": (time.perf_counter() - t0) / 60,
            "per_class_f1": {c: s["per_class"][c]["f1"] for c in CLASSES},
        }
        if iv is not None:
            ipred = np.asarray(CLASSES)[
                predict(model, iv_enc, iv_codes, device, bs * 4).argmax(axis=1)]
            iy = iv["label4"].to_numpy()
            rec["inner_val_macro_f1"] = score(iy, ipred)["macro_f1"]
            rec["inner_val_unseen_macro_f1"] = score(
                iy[iv_unseen], ipred[iv_unseen])["macro_f1"]

        hist.append(rec)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("\n에폭 {}  loss {:.4f}  val macro F1 {:.4f}  {:.1f}분  {}".format(
            epoch, rec["train_loss"], s["macro_f1"], rec["minutes"],
            " ".join("{}={:.3f}".format(c[:4], s["per_class"][c]["f1"]) for c in CLASSES)))
        if iv is not None:
            print("  inner-val {:.4f} | 미등장 {:.4f}".format(
                rec["inner_val_macro_f1"], rec["inner_val_unseen_macro_f1"]))

        if rec[metric] > best:
            best, best_epoch = rec[metric], epoch
            torch.save(model.state_dict(), rundir / "best.pt")
            print("  베스트 갱신 -> {}".format(rundir / "best.pt"))

        save_state(last_path, model, opt, sched, epoch, best, best_epoch, hist, steps, fp)
        print(f"  체크포인트 저장 -> {last_path} (여기서 꺼져도 이어서 학습 가능)")
        render(rundir, args.exp_id, hist, steps, {
            "state": f"에폭 {epoch} 완료", "epoch": epoch, "metric": metric,
            "max_epochs": tcfg["max_epochs"], "step": len(loader), "total_steps": len(loader),
            "loss": rec["train_loss"], "best": best, "best_epoch": best_epoch,
            "elapsed": (time.perf_counter() - t_all) / 60, "eta": 0.0})

        if epoch - best_epoch >= patience:
            print(f"  조기 종료: {patience} 에폭 동안 개선 없음 (베스트 에폭 {best_epoch})")
            break

    model.load_state_dict(torch.load(rundir / "best.pt", map_location=device))
    ckpt = rundir / "checkpoint"
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    print(f"\n베스트 에폭 {best_epoch}, {metric} {best:.4f} -> {ckpt}")

    for split in ["val", "test"]:
        part = pd.read_parquet(processed / (split + ".parquet"), columns=cols)
        enc, codes = encode_unique(build_text(part, dcfg["input_mode"]), tok, dcfg["max_length"])
        write_preds(predict(model, enc, codes, device, bs * 4), part,
                    rundir / f"preds_{split}.parquet")
        print(f"{split:5s} {len(part):>9,}행 -> {rundir}/preds_{split}.parquet")

    render(rundir, args.exp_id, hist, steps, {
        "state": "학습 완료", "epoch": best_epoch, "max_epochs": tcfg["max_epochs"],
        "metric": metric,
        "step": len(loader), "total_steps": len(loader), "loss": hist[-1]["train_loss"],
        "best": best, "best_epoch": best_epoch,
        "elapsed": (time.perf_counter() - t_all) / 60, "eta": 0.0})
    print(f"\n채점: uv run python -m src.evaluate --exp-id {args.exp_id}")


if __name__ == "__main__":
    main()
