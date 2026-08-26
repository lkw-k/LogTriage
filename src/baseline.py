"""E1. TF-IDF + 로지스틱 회귀.

로그 이상탐지에서는 단순 기법이 딥러닝을 이기는 경우가 자주 보고된다.
E2(BERT)가 이 점수를 못 이기면 딥러닝을 쓸 이유를 다시 생각해야 한다 (SPEC 4-2).

출력은 runs/<exp_id>/preds_{val,test}.parquet 이고 evaluate 가 채점한다.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config

CLASSES = ["normal", "kernel_mem", "kernel_ops", "app"]


def collapse(df):
    """같은 (메시지, 라벨) 을 묶고 개수를 sample_weight 로 준다.

    train 은 고유 템플릿 16,945종에 194.7배 중복이다. 묶어서 가중치를 주면
    가중 손실이 원본과 완전히 동일하면서 행 수가 200배 줄어든다.
    """
    g = df.groupby(["msg_norm", "label4"], observed=True).size().reset_index(name="w")
    return g["msg_norm"].to_numpy(), g["label4"].to_numpy(), g["w"].to_numpy().astype(float)


def balanced_class_weight(y, w):
    """w_c = N / (K * n_c). 묶기 전 원본 행 수(w의 합)로 계산해야 한다.

    묶인 행을 세면 kernel_mem 이 9종이라 비율이 완전히 달라진다.
    """
    total = w.sum()
    out = {}
    for c in CLASSES:
        n = w[y == c].sum()
        if n > 0:
            out[c] = total / (len(CLASSES) * n)
    return out


def fit(all_texts, texts, y, w, seed):
    """IDF 는 원본 행 전체에서, 회귀는 묶은 행 + sample_weight 로.

    묶은 행으로 벡터라이저까지 학습하면 문서 빈도가 달라져 원본과 다른 모델이 된다
    (`generating core.[NUM]` 하나가 train 의 51.7% 다). IDF 를 원본에서 뽑으면
    전체 행으로 학습한 것과 결과가 완전히 같으면서 회귀만 200배 빨라진다.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit(all_texts)
    x = vec.transform(texts)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight=balanced_class_weight(y, w),
        random_state=seed,
    )
    clf.fit(x, y, sample_weight=w)
    return vec, clf


def predict_frame(vec, clf, df):
    proba = clf.predict_proba(vec.transform(df["msg_norm"]))
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in clf.classes_])
    out["y_true"] = df["label4"].to_numpy()
    out["y_pred"] = clf.classes_[proba.argmax(axis=1)]
    out["msg_norm"] = df["msg_norm"].to_numpy()
    return out[["y_true", "y_pred", "msg_norm"] + [f"p_{c}" for c in CLASSES]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="E1")
    ap.add_argument("--input", help="기본값은 processed/train_sampled.parquet")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    processed = Path(cfg["paths"]["processed"])
    seed = cfg["seed"]
    train_path = Path(args.input) if args.input else processed / "train_sampled.parquet"

    df = pd.read_parquet(train_path, columns=["msg_norm", "label4"])
    texts, y, w = collapse(df)
    print(f"train {len(df):,}줄 -> 묶어서 {len(texts):,}행 (sample_weight 합 {w.sum():,.0f})")

    t = time.perf_counter()
    vec, clf = fit(df["msg_norm"], texts, y, w, seed)
    print(f"학습 {time.perf_counter() - t:.1f}초 | 피처 {len(vec.vocabulary_):,}개")
    print("class_weight:", {k: round(v, 3) for k, v in balanced_class_weight(y, w).items()})

    rundir = Path(cfg["paths"]["runs"]) / args.exp_id
    rundir.mkdir(parents=True, exist_ok=True)
    for split in ["val", "test"]:
        part = pd.read_parquet(processed / f"{split}.parquet", columns=["msg_norm", "label4"])
        out = predict_frame(vec, clf, part)
        out.to_parquet(rundir / f"preds_{split}.parquet", index=False)
        print(f"{split:5s} {len(out):>9,}행 -> {rundir}/preds_{split}.parquet")

    print(f"\n채점: uv run python -m src.evaluate --exp-id {args.exp_id}")


if __name__ == "__main__":
    main()
