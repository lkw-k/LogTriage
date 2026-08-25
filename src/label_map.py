"""원본 41 카테고리 -> 4클래스. 매핑은 configs/base.yaml 이 정본이다."""

import argparse
from pathlib import Path

import pandas as pd

from src import config

CLASSES = ["normal", "kernel_mem", "kernel_ops", "app"]


def build_lookup(label_map):
    lookup = {}
    for cls, cats in label_map.items():
        for cat in cats:
            lookup[cat] = cls
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    lookup = build_lookup(cfg["label_map"])
    holdout = set(cfg["holdout_rare"])

    df = pd.read_parquet(args.input)
    seen = set(df["label"].unique())

    unmapped = seen - set(lookup) - holdout
    if unmapped:
        raise SystemExit(
            f"매핑되지 않은 카테고리 {len(unmapped)}종: {sorted(unmapped)}\n"
            "configs/base.yaml 의 label_map 또는 holdout_rare 에 추가할 것."
        )

    is_holdout = df["label"].isin(holdout)
    df_hold = df[is_holdout].copy()
    df_main = df[~is_holdout].copy()
    df_main["label4"] = df_main["label"].map(lookup)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_main.to_parquet(out, index=False)
    hold_out = out.with_name(out.stem.replace("_labeled", "") + "_holdout.parquet")
    df_hold.to_parquet(hold_out, index=False)

    counts = df_main["label4"].value_counts()
    total = len(df_main)
    alerts = total - counts.get("normal", 0)
    print("미매핑 카테고리 : 0종")
    print(f"학습 대상       : {total:,}줄")
    for c in CLASSES:
        n = counts.get(c, 0)
        share = f"{n / alerts * 100:5.1f}% of alerts" if c != "normal" else ""
        print(f"  {c:<11}: {n:>9,}  ({n / total * 100:5.2f}%)  {share}")
    print(f"holdout_rare    : {len(df_hold):,}줄 -> {hold_out}")
    print(f"출력            : {out}")


if __name__ == "__main__":
    main()
