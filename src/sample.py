"""train 구성 조정. val/test 는 절대 건드리지 않는다 (CLAUDE.md 절대규칙 2)."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src import config

CLASSES = ["normal", "kernel_mem", "kernel_ops", "app"]
STRATEGIES = ["none", "ratio", "balanced", "dedup", "cap"]


def apply_strategy(df, strategy, *, ratio=3, cap=2000, seed=42):
    """train 만 받는다. 호출자가 val/test 를 넘기지 않도록 할 것."""
    if strategy == "none":
        return df

    if strategy == "dedup":
        # E6. 완전 중복 줄 제거. 클래스별 고유 템플릿이 9~24종뿐이라
        # 알럿이 두 자릿수로 줄어든다. 비교용이며 기본값으로 쓰지 말 것.
        return df.drop_duplicates(["label4", "msg_norm"])

    if strategy == "cap":
        # 템플릿당 상한. 194.7배 중복만 깎고 클래스 비율은 거의 유지된다.
        shuffled = df.sample(frac=1.0, random_state=seed)
        keep = shuffled.groupby(["label4", "msg_norm"], observed=True).cumcount() < cap
        return shuffled[keep].sort_index()

    if strategy == "ratio":
        alerts = df[df["label4"] != "normal"]
        normal = df[df["label4"] == "normal"]
        target = min(len(normal), len(alerts) * ratio)
        return pd.concat([alerts, normal.sample(target, random_state=seed)]).sort_index()

    if strategy == "balanced":
        n = df["label4"].value_counts().min()
        parts = [g.sample(n, random_state=seed) for _, g in df.groupby("label4", observed=True)]
        return pd.concat(parts).sort_index()

    raise SystemExit(f"모르는 전략: {strategy}. {STRATEGIES} 중 하나여야 한다.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strategy", choices=STRATEGIES)
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    scfg = cfg["sample"]
    strategy = args.strategy or scfg["strategy"]
    seed = cfg["seed"]

    src = Path(args.input)
    # 규칙 2 검증용. sample 은 이 파일들을 열지도 않는다.
    others = {n: src.with_name(f"{n}.parquet") for n in ["val", "test"]}
    before = {n: sha256(p) for n, p in others.items() if p.exists()}

    df = pd.read_parquet(src)
    out = apply_strategy(
        df, strategy, ratio=scfg["ratio_normal_to_alert"], cap=scfg["cap_per_template"], seed=seed
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    print(f"전략 {strategy}  (seed {seed})")
    print(f"train {len(df):,} -> {len(out):,}줄  ({len(out) / len(df) * 100:.2f}%)")
    print(f"  {'':<11}{'before':>22}{'after':>22}")
    for c in CLASSES:
        b = int((df["label4"] == c).sum())
        a = int((out["label4"] == c).sum())
        print(
            f"  {c:<11}{b:>12,} ({b / len(df) * 100:5.2f}%)"
            f"{a:>12,} ({a / len(out) * 100:5.2f}%)"
        )
    print(f"고유 템플릿 {df['msg_norm'].nunique():,} -> {out['msg_norm'].nunique():,}")

    after = {n: sha256(p) for n, p in others.items() if p.exists()}
    print("\nval/test 무결성 (규칙 2)")
    for n in before:
        ok = before[n] == after[n]
        print(f"  {n}.parquet {before[n][:16]}...  {'변경 없음' if ok else '변경됨!'}")
    if before != after:
        raise SystemExit("val/test 가 변경되었다. 절대규칙 2 위반.")
    print(f"\n출력: {args.output}")


if __name__ == "__main__":
    main()
