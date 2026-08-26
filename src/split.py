"""시간순 70/15/15 분할. 랜덤 분할 금지 (SPEC 3-4)."""

import argparse
from pathlib import Path

import pandas as pd

from src import config

CLASSES = ["normal", "kernel_mem", "kernel_ops", "app"]
SPLITS = ["train", "val", "test"]

# 어느 클래스든 한 구간에서 이 값 미만이면 그 클래스의 F1 은 표본이 얇아
# 해석할 수 없다. 진행하지 말고 사람에게 보고한다 (CLAUDE.md 중단 조건).
MIN_ROWS = 100


def snap(ts, i):
    """같은 타임스탬프가 두 구간에 걸치지 않도록 경계를 다음 타임스탬프로 민다.

    행 번호로 그냥 자르면 한 초에 몰린 수백 줄이 train 과 val 로 쪼개진다.
    BGL 은 초 단위 해상도라 버스트 구간에서 실제로 일어난다.
    """
    while 0 < i < len(ts) and ts[i] == ts[i - 1]:
        i += 1
    return i


def cut(df, ratios, by="unix_ts"):
    """by 오름차순 정렬 후 앞에서부터 ratios 대로 자른다."""
    df = df.sort_values(by, kind="stable").reset_index(drop=True)
    ts = df[by].to_numpy()
    n = len(df)
    i1 = snap(ts, int(n * ratios[0]))
    i2 = snap(ts, int(n * (ratios[0] + ratios[1])))
    if not 0 < i1 < i2 < n:
        raise SystemExit(
            f"경계가 무너졌다: i1={i1} i2={i2} n={n}. 타임스탬프가 한쪽에 몰렸는지 확인할 것."
        )
    return {"train": df.iloc[:i1].copy(), "val": df.iloc[i1:i2].copy(), "test": df.iloc[i2:].copy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)["split"]
    by, ratios = cfg["by"], cfg["ratios"]

    df = pd.read_parquet(args.input)
    parts = cut(df, ratios, by)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(outdir / f"{name}.parquet", index=False)

    total = sum(len(p) for p in parts.values())
    print(f"입력 {len(df):,}줄 -> 분할 {total:,}줄  (기준 {by}, 비율 {ratios})")

    print("\n구간별 기간")
    for name in SPLITS:
        p = parts[name]
        lo, hi = p[by].iloc[0], p[by].iloc[-1]
        d0 = pd.to_datetime(lo, unit="s").strftime("%Y-%m-%d")
        d1 = pd.to_datetime(hi, unit="s").strftime("%Y-%m-%d")
        print(f"  {name:<5}: {len(p):>9,}줄  ({len(p) / total * 100:5.2f}%)  {d0} ~ {d1}")

    print("\n구간별 클래스 분포")
    print(f"  {'':<11}" + "".join(f"{n:>21}" for n in SPLITS))
    for c in CLASSES:
        row = f"  {c:<11}"
        for name in SPLITS:
            p = parts[name]
            n = int((p["label4"] == c).sum())
            row += f"{n:>12,} ({n / len(p) * 100:5.2f}%)"
        print(row)

    # 검증 1: 시간 경계가 겹치지 않는가
    print("\n시간 경계")
    ok_bound = True
    for a, b in zip(SPLITS, SPLITS[1:]):
        hi, lo = parts[a][by].max(), parts[b][by].min()
        ok = hi < lo
        ok_bound &= ok
        print(f"  {a}.max={hi} < {b}.min={lo}  {'OK' if ok else '겹침!'}")

    # 검증 2: 한 줄이 두 구간에 들어가지 않는가
    print(f"  행 수 합계 {total:,} == 입력 {len(df):,}  {'OK' if total == len(df) else '불일치!'}")

    # 검증 3: 클래스별 최소 표본
    print(f"\n최소 표본 게이트 (구간당 {MIN_ROWS}줄)")
    thin = []
    for c in CLASSES:
        for name in SPLITS:
            n = int((parts[name]["label4"] == c).sum())
            if n < MIN_ROWS:
                thin.append((c, name, n))
    if thin:
        for c, name, n in thin:
            print(f"  {c} / {name}: {n}줄")
    else:
        print("  통과 — 모든 클래스가 세 구간 모두 기준 이상")

    print(f"\n출력: {outdir}/{{train,val,test}}.parquet")

    if not ok_bound or total != len(df):
        raise SystemExit("분할 검증 실패. 위 출력을 확인할 것.")
    if thin:
        raise SystemExit(
            "표본이 얇은 클래스가 있다. 다음 단계로 넘어가기 전에 사람에게 보고할 것 "
            "(CLAUDE.md 중단 조건)."
        )


if __name__ == "__main__":
    main()
