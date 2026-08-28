"""1분 발생량의 EWMA z-score (SPEC 3-9).

**척도는 과거 잔차로만 잡는다.** 현재 잔차를 자기 분모에 넣으면 큰 스파이크가
자기 표준편차를 같이 키워 z 에 상한이 생긴다. BGL val 에서 z 최대가 5.6 이었고,
임계를 6 으로 올리면 어떤 급증도 영영 안 잡혔다.

**최소 기준선 가드가 필요하다.** BGL 은 val 96,126분 중 로그가 있는 분이 9,530개
(90%가 0건)다. 조용한 구간에서는 잔차 표준편차가 0 에 붙어 z 가 폭주한다
(가드 없이 |z|>3 이 하루 20건). 비율 판정의 min_count 와 같은 취지다.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def ewma_z(counts, span):
    """-> (baseline, z). counts 는 시간순 1분 건수 시리즈. 둘 다 과거만 본다."""
    baseline = counts.ewm(span=span, adjust=False).mean().shift(1)
    resid = counts - baseline
    scale = resid.ewm(span=span, adjust=False).std().shift(1)
    z = resid / scale.replace(0.0, np.nan)
    return baseline.bfill(), z.fillna(0.0)


def flag(z, threshold):
    if z > threshold:
        return "spike"
    if z < -threshold:
        return "drop"
    return "normal"


def watchable(baseline, min_baseline):
    """기준선이 이 값 미만인 구간에서는 트래픽 판정을 하지 않는다."""
    return baseline >= min_baseline


def zero_run(counts):
    """각 분에서 끝나는 연속 0 구간의 길이. 0건이 아닌 분은 0."""
    is_zero = counts == 0
    return is_zero.groupby((~is_zero).cumsum()).cumsum()


def outage(counts, baseline, min_baseline, min_minutes):
    """연속 0 이 min_minutes 이상이고, 0 이 시작되기 직전 기준선이 살아 있으면 무응답.

    **가드는 현재 기준선이 아니라 0 구간 직전의 기준선을 본다.** EWMA 는 정지를 새
    평시로 학습하므로 현재 기준선은 정지가 길어질수록 0 으로 내려간다. `watchable` 을
    그대로 쓰면 NASA 허리케인에서 정지 25분 만에 감시가 꺼져 남은 37.6시간을 놓쳤다.

    z-score 로는 이 상황을 표현할 수 없다. 잔차도 척도도 같이 줄어 z 가 -2.5 에서
    멈춘다 (NASA 실측 최저 -2.48, 임계 -3 미달). z 는 "평소보다 적다"를 재는 도구지
    "아무것도 안 온다"를 재는 도구가 아니다. experiments.md 2026-08-28 항목 참조.
    """
    before = baseline.where(counts != 0).ffill()
    return (zero_run(counts) >= min_minutes) & (before >= min_baseline)


def events(fired):
    """연속 발화 구간의 시작 시각 목록. 분 단위 플래그를 사람이 읽는 단위로 접는다."""
    return fired.index[fired & ~fired.shift(1, fill_value=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="parse_nasa 가 만든 분 단위 parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--expect-outage", nargs=2, metavar=("START", "END"),
                    help="이 구간(UTC)이 무응답으로 잡히는지 PASS/FAIL 로 검증")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)["traffic"]
    counts = pd.read_parquet(args.input).set_index("minute")["count"].astype(float)

    baseline, z = ewma_z(counts, cfg["ewma_span"])
    watch = watchable(baseline, cfg["min_baseline"])
    down = outage(counts, baseline, cfg["min_baseline"], cfg["outage_minutes"])
    spike, drop = watch & (z > cfg["z_threshold"]), watch & (z < -cfg["z_threshold"])

    out = pd.DataFrame({
        "minute": counts.index, "count": counts.astype(int).to_numpy(),
        "baseline": baseline.round(2).to_numpy(),
        # 감시 대상이 아닌 구간의 z 는 발산한다 (infer.traffic_view 와 같은 이유).
        "z": z.where(watch).round(2).to_numpy(),
        "flag": np.select([down, spike, drop], ["outage", "spike", "drop"], "normal"),
    })
    out = out[out["flag"] != "normal"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    days = len(counts) / 1440
    print(f"버킷 {len(counts):,} | {days:.1f}일 | watchable {int(watch.sum()):,} "
          f"({watch.mean() * 100:.1f}%)")
    for name, fired in [("급증 spike", spike), ("급감 drop", drop), ("무응답 outage", down)]:
        ev = events(fired)
        print(f"{name:<14} 발화 {int(fired.sum()):>6}분 | 이벤트 {len(ev):>4}건 "
              f"| {len(ev) / days:.2f}건/일")

    if args.expect_outage:
        lo, hi = (pd.Timestamp(t) for t in args.expect_outage)
        seg = down.loc[lo:hi]
        hit = int(seg.sum())
        first = seg[seg].index[0] if hit else None
        print(f"\n기대 구간 {lo} ~ {hi} ({len(seg):,}분)")
        print(f"  발화 {hit:,}분 ({hit / len(seg) * 100:.1f}%) | 첫 발화 {first}")
        print(f"  최저 z {z.loc[lo:hi].min():.2f} | watchable {int(watch.loc[lo:hi].sum())}분")
        print(f"  => {'PASS' if hit else 'FAIL'}")
    print(f"\n출력: {args.output} ({len(out):,}행)")


if __name__ == "__main__":
    main()
