"""nasa.csv -> 1분 단위 발생량 parquet (SPEC 3-9).

**원본은 호스트 알파벳순으로 정렬돼 있다** (`***.novo.dk` ... `zzzzzzzz.mindspring.com`).
시간순이 아니므로 리샘플 전에 반드시 `time` 으로 정렬한다.

**0 채움은 [첫 기록 분, 마지막 기록 분] 안에서만 한다.** 날짜 경계로 반올림하면 트레이스가
끝난 뒤에 0 이 붙어, 실제로는 없는 정지 구간이 만들어진다 (개발 중 1,200분짜리 유령 정지를
만들어냈다).

`time` 은 진짜 UTC epoch 이고 NASA 원본 로그의 타임스탬프는 EDT(UTC-4) 다. parquet 은 UTC
로 저장하고, 원 로그와 대조할 수 있도록 stdout 에만 EDT 를 병기한다.
"""

import argparse
from pathlib import Path

import pandas as pd

EDT_OFFSET = pd.Timedelta(hours=4)   # NASA 원본 로그 타임스탬프는 -0400


def per_minute(df):
    """-> (분 단위 건수 Series, 정렬된 df). 인덱스는 UTC, 빈 분은 0 으로 채운다."""
    df = df.sort_values("time")
    ts = pd.to_datetime(df["time"].to_numpy(), unit="s")
    return pd.Series(1, index=ts).resample("1min").sum(), df


def longest_zero_run(counts):
    is_zero = counts == 0
    if not is_zero.any():
        return 0
    return int(is_zero.groupby((~is_zero).cumsum()).sum().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # url 컬럼은 읽지 않는다 (230MB 중 대부분). Unnamed: 0 도 usecols 로 자연히 빠진다.
    df = pd.read_csv(args.input, usecols=["time", "response", "bytes"])
    total = len(df)
    df = df.dropna(subset=["response", "bytes"])
    dropped = total - len(df)

    counts, df = per_minute(df)
    counts.index.name = "minute"
    counts.rename("count").reset_index().to_parquet(out, index=False)

    lo, hi = df["time"].min(), df["time"].max()
    lo, hi = pd.Timestamp(lo, unit="s"), pd.Timestamp(hi, unit="s")
    n_zero = int((counts == 0).sum())
    print(f"총 행 수      : {total:,}")
    print(f"결측 제거     : {dropped:,} ({dropped / total * 100:.4f}%)")
    print(f"시각 범위 UTC : {lo} ~ {hi}")
    print(f"시각 범위 EDT : {lo - EDT_OFFSET} ~ {hi - EDT_OFFSET}")
    print(f"분 버킷       : {len(counts):,} ({len(counts) / 1440:.1f}일)")
    print(f"합계 일치     : {int(counts.sum()):,} vs {len(df):,} "
          f"-> {'OK' if int(counts.sum()) == len(df) else '불일치'}")
    print(f"0건 분        : {n_zero:,} ({n_zero / len(counts) * 100:.2f}%)")
    print(f"최장 연속 0   : {longest_zero_run(counts):,}분")
    print(f"출력          : {out}")


if __name__ == "__main__":
    main()
