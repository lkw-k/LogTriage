import pandas as pd

from src.traffic.parse_nasa import longest_zero_run, per_minute


def frame(times):
    return pd.DataFrame({"time": times, "response": 200, "bytes": 1})


def test_resample_sorts_first():
    """원본 nasa.csv 는 호스트 알파벳순이라 time 이 뒤죽박죽이다.

    정렬 없이 resample 하면 pandas 가 뒤섞인 인덱스로 잘못된 구간을 만든다.
    """
    base = 804571200
    shuffled = [base + 300, base, base + 60, base + 120]
    counts, _ = per_minute(frame(shuffled))
    assert counts.index.is_monotonic_increasing
    assert counts.tolist() == [1, 1, 1, 0, 0, 1]


def test_zero_fill_stops_at_last_record():
    """마지막 기록 뒤로 0 을 채우면 존재하지 않는 정지 구간이 생긴다."""
    base = 804571200
    counts, _ = per_minute(frame([base, base + 120]))
    assert len(counts) == 3
    assert counts.index[-1] == pd.Timestamp(base + 120, unit="s")


def test_longest_zero_run():
    idx = pd.date_range("1995-07-01", periods=7, freq="1min")
    assert longest_zero_run(pd.Series([1, 0, 0, 1, 0, 0, 0], index=idx)) == 3
    assert longest_zero_run(pd.Series([1, 1, 1, 1, 1, 1, 1], index=idx)) == 0
