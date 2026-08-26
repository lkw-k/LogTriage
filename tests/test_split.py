import pandas as pd

from src.split import cut

RATIOS = [0.70, 0.15, 0.15]


def make(ts):
    return pd.DataFrame(
        {"unix_ts": list(ts), "label4": ["normal"] * len(ts), "row_id": range(len(ts))}
    )


def test_boundaries_are_time_ordered():
    parts = cut(make(range(100)), RATIOS)
    assert parts["train"]["unix_ts"].max() < parts["val"]["unix_ts"].min()
    assert parts["val"]["unix_ts"].max() < parts["test"]["unix_ts"].min()


def test_no_row_in_two_splits():
    parts = cut(make(range(100)), RATIOS)
    ids = [set(p["row_id"]) for p in parts.values()]
    assert sum(len(s) for s in ids) == 100
    assert set.union(*ids) == set(range(100))


def test_unsorted_input_is_sorted_before_cutting():
    parts = cut(make(list(range(50, 100)) + list(range(50))), RATIOS)
    assert parts["train"]["unix_ts"].is_monotonic_increasing
    assert parts["train"]["unix_ts"].max() < parts["val"]["unix_ts"].min()


def test_same_timestamp_never_straddles_a_boundary():
    """한 초에 몰린 줄이 두 구간으로 쪼개지면 안 된다.

    행 번호로 그냥 자르면 70% 지점이 ts=1 뭉치 한가운데(70번째)에 떨어져
    같은 초의 줄이 train 과 val 에 나뉜다.
    """
    ts = [1] * 80 + [2] * 10 + list(range(3, 13))
    parts = cut(make(ts), RATIOS)
    assert len(parts["train"]) == 80
    assert set(parts["train"]["unix_ts"]) == {1}
    assert parts["val"]["unix_ts"].min() == 2
