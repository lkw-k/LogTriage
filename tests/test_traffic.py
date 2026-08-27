import numpy as np
import pandas as pd

from src.traffic.detect import ewma_z, flag, watchable


def series(values):
    idx = pd.date_range("2005-06-03", periods=len(values), freq="1min")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_z_is_not_capped_by_its_own_spike():
    """척도를 과거 잔차로만 잡아야 큰 급증이 큰 z 로 나온다.

    현재 잔차를 분모에 포함하면 스파이크가 자기 표준편차를 같이 키워 z 에 상한이
    생긴다. BGL val 에서 z 최대가 5.6 이라 임계 6 에서는 아무것도 안 잡혔다.
    """
    x = series([100] * 200 + [1] * 5 + [100] * 5 + [100000])
    _, z = ewma_z(x, span=60)
    assert z.iloc[-1] > 10, f"급증 z 가 {z.iloc[-1]:.1f} 로 눌렸다"


def test_baseline_uses_only_the_past():
    """기준선이 현재 값을 포함하면 급증이 기준선을 같이 끌어올린다."""
    x = series([10] * 100 + [500])
    baseline, _ = ewma_z(x, span=60)
    assert baseline.iloc[-1] < 20, "급증 시점의 기준선이 그 급증을 반영했다"


def test_watchable_guards_quiet_stretches():
    baseline = series([0, 3, 19, 20, 100])
    got = watchable(baseline, 20).tolist()
    assert got == [False, False, False, True, True]


def test_flag_directions():
    assert flag(4.0, 3.0) == "spike"
    assert flag(-4.0, 3.0) == "drop"
    assert flag(1.0, 3.0) == "normal"
