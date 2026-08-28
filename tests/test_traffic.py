import numpy as np
import pandas as pd

from src.traffic.detect import ewma_z, flag, outage, watchable, zero_run


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


def test_zero_run_counts_the_current_stretch():
    x = series([5, 0, 0, 3, 0, 0, 0])
    assert zero_run(x).tolist() == [0, 1, 2, 0, 1, 2, 3]


def fired(values, min_minutes=5, min_baseline=20):
    x = series(values)
    baseline, _ = ewma_z(x, span=60)
    return outage(x, baseline, min_baseline, min_minutes)


def test_outage_fires_after_a_healthy_baseline_goes_silent():
    got = fired([50] * 200 + [0] * 10)
    assert not got.iloc[:204].any(), "5분 미만인데 발화했다"
    assert got.iloc[204:].all(), "5분째부터 발화해야 한다"


def test_outage_does_not_fire_in_a_quiet_stretch():
    """BGL 형태 회귀. 무로그 분이 90% 인 로그에서 조용한 구간은 장애가 아니다.

    기준선이 계속 min_baseline 아래면 애초에 감시 대상이 아니다.
    """
    assert not fired([3] * 200 + [0] * 500).any()


def test_outage_ignores_a_short_gap():
    assert not fired([50] * 200 + [0] * 4 + [50] * 10).any()


def test_outage_guard_uses_the_baseline_before_the_gap():
    """가드는 현재 기준선이 아니라 0 이 시작되기 직전의 기준선을 봐야 한다.

    EWMA 는 정지를 새 평시로 학습하므로 현재 기준선은 정지가 길어질수록 0 으로
    내려간다. 현재 기준선(=watchable)을 쓰면 NASA 허리케인에서 정지 25분 만에
    감시가 꺼져 남은 37.6시간을 놓쳤다. 이 테스트가 그 실패를 고정한다.
    """
    x = series([50] * 200 + [0] * 600)
    baseline, _ = ewma_z(x, span=60)

    assert baseline.iloc[-1] < 20, "전제: 정지가 길어져 현재 기준선이 가드 아래로 내려간다"
    assert not watchable(baseline, 20).iloc[-1], "전제: watchable 은 이미 꺼져 있다"
    assert outage(x, baseline, 20, 5).iloc[-1], "현재 기준선을 쓰면 여기서 실패한다"
