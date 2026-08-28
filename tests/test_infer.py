import pandas as pd

from src.infer import judge, traffic_view

CAL = {"min_count": 5,
       "class_ratio_mean": {"kernel_mem": 0.003, "kernel_ops": 0.0008, "app": 0.0005},
       "class_ratio_std": {"kernel_mem": 0.002, "kernel_ops": 0.0006, "app": 0.0004}}
ICFG = {"window": "1min", "z_warning": 3.0, "z_critical": 5.0,
        "unknown_ratio_warning": 0.10}


def row(z, watch, down=False, run=0, **counts):
    base = {"normal": 0, "kernel_mem": 0, "kernel_ops": 0, "app": 0, "unknown": 0}
    return pd.Series({**base, **counts, "traffic_z": z, "traffic_watch": watch,
                      "traffic_outage": down, "traffic_zero_run": run})


def test_z_is_withheld_when_not_watchable():
    """감시 대상이 아닌 구간의 z 는 출력하지 않는다.

    긴 무로그 구간 뒤에는 잔차 척도가 0 에 수렴해 z 가 발산한다. NASA 62일 실측에서
    max|z| 가 5.26e+35 였고, 그 분들은 count>0 이라 run() 의 total==0 스킵을 통과해
    windows.jsonl 에 그대로 찍혔다. 판정은 watchable 게이트가 막지만 출력은 못 막았다.
    """
    got = traffic_view(row(5.255e35, False), 45, 0.0, 3.0)
    assert got["z"] is None
    assert got["flag"] == "low_volume"


def test_z_is_reported_when_watchable():
    got = traffic_view(row(3.81, True), 1204, 800.23, 3.0)
    assert got == {"count": 1204, "baseline": 800.2, "z": 3.81, "flag": "spike"}


def test_outage_is_critical_even_though_traffic_watch_is_off():
    """완전 정지에서는 기준선이 내려가 traffic_watch 가 꺼진다.

    무응답 검사를 watchable 조기 반환 뒤에 두면 이 경우에 절대 도달하지 못한다.
    NASA 허리케인이 정확히 이 모양이었다 — 정지 25분 만에 감시가 꺼지고 남은
    37.6시간이 판정에서 빠졌다.
    """
    level, reasons = judge(row(0.0, watch=False, down=True, run=2263), 0, CAL, ICFG)
    assert level == "critical"
    assert reasons == ["무응답 2263분 연속 — 서비스 정지 의심"]


def test_quiet_bucket_without_outage_stays_ok():
    level, reasons = judge(row(-9.9, watch=False), 0, CAL, ICFG)
    assert (level, reasons) == ("ok", [])


def test_outage_flag_wins_over_low_volume():
    got = traffic_view(row(0.0, watch=False, down=True, run=2263), 0, 0.4, 3.0)
    assert got["flag"] == "outage"
    assert got["z"] is None
