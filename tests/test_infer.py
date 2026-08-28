import pandas as pd

from src.infer import traffic_view


def row(z, watch):
    return pd.Series({"traffic_z": z, "traffic_watch": watch})


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
