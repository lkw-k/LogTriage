import numpy as np
import pandas as pd

from src.calibrate import bucket_ratios, pick_threshold
from src.dataset import CLASSES


def make_proba(rows):
    """(정답, 예측클래스, 신뢰도) -> (proba, y_true). 나머지 확률은 균등 배분한다."""
    proba, y = [], []
    for true_c, pred_c, conf in rows:
        p = np.full(len(CLASSES), (1.0 - conf) / (len(CLASSES) - 1))
        p[CLASSES.index(pred_c)] = conf
        proba.append(p)
        y.append(true_c)
    return np.array(proba), np.array(y)


def test_pick_threshold_rejects_wiping_out_a_class():
    """클래스 천장보다 높은 임계는 고르지 않는다.

    E2w 에서 실제로 밟은 버그다. normal 은 0.9999 를 넘고 app 은 0.999 가 천장이라
    임계 0.9999 가 app 을 통째로 지운다. 살아남은 클래스만 평균하면 그게 1등이 된다.
    """
    rows = [("normal", "normal", 0.99999)] * 96 + [("app", "app", 0.999)] * 4
    thr, table, best = pick_threshold(*make_proba(rows))

    assert thr < 0.9999, "app 을 전부 지우는 임계를 골랐다"
    wipe = [r for r in table if r[0] == 0.9999][0]
    assert wipe[3] < len(CLASSES), "0.9999 에서 클래스가 사라지지 않았다면 픽스처가 틀렸다"
    assert wipe[2] < best[2], "클래스를 지운 후보가 더 높은 점수를 받았다"


def test_pick_threshold_respects_unknown_budget():
    """unknown 이 5% 를 넘는 임계는 후보에서 빠진다."""
    rows = [("normal", "normal", 0.9)] * 80 + [("app", "app", 0.7)] * 20
    thr, table, _ = pick_threshold(*make_proba(rows))
    assert thr <= 0.7, "app 20% 를 unknown 으로 보내는 임계를 골랐다"


def test_bucket_ratios_excludes_window_and_small_buckets():
    """장애 구간과 min_count 미만 버킷이 기준선에서 빠진다."""
    # 9시 = 정상 10건, 10시 = 장애 10건(전부 kernel_mem), 11시 = 2건뿐.
    # unix_ts 는 UTC 로 해석되므로(bucket_ratios 의 to_datetime(unit="s")) 제외 구간
    # 문자열도 같은 변환으로 만든다. 로컬 시간대를 타면 테스트가 환경마다 달라진다.
    base = int(pd.Timestamp("2005-06-12 00:00", tz="UTC").timestamp())
    hour = 3600
    ts, pred = [], []
    for h, cls, n in [(9, "normal", 10), (10, "kernel_mem", 10), (11, "normal", 2)]:
        ts += [base + h * hour] * n
        pred += [cls] * n
    label = str(pd.to_datetime(base + 10 * hour, unit="s"))
    end = str(pd.to_datetime(base + 11 * hour, unit="s"))
    ratios, n_all, n_win, n_kept = bucket_ratios(
        pd.Series(ts), np.array(pred), "1min", [(label, end)], min_count=5)

    assert n_all == 3
    assert n_win == 1, "장애 구간 버킷이 빠지지 않았다"
    assert n_kept == 1, "2건짜리 버킷이 기준선에 남았다"
    assert ratios["kernel_mem"].max() == 0.0, "장애 버킷이 기준선에 섞였다"
