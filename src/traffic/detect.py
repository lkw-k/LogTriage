"""1분 발생량의 EWMA z-score (SPEC 3-9).

**척도는 과거 잔차로만 잡는다.** 현재 잔차를 자기 분모에 넣으면 큰 스파이크가
자기 표준편차를 같이 키워 z 에 상한이 생긴다. BGL val 에서 z 최대가 5.6 이었고,
임계를 6 으로 올리면 어떤 급증도 영영 안 잡혔다.

**최소 기준선 가드가 필요하다.** BGL 은 val 96,126분 중 로그가 있는 분이 9,530개
(90%가 0건)다. 조용한 구간에서는 잔차 표준편차가 0 에 붙어 z 가 폭주한다
(가드 없이 |z|>3 이 하루 20건). 비율 판정의 min_count 와 같은 취지다.
"""

import numpy as np


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
