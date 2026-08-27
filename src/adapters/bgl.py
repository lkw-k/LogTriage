"""BGL 포맷 어댑터. 원본 로그 줄 -> (unix_ts, message) (SPEC 5-2).

분류기는 BGL 포맷으로 학습했다. 다른 포맷을 넣으려면 이 파일과 같은 모양으로
어댑터를 추가한다. 어댑터 없이 다른 포맷을 넣으면 결과는 무의미하다.

파싱 실패한 줄은 버리지 않고 세어서 돌려준다. 실패율 자체가 이상 신호일 수 있다.
"""

from src.parse_bgl import parse_line

NAME = "bgl"


def parse(lines):
    """-> (rows, unparseable). rows 는 (unix_ts, message) 리스트."""
    rows, bad = [], []
    for line in lines:
        if not line.strip():
            continue
        parts = parse_line(line)
        if parts is None:
            bad.append(line.rstrip())
            continue
        rows.append((parts[1], parts[9]))
    return rows, bad
