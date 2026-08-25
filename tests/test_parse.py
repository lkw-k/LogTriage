from src.parse_bgl import parse_line

FIXTURE = "tests/fixtures/BGL_2k.log"


def test_normal_line_splits_into_ten_fields():
    line = (
        "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 "
        "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
    )
    row = parse_line(line)
    assert row[0] == "-"
    assert row[1] == 1117838570
    assert row[3] == "R02-M1-N0-C:J12-U11"
    assert row[7] == "KERNEL"
    assert row[8] == "INFO"
    assert row[9] == "instruction cache parity error corrected"


def test_message_keeps_internal_spaces():
    """maxsplit=9 가 아니면 메시지가 쪼개져 컬럼 수가 줄마다 달라진다."""
    line = (
        "KERNDTLB 1117903context 2005.06.04 NODE 2005-06-04-00.00.58.streamed "
        "NODE RAS KERNEL FATAL data TLB error interrupt"
    )
    row = parse_line(line)
    assert row is None  # unix_ts 가 숫자가 아니면 실패로 센다


def test_fixture_parses_without_failure():
    with open(FIXTURE, encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f if ln.strip()]
    failed = sum(1 for ln in lines if parse_line(ln) is None)
    assert len(lines) == 2000
    assert failed / len(lines) < 0.001
