import re

from src.normalize import RULES, normalize


def test_node_id_becomes_node_token():
    assert normalize("R02-M1-N0-C:J12-U11 ddr error") == "[NODE] ddr error"


def test_remaining_digits_become_num():
    assert normalize("total of 128 errors") == "total of [NUM] errors"


def test_hex_and_ip():
    assert normalize("addr 0x1a2b from 172.16.96.116") == "addr [HEX] from [IP]"


def test_real_path_is_replaced():
    msg = "ciod: Error creating node map from file /p/gb2/pakin1/sweep.map"
    assert normalize(msg) == "ciod: Error creating node map from file [PATH]"


def test_node_id_survives_because_digits_go_last():
    """숫자 치환을 먼저 하면 노드 ID가 R[NUM]-M[NUM]-... 로 부서진다."""
    reversed_rules = [RULES[-1]] + RULES[:-1]
    msg = "R02-M1-N0-C:J12-U11 ddr error"
    broken = msg
    for _, pat, repl in reversed_rules:
        broken = pat.sub(repl, broken)
    assert broken != normalize(msg)
    assert "[NODE]" not in broken


def test_slash_inside_a_word_is_not_a_path():
    """단어 중간의 슬래시를 경로로 먹으면 분류 단서가 사라진다."""
    assert normalize("force load/store alignment") == "force load/store alignment"
    assert normalize("Controlling BG/L rows") == "Controlling BG/L rows"
    assert "Torus/Tree/GI" in normalize("Torus/Tree/GI read error 0")


def test_naive_path_regex_would_break_those():
    """가드가 없는 정규식이 실제로 망가뜨리는지 확인 — 회귀 방지용."""
    naive = re.compile(r"(/[\w.\-]+)+")
    assert naive.sub("[PATH]", "force load/store alignment") == "force load[PATH] alignment"
