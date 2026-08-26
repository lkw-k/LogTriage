import hashlib

import pandas as pd
import pytest

from src.sample import apply_strategy, main


def make_train():
    """normal 100 / kernel_mem 50 / kernel_ops 10+10 / app 4."""
    spec = [
        ("normal", "generating core.[NUM]", 100),
        ("kernel_mem", "data TLB error interrupt", 50),
        ("kernel_ops", "rts panic!", 10),
        ("kernel_ops", "rts: kernel terminated", 10),
        ("app", "ciod: failed to read message", 4),
    ]
    rows = [{"label4": c, "msg_norm": m, "unix_ts": i} for c, m, n in spec for i in range(n)]
    return pd.DataFrame(rows)


def test_none_is_identity():
    df = make_train()
    assert len(apply_strategy(df, "none")) == len(df)


def test_cap_limits_rows_per_template():
    out = apply_strategy(make_train(), "cap", cap=5)
    per = out.groupby(["label4", "msg_norm"]).size()
    assert per.max() == 5
    assert len(out) == 5 + 5 + 5 + 5 + 4  # app 은 4줄뿐이라 그대로


def test_dedup_leaves_one_row_per_template():
    out = apply_strategy(make_train(), "dedup")
    assert len(out) == 5
    assert int((out["label4"] == "kernel_ops").sum()) == 2


def test_balanced_equalizes_classes():
    out = apply_strategy(make_train(), "balanced")
    assert out["label4"].value_counts().nunique() == 1


def test_ratio_keeps_all_alerts_and_caps_normal():
    out = apply_strategy(make_train(), "ratio", ratio=3)
    alerts = int((out["label4"] != "normal").sum())
    assert alerts == 74  # 알럿은 하나도 안 버린다
    assert int((out["label4"] == "normal").sum()) == min(100, 74 * 3)


def test_val_and_test_files_are_untouched(tmp_path, monkeypatch):
    """절대규칙 2 — sample 은 train 만 바꾼다."""
    make_train().to_parquet(tmp_path / "train.parquet", index=False)
    other = pd.DataFrame({"label4": ["normal"], "msg_norm": ["x"], "unix_ts": [1]})
    for name in ["val", "test"]:
        other.to_parquet(tmp_path / f"{name}.parquet", index=False)
    digest = {
        n: hashlib.sha256((tmp_path / f"{n}.parquet").read_bytes()).hexdigest()
        for n in ["val", "test"]
    }

    monkeypatch.setattr(
        "sys.argv",
        ["sample", "--input", str(tmp_path / "train.parquet"),
         "--output", str(tmp_path / "train_sampled.parquet"), "--strategy", "cap"],
    )
    main()

    for n in ["val", "test"]:
        now = hashlib.sha256((tmp_path / f"{n}.parquet").read_bytes()).hexdigest()
        assert now == digest[n]


def test_unknown_strategy_is_a_hard_error():
    with pytest.raises(SystemExit):
        apply_strategy(make_train(), "shuffle")
