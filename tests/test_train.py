import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import CLASSES, LogDataset, build_text, encode_unique
from src.train import class_weights, inner_split, write_preds


class FakeTok:
    """실제 토크나이저 없이 인코딩 계약만 흉내낸다."""

    def __call__(self, texts, truncation, max_length, padding, return_tensors):
        n = len(texts)
        return {
            "input_ids": torch.arange(n * max_length).reshape(n, max_length) % 100,
            "attention_mask": torch.ones(n, max_length, dtype=torch.long),
        }


def make_df():
    spec = [("normal", "generating core.[NUM]", 40), ("normal", "other line", 10),
            ("kernel_mem", "data TLB error interrupt", 8), ("app", "ciod: failed", 2)]
    rows = [{"label4": c, "msg_norm": m, "component": "KERNEL", "level": "INFO"}
            for c, m, n in spec for _ in range(n)]
    return pd.DataFrame(rows)


def test_build_text_modes():
    df = make_df()
    assert build_text(df, "msg_only")[0] == "generating core.[NUM]"
    assert build_text(df, "with_meta")[0] == "KERNEL INFO generating core.[NUM]"
    with pytest.raises(SystemExit):
        build_text(df, "msg_and_node")


def test_encode_unique_tokenizes_each_template_once():
    df = make_df()
    enc, codes = encode_unique(build_text(df, "msg_only"), FakeTok(), 8)
    assert len(enc["input_ids"]) == 4          # 고유 템플릿 4종
    assert len(codes) == len(df)               # 행은 전부 유지
    uniq = df.msg_norm.to_numpy()
    assert all(uniq[i] == uniq[j] for i in range(len(df)) for j in range(len(df))
               if codes[i] == codes[j])


def test_dataset_row_maps_to_its_own_template_and_label():
    df = make_df()
    ds = LogDataset(df, FakeTok(), 8, "msg_only")
    assert len(ds) == len(df)
    last = ds[len(df) - 1]
    assert last["labels"].item() == CLASSES.index("app")
    assert torch.equal(ds[0]["input_ids"], ds[39]["input_ids"])   # 같은 템플릿
    assert not torch.equal(ds[0]["input_ids"], ds[49]["input_ids"])


def test_class_weight_formula():
    """w_c = N / (K * n_c). 없는 클래스가 있어도 0 으로 나누지 않는다."""
    labels = make_df()["label4"].to_numpy()
    w = class_weights(labels, torch.device("cpu")).numpy()
    n = len(labels)
    assert w[CLASSES.index("normal")] == pytest.approx(n / (4 * 50))
    assert w[CLASSES.index("app")] == pytest.approx(n / (4 * 2))
    assert np.isfinite(w).all()


def uneven_frames():
    """sample 이 시간축을 따라 불균등하게 행을 지운 상황.

    ref 는 지우기 전(train.parquet), df 는 지운 뒤(train_cap.parquet)에 해당한다.
    알럿은 ts 45~70 에만 있다.
    """
    ref = np.concatenate([np.repeat(np.arange(1, 51), 50), np.arange(51, 101)])
    rows = [{"unix_ts": t, "label4": "kernel_ops" if 45 <= t <= 70 else "normal",
             "msg_norm": f"t{t}"} for t in range(1, 101)]
    return ref, pd.DataFrame(rows)


def test_inner_split_cuts_on_unix_ts_not_row_position():
    """행 위치로 자르면 알럿이 전부 inner-train 으로 넘어가 미등장이 0 이 된다."""
    ref, df = uneven_frames()
    _, va, t = inner_split(df, ref, 0.2)
    assert t == 41
    assert (va["label4"] == "kernel_ops").sum() == 26

    _, by_position, t_pos = inner_split(df, df["unix_ts"].to_numpy(), 0.2)
    assert t_pos == 81
    assert (by_position["label4"] == "kernel_ops").sum() == 0


def test_inner_split_puts_no_row_in_both_halves():
    ref, df = uneven_frames()
    tr, va, t = inner_split(df, ref, 0.2)
    assert len(tr) + len(va) == len(df)
    assert tr["unix_ts"].max() < t <= va["unix_ts"].min()


def test_unseen_subset_is_judged_against_inner_train_only():
    ref, df = uneven_frames()
    tr, va, _ = inner_split(df, ref, 0.2)
    unseen = ~va["msg_norm"].isin(set(tr["msg_norm"])).to_numpy()
    assert unseen.all()                       # 템플릿이 ts 마다 달라 전부 미등장
    df2 = pd.concat([df, df.iloc[:1].assign(unix_ts=99)])   # t1 이 뒤에 다시 등장
    tr2, va2, _ = inner_split(df2, ref, 0.2)
    reappeared = va2["msg_norm"] == "t1"
    assert reappeared.sum() == 1
    assert va2.loc[reappeared, "msg_norm"].isin(set(tr2["msg_norm"])).all()


def test_write_preds_matches_what_evaluate_reads(tmp_path):
    df = make_df()
    rng = np.random.default_rng(42)
    probs = rng.random((len(df), 4))
    probs /= probs.sum(axis=1, keepdims=True)
    pred = write_preds(probs, df, tmp_path / "p.parquet")
    out = pd.read_parquet(tmp_path / "p.parquet")
    assert list(out.columns) == ["y_true", "y_pred", "msg_norm"] + [f"p_{c}" for c in CLASSES]
    assert (out["y_pred"].to_numpy() == pred).all()
    assert (out["y_pred"].to_numpy() == np.asarray(CLASSES)[probs.argmax(axis=1)]).all()
