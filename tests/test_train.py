import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import CLASSES, LogDataset, build_text, encode_unique
from src.train import class_weights, write_preds


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
