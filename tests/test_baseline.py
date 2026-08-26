import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.baseline import CLASSES, balanced_class_weight, collapse, fit, predict_frame

SPEC = [
    ("normal", "generating core.[NUM]", 60),
    ("normal", "[NUM] double-hummer alignment exceptions", 20),
    ("kernel_mem", "data TLB error interrupt", 12),
    ("kernel_mem", "data storage interrupt", 4),
    ("kernel_ops", "rts panic! - stopping execution", 6),
    ("app", "ciod: failed to read message prefix", 3),
]


def make_df():
    rows = [{"label4": c, "msg_norm": m} for c, m, n in SPEC for _ in range(n)]
    return pd.DataFrame(rows)


def test_collapse_preserves_every_row_as_weight():
    df = make_df()
    texts, y, w = collapse(df)
    assert len(texts) == len(SPEC)
    assert w.sum() == len(df)
    for c in CLASSES:
        assert w[y == c].sum() == int((df["label4"] == c).sum())


def test_class_weight_uses_original_row_counts_not_collapsed_ones():
    """묶인 행으로 세면 kernel_mem 이 2종뿐이라 비율이 완전히 달라진다."""
    df = make_df()
    texts, y, w = collapse(df)
    real = balanced_class_weight(y, w)
    wrong = balanced_class_weight(y, np.ones_like(w))
    assert real["kernel_mem"] != wrong["kernel_mem"]
    n = len(df)
    for c in CLASSES:
        assert real[c] == n / (4 * int((df["label4"] == c).sum()))


def test_collapsed_fit_equals_fitting_on_every_row():
    """묶기 + sample_weight 가 전체 행 학습과 같은 모델을 내는지 — 이게 깨지면 200배 축소가 무효."""
    df = make_df()
    texts, y, w = collapse(df)
    _, clf_fast = fit(df["msg_norm"], texts, y, w, seed=42)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_full = vec.fit_transform(df["msg_norm"])
    y_full = df["label4"].to_numpy()
    clf_full = LogisticRegression(
        max_iter=1000,
        class_weight=balanced_class_weight(y_full, np.ones(len(df))),
        random_state=42,
    ).fit(x_full, y_full)

    assert clf_fast.classes_.tolist() == clf_full.classes_.tolist()
    assert np.allclose(clf_fast.coef_, clf_full.coef_, atol=1e-4)


def test_predict_frame_has_the_columns_evaluate_needs():
    df = make_df()
    texts, y, w = collapse(df)
    vec, clf = fit(df["msg_norm"], texts, y, w, seed=42)
    out = predict_frame(vec, clf, df.head(20))
    assert list(out.columns) == ["y_true", "y_pred", "msg_norm"] + [f"p_{c}" for c in CLASSES]
    assert np.allclose(out[[f"p_{c}" for c in CLASSES]].sum(axis=1), 1.0)
