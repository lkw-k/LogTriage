import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    CLASSES,
    PROB_COLS,
    main,
    prf,
    score,
    seen_mask,
    tune_weights,
    weighted_pred,
)

CFG = """
seed: 42
paths:
  processed: {processed}
  runs: {runs}
evaluate:
  n_error_samples: 5
"""


def test_prf_on_a_known_case():
    y_true = np.array(["normal", "app", "app", "normal"])
    y_pred = np.array(["normal", "app", "normal", "normal"])
    m = prf(y_true, y_pred, "app")
    assert (m["tp"], m["fp"], m["support"]) == (1, 0, 2)
    assert m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3)


def test_macro_f1_is_the_plain_mean_of_four_class_f1():
    y_true = np.array(["normal"] * 8 + ["app", "kernel_mem"])
    y_pred = np.array(["normal"] * 9 + ["kernel_mem"])
    s = score(y_true, y_pred)
    assert s["macro_f1"] == pytest.approx(sum(s["per_class"][c]["f1"] for c in CLASSES) / 4)


def test_constant_normal_scores_the_expected_floor():
    """전부 normal 예측이면 macro F1 = p / (2(1+p)). 정상 92.7% 면 0.24 근처."""
    n_norm, n_other = 934, 66
    y_true = np.array(["normal"] * n_norm + ["app"] * n_other)
    y_pred = np.array(["normal"] * (n_norm + n_other))
    p = n_norm / (n_norm + n_other)
    assert score(y_true, y_pred)["macro_f1"] == pytest.approx(p / (2 * (1 + p)))


def test_weight_tuning_beats_argmax_when_a_class_is_rare():
    rng = np.random.default_rng(42)
    y = np.array(["normal"] * 400 + ["app"] * 8)
    probs = np.zeros((len(y), 4))
    probs[:, 0] = rng.uniform(0.45, 0.9, len(y))
    probs[y == "app", 0] = rng.uniform(0.40, 0.55, 8)   # app 이 argmax 를 못 이김
    probs[:, CLASSES.index("app")] = 1 - probs[:, 0]
    base = score(y, weighted_pred(probs, np.ones(4)))["macro_f1"]
    _, tuned = tune_weights(y, probs)
    assert tuned > base


def write_cfg(tmp_path):
    proc, runs = tmp_path / "processed", tmp_path / "runs"
    proc.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CFG.format(processed=proc.as_posix(), runs=runs.as_posix()), encoding="utf-8")
    frame = lambda labels: pd.DataFrame(  # noqa: E731
        {"label4": labels, "msg_norm": [f"m{i}" for i in range(len(labels))]}
    )
    for name in ["train", "val", "test"]:
        frame(["normal"] * 90 + ["app"] * 6 + ["kernel_mem"] * 3 + ["kernel_ops"] * 1).to_parquet(
            proc / f"{name}.parquet", index=False
        )
    return cfg, runs


def test_majority_baseline_runs_end_to_end(tmp_path, monkeypatch):
    cfg, runs = write_cfg(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate", "--exp-id", "E0", "--baseline", "majority", "--config", str(cfg)],
    )
    main()
    out = runs / "E0"
    assert (out / "cm.png").exists() and (out / "errors.csv").exists()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["argmax"]["per_class"]["app"]["f1"] == 0.0
    assert m["argmax"]["per_class"]["normal"]["support"] == 90
    assert "calibrated" not in m          # 확률이 없으면 보정하지 않는다
    assert "accuracy" not in json.dumps(m)  # 절대규칙 4


def test_missing_prediction_file_is_a_hard_error(tmp_path, monkeypatch):
    cfg, _ = write_cfg(tmp_path)
    monkeypatch.setattr("sys.argv", ["evaluate", "--exp-id", "E9", "--config", str(cfg)])
    with pytest.raises(SystemExit):
        main()


def test_calibration_is_fitted_on_val_not_test(tmp_path, monkeypatch):
    """test 예측만 바꿔도 가중치는 그대로여야 한다 — val 에서만 맞추므로."""
    cfg, runs = write_cfg(tmp_path)
    rng = np.random.default_rng(0)
    out = runs / "E1"
    out.mkdir(parents=True)
    for sp in ["val", "test"]:
        y = np.array(["normal"] * 90 + ["app"] * 10)
        pr = rng.uniform(0.3, 0.9, (100, 4))
        pr /= pr.sum(axis=1, keepdims=True)
        d = pd.DataFrame(pr, columns=PROB_COLS)
        d["y_true"] = y
        d["y_pred"] = weighted_pred(pr, np.ones(4))
        d["msg_norm"] = [f"m{i}" for i in range(100)]
        d.to_parquet(out / f"preds_{sp}.parquet", index=False)

    monkeypatch.setattr("sys.argv", ["evaluate", "--exp-id", "E1", "--config", str(cfg)])
    main()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["calibrated"]["tuned_on"] == "val"
    assert len(m["calibrated"]["weights"]) == 4


def test_seen_mask_is_computed_against_the_train_reference(tmp_path):
    ref = tmp_path / "train.parquet"
    pd.DataFrame({"msg_norm": ["a", "b", "b"]}).to_parquet(ref, index=False)
    assert seen_mask(pd.Series(["a", "c", "b", "d"]), ref).tolist() == [True, False, True, False]


def split_preds_frame():
    """기등장/미등장 각각에 4클래스를 모두 넣는다. 한쪽만 실수하게 만든다."""
    labels = ["normal"] * 40 + ["app"] * 4 + ["kernel_mem"] * 3 + ["kernel_ops"] * 3
    y = np.array(labels * 2)
    pred = y.copy()
    pred[50 + 40 : 50 + 44] = "normal"          # 미등장 쪽 app 4건만 틀린다
    msgs = [f"m{i}" for i in range(50)] + [f"u{i}" for i in range(50)]
    return pd.DataFrame({"y_true": y, "y_pred": pred, "msg_norm": msgs})


def test_metrics_report_seen_and_unseen_template_subsets(tmp_path, monkeypatch):
    """미등장 템플릿 부분집합이 모델 비교의 판정 기준이다 (docs/RISKS.md 9)."""
    cfg, runs = write_cfg(tmp_path)          # 참조 train 의 템플릿은 m0..m99
    out = runs / "E2"
    out.mkdir(parents=True)
    split_preds_frame().to_parquet(out / "preds_test.parquet", index=False)

    monkeypatch.setattr("sys.argv", ["evaluate", "--exp-id", "E2", "--config", str(cfg)])
    main()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["seen"]["n_rows"] == 50 and m["unseen"]["n_rows"] == 50
    assert m["seen"]["macro_f1"] == 1.0
    assert m["unseen"]["macro_f1"] < 1.0
    assert m["unseen"]["per_class"]["app"]["f1"] == 0.0


def test_train_ref_can_be_overridden(tmp_path, monkeypatch):
    """참조 train 을 바꾸면 미등장 집합도 바뀐다."""
    cfg, runs = write_cfg(tmp_path)
    out = runs / "E2"
    out.mkdir(parents=True)
    split_preds_frame().to_parquet(out / "preds_test.parquet", index=False)
    ref = tmp_path / "everything.parquet"
    pd.DataFrame({"msg_norm": [f"m{i}" for i in range(50)] + [f"u{i}" for i in range(50)]}
                 ).to_parquet(ref, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["evaluate", "--exp-id", "E2", "--config", str(cfg), "--train-ref", str(ref)],
    )
    main()
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["seen"]["n_rows"] == 100 and m["unseen"]["n_rows"] == 0
