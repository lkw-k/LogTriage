"""평시 기준선과 conf_threshold 를 미리 계산해 calibration.json 에 남긴다 (SPEC 5-4).

추론 시점에 기준을 실시간으로 잡으면 장애가 지속될 때 그 상태를 평시로 학습해버린다.
그래서 기준선은 train 구간에서 한 번만 계산한다.

두 가지를 정한다.

1. `class_ratio_mean/std` — 1분 버킷의 클래스 비율 분포. `infer` 가 z 점수를 낸다.
   `infer.baseline_exclude` 구간은 뺀다. train 의 kernel_mem 99.0% 가 3일짜리 장애
   하나라, 넣으면 mean 과 std 를 둘 다 끌어올려 임계가 0.61 이 된다.
   버킷 로그 수가 `infer.min_count` 미만인 것도 뺀다 — 8건짜리 버킷의 비율은
   0 아니면 0.125 라 분포를 왜곡한다.

2. `conf_threshold` — 최대 확률이 이 값 미만이면 `unknown`. **val 에서만 고른다.**
   test 에서 고르면 그 순간 test 가 오염된다 (SPEC 5-5).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, predictor
from src.dataset import CLASSES

ALERT = [c for c in CLASSES if c != "normal"]
UNKNOWN_BUDGET = 0.05   # SPEC 5-5. unknown 비율이 이 값을 넘지 않는 선에서 고른다.


def bucket_ratios(ts, pred, window, exclude, min_count):
    """1분 버킷별 클래스 비율. 제외 구간과 저건수 버킷을 뺀 뒤 돌려준다."""
    b = pd.to_datetime(ts, unit="s").dt.floor(window)
    g = pd.crosstab(b, pd.Categorical(pred, categories=CLASSES))
    n = g.sum(axis=1)
    keep = pd.Series(True, index=g.index)
    for start, end in exclude:
        keep &= ~((g.index >= pd.Timestamp(start)) & (g.index < pd.Timestamp(end)))
    dropped_window = int((~keep).sum())
    keep &= n >= min_count
    return g[keep].div(n[keep], axis=0), len(g), dropped_window, int(keep.sum())


def pick_threshold(proba, y_true):
    """unknown 비율 <= 5% 인 후보 중 나머지 예측의 macro precision 이 최대인 값.

    **macro 는 항상 4클래스 고정이다.** 살아남은 클래스만 평균하면 클래스를 통째로
    지울수록 점수가 오르는 지표가 된다. E2w 는 클래스별 신뢰도 천장이 달라
    (normal 0.999987 / app 0.999817 / kernel_ops 0.999730) 임계 0.9999 가 알럿
    3클래스를 전부 지우는데, 그때 살아남은 것만 평균하면 0.9567 로 1등이 된다.
    예측이 하나도 안 남은 클래스는 정밀도 0 으로 센다.
    """
    conf, pred = proba.max(axis=1), np.asarray(CLASSES)[proba.argmax(axis=1)]
    rows = []
    for t in [0.0, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]:
        known = conf >= t
        unknown_ratio = 1.0 - known.mean()
        precs, alive = [], 0
        for c in CLASSES:
            hit = known & (pred == c)
            precs.append(float((y_true[hit] == c).mean()) if hit.sum() else 0.0)
            alive += bool(hit.sum())
        rows.append((t, unknown_ratio, float(np.mean(precs)), alive))
    ok = [r for r in rows if r[1] <= UNKNOWN_BUDGET and r[3] == len(CLASSES)]
    best = max(ok, key=lambda r: (r[2], -r[1])) if ok else rows[0]
    return best[0], rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--output", help="기본값은 runs/<exp_id>/calibration.json")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    icfg, processed = cfg["infer"], Path(cfg["paths"]["processed"])
    rundir = Path(cfg["paths"]["runs"]) / args.exp_id
    out_path = Path(args.output) if args.output else rundir / "calibration.json"

    pred_model, kind = predictor.load(args.exp_id, cfg)
    print(f"exp {args.exp_id} | 모델 {kind}")

    tr = pd.read_parquet(processed / "train.parquet", columns=["unix_ts", "msg_norm"])
    proba = predictor.predict_unique(pred_model, tr["msg_norm"].to_numpy())
    pred = np.asarray(CLASSES)[proba.argmax(axis=1)]
    print(f"train {len(tr):,}행 예측 | 고유 {tr['msg_norm'].nunique():,}종")

    exclude = [tuple(w) for w in icfg.get("baseline_exclude", [])]
    ratios, n_all, n_win, n_kept = bucket_ratios(
        tr["unix_ts"], pred, icfg["window"], exclude, icfg["min_count"])
    print(f"1분 버킷 {n_all:,}개 -> 장애 구간 {n_win:,}개 제외 -> "
          f"{icfg['min_count']}건 미만 제외 -> 기준선 {n_kept:,}개")
    for start, end in exclude:
        print(f"  제외 구간 {start} ~ {end}")

    mean = {c: float(ratios[c].mean()) for c in ALERT}
    std = {c: float(ratios[c].std()) for c in ALERT}
    print(f"\n{'클래스':<12}{'mean':>10}{'std':>10}{'z>3 임계':>12}{'z>5 임계':>12}")
    for c in ALERT:
        print(f"{c:<12}{mean[c]:>10.6f}{std[c]:>10.6f}"
              f"{mean[c] + icfg['z_warning'] * std[c]:>12.4f}"
              f"{mean[c] + icfg['z_critical'] * std[c]:>12.4f}")

    val = pd.read_parquet(processed / "val.parquet", columns=["msg_norm", "label4"])
    vproba = predictor.predict_unique(pred_model, val["msg_norm"].to_numpy())
    thr, rows, best = pick_threshold(vproba, val["label4"].to_numpy())
    print(f"\nconf_threshold 후보 (val {len(val):,}행)")
    print(f"{'임계':>9}{'unknown':>10}{'macro P':>10}{'살아있는 클래스':>16}   비고")
    for t, ur, p, k in rows:
        note = ""
        if ur > UNKNOWN_BUDGET:
            note = "unknown 예산 초과"
        elif k < len(CLASSES):
            note = f"{len(CLASSES) - k}개 클래스가 통째로 사라짐"
        elif t == thr:
            note = "<- 선택"
        print(f"{t:>9}{ur * 100:>9.2f}%{p:>10.4f}{k:>16}   {note}")

    payload = {
        "exp_id": args.exp_id,
        "model_kind": kind,
        "class_ratio_mean": mean,
        "class_ratio_std": std,
        "traffic_ewma_span": cfg["traffic"]["ewma_span"],
        "conf_threshold": thr,
        "window": icfg["window"],
        "min_count": icfg["min_count"],
        "baseline": {
            "source": "train.parquet",
            "buckets_total": n_all,
            "buckets_excluded_window": n_win,
            "buckets_used": n_kept,
            "exclude": [list(w) for w in exclude],
        },
        "threshold_selection": {
            "split": "val",
            "unknown_budget": UNKNOWN_BUDGET,
            "unknown_ratio": best[1],
            "macro_precision": best[2],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n출력: {out_path}")


if __name__ == "__main__":
    main()
