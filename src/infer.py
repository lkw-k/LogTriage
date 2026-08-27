"""추론 파이프라인. 원본 로그 -> 1분 구간별 판정 (SPEC 5-2 ~ 5-4).

줄 단위 분류와 구간 단위 발생량은 단위가 달라 그대로 못 합친다.
**1분 버킷을 공통 키로** 줄 예측을 집계한 뒤 트래픽 신호와 조인한다.

판정 기준선은 calibrate 가 만든 calibration.json 에서 읽는다. 추론 시점에
실시간으로 잡으면 장애가 지속될 때 그 상태를 평시로 학습해버린다 (SPEC 5-4).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, predictor
from src.adapters import bgl
from src.dataset import CLASSES
from src.normalize import normalize
from src.traffic.detect import ewma_z, flag, watchable

ALERT = [c for c in CLASSES if c != "normal"]
ADAPTERS = {bgl.NAME: bgl}
LEVELS = {"ok": 0, "warning": 1, "critical": 2}


def zscore(ratio, mean, std):
    """std 가 0 인 클래스는 평시에 한 번도 안 나온 것이다. 넘으면 무한대로 본다."""
    if std > 0:
        return (ratio - mean) / std
    return float("inf") if ratio > mean else 0.0


def judge(row, n, cal, icfg):
    """SPEC 5-4 판정표. -> (level, reasons)"""
    level, reasons = "ok", []

    def raise_to(new, why):
        nonlocal level
        if LEVELS[new] > LEVELS[level]:
            level = new
        reasons.append(why)

    # 비율 판정은 최소 건수를 넘은 버킷에서만. 8건짜리 버킷은 한 건에 비율이 튄다.
    if n >= cal["min_count"]:
        for c in ALERT:
            ratio = row[c] / n
            z = zscore(ratio, cal["class_ratio_mean"][c], cal["class_ratio_std"][c])
            if z > icfg["z_critical"]:
                raise_to("critical", f"{c} 비율 {ratio:.1%} (평시 "
                                     f"{cal['class_ratio_mean'][c]:.1%}, z={z:.1f})")
            elif z > icfg["z_warning"]:
                raise_to("warning", f"{c} 비율 {ratio:.1%} (평시 "
                                    f"{cal['class_ratio_mean'][c]:.1%}, z={z:.1f})")
        u = row["unknown"] / n
        if u > icfg["unknown_ratio_warning"]:
            raise_to("warning", f"unknown 비율 {u:.1%} — 미지의 로그 패턴 유입")

    tz = row["traffic_z"]
    if not row["traffic_watch"]:
        return level, reasons
    if tz < -icfg["z_warning"]:
        raise_to("critical", f"트래픽 급감 z={tz:.1f} — 서비스 중단 의심")
    elif tz > icfg["z_warning"]:
        raise_to("warning", f"트래픽 급증 z={tz:.1f}")
    return level, reasons


def run(lines, cal, cfg, adapter):
    icfg = cfg["infer"]
    rows, bad = adapter.parse(lines)
    if not rows:
        raise SystemExit("파싱된 줄이 0 이다. 어댑터가 포맷과 맞는지 확인할 것.")

    df = pd.DataFrame(rows, columns=["unix_ts", "message"])
    df["msg_norm"] = [normalize(m) for m in df["message"]]

    model, _ = predictor.load(cal["exp_id"], cfg)
    proba = predictor.predict_unique(model, df["msg_norm"].to_numpy())
    df["conf"] = proba.max(axis=1)
    df["pred"] = np.asarray(CLASSES)[proba.argmax(axis=1)]
    df.loc[df["conf"] < cal["conf_threshold"], "pred"] = "unknown"
    df["bucket"] = pd.to_datetime(df["unix_ts"], unit="s").dt.floor(icfg["window"])

    counts = pd.crosstab(df["bucket"], df["pred"]).reindex(
        columns=CLASSES + ["unknown"], fill_value=0)
    counts = counts.reindex(
        pd.date_range(counts.index.min(), counts.index.max(), freq=icfg["window"]),
        fill_value=0)
    n = counts.sum(axis=1)
    baseline, z = ewma_z(n, cal["traffic_ewma_span"])
    counts["traffic_z"] = z
    counts["traffic_watch"] = watchable(baseline, cfg["traffic"]["min_baseline"])

    windows = []
    for ts, row in counts.iterrows():
        total = int(n.loc[ts])
        if total == 0:
            continue
        level, reasons = judge(row, total, cal, icfg)
        part = df[df["bucket"] == ts]
        top = part[part["pred"].isin(ALERT)].nlargest(icfg["top_samples"], "conf")
        windows.append({
            "window_start": ts.isoformat(),
            "n_logs": total,
            "class_counts": {c: int(row[c]) for c in CLASSES + ["unknown"]},
            "anomaly_ratio": round(float(sum(row[c] for c in ALERT) / total), 4),
            "traffic": {
                "count": total,
                "baseline": round(float(baseline.loc[ts]), 1),
                "z": round(float(row["traffic_z"]), 2),
                "flag": (flag(float(row["traffic_z"]), cfg["traffic"]["z_threshold"])
                         if row["traffic_watch"] else "low_volume"),
            },
            "alert": {"level": level, "reasons": reasons},
            "top_samples": [
                {"raw": r.message, "pred": r.pred, "conf": round(float(r.conf), 4)}
                for r in top.itertuples()
            ],
        })
    return df, windows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--input", required=True, help="원본 로그 파일")
    ap.add_argument("--output", help="기본값은 runs/<exp_id>/windows.jsonl")
    ap.add_argument("--adapter", default=bgl.NAME, choices=sorted(ADAPTERS))
    ap.add_argument("--calibration", help="기본값은 runs/<exp_id>/calibration.json")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    rundir = Path(cfg["paths"]["runs"]) / args.exp_id
    cal_path = Path(args.calibration) if args.calibration else rundir / "calibration.json"
    if not cal_path.exists():
        raise SystemExit(f"{cal_path} 가 없다. "
                         f"calibrate --exp-id {args.exp_id} 를 먼저 돌려야 한다.")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal["exp_id"] != args.exp_id:
        raise SystemExit(f"calibration.json 은 {cal['exp_id']} 것이다. --exp-id 와 맞지 않는다.")

    lines = Path(args.input).read_text(encoding="utf-8", errors="replace").splitlines()
    df, windows, bad = run(lines, cal, cfg, ADAPTERS[args.adapter])
    print(f"입력 {len(lines):,}줄 | 파싱 실패 {len(bad):,}줄 "
          f"({len(bad) / max(len(lines), 1) * 100:.2f}%) | 어댑터 {args.adapter}")
    print(f"모델 {cal['exp_id']} | conf_threshold {cal['conf_threshold']} "
          f"| unknown {int((df['pred'] == 'unknown').sum()):,}줄")

    out = Path(args.output) if args.output else rundir / "windows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for w in windows:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    preds_path = out.with_name("predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r in df.itertuples():
            f.write(json.dumps({"ts": int(r.unix_ts), "pred": r.pred,
                                "conf": round(float(r.conf), 4), "raw": r.message},
                               ensure_ascii=False) + "\n")

    lv = pd.Series([w["alert"]["level"] for w in windows]).value_counts()
    span = (pd.Timestamp(windows[-1]["window_start"])
            - pd.Timestamp(windows[0]["window_start"])).total_seconds() / 86400
    print(f"\n구간 {len(windows):,}개 | " + " ".join(
        f"{k} {int(lv.get(k, 0)):,}" for k in ["ok", "warning", "critical"]))
    if span > 0:
        fired = len(windows) - int(lv.get("ok", 0))
        print(f"기간 {span:.1f}일 | 알림 {fired:,}건 = 하루 {fired / span:.1f}건")
    print(f"출력: {out}, {preds_path}")


if __name__ == "__main__":
    main()
