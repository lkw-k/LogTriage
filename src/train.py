"""E2. BERT 파인튜닝. 하이퍼파라미터의 정본은 configs/base.yaml 의 train 절이다 (SPEC 4-1).

베스트 체크포인트는 val macro F1 기준으로 고른다. 다만 val 은 test 의 대리 지표가
약하다는 것이 E1 에서 확인됐다 (docs/RISKS.md A) — 에폭별 로그를 반드시 같이 볼 것.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src import config
from src.dataset import CLASSES, LogDataset, build_text, encode_unique
from src.evaluate import score


def class_weights(labels, device):
    """w_c = N / (K * n_c). SPEC 4-1."""
    counts = np.array([(labels == c).sum() for c in CLASSES], dtype=float)
    w = len(labels) / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, enc, codes, device, batch_size):
    """고유 텍스트만 통과시키고 행으로 되돌린다. 예측은 텍스트에만 의존한다."""
    model.eval()
    ids, mask = enc["input_ids"], enc["attention_mask"]
    out = []
    for i in range(0, len(ids), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(
                input_ids=ids[i : i + batch_size].to(device),
                attention_mask=mask[i : i + batch_size].to(device),
            ).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out).numpy()[codes]


def write_preds(probs, df, path):
    pred = np.asarray(CLASSES)[probs.argmax(axis=1)]
    out = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    out["y_true"] = df["label4"].to_numpy()
    out["y_pred"] = pred
    out["msg_norm"] = df["msg_norm"].to_numpy()
    out[["y_true", "y_pred", "msg_norm"] + [f"p_{c}" for c in CLASSES]].to_parquet(
        path, index=False
    )
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--input", help="기본값은 processed/train_sampled.parquet")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    tcfg, dcfg = cfg["train"], cfg["dataset"]
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = Path(cfg["paths"]["processed"])
    rundir = Path(cfg["paths"]["runs"]) / args.exp_id
    rundir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.input) if args.input else processed / "train_sampled.parquet"

    cols = ["msg_norm", "label4"] + (
        ["component", "level"] if dcfg["input_mode"] == "with_meta" else []
    )
    tr = pd.read_parquet(train_path, columns=cols)
    val = pd.read_parquet(processed / "val.parquet", columns=cols)
    print(f"train {len(tr):,}줄 ({train_path.name}) | val {len(val):,}줄 | device {device}")

    tok = AutoTokenizer.from_pretrained(tcfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg["model_name"], num_labels=tcfg["num_labels"]
    ).to(device)

    ds = LogDataset(tr, tok, dcfg["max_length"], dcfg["input_mode"])
    bs = tcfg["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)
    val_enc, val_codes = encode_unique(build_text(val, dcfg["input_mode"]), tok, dcfg["max_length"])
    print(f"고유 텍스트 train {len(ds.input_ids):,}종 / val {len(val_enc['input_ids']):,}종")

    w = class_weights(tr["label4"].to_numpy(), device) if tcfg["class_weight"] else None
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=tcfg["weight_decay"]
    )
    total = len(loader) * tcfg["max_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(total * tcfg["warmup_ratio"]), total)
    print(f"배치 {bs} | 에폭당 {len(loader):,} 스텝 | 최대 {tcfg['max_epochs']} 에폭")

    best, best_epoch, best_state, patience = -1.0, 0, None, tcfg["early_stopping"]["patience"]
    log_path = rundir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        t0, run = time.perf_counter(), 0.0
        for step, batch in enumerate(loader, 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
            loss = lossf(logits.float(), batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if step % 200 == 0 or step == len(loader):
                done = step / len(loader)
                eta = (time.perf_counter() - t0) / done * (1 - done)
                print(
                    f"  에폭 {epoch} {step:>6,}/{len(loader):,} ({done * 100:5.1f}%) "
                    f"loss {run / step:.4f}  남은 {eta / 60:.1f}분",
                    end="\r",
                )

        probs = predict(model, val_enc, val_codes, device, bs * 4)
        s = score(val["label4"].to_numpy(), np.asarray(CLASSES)[probs.argmax(axis=1)])
        mins = (time.perf_counter() - t0) / 60
        rec = {
            "epoch": epoch,
            "train_loss": run / len(loader),
            "val_macro_f1": s["macro_f1"],
            "minutes": mins,
            "per_class_f1": {c: s["per_class"][c]["f1"] for c in CLASSES},
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"\n에폭 {epoch}  loss {rec['train_loss']:.4f}  val macro F1 {s['macro_f1']:.4f}  "
            f"{mins:.1f}분  " + " ".join(f"{c[:4]}={s['per_class'][c]['f1']:.3f}" for c in CLASSES)
        )

        if s["macro_f1"] > best:
            best, best_epoch = s["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  베스트 갱신 (에폭 {epoch})")
        elif epoch - best_epoch >= patience:
            print(f"  조기 종료: {patience} 에폭 동안 개선 없음 (베스트 에폭 {best_epoch})")
            break

    model.load_state_dict(best_state)
    ckpt = rundir / "checkpoint"
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    print(f"\n베스트 에폭 {best_epoch}, val macro F1 {best:.4f} -> {ckpt}")

    for split in ["val", "test"]:
        part = pd.read_parquet(processed / f"{split}.parquet", columns=cols)
        enc, codes = encode_unique(build_text(part, dcfg["input_mode"]), tok, dcfg["max_length"])
        probs = predict(model, enc, codes, device, bs * 4)
        write_preds(probs, part, rundir / f"preds_{split}.parquet")
        print(f"{split:5s} {len(part):>9,}행 -> {rundir}/preds_{split}.parquet")

    print(f"\n채점: uv run python -m src.evaluate --exp-id {args.exp_id}")


if __name__ == "__main__":
    main()
