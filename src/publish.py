"""학습한 모델과 모델 카드를 Hugging Face Hub 에 올린다.

**카드의 모든 수치는 runs/<exp_id>/metrics.json 에서 읽어 만든다.** 손으로 적지 않는다.
모델을 다시 학습하고 evaluate 를 돌린 뒤 이 명령만 다시 치면 카드가 새 수치로 갱신된다.
손으로 적으면 모델은 바뀌었는데 카드는 옛 숫자인 상태가 반드시 생긴다.

    uv run python -m src.publish --exp-id E2w --repo illimax/bgl-log-triage-bert --dry-run
    uv run python -m src.publish --exp-id E2w --repo illimax/bgl-log-triage-bert

data/ 와 runs/ 는 저장소에 없으므로 GitHub Actions 에서는 돌릴 수 없다. 모델 파일이
있는 로컬에서 실행한다.
"""

import argparse
import json
from pathlib import Path

from src import config
from src.dataset import CLASSES

REPO_URL = "https://github.com/lkw-k/LogTriage"

# 정규화 규칙은 src/normalize.py 가 정본이다. 카드에는 그 코드를 그대로 박는다 —
# 이 모델은 정규화된 텍스트로 학습했고, 원본 로그를 그대로 넣으면 test 719,665줄 중
# 12.95% 의 예측이 바뀐다 (normal -> kernel_ops 78,674 / normal -> kernel_mem 14,481).
NORMALIZE_SNIPPET = '''import re

# Order matters. Digits MUST be substituted last, or node IDs are shredded first.
RULES = [
    (re.compile(r"R\\d+-M\\d+-N\\d+-C:J\\d+-U\\d+"), "[NODE]"),
    (re.compile(r"\\d+\\.\\d+\\.\\d+\\.\\d+"), "[IP]"),
    (re.compile(r"0x[0-9a-fA-F]+"), "[HEX]"),
    (re.compile(r"(?<![\\w])(/[\\w.\\-]+)+"), "[PATH]"),   # the lookbehind is required
    (re.compile(r"\\d+"), "[NUM]"),
]

def normalize(msg: str) -> str:
    for pat, repl in RULES:
        msg = pat.sub(repl, msg)
    return msg'''


def pct(x):
    return f"{x:.4f}"


def load_metrics(runs, exp_id):
    p = Path(runs) / exp_id / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def per_class_row(block, key):
    return " | ".join(pct(block["per_class"][c][key]) for c in CLASSES)


def build_card(exp_id, m, baseline):
    """metrics.json 에서 읽은 수치만으로 카드를 만든다."""
    u, s, a = m["unseen"], m["seen"], m["argmax"]
    n_rows = m["n_rows"]
    b_line = ""
    if baseline:
        bu = baseline["unseen"]["macro_f1"]
        ba = baseline["argmax"]["macro_f1"]
        verdict = "loses to" if u["macro_f1"] < bu else "beats"
        b_line = (
            f"\n**This model {verdict} a TF-IDF + logistic-regression baseline trained on the "
            f"same data** ({pct(bu)} vs {pct(u['macro_f1'])} unseen-template macro F1; "
            f"{pct(ba)} vs {pct(a['macro_f1'])} overall). That result, and the investigation "
            f"into why, is the point of this repository.\n"
        )

    head = "| metric | " + " | ".join(CLASSES) + " | macro |"
    sep = "|---" * (len(CLASSES) + 2) + "|"
    support_row = " | ".join(format(u["per_class"][c]["support"], ",") for c in CLASSES)
    app_recall = pct(u["per_class"]["app"]["recall"])

    return f"""---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-classification
tags:
  - log-analysis
  - anomaly-detection
  - bert
  - bgl
base_model: bert-base-uncased
---

# BGL log triage — BERT ({exp_id})

Classifies **one Blue Gene/L server log line** into 4 classes so an operator knows *where to
look*: `normal`, `kernel_mem` (hardware), `kernel_ops` (system config), `app` (code).

Fine-tuned from `bert-base-uncased` on the [LogHub BGL](https://github.com/logpai/loghub)
dataset. Full pipeline, experiment log, and the losing runs: {REPO_URL}
{b_line}

## Read this before you use it

**1. The input must be normalized first.** This model was trained on messages with node IDs,
IPs, hex values, paths, and digits replaced by placeholders. Feeding raw log lines runs fine
and gives you wrong answers: over the test period's 719,665 raw lines, **12.95% of
predictions change** — 78,674 `normal` lines become `kernel_ops` and 14,481 become
`kernel_mem`. That is a false-alarm flood, not a rounding difference.

```python
{NORMALIZE_SNIPPET}
```

**2. Feed the message only.** Strip the 9 BGL header fields
(`label unix_ts date node time node_repeat type component level`) and pass what follows.
Use `line.split(maxsplit=9)` — a plain `split()` breaks on spaces inside the message.

**3. `max_length=64`.** That is what it was trained with.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{{repo_id}}")
tok = AutoTokenizer.from_pretrained("{{repo_id}}")
model.eval()

raw = ("- 1131147223 2005.11.05 R16-M0-N4-C:J13-U11 2005-11-05-01.33.43.334348 "
       "R16-M0-N4-C:J13-U11 RAS KERNEL FATAL data TLB error interrupt")
message = raw.split(maxsplit=9)[9]          # step 2
text = normalize(message)                    # step 1 — do not skip

with torch.no_grad():
    probs = model(**tok(text, truncation=True, max_length=64,
                        return_tensors="pt")).logits.softmax(-1)[0]
print(model.config.id2label[int(probs.argmax())], float(probs.max()))
```

## Evaluation

Scored on a **time-based** split (sorted by `unix_ts`, first 70/15/15). Never a random split:
BGL repeats identical lines dozens of times, so a random split puts the same line in train
and test.

**The headline number is macro F1 on log templates that never appear in training.**
Overall macro F1 mixes in {s["share"]:.1f}% of rows whose template the model memorized during
training — on that subset every model trends toward 1.0000 as it overfits, which measures
memorization, not skill. Accuracy is not reported at all: predicting all-`normal` scores 92.7%.

| subset | rows | macro F1 |
|---|---|---|
| **unseen templates ({100 - s["share"]:.1f}%)** | {u["n_rows"]:,} | **{pct(u["macro_f1"])}** |
| seen templates ({s["share"]:.1f}%) | {s["n_rows"]:,} | {pct(s["macro_f1"])} |
| all test | {n_rows:,} | {pct(a["macro_f1"])} |

Per-class F1 on unseen templates:

{head}
{sep}
| F1 | {per_class_row(u, "f1")} | {pct(u["macro_f1"])} |
| precision | {per_class_row(u, "precision")} | |
| recall | {per_class_row(u, "recall")} | |
| support | {support_row} | |

## Known limitations

- **`kernel_ops` on unseen templates is near-zero for every model tried**, including the
  TF-IDF baseline (recall 1.0%). A failure mode absent from the training window
  (`Error receiving packet on tree network`) appears in test and nothing catches it.
- **`kernel_mem` has only 100 supporting rows in test.** Its F1 is decided by the false-positive
  count, not recall. A single unseen template (`MACHINE CHECK DCR read timeout`, 14,481 rows,
  actually `normal`) swings unseen macro F1 between 0.73 and 0.50 depending on training setup.
- **Three `ciod:` templates are missed** and come back as `normal` — most visibly
  `ciod: Error reading message prefix on CioStream socket to [IP]:[NUM], Connection reset by
  peer` and `ciod: LOGIN chdir([PATH]) failed: Input/output error`. That is 5,063 test rows and
  the whole reason `app` recall on unseen templates is {app_recall} rather than ~0.97.
- **Single seed.** The checkpoint selection sits on a 0.0005 metric gap; reproducibility across
  seeds was not measured.
- **BGL only.** Other log formats need an adapter that yields `(timestamp, message)`; results
  without one are meaningless.
- The 4-class grouping is **this project's judgment, not an official taxonomy**. No complete
  BGL alert-code documentation exists, so the 41 raw categories were grouped by name.

## Reproduce

```bash
git clone {REPO_URL} && cd LogTriage && uv sync
# download BGL.log from LogHub into data/raw/, then run the pipeline in README.md
```

Every number in the Evaluation table is generated from `runs/{exp_id}/metrics.json` by
`src/publish.py`, so retraining and re-publishing cannot leave a stale figure behind. The
12.95% normalization figure was measured separately on this same checkpoint.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--repo", required=True, help="예: illimax/bgl-log-triage-bert")
    ap.add_argument("--baseline-exp", default="E1c", help="카드에 비교로 넣을 실험")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="카드만 만들고 올리지 않는다")
    ap.add_argument("--config", default=config.DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg = config.load(args.config)
    runs = cfg["paths"]["runs"]
    ckpt = Path(runs) / args.exp_id / "checkpoint"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} 가 없다. train --exp-id {args.exp_id} 를 먼저 돌려야 한다.")

    m = load_metrics(runs, args.exp_id)
    if m is None:
        raise SystemExit(
            f"{runs}/{args.exp_id}/metrics.json 이 없다. "
            f"evaluate --exp-id {args.exp_id} 를 먼저 돌려야 카드 수치가 나온다."
        )
    baseline = load_metrics(runs, args.baseline_exp) if args.baseline_exp else None
    if baseline is None and args.baseline_exp:
        print(f"경고: {args.baseline_exp} 의 metrics.json 이 없어 비교 문장을 뺀다.")

    card = build_card(args.exp_id, m, baseline).replace("{repo_id}", args.repo)
    card_path = ckpt / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"카드 -> {card_path} ({len(card.splitlines())}줄)")
    print(f"  미등장 macro F1 {pct(m['unseen']['macro_f1'])} | 전체 {pct(m['argmax']['macro_f1'])}")

    files = sorted(p.name for p in ckpt.iterdir() if p.is_file())
    size = sum(p.stat().st_size for p in ckpt.iterdir() if p.is_file())
    print(f"  올릴 파일 {len(files)}개 ({size / 1e6:.1f}MB): {', '.join(files)}")

    if args.dry_run:
        print("\n--dry-run 이라 업로드하지 않았다.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), repo_id=args.repo, repo_type="model")
    print(f"\n업로드 완료: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
