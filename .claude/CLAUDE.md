# log-anomaly-classifier

Classify one server log line into 4 classes (normal / kernel_mem / kernel_ops / app).
Side module: per-minute log volume anomaly detector.

Full design: `docs/SPEC.md`. General coding principles: `~/.claude/CLAUDE.md` (already loaded).
This file = project rules only.

## Stack

Python 3.11 · **uv** (never call `pip`) · pandas/pyarrow · torch/transformers ·
scikit-learn/matplotlib · pyyaml · pytest · ruff. CLI = `argparse` only, no CLI lib.

## Commands

```bash
uv run python -m src.parse_bgl --input data/raw/BGL.log --output data/interim/bgl_parsed.parquet
uv run python -m src.label_map --input data/interim/bgl_parsed.parquet --output data/interim/bgl_labeled.parquet
uv run python -m src.normalize --input data/interim/bgl_labeled.parquet --output data/interim/bgl_norm.parquet
uv run python -m src.split     --input data/interim/bgl_norm.parquet --outdir data/processed
uv run python -m src.sample    --input data/processed/train.parquet --output data/processed/train_sampled.parquet --strategy none
uv run python -m src.train     --config configs/base.yaml --exp-id E2
uv run python -m src.evaluate  --exp-id E2
uv run python -m src.calibrate --exp-id E2 --output runs/E2/calibration.json
uv run python -m src.infer     --exp-id E2 --input data/raw/sample.log --output runs/E2/windows.jsonl
uv run python -m src.traffic.parse_nasa --input data/raw/nasa.csv --output data/interim/nasa_per_min.parquet
uv run python -m src.traffic.detect --input data/interim/nasa_per_min.parquet --output runs/traffic/anomalies.csv --expect-outage "1995-08-01 18:52" "1995-08-03 08:36"
uv run pytest && uv run ruff check src/
```

## Hard rules — breaking these invalidates every number

1. **No random split.** Sort by `unix_ts`, cut first 70/15/15. BGL repeats identical lines
   dozens of times; random split puts the same line in train and test.
2. **Never touch val/test class ratios.** Sampling applies to train only. If shrinking
   val/test, stratify to keep the original ratio (normal ~92.7%).
3. **`label_map` lives only in `configs/base.yaml`.** Never copy it into code or docs.
   Unmapped category → hard error, never silent pass-through.
4. **macro F1 is the headline metric.** Never accuracy — predicting all-normal scores 92.7%.
5. **Never commit `data/` or `runs/`.** README gets the download URL, not the data.
6. **`level`/`component` are not default inputs.** `level` (FATAL/INFO) nearly duplicates the
   label and inflates scores. Use only in E5; report `msg_only` as headline.
7. **Seed 42** everywhere.

## Commit small and often

Pipeline errors flow silently into downstream numbers. One commit per stage means you can
roll back one step to find where it broke; batched commits make the cause unrecoverable.

- Commit the moment a stage passes its done-condition. Never batch stages.
- Before commit: `uv run ruff check src/` + `uv run pytest`. Red → no commit.
- **Never commit code you have not run.** A written-but-unexecuted module is unfinished.
- One concern per commit. Don't mix module code with doc edits.
- New numbers → the `experiments.md` line goes in that same commit.
- **Data-policy changes (dropping rows, regex edits) get their own commit** so they can be
  reverted alone when numbers shift.
- `git status --short` before commit (rule 5).

## Data

**BGL.log** — space-separated, 9 headers + message:
`label unix_ts date node time node_repeat type component level message...`
`label` `-` = normal, 41 other codes = alert. **`parts = line.split(maxsplit=9)` — always
maxsplit=9**, plain `split()` breaks on spaces inside the message.

**34,470 lines (0.726%) have 9 headers and an empty message.** Valid records, not parse
failures — `parse_bgl` keeps them with `message=""`. All 100% `normal`, so keeping them in
training (as `""` or `[EMPTY]`) would make "empty ⇒ normal" a perfect shortcut that inflates
macro F1. **Decided: `label_map` splits them into `bgl_empty_msg.parquet`, out of training.**
Reversing this needs human approval; log count + share + per-class distribution to stdout and
`experiments.md`. See the entry in `experiments.md`.

**nasa.csv** (volume detector only) — `host,time,method,url,response,bytes`.
Drop `Unnamed: 0`; drop rows with null `response`/`bytes`.

## Classes

4,747,963 lines total, 348,460 alerts (7.34%). 41 raw categories → 4 classes.

| class | count | of alerts | operator looks at |
|---|---|---|---|
| `normal` | 4,399,503 | — | — |
| `kernel_mem` | 218,075 | 62.6% | hardware |
| `kernel_ops` | 66,269 | 19.0% | system config |
| `app` | 63,845 | 18.3% | code |

Grouping is **this project's judgment, not an official taxonomy** — no complete BGL alert-code
doc exists, so codes were grouped by name. Some borders are arguable (`KERNPOW` could be
`kernel_mem`). State this in README.

`holdout_rare` (LINK/MAS/MON/MMCS, 271 lines) is excluded from train/eval and from macro F1.
Kept as categories the model never saw, to check they come out as `unknown` at inference.

**Remap trigger:** if the confusion matrix shows heavy **bidirectional** `kernel_mem` ↔
`kernel_ops` error, the mapping boundary may be wrong, not the model. Report it; never remap
on your own.

## Normalization — order is fixed

| # | target | regex | → |
|---|---|---|---|
| 1 | node id | `R\d+-M\d+-N\d+-C:J\d+-U\d+` | `[NODE]` |
| 2 | ip | `\d+\.\d+\.\d+\.\d+` | `[IP]` |
| 3 | hex | `0x[0-9a-fA-F]+` | `[HEX]` |
| 4 | path | `(?<![\w])(/[\w.\-]+)+` | `[PATH]` |
| 5 | remaining digits | `\d+` | `[NUM]` |

**Digits last** — substituting them first shatters node ids.

**Keep the `(?<![\w])` guard on the path rule.** Without it a slash inside a word matches:
`load/store` → `load[PATH]`, `BG/L` → `BG[PATH]`, `Torus/Tree/GI` → `Torus[PATH]`. That is
input corruption, not a model problem — `load/store` is the memory-access term that points at
`kernel_mem`. Real paths start after a space, `=` or `(`, so the guard doesn't affect them.

`normalize` reports three checks: (1) non-empty messages that became empty — **must be 0**,
(2) 200 before/after `[PATH]` samples dumped for review, (3) unique-message drop.
Baseline on full data: 358,329 → 24,693 unique (93.1%), 0 emptied, 0 mid-word `[PATH]`.
If (1) ≠ 0, report the dump before changing any regex.

## Done conditions

| module | done when |
|---|---|
| `parse_bgl` | parse failure < 0.1%; prints total lines + alert share |
| `label_map` | 0 unmapped; prints 4-class distribution; 271 holdout_rare split out |
| `normalize` | unique messages drop sharply; prints the 3 measurements above |
| `split` | prints per-split class distribution; no time-boundary overlap |
| `sample` | train only changed; val/test file hashes unchanged |
| `train` | best checkpoint by val macro F1; per-epoch log |
| `evaluate` | writes `metrics.json`, `cm.png`, `errors.csv` (50 misclassified) |
| `infer` | writes `windows.jsonl` with alert level + supporting logs per window |

## Tests

Fixture: `tests/fixtures/BGL_2k.log` (LogHub 2k sample).
`test_parse` (3 known lines split correctly) · `test_normalize` (node→`[NODE]`, digits→`[NUM]`,
plus a case that fails if rule order is reversed) · `test_split` (`unix_ts` monotonic at
boundaries, no row in two splits) · `test_sample` (val/test untouched).
Run the whole pipeline on the fixture before touching full data.

## Build order

`parse_bgl` → `label_map` → `normalize` → `sample` → **`evaluate` (build before training)** →
**E0** (majority-class constant) → **E1** (TF-IDF + logistic regression) → **E2** (BERT
baseline) → `calibrate` → `infer` → volume detector (independent, any time).

**Never skip E1.** Simple methods beating deep learning is common in log anomaly detection.
Without E1 there is no way to tell whether E2's number is good.

## Stop and ask

- You don't know where the raw log file is — **do not go find it yourself**
- Any decision that drops rows — report count + share + per-class distribution first
- A class has < 100 rows in any split after the time-based split
- Heavy bidirectional `kernel_mem` ↔ `kernel_ops` confusion
- Parse failure rate > 0.1%
- Alert share far from 7.3% (file may be truncated)
- Quiet-period false-alarm rate far above target (5/day)
