# log-anomaly-classifier

[![CI](https://github.com/lkw-k/LogTriage/actions/workflows/ci.yml/badge.svg)](https://github.com/lkw-k/LogTriage/actions/workflows/ci.yml)

Blue Gene/L 서버 로그 **한 줄**을 4클래스(`normal` / `kernel_mem` / `kernel_ops` / `app`)로
분류하고, 1분 구간 단위로 경보를 낸다. 정상/이상 2분류가 아니라 **원인 계열**을 구분해
운영자가 어디를 봐야 할지 알려주는 것이 목표다.

모델: [huggingface.co/illimax/bgl-log-triage-bert](https://huggingface.co/illimax/bgl-log-triage-bert)

## 결과

**BERT 파인튜닝 5판이 TF-IDF + 로지스틱 회귀에 한 번도 이기지 못했다.**

| exp_id | 변경점 | 미등장 F1 | 전체 F1 |
|---|---|---|---|
| E0 | 다수 클래스 고정 (하한선) | 0.2411 | 0.2415 |
| E1 | TF-IDF + 로지스틱 회귀 | 0.5192 | 0.8067 |
| **E1c** | E1 에서 샘플링만 cap(2000) 으로 | **0.7451** | **0.9316** |
| E2c | E1c 에서 모델만 BERT 로 | 0.4998 | 0.6829 |
| E2i | E2c 에서 체크포인트 선택 기준만 교체 | 0.5049 | 0.6429 |
| **E2w** | E2i 에서 클래스 가중치만 끔 (BERT 최고) | 0.7330 | 0.8289 |
| E2 | BERT, 샘플링 none | 0.5144 | 0.7760 |
| E3 | E2 에서 샘플링만 ratio(3) 로 | 0.5011 | 0.6410 |

로그 이상탐지에서 단순 기법이 딥러닝을 이기는 건 자주 보고된다. 그래서 BERT 를 만지기 전에
E0 → E1 을 먼저 세웠고, 그 덕에 E2c 의 0.6829 를 "괜찮은 점수"로 착각하지 않았다.

## 판정 기준은 미등장 템플릿 macro F1 이다

BGL 은 같은 로그 템플릿이 수십~수백 번 반복된다. **test 행의 24.4% 는 train 에도 있던
템플릿이라, 그 구간 점수는 성능이 아니라 암기의 확인이다.** 실제로 기등장 구간 F1 은
E0 0.2431 → E1 0.8988 → E1c 0.9993 → E2c **1.0000** 으로 단조 증가한다. 모델이 좋아진 게
아니라 암기가 완성돼 가는 것이다.

`evaluate` 가 `--train-ref` 의 템플릿 집합으로 test 를 쪼개 `metrics.json` 에
`seen` / `unseen` 으로 남긴다. accuracy 는 보고하지 않는다 — test 를 전부 `normal` 로 찍기만 해도
93.5% 가 나온다 (test 의 정상 비율 660,735/706,972).

## BERT 는 왜 졌나

E2c 의 오분류는 흩어져 있지 않았다. **미등장 템플릿 1종(`MACHINE CHECK DCR read timeout`,
14,481건, 실제로는 정상)이 전부였다.** BGL 에서 machine check 를 포함한 40,026행(대소문자 무관) 중
39,684행(99.1%)이 정상인데, `kernel_mem` 의 test support 가 100건뿐이라 이 한 템플릿의
분류 결과가 미등장 macro F1 을 **0.73 ↔ 0.50** 사이에서 통째로 흔든다.

발화 조건 두 개를 실험으로 분리했다.

1. 학습셋에서 `kernel_mem` 의 실효 비중이 높을 때. 클래스 가중치가 14.84 로 올리든(E2c),
   `ratio` 샘플링이 비중을 21.29% 로 올리든(E3) 결과가 같다.
2. 비중이 낮아도 2에폭 이상 돌릴 때. 가중치를 껐는데도 E2w 의 에폭 3 에서 재발한다.

E2w 의 에폭 1 은 둘 다 피한 유일한 지점이고, 선택 지표가 에폭 1과 3을 **0.0005 차이**로만
구분했다. 즉 **운이 좋았던 것에 가깝다.** cap / none / ratio 세 샘플링 중 어느 것도 세 알럿
클래스를 동시에 살리지 못해 이 축은 닫았다. 근거는 [experiments.md](experiments.md).

## 추론 파이프라인

줄 단위 분류와 구간 단위 발생량은 단위가 달라 그대로 못 합친다. **1분 버킷을 공통 키로**
집계한 뒤 조인한다.

```
원본 로그
  ├─ [줄]   adapters → normalize → 모델 → (class, confidence)
  │                                          └→ 1분 버킷 집계 ─┐
  └─ [구간] 1분 건수 → EWMA z-score ────────────────────────→ 병합 → windows.jsonl
```

판정 기준선은 `calibrate` 가 train 구간에서 미리 계산해 `calibration.json` 에 남긴다.
추론 시점에 실시간으로 잡으면 장애가 지속될 때 그 상태를 평시로 학습해버린다.

test 구간 원본 로그 719,665줄을 통과시킨 결과:

| | 알림/일 | 탐지/일 | 오탐/일 | 재현율 |
|---|---|---|---|---|
| **E1c** | 5.1 | 3.7 | **1.4** | 59.3% |
| E2w | 5.7 | 3.4 | 2.3 | 53.8% |

목표였던 **평시 오탐 1일 5건 이하**는 둘 다 만족한다. 다만 **재현율 59.3% 는 자랑할 수치가
아니다.** 여기서 "이상 구간"은 알럿 라벨이 1행이라도 있는 버킷이라 기준이 매우 느슨한데도
40% 를 놓친다. 오탐 예산을 지키기 위해 재현율을 포기한 결과다.

## 실행

원본 로그는 용량·재배포 조건 때문에 저장소에 없다. [LogHub](https://github.com/logpai/loghub)
에서 BGL 을 받아 `data/raw/BGL.log` 에 둔다 (4,747,963줄 / 알럿 348,460건 = 7.34%).

**데이터 출처.** BGL 은 Lawrence Livermore National Labs(LLNL)의 BlueGene/L 슈퍼컴퓨터
(프로세서 131,072개)에서 수집된 로그다. 첫 열의 `-` 가 정상, 나머지 41개 코드가 알럿이다.
원 논문은 Oliner & Stearley, *What Supercomputers Say: A Study of Five System Logs*, DSN 2007.
배포처인 LogHub 는 **연구·학술 목적으로 자유롭게 쓸 수 있으나, 사용 시 저장소 URL 을 밝히고
아래를 인용할 것**을 요구한다.

> Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
> *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.*
> IEEE ISSRE 2023. [arXiv:2008.06448](https://arxiv.org/abs/2008.06448)

```bash
uv sync    # Python 3.11. pip 을 직접 호출하지 않는다.
```

`transformers` 는 **5.x** 다 (4.x 예제는 그대로 안 돈다). `torch` 는 CUDA 인덱스(cu126)로
고정돼 있어 **Linux / Windows 만 지원한다 — macOS 는 cu126 빌드가 없어 `uv sync` 가 실패한다.**

```bash
# 전처리 (한 번만)
uv run python -m src.parse_bgl  --input data/raw/BGL.log --output data/interim/bgl_parsed.parquet
uv run python -m src.label_map  --input data/interim/bgl_parsed.parquet --output data/interim/bgl_labeled.parquet
uv run python -m src.normalize  --input data/interim/bgl_labeled.parquet --output data/interim/bgl_norm.parquet
uv run python -m src.split      --input data/interim/bgl_norm.parquet --outdir data/processed
uv run python -m src.sample     --input data/processed/train.parquet --output data/processed/train_cap.parquet --strategy cap

# 대표 모델 E1c (CPU 몇 분) / BERT 최고 성적 E2w (GPU 약 20분)
uv run python -m src.baseline   --exp-id E1c --input data/processed/train_cap.parquet
uv run python -m src.evaluate   --exp-id E1c
uv run python -m src.train      --exp-id E2w --input data/processed/train_cap.parquet --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --no-class-weight
uv run python -m src.evaluate   --exp-id E2w

# 추론
uv run python -m src.calibrate  --exp-id E1c
uv run python -m src.infer      --exp-id E1c --input tests/fixtures/BGL_2k.log

uv run pytest && uv run ruff check src/
```

각 단계는 파일을 남기고 끝나며, 다음 단계는 앞 단계의 출력만 읽는다. `train` 은 에폭마다
`last.pt` 에 전체 상태를 남겨 중간에 꺼져도 같은 명령으로 이어서 학습한다.

## 숫자를 믿을 수 있게 하는 규칙

어기면 위의 모든 수치가 무효다.

- **랜덤 분할 금지.** `unix_ts` 오름차순 70/15/15. BGL 은 동일한 줄이 수십 번 반복되므로
  랜덤 분할은 같은 줄을 train 과 test 에 동시에 넣는다.
- **val/test 의 클래스 비율은 건드리지 않는다.** 샘플링은 train 에만. test 에서 정상을
  덜어내면 같은 모델의 `kernel_mem` F1 이 0.190 → 0.842 로 뛴다.
- **임계값은 val 에서 고르고 test 는 마지막에 한 번만 본다.** E2w 는 test 성적이 val 보다
  좋게 나왔다 — test 에 맞추지 않았다는 증거다.
- **행을 버리는 결정은 건수·비율·클래스 분포와 함께 기록하고 별도 커밋으로 남긴다.**
  빈 메시지 34,470줄(전부 정상이라 "비었다 ⇒ 정상" 지름길이 된다), 평시 기준선에서
  06-12~06-14 장애 3일(train `kernel_mem` 의 99.0%).

## 클래스

| 클래스 | 의미 | 운영자가 볼 곳 | 알럿 중 비중 |
|---|---|---|---|
| `normal` | 정상 | — | — |
| `kernel_mem` | 메모리·주소변환 오류 | 하드웨어 | 62.6% |
| `kernel_ops` | 마운트·종료·복구 실패 | 시스템 설정 | 19.0% |
| `app` | 애플리케이션 오류 | 코드 | 18.3% |

> **공식 분류가 아니라 이 프로젝트가 정한 기준이다.** BGL 알럿 코드의 완전한 문서가 없어
> 코드명으로 41개 카테고리를 묶었고 경계가 애매한 항목이 있다. 원본 카테고리 자체도 2007년
> 반자동 분류 결과라 라벨 신뢰도에 한계가 있다. 매핑은 [`configs/base.yaml`](configs/base.yaml).

`LINK*` / `MAS*` / `MON*` / `MMCS` 271건은 `holdout_rare` 로 분리해 학습·평가에서 뺐다.
모델이 한 번도 못 본 카테고리로 남겨 `unknown` 검증에 쓰려던 것인데, **아래 한계를 보라.**

## 한계 — 안 한 것과 안 되는 것

- **도메인 코퍼스 MLM 사전학습(E7)을 안 했다.** "BERT 가 졌다"는 결론에 대해 "도메인
  사전학습을 했다면?" 은 답하지 못한다. 진단상 원인이 템플릿 1종의 사전확률이라 기대값이
  낮다고 판단해 미뤘을 뿐, 확인하지 않았다.
- **미등장 `kernel_ops` 는 모든 모델에서 F1 0.00~0.05 다.** E1c 도 recall 1.0%. test 구간에
  새 장애 유형(`Error receiving packet on tree network`)이 등장하는데 아무도 못 잡는다.
  이 프로젝트의 최대 미해결 문제이고, 샘플링 말고 다른 시도를 하지 않았다.
- **`kernel_mem` 의 test support 는 100건뿐이다.** F1 이 재현율이 아니라 오탐률로 결정된다.
  정상 660,735건 중 오탐 66건 이하라야 F1 0.65 다. 표본이 얇아 이 클래스 수치는 불안정하다.
- **시드 1개.** E2w 의 에폭 1 선택은 0.0005 차이 위에 서 있는데 재현성을 확인하지 않았다.
- **`holdout_rare` 검증은 통과하지 못한다.** E1c 는 `conf_threshold` 가 0.0 으로 뽑혀
  `unknown` 이 아예 꺼진다 (val 에서 임계를 올려도 macro precision 이 0.9980 으로 평평).
  E2w 는 동작하지만 대표 모델이 아니다.
- **트래픽 감지기의 NASA 검증(`parse_nasa`)은 미착수다.** 데이터를 확보하지 못했다.
  EWMA z-score 자체는 `infer` 안에서 동작하지만, BGL 은 로그가 있는 분이 9% 뿐이라
  (train 11,293분 / 125,687분) 최소 발생량 가드 없이는 쓸 수 없었다.
- **E4~E6 은 돌리지 않았다.** E4(balanced)는 E3 보다 나쁠 것이 예측돼 생략했고, 예측일 뿐이다.
- **BGL 단일 시스템 로그다.** 다른 포맷은 `src/adapters/` 에 `(timestamp, message)` 변환기를
  추가해야 하고, 어댑터 없이 넣은 결과는 무의미하다.

## CI / 배포

CI 는 매 푸시마다 리눅스에서 `uv sync --frozen` → `ruff` → `pytest` → 픽스처 파이프라인
스모크를 돌린다. "내 컴퓨터에서만 되는" 상태를 막는 게 목적이라 락파일을 그대로 쓴다.

모델 배포는 로컬에서 한다 (`runs/` 가 저장소에 없어 Actions 에서 모델 파일을 볼 수 없다).

```bash
uv run python -m src.publish --exp-id E2w --repo illimax/bgl-log-triage-bert --dry-run
uv run python -m src.publish --exp-id E2w --repo illimax/bgl-log-triage-bert
```

**모델 카드의 평가 수치는 `runs/<exp_id>/metrics.json` 에서 생성한다.** 손으로 적지 않으므로
재학습 → `evaluate` → `publish` 만 하면 갱신된다.

> **HF 모델을 쓸 때는 정규화를 먼저 해야 한다.** 원본 로그를 그대로 넣으면 test 719,665줄 중
> **12.95% 의 예측이 바뀐다** (`normal`→`kernel_ops` 78,674 / `normal`→`kernel_mem` 14,481).
> 에러 없이 조용히 틀리므로 제일 위험하다. 규칙은 모델 카드에 코드째 실려 있다.

## 문서

[docs/SPEC.md](docs/SPEC.md) 설계 · [experiments.md](experiments.md) 실험 기록 ·
[docs/RISKS.md](docs/RISKS.md) 위기 기록 · [PROGRESS.md](PROGRESS.md) 진행 상황
