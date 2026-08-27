# log-anomaly-classifier

서버 로그 **한 줄**을 받아 정상 여부와 **이상의 원인 계열(4클래스)** 을 분류하고,
1분 구간 단위로 경보를 낸다.

기존 로그 이상탐지 모델은 대부분 정상/이상 2클래스다. 이 프로젝트는 원인까지 구분해
운영자가 알림을 받았을 때 **어디를 봐야 할지** 바로 알 수 있게 하는 것이 목표다.

---

## 결론부터: BERT 는 TF-IDF 에 졌다

**BERT 파인튜닝 5판이 TF-IDF + 로지스틱 회귀 베이스라인을 한 번도 이기지 못했다.**

| | 미등장 템플릿 macro F1 | 전체 macro F1 |
|---|---|---|
| **E1c — TF-IDF + 로지스틱 회귀** | **0.7451** | **0.9316** |
| E2w — BERT (최고 성적) | 0.7330 | 0.8289 |
| E0 — 다수 클래스 고정 (하한선) | 0.2411 | 0.2415 |

운영 지표도 같은 방향이다. 두 모델을 **같은 추론 파이프라인**에 세우고 test 구간
원본 로그 719,665줄을 통과시킨 결과:

| | 알림/일 | 탐지/일 | 오탐/일 | 재현율 |
|---|---|---|---|---|
| **E1c** | 5.1 | **3.7** | **1.4** | **59.3%** |
| E2w | 5.7 | 3.4 | 2.3 | 53.8% |

**E1c 가 더 많이 잡고 덜 틀린다.** 이건 실패가 아니라 결과다. 로그 이상탐지에서 단순
기법이 딥러닝을 이기는 것은 자주 보고되며, 그래서 이 프로젝트는 **BERT 를 만지기 전에
E0(다수 클래스) → E1(TF-IDF) 베이스라인을 먼저 세우도록** 빌드 순서를 정했다.
E1 이 없었다면 E2c 의 0.6829 를 보고 "이 정도면 괜찮다"고 넘어갔을 것이다.

## 왜 "미등장 템플릿" macro F1 인가

**전체 macro F1 은 단독으로 아무것도 말해주지 않는다.** BGL 은 같은 로그 템플릿이
수십~수백 번 반복되고, test 행의 24.4% 는 train 에도 있던 템플릿이다. 그 구간 점수는
성능이 아니라 **암기의 확인**이다.

| exp_id | 기등장 172,357행 (24.4%) | 미등장 534,615행 (75.6%) | 전체 |
|---|---|---|---|
| E0 | 0.2431 | 0.2411 | 0.2415 |
| E1 | 0.8988 | 0.5192 | 0.8067 |
| E1c | 0.9993 | **0.7451** | 0.9316 |
| E2c | **1.0000** | 0.4998 | 0.6829 |

기등장 열은 0.2431 → **1.0000** 으로 단조 증가한다. 모델이 좋아진 게 아니라 암기가
완성돼 가는 것이다. 전체 F1 은 이 둘을 24 대 76 으로 섞은 값이다.

`evaluate` 는 `--train-ref` 의 템플릿 집합으로 test 를 쪼개 `metrics.json` 에
`seen` / `unseen` 으로 남긴다. 애드혹 계산이 아니라 재현 가능한 코드 산출물이다.

## 실험 기록

각 실험은 **변수를 하나만** 바꾼다. 실패한 실험도 지우지 않는다.

| exp_id | 변경점 | 샘플링 | macro F1 | 미등장 F1 |
|---|---|---|---|---|
| E0 | 다수 클래스 고정 (하한선) | — | 0.2415 | 0.2411 |
| E1 | TF-IDF + 로지스틱 회귀 | none | 0.8067 | 0.5192 |
| **E1c** | E1 에서 샘플링만 cap 으로 | cap(2000) | **0.9316** | **0.7451** |
| E2c | E1c 에서 모델만 BERT 로 | cap(2000) | 0.6829 | 0.4998 |
| E2i | E2c 에서 체크포인트 선택 기준만 교체 | cap(2000) | 0.6429 | 0.5049 |
| **E2w** | E2i 에서 클래스 가중치만 끔 | cap(2000) | 0.8289 | 0.7330 |
| E2 | BERT 대표 성능 | none | 0.7760 | 0.5144 |
| E3 | E2 에서 샘플링만 ratio 로 | ratio(3) | 0.6410 | 0.5011 |

전체 기록과 근거는 [experiments.md](experiments.md) 에 있다.

## 무엇이 BERT 를 무너뜨렸나

E2c 의 오분류는 흩어져 있지 않았다. **미등장 템플릿 1종이 전부였다.**

```
MACHINE CHECK DCR read timeout (mc=e[NUM]x iar [HEX] lr [HEX])   실제 normal, 14,481건
```

BGL 에서 "MACHINE CHECK" 를 포함한 40,026행 중 **39,684행(99.1%)이 정상**이다.
machine check 는 대부분 정보성 메시지다. 그런데 `kernel_mem` 의 test support 가
100건뿐이라(전체의 0.01%), 이 템플릿 하나를 `kernel_mem` 으로 보내는 순간 precision 이
0.0069 로 떨어지고 **미등장 macro F1 이 0.73 ↔ 0.50 사이에서 통째로 흔들린다.**

발화 조건은 두 개다.

| 실험 | 샘플링 | 학습셋 kernel_mem 비중 | 클래스 가중치 | 실효 비중 | 오탐 | 미등장 F1 |
|---|---|---|---|---|---|---|
| E2c / E2i | cap(2000) | 1.68% | ×14.84 | ~25% | 14,482 | 0.4998 |
| **E2w (에폭 1)** | cap(2000) | 1.68% | 끔 | **1.68%** | **0** | **0.7330** |
| E2w (에폭 3) | cap(2000) | 1.68% | 끔 | 1.68% | 14,481 | 0.4848 |
| E2 | none | 6.60% | 끔 | 6.60% | 0 | 0.5144 |
| E3 | ratio(3) | **21.29%** | 끔 | **21.29%** | 14,481 | 0.5011 |

1. **학습셋에서 `kernel_mem` 의 실효 비중이 높을 때.** `w_c = N/(K*n_c)` 가 가중치를
   14.84 로 올리든, `ratio` 샘플링이 정상만 깎아 비중을 21.29% 로 올리든 결과는 같다.
   **클래스 가중치가 하던 일을 샘플링이 그대로 재현했다.**
2. **비중이 낮아도 2에폭 이상 돌릴 때.** 가중치를 껐는데도 에폭 3 에서 오탐이 돌아온다.

E2w 의 에폭 1 은 두 조건을 모두 피한 유일한 지점이고, **그래서 칼날 위다** —
정직한 선택 지표조차 에폭 1과 3을 0.0005 차이로만 구분했다.

**샘플링 축은 닫았다.** cap / none / ratio 세 지점 중 어느 것도 세 알럿 클래스를 동시에
살리지 못한다. cap 은 `ciod` app 템플릿을 자르고, none 은 미등장 알럿 recall 을 0 으로
만들고, ratio 는 `kernel_mem` 오탐을 부른다.

## 클래스 정의

| 클래스 | 의미 | 운영자가 볼 곳 | 알럿 중 비중 |
|---|---|---|---|
| `normal` | 정상 | — | — |
| `kernel_mem` | 메모리·주소변환 오류 | 하드웨어 | 62.6% |
| `kernel_ops` | 마운트·종료·복구 실패 | 시스템 설정 | 19.0% |
| `app` | 애플리케이션 오류 | 코드 | 18.3% |

추론 시 최대 softmax 확률이 임계값 미만이면 4클래스로 강제하지 않고 `unknown` 으로 표시한다.
임계값은 **val 에서만** 고른다.

> **이 4클래스 그룹핑은 공식 분류가 아니라 이 프로젝트에서 정한 기준이다.**
> BGL 알럿 코드에 대한 완전한 공식 문서가 없어 코드명을 근거로 판단했다.
> 경계가 애매한 항목이 있다 (예: `KERNPOW` 는 전원 관련이라 `kernel_mem` 으로 봐도 무방).
> 원본 41개 카테고리는 2007년 반자동 분류 결과이므로 라벨 자체의 신뢰도에도 한계가 있다.
> 전체 매핑은 [`configs/base.yaml`](configs/base.yaml) 에 있다.

`LINK*` / `MAS*` / `MON*` / `MMCS` 계열 271건은 `holdout_rare` 로 분리해 학습·평가에서 제외한다.
버리는 것이 아니라 **모델이 한 번도 보지 못한 카테고리**로 남겨,
추론 시 `unknown` 으로 분류되는지 확인하는 데 쓴다.

## 방법론에서 지킨 것

이 프로젝트의 숫자를 믿을 수 있게 하는 규칙들이다. 어기면 아래 모든 수치가 무효다.

- **랜덤 분할 금지.** `unix_ts` 오름차순 정렬 후 앞에서부터 70/15/15 로 자른다. BGL 은
  동일한 줄이 수십 번 반복되므로 랜덤 분할은 같은 줄을 train 과 test 에 동시에 넣는다.
- **val/test 의 클래스 비율은 건드리지 않는다.** 샘플링은 train 에만 적용한다.
  test 에서 정상을 덜어내면 같은 모델의 `kernel_mem` F1 이 0.190 → 0.842 로 뛴다.
  모델이 아니라 지운 행 수를 재는 숫자가 된다.
- **임계값은 val 에서 고르고 test 는 마지막에 한 번만 본다.** E2w 는 test 성적이 val 보다
  좋게 나왔다 — test 에 맞춘 흔적이 없다는 증거다.
- **accuracy 는 보고하지 않는다.** 전부 정상으로 찍어도 92.7% 가 나온다.
- **빈 메시지 34,470줄(0.726%)은 학습에서 뺐다.** 전부 정상이라 "메시지가 비었다 ⇒ 정상"
  이 100% 정확한 지름길 피처가 된다. 버리지 않고 별도 파일로 분리했다.
- **평시 기준선에서 06-12~06-14 장애 3일을 뺐다.** train `kernel_mem` 의 99.0% 가 이
  사흘에 몰려 있어, 넣으면 경보 임계가 0.6068 이 되어 **한 구간의 61% 가 `kernel_mem`
  이어야** 경보가 뜬다. 학습 데이터는 건드리지 않았다.

행을 버리는 결정은 전부 건수·비율·클래스 분포와 함께 [experiments.md](experiments.md) 에
기록했고, 되돌릴 수 있도록 별도 커밋으로 남겼다.

## 추론 파이프라인

줄 단위 분류와 구간 단위 발생량은 단위가 달라 그대로 합칠 수 없다.
**1분 버킷을 공통 키로** 삼아 줄 예측을 집계한 뒤 트래픽 신호와 조인한다.

```
원본 로그
  ├─ [줄 단위]  adapters → normalize → 모델 → (class, confidence)
  │                                              └→ 1분 버킷 집계 ─┐
  └─ [구간 단위] 1분 건수 → EWMA z-score ───────────────────────→ 병합 → windows.jsonl
```

판정 기준선은 추론 시점에 실시간으로 잡지 않는다. `calibrate` 가 train 구간에서 미리
계산해 `calibration.json` 에 남긴다 — 실시간으로 잡으면 장애가 지속될 때 그 상태를
평시로 학습해버린다.

출력 (`windows.jsonl`, 한 줄 = 1분 구간):

```json
{"window_start": "2005-11-05T09:13:00", "n_logs": 1204,
 "class_counts": {"normal": 1180, "kernel_mem": 20, "kernel_ops": 1, "app": 3, "unknown": 0},
 "anomaly_ratio": 0.0199,
 "traffic": {"count": 1204, "baseline": 800.2, "z": 3.8, "flag": "spike"},
 "alert": {"level": "warning", "reasons": ["kernel_mem 비율 1.7% (평시 0.1%, z=3.8)"]},
 "top_samples": [{"raw": "...", "pred": "kernel_mem", "conf": 0.97}]}
```

알림만 주고 끝내면 쓸모가 없어서, 근거가 된 실제 로그를 `top_samples` 로 같이 낸다.

**실사용 기준은 평시 구간 오탐 1일 5건 이하다.** test 구간에서 E1c 1.4건/일, E2w 2.3건/일.

## 데이터

원본 로그는 용량이 크고 재배포 조건이 있어 저장소에 포함하지 않는다. 직접 받아서 `data/raw/` 에 둔다.

| 파일 | 용도 | 출처 |
|---|---|---|
| `data/raw/BGL.log` | 분류기 학습·평가 (4,747,963줄 / 알럿 7.34%) | [LogHub](https://github.com/logpai/loghub) — BGL |
| `data/raw/nasa.csv` | 발생량 감지기 검증 (미착수) | NASA-HTTP (1995) 접근 로그 |

LogHub 데이터를 쓸 때는 원 논문 인용 조건이 있다. NASA-HTTP 는 자유 재배포가 가능하나
Kaggle 가공본을 쓴 경우 그 사실을 명시한다.

테스트 픽스처로는 LogHub 이 제공하는 2천 줄 샘플 `BGL_2k.log` 를 `tests/fixtures/` 에 둔다.

## 설치

```bash
uv sync
```

Python 3.11. `pip` 을 직접 호출하지 않는다.

의존성은 재현성을 위해 `pyproject.toml` 에 정확한 버전으로 고정했다.
특히 **`transformers` 는 5.x 다.** 인터넷에 도는 4.x 예제(`Trainer` 인자, 토크나이저 호출 방식)를
그대로 옮기면 동작하지 않는다. 5.x 문서를 기준으로 작성한다.

`torch` 는 PyPI 기본 휠이 CPU 전용이라 `pyproject.toml` 에서 CUDA 인덱스(cu126)를
명시한다. 개발은 `torch==2.13.0+cu126` / RTX 4070 에서 했다.

## 실행

전처리 (한 번만):

```bash
uv run python -m src.parse_bgl  --input data/raw/BGL.log --output data/interim/bgl_parsed.parquet
uv run python -m src.label_map  --input data/interim/bgl_parsed.parquet --output data/interim/bgl_labeled.parquet
uv run python -m src.normalize  --input data/interim/bgl_labeled.parquet --output data/interim/bgl_norm.parquet
uv run python -m src.split      --input data/interim/bgl_norm.parquet --outdir data/processed
uv run python -m src.sample     --input data/processed/train.parquet --output data/processed/train_cap.parquet --strategy cap
```

대표 모델 E1c (TF-IDF, CPU 몇 분):

```bash
uv run python -m src.baseline   --exp-id E1c --input data/processed/train_cap.parquet
uv run python -m src.evaluate   --exp-id E1c
```

BERT 최고 성적 E2w (GPU 약 20분):

```bash
uv run python -m src.train      --exp-id E2w --input data/processed/train_cap.parquet --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --no-class-weight
uv run python -m src.evaluate   --exp-id E2w
```

추론:

```bash
uv run python -m src.calibrate  --exp-id E1c
uv run python -m src.infer      --exp-id E1c --input tests/fixtures/BGL_2k.log
```

각 단계는 파일을 남기고 끝난다. 다음 단계는 앞 단계의 출력만 읽는다.
`train` 은 에폭마다 `last.pt` 에 전체 상태를 남기므로, 중간에 꺼져도 같은 명령을 다시
실행하면 이어서 학습한다. 진행 상황은 `runs/<exp_id>/progress.html` 을 열어두면 5초마다 갱신된다.

```bash
uv run pytest && uv run ruff check src/
```

## 한계

- **BERT 가 이 문제를 못 푼 이유를 표현력 부족으로 결론짓지 않았다.** 진단상 원인은
  템플릿 1종의 사전확률이다. 도메인 코퍼스 MLM 사전학습(E7)은 **시도하지 않았다.**
- **미등장 `kernel_ops` 는 세 모델 모두 F1 0.02~0.05 다.** E1c 도 recall 1.0% 다.
  test 구간에 새 장애 유형(`Error receiving packet on tree network`)이 등장하는데
  어느 모델도 잡지 못한다. 이 프로젝트의 최대 미해결 문제다.
- **`kernel_mem` 의 test support 가 100건뿐이다.** F1 이 재현율이 아니라 오탐률로
  결정된다. 정상 660,735건 중 오탐 66건 이하라야 F1 0.65 가 나온다.
- **시드 1개.** E2w 의 에폭 1 선택은 지표 차이 0.0005 위에 서 있는데 재현성을 확인하지 않았다.
- **E1c 에서는 `unknown` 이 꺼진다.** val 에서 임계를 올려도 macro precision 이 0.9980 으로
  평평해 얻는 게 없다. `holdout_rare` 271건 검증은 E1c 로는 통과할 수 없다.
- BGL 단일 시스템 로그로 학습했다. 다른 포맷을 넣으려면 `src/adapters/` 에
  `(timestamp, message)` 로 변환하는 파서를 추가해야 하며, 어댑터 없이 넣은 결과는 무의미하다.
- 발생량 감지기의 NASA 검증(`parse_nasa`)은 데이터 미확보로 미착수다.
  EWMA z-score 자체는 `infer` 안에서 동작한다.
- 4클래스 매핑은 위에 적은 대로 이 프로젝트가 정한 기준이다.

## 문서

- 설계: [docs/SPEC.md](docs/SPEC.md)
- 실험 기록: [experiments.md](experiments.md)
- 위기 기록(잡은 것 / 아직 열려 있는 것): [docs/RISKS.md](docs/RISKS.md)
- 진행 상황: [PROGRESS.md](PROGRESS.md)
- 작업 규칙: [.claude/CLAUDE.md](.claude/CLAUDE.md)
