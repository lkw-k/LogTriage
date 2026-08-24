# 로그 이상 원인 분류 — 파이프라인 및 학습 계획

## 0. 프로젝트 정의

서버 로그 한 줄을 입력받아 **정상 여부 + 이상의 원인 계열**을 분류하는 모델.
기존 로그 이상탐지 모델은 대부분 정상/이상 2클래스. 이 프로젝트는 원인까지 구분해서
운영자가 알림을 받았을 때 어디를 봐야 할지 바로 알 수 있게 하는 것이 목표.

부가 모듈로 트래픽 급증/급감 감지기를 붙인다. 트래픽은 로그 한 줄로 판단할 수 없는
시간 구간의 성질이므로 모델이 아니라 통계 규칙으로 처리한다.

| | 로그 분류기 | 발생량 감지기 |
|---|---|---|
| 입력 | 로그 한 줄 | 1분당 로그 발생 건수 |
| 출력 | 4클래스 | 이상 구간 플래그 |
| 방식 | BERT 계열 파인튜닝 | EWMA + z-score |
| 학습 데이터 | BGL (LogHub) | 불필요 |
| 검증 데이터 | BGL test 구간 | NASA-HTTP |

**발생량 감지기는 "요청 수"가 아니라 "로그 발생 건수"로 일반화한다.**
분류기는 BGL로 학습하고 감지기를 NASA로 만들면 서로 다른 시스템이라 같은 로그에
두 신호를 붙일 수 없다. 타임스탬프만 있으면 되는 형태로 일반화하면 BGL에도 그대로 적용된다.
BGL에서는 로그 폭주(log storm) 감지가 되는데, 이는 실제 장애의 전조 신호다.
NASA-HTTP는 정답 구간(허리케인 정지)이 있으므로 **감지기 검증용**으로만 쓴다.

---

## 1. 레포 구조

```
log-anomaly-classifier/
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/            # BGL.log, nasa.csv          (git 제외)
│   ├── interim/        # 파싱 결과 parquet          (git 제외)
│   └── processed/      # train/val/test             (git 제외)
├── src/
│   ├── parse_bgl.py
│   ├── label_map.py
│   ├── normalize.py
│   ├── split.py
│   ├── sample.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py    # 평시 기준선 산출 → calibration.json
│   ├── infer.py        # 최종 추론 진입점
│   ├── adapters/       # 로그 포맷별 파서
│   │   └── bgl.py
│   └── traffic/
│       ├── parse_nasa.py
│       └── detect.py
├── notebooks/
│   └── 00_explore.ipynb
├── experiments.md
└── README.md
```

`.gitignore` 에 `data/` 전체를 넣는다. 원본 로그는 용량이 크고 재배포 조건이 있으므로
레포에 올리지 않고 README에 다운로드 경로만 적는다.

---

## 2. 데이터 흐름

```
BGL.log
  ├─[parse_bgl]──→ interim/bgl_parsed.parquet      10컬럼 분리
  ├─[label_map]──→ interim/bgl_labeled.parquet     + label4 컬럼
  ├─[normalize]──→ interim/bgl_norm.parquet        + msg_norm 컬럼
  ├─[split]──────→ processed/{train,val,test}.parquet   시간순 70/15/15
  ├─[sample]─────→ processed/train_sampled.parquet      train만 축소
  ├─[train]──────→ runs/<exp_id>/checkpoint/
  └─[evaluate]───→ runs/<exp_id>/{metrics.json, cm.png, errors.csv}

nasa.csv
  ├─[parse_nasa]─→ interim/nasa_per_min.parquet    1분 단위 집계
  └─[detect]─────→ runs/traffic/anomalies.csv
```

각 단계는 파일을 남기고 끝난다. 다음 단계는 앞 단계 출력만 읽는다.
학습에서 문제가 생겨도 앞 단계를 다시 돌릴 필요가 없고, 어디가 잘못됐는지 좁혀 들어갈 수 있다.

---

## 3. 단계별 명세

### 3-1. parse_bgl.py

- 입력: `data/raw/BGL.log`
- 출력: `interim/bgl_parsed.parquet`
- 컬럼: `label, unix_ts, date, node, time, node_repeat, type, component, level, message`

```python
parts = line.split(maxsplit=9)   # 헤더 9개 + 메시지 통째로
```

`split()` 을 그냥 쓰면 메시지 안의 공백 때문에 줄마다 컬럼 수가 달라진다.

**검증 항목**
- 총 줄 수 (원본 기준 4,747,963)
- 파싱 실패율 < 0.1%
- `label != "-"` 비율이 7.3% 근처
- `unix_ts` 최소/최대 → 데이터 기간 확인

세 번째가 크게 어긋나면 파일이 잘린 것이므로 재다운로드.

### 3-2. label_map.py

- 입력: `bgl_parsed.parquet`
- 출력: `bgl_labeled.parquet` (+ `label4` 컬럼)

41개 원본 카테고리를 4개로 묶는다. 매핑은 `configs/base.yaml` 에 명시한다.

```yaml
label_map:
  normal:   ["-"]
  kernel_hw: [KERNDTLB, KERNSTOR, KERNTERM, KERNMNTF, ...]
  app:       [APPSEV, APPBUSY, ...]
  other:     [...]
```

**규칙: 매핑되지 않은 카테고리가 하나라도 있으면 에러로 중단한다.**
조용히 `other` 로 넣으면 나중에 성능이 이상할 때 원인을 못 찾는다.

> 이 매핑은 실제 라벨 분포를 확인한 뒤 확정한다. (현재 미정)

### 3-3. normalize.py

- 입력: `bgl_labeled.parquet`
- 출력: `bgl_norm.parquet` (+ `msg_norm` 컬럼)

정규화하지 않으면 모델이 노드 ID나 메모리 주소를 통째로 외워버린다.
학습 구간에만 나온 노드 ID를 이상의 근거로 삼게 되고, 새 노드에서는 전혀 동작하지 않는다.

**적용 순서 (순서가 중요하다)**

| 순서 | 대상 | 정규식 | 치환 |
|---|---|---|---|
| 1 | 노드 ID | `R\d+-M\d+-N\d+-C:J\d+-U\d+` | `[NODE]` |
| 2 | IP 주소 | `\d+\.\d+\.\d+\.\d+` | `[IP]` |
| 3 | 16진수 | `0x[0-9a-fA-F]+` | `[HEX]` |
| 4 | 경로 | `(/[\w.\-]+)+` | `[PATH]` |
| 5 | 남은 숫자 | `\d+` | `[NUM]` |

숫자 치환을 먼저 하면 노드 ID가 `R[NUM]-M[NUM]-...` 로 부서지므로 반드시 마지막.

**검증**: 정규화 전후 고유 메시지 종류 수를 비교한다. 크게 줄어들면(수만 → 수천) 정상 동작.

### 3-4. split.py

- 입력: `bgl_norm.parquet`
- 출력: `processed/{train,val,test}.parquet`

`unix_ts` 오름차순 정렬 후 앞에서부터 70 / 15 / 15 로 자른다.

**랜덤 분할 금지.** BGL은 동일한 로그 줄이 수십 회 반복되므로 랜덤으로 나누면
완전히 같은 줄이 학습셋과 테스트셋에 동시에 들어간다. 그 상태의 F1은 의미가 없다.

**여기서 클래스 비율을 조작하지 않는다.** 원본 그대로 저장한다.

**검증**: 세 구간의 `label4` 분포를 각각 출력. val과 test 분포가 크게 다르면 경계를 조정한다.

### 3-5. sample.py

- 입력: `processed/train.parquet`
- 출력: `processed/train_sampled.parquet`
- **train에만 적용한다. val / test는 손대지 않는다.**

`--strategy` 인자:

| 값 | 내용 |
|---|---|
| `none` | 원본 유지 (class weight로 대응) |
| `ratio` | 정상을 이상 합계의 3배까지만 남김 |
| `balanced` | 최소 클래스 크기에 맞춰 전부 하향 |

val / test 크기를 줄이고 싶으면 **층화 샘플링으로 비율을 유지한 채** 줄인다.
예: test 10만 줄 → 정상 9만 3천 / 이상 7천.
비율을 1:1로 맞추면 실제 운영(정상 92.7%)과 달라져 F1이 부풀려진다.

**검증**: 출력 파일의 클래스별 개수와 비율 출력.

### 3-6. dataset.py

- 토크나이징 담당
- 입력 텍스트 구성은 설정으로 선택:
  - `msg_only`: `msg_norm`
  - `with_meta`: `component + " " + level + " " + msg_norm`
- `max_length`: `msg_norm` 토큰 길이의 95 백분위수로 결정 (BGL은 64 전후 예상)

**주의**: `with_meta` 에서 `level`(FATAL/INFO 등)을 넣으면 성능이 크게 오를 수 있는데,
이는 실력이 아니라 정답에 가까운 힌트를 준 것에 가깝다.
두 설정을 모두 돌리고 **`msg_only` 를 대표 성능으로 보고**한다.

### 3-7. train.py

아래 4절 참조.

### 3-8. evaluate.py

- 클래스별 precision / recall / F1
- **macro F1** (대표 지표)
- 4×4 혼동행렬 (`cm.png`)
- **오분류 샘플 상위 50개를 `errors.csv` 로 덤프** — 다음 개선 방향은 여기서 나온다

Accuracy는 보고하지 않는다. 정상이 92.7%라 전부 정상으로 찍어도 92.7%가 나온다.

### 3-9. traffic/

**parse_nasa.py**
- `time`(유닉스 초) → datetime → 1분 단위 `resample().size()`
- 결측 행 제거, `Unnamed: 0` 삭제

**detect.py**
- EWMA(span=60)로 기준선 계산
- 잔차의 이동 표준편차로 z-score
- `|z| > 3` 이면 이상 플래그

**검증**: 1995-08-01 14:52 ~ 08-03 04:36 구간(허리케인으로 서버 정지)이 이상으로 잡히는지 확인.
안 잡히면 파라미터 조정. 라벨 없이도 정답을 아는 구간이므로 테스트용으로 쓴다.

---

## 4. 학습 계획

### 4-1. 기본 설정

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | `bert-base-uncased` | 로그는 영문 시스템 메시지 |
| 헤드 | Linear (768 → 4) | |
| 손실 | CrossEntropy + class weight | `w_c = N / (K * n_c)` |
| 옵티마이저 | AdamW | weight_decay 0.01 |
| 학습률 | 2e-5 | |
| 스케줄러 | linear + warmup 10% | |
| 배치 | 32 | |
| 에폭 | 최대 5 | |
| 조기 종료 | val macro F1, patience 2 | |
| 체크포인트 | val macro F1 최고 지점 | |
| 시드 | 42 (고정) | 재현성 |

class weight는 `sample.py` 에서 `none` 을 골랐을 때 필수, `balanced` 를 골랐을 때는 불필요.

### 4-2. 실험 순서

각 실험은 하나의 변수만 바꾼다. 여러 개를 동시에 바꾸면 무엇이 효과를 냈는지 알 수 없다.

| # | 목적 | 설정 | 확인할 것 |
|---|---|---|---|
| E0 | 하한선 | 다수 클래스 항상 예측 | macro F1 하한 (0.25 근처) |
| E1 | 고전 기법 하한선 | TF-IDF + 로지스틱 회귀 | 딥러닝이 이걸 못 이기면 문제 |
| E2 | 베이스라인 | BERT + `msg_only` + `none` + class weight | **대표 성능** |
| E3 | 샘플링 비교 | E2에서 `ratio` 로 변경 | 정상 축소가 도움되는지 |
| E4 | 샘플링 비교 | E2에서 `balanced` 로 변경 | 과도한 축소의 부작용 |
| E5 | 메타 정보 | E2에서 `with_meta` 로 변경 | 참고용 (대표 아님) |
| E6 | 중복 제거 | train에서 완전 동일 줄 제거 | 반복 로그의 영향 |
| E7 | 프리트레이닝 | 로그 코퍼스 MLM 후 E2 재실행 | **E2 대비 몇 %p 올랐는가** |

**E1을 반드시 먼저 한다.** 로그 이상탐지 분야에서는 단순 기법이 딥러닝을 이기는 경우가
자주 보고되었다. E1이 E2와 비슷하면 딥러닝을 쓸 이유를 다시 생각해야 한다.

### 4-3. 프리트레이닝 (E7)

E2가 끝난 뒤에 진행한다. 베이스라인 없이는 프리트레이닝이 도움됐다는 것을 증명할 수 없다.

- 코퍼스: BGL train 구간 + Thunderbird + HPC + OpenStack (LogHub)
- 방식: MLM, 마스킹 비율 15%
- 이후 동일한 조건으로 4클래스 파인튜닝 → E2와 macro F1 비교

기대 효과는 **처음 보는 로그 템플릿에 대한 강건성**이다.
BGL은 학습 구간에 없던 템플릿이 테스트 구간에 1,000개 이상 등장하므로,
전체 F1보다 **미등장 템플릿만 골라낸 부분집합에서의 F1**을 따로 측정한다.
이 수치가 오르면 프리트레이닝이 의미 있었다는 뜻이다.

### 4-4. 실험 기록

매 실험마다 `experiments.md` 에 한 줄씩 남긴다.

| exp_id | 날짜 | 변경점 | 샘플링 | 입력 | macro F1 | 미등장템플릿 F1 | 비고 |
|---|---|---|---|---|---|---|---|
| E2 | | 베이스라인 | none | msg_only | | | |

기록하지 않으면 나중에 무엇이 좋았는지 기억하지 못한다.

---

## 5. 추론 파이프라인 (입력 → 출력)

학습 파이프라인과 별개로, 완성된 결과물이 실제로 무엇을 받고 무엇을 내놓는지 정의한다.
이것이 이 프로젝트의 최종 산출물이다.

### 5-1. 두 신호의 결합 지점

분류기는 **로그 한 줄** 단위로, 트래픽 감지기는 **1분 구간** 단위로 결과를 낸다.
단위가 다르므로 그대로는 합칠 수 없다.

해결: **1분 시간 구간을 공통 키로 삼는다.**
줄 단위 예측을 같은 1분 버킷으로 집계한 뒤, 트래픽 신호와 조인한다.

```
raw log lines
   │
   ├─→ [줄 단위]  parse → normalize → tokenize → model → (class, confidence)
   │                                                          │
   │                                          1분 버킷으로 집계 │
   │                                                          ▼
   └─→ [구간 단위] timestamp → 1분 요청수 → EWMA z-score ──→ 병합 → 최종 출력
```

### 5-2. infer.py — 입력 계약

```python
predict(lines: list[str], window: str = "1min") -> InferResult
```

- 입력: 원본 로그 줄의 리스트 (파일 통째로 또는 스트림 버퍼)
- 파싱 실패한 줄은 버리지 않고 `unparseable` 로 별도 집계한다.
  파싱 실패율 자체가 이상 신호일 수 있다.

**포맷 어댑터**: 분류기는 BGL 포맷으로 학습했다. 다른 포맷의 로그를 넣으려면
`adapters/` 에 파서를 추가해 `(timestamp, message)` 형태로 변환한 뒤 넣는다.
어댑터 없이 다른 포맷을 넣으면 결과는 무의미하다.

### 5-3. 출력 계약

**줄 단위 출력** (`predictions.jsonl`)

```json
{"ts": 1117838570, "pred": "kernel_hw", "conf": 0.94, "raw": "KERNDTLB ..."}
```

**구간 단위 출력** (`windows.jsonl`) — 사람이 실제로 보는 것

```json
{
  "window_start": "2005-06-03T15:42:00",
  "n_logs": 1204,
  "class_counts": {"normal": 1180, "kernel_hw": 20, "app": 3, "other": 1, "unknown": 0},
  "anomaly_ratio": 0.0199,
  "traffic": {"count": 1204, "baseline": 800.2, "z": 3.8, "flag": "spike"},
  "alert": {
    "level": "warning",
    "reasons": [
      "kernel_hw 비율 1.7% (평시 0.3%)",
      "트래픽 급증 z=3.8"
    ]
  },
  "top_samples": [
    {"raw": "KERNDTLB ... data TLB error interrupt", "pred": "kernel_hw", "conf": 0.97}
  ]
}
```

`top_samples` 는 그 구간에서 확신도가 가장 높은 이상 로그 3건.
알림만 주고 끝내면 쓸모가 없고, 근거가 되는 실제 로그를 같이 보여줘야 한다.

### 5-4. 판정 규칙

**평시 기준선은 학습 구간에서 미리 계산해 `calibration.json` 으로 저장한다.**
추론 시점에 실시간으로 기준을 잡으면, 장애가 지속될 때 그 상태를 평시로 학습해버린다.

```json
{
  "class_ratio_mean": {"kernel_hw": 0.003, "app": 0.0005, "other": 0.0008},
  "class_ratio_std":  {"kernel_hw": 0.002, "app": 0.0004, "other": 0.0006},
  "traffic_ewma_span": 60,
  "conf_threshold": 0.65
}
```

| 조건 | 판정 |
|---|---|
| 모든 지표 평시 범위 | `ok` |
| 특정 클래스 비율 z > 3 | `warning` + 해당 클래스명 |
| 특정 클래스 비율 z > 5 | `critical` |
| 트래픽 z > 3 | `warning` (급증) |
| 트래픽 z < -3 | `critical` (급감 — 서비스 중단 의심) |
| `unknown` 비율 > 10% | `warning` (미지의 로그 패턴 유입) |

비율 기준만 쓰면 로그가 적은 구간에서 한 건만 떠도 비율이 튄다.
**최소 건수 조건(예: 5건 이상)을 같이 걸어** 오탐을 막는다.

### 5-5. 미분류(`unknown`) 클래스

모델의 최대 softmax 확률이 `conf_threshold` 미만이면 4클래스 중 하나로 강제하지 않고
`unknown` 으로 표시한다.

BGL은 테스트 구간에만 등장하는 로그 템플릿이 1,000개 이상이다.
처음 보는 로그에 모델이 확신 없이 아무 클래스나 붙이는 것보다,
"모르겠다"고 표시하고 사람이 보게 하는 편이 운영상 안전하다.

`conf_threshold` 는 **val 셋에서** 결정한다. test에서 고르면 그 순간 test가 오염된다.
기준: `unknown` 비율이 5% 이하로 유지되는 선에서 나머지 예측의 정밀도가 최대가 되는 값.

### 5-6. 추론 결과 검증

배포 전 확인:

- 학습에 쓰지 않은 test 구간 전체를 `infer.py` 에 통과시켜 `windows.jsonl` 생성
- 실제 이상이 몰린 구간이 `warning` 이상으로 뜨는지
- **평시 구간에서 `warning` 이 얼마나 뜨는지** — 이게 오탐률이고, 실사용 가능 여부를 결정한다
- NASA 로그의 허리케인 정지 구간(1995-08-01 14:52 ~ 08-03 04:36)이 `critical`(급감)으로 뜨는지

하루치 평시 로그를 넣었을 때 알림이 수십 개씩 뜨면 아무도 안 쓴다.
**목표: 평시 구간 오탐 1일 5건 이하.** 이 수치를 README에 적는다.

---

## 6. 공개 계획

**Hugging Face**
- 모델 카드에 명시: 학습 데이터, 4클래스 정의, macro F1, 혼동행렬, 정규화 규칙
- **한계를 반드시 적는다**: BGL 단일 시스템 로그로 학습했으므로 다른 시스템 로그에서는
  성능이 떨어질 수 있다. 개념 변화에 취약하다.
- 추론 예시 코드 3줄 포함

**GitHub**
- 전처리부터 평가까지 전 과정 재현 가능하게
- `experiments.md` 를 그대로 공개 (실패한 실험 포함)
- 데이터는 다운로드 링크만, 원본 파일은 커밋하지 않음
- 출처 표기: LogHub은 원 논문 인용 조건이 있고, NASA-HTTP는 자유 재배포 가능하나
  Kaggle 가공본을 쓴 경우 그 사실을 명시

---

## 7. 알려진 위험 요소

| 위험 | 내용 | 대응 |
|---|---|---|
| 개념 변화 | 테스트 구간에만 등장하는 템플릿 1,000개 이상 | 한계로 명시, E7에서 개선 시도 |
| 클래스 불균형 | KERNDTLB 하나가 전체 이상의 44% | class weight, macro F1로 평가 |
| 데이터 누수 | 동일 줄 반복 | 시간순 분할 필수 |
| 기존 성능 포화 | BGL은 일반 BERT로도 F1 0.96~0.99 | 성능이 아니라 **원인 분류**가 차별점 |
| 라벨 신뢰도 | 41개 카테고리는 2007년 반자동 분류 결과 | 원본 기준을 그대로 따르고 출처 명시 |

---

## 8. 다음 할 일

1. `parse_bgl.py` 로 라벨 분포 확인 (첫 컬럼 기준, 41종 나와야 함)
2. 분포 보고 4클래스 매핑 확정 → `configs/base.yaml` 작성
3. E0 / E1 먼저 돌려서 하한선 확보
4. E2 베이스라인
