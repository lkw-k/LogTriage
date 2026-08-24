# log-anomaly-classifier

서버 로그 **한 줄**을 받아 정상 여부와 **이상의 원인 계열(4클래스)** 을 분류한다.
부가 모듈로 분당 로그 발생량 이상 감지기를 붙인다.

기존 로그 이상탐지 모델은 대부분 정상/이상 2클래스다. 이 프로젝트는 원인까지 구분해
운영자가 알림을 받았을 때 **어디를 봐야 할지** 바로 알 수 있게 하는 것이 목표다.

- 설계 문서: [docs/SPEC.md](docs/SPEC.md)
- 작업 규칙: [CLAUDE.md](CLAUDE.md)
- 실험 기록: [experiments.md](experiments.md)

## 클래스 정의

| 클래스 | 의미 | 운영자가 볼 곳 |
|---|---|---|
| `normal` | 정상 | — |
| `kernel_mem` | 메모리·주소변환 오류 | 하드웨어 |
| `kernel_ops` | 마운트·종료·복구 실패 | 시스템 설정 |
| `app` | 애플리케이션 오류 | 코드 |

추론 시 최대 softmax 확률이 임계값 미만이면 4클래스로 강제하지 않고 `unknown` 으로 표시한다.

> **이 4클래스 그룹핑은 공식 분류가 아니라 이 프로젝트에서 정한 기준이다.**
> BGL 알럿 코드에 대한 완전한 공식 문서가 없어 코드명을 근거로 판단했다.
> 경계가 애매한 항목이 있다 (예: `KERNPOW` 는 전원 관련이라 `kernel_mem` 으로 봐도 무방).
> 원본 41개 카테고리는 2007년 반자동 분류 결과이므로 라벨 자체의 신뢰도에도 한계가 있다.
> 전체 매핑은 [`configs/base.yaml`](configs/base.yaml) 에 있다.

`LINK*` / `MAS*` / `MON*` / `MMCS` 계열 271건은 `holdout_rare` 로 분리해 학습·평가에서 제외한다.
버리는 것이 아니라 **모델이 한 번도 보지 못한 카테고리**로 남겨,
추론 시 `unknown` 으로 분류되는지 확인하는 데 쓴다.

## 데이터

원본 로그는 용량이 크고 재배포 조건이 있어 저장소에 포함하지 않는다. 직접 받아서 `data/raw/` 에 둔다.

| 파일 | 용도 | 출처 |
|---|---|---|
| `data/raw/BGL.log` | 분류기 학습·평가 (4,747,963줄 / 알럿 7.34%) | [LogHub](https://github.com/logpai/loghub) — BGL |
| `data/raw/nasa.csv` | 발생량 감지기 검증 | NASA-HTTP (1995) 접근 로그 |

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

GPU 학습이 필요하면 CPU 빌드 torch 를 CUDA 빌드로 교체한다 (기본은 CPU 빌드).
`torch==2.13.0` 은 로컬 버전 태그를 포함하므로 `2.13.0+cu124` 같은 CUDA 빌드도 이 핀에 맞는다.

## 실행

```bash
uv run python -m src.parse_bgl   --input data/raw/BGL.log --output data/interim/bgl_parsed.parquet
uv run python -m src.label_map   --input data/interim/bgl_parsed.parquet --output data/interim/bgl_labeled.parquet
uv run python -m src.normalize   --input data/interim/bgl_labeled.parquet --output data/interim/bgl_norm.parquet
uv run python -m src.split       --input data/interim/bgl_norm.parquet --outdir data/processed
uv run python -m src.sample      --input data/processed/train.parquet --output data/processed/train_sampled.parquet --strategy none
uv run python -m src.train       --config configs/base.yaml --exp-id E2
uv run python -m src.evaluate    --exp-id E2
uv run python -m src.calibrate   --exp-id E2 --output runs/E2/calibration.json
uv run python -m src.infer       --exp-id E2 --input data/raw/sample.log --output runs/E2/windows.jsonl
```

각 단계는 파일을 남기고 끝난다. 다음 단계는 앞 단계의 출력만 읽는다.

```bash
uv run pytest
uv run ruff check src/
```

## 평가 기준

- 대표 지표는 **macro F1**. accuracy 는 보고하지 않는다 — 전부 정상으로 찍어도 92.7% 가 나온다.
- 시간순 분할(`unix_ts` 오름차순 70/15/15). 랜덤 분할은 동일한 줄이 train/test 에 동시에
  들어가므로 금지한다.
- val / test 의 클래스 비율은 조작하지 않는다. 샘플링은 train 에만 적용한다.
- 실사용 기준: **평시 구간 오탐 1일 5건 이하.**

## 한계

- BGL 단일 시스템 로그로 학습했다. 다른 시스템 로그에서는 성능이 떨어진다.
  다른 포맷을 넣으려면 `src/adapters/` 에 `(timestamp, message)` 로 변환하는 파서를 추가해야 하며,
  어댑터 없이 넣은 결과는 무의미하다.
- 개념 변화에 취약하다. BGL 은 학습 구간에 없던 로그 템플릿이 테스트 구간에 1,000개 이상 등장한다.
- 4클래스 매핑은 위에 적은 대로 이 프로젝트가 정한 기준이다.

## 상태

초기 세팅 단계. 모듈은 아직 구현되지 않았다.
