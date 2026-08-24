# log-anomaly-classifier

서버 로그 한 줄을 받아 **정상 여부 + 이상의 원인 계열(4클래스)** 을 분류하는 모델.
부가 모듈로 분당 로그 발생량 이상 감지기를 붙인다.

전체 설계는 `docs/SPEC.md` 를 읽을 것. 이 파일은 규칙과 명령어만 담는다.

---

## 스택

| 항목 | 값 |
|---|---|
| Python | 3.11 |
| 패키지 관리 | **uv** (`pip` 직접 호출 금지) |
| 데이터 | pandas, pyarrow |
| 모델 | torch, transformers |
| 평가 | scikit-learn, matplotlib |
| 설정 | pyyaml |
| 테스트 | pytest |
| 린트 | ruff |

```bash
uv init
uv add pandas pyarrow torch transformers scikit-learn matplotlib pyyaml tqdm
uv add --dev pytest ruff
```

CLI는 `argparse` 로 통일한다. 별도 CLI 라이브러리를 추가하지 말 것.

---

## 명령어

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

uv run pytest
uv run ruff check src/
```

---

## 절대 규칙

이 규칙을 어기면 결과 수치가 전부 무효가 된다.

1. **랜덤 분할 금지.** `unix_ts` 오름차순 정렬 후 앞 70/15/15 로 자른다.
   BGL은 동일한 로그 줄이 수십 회 반복되므로 랜덤 분할 시 동일 줄이 train/test에 동시에 들어간다.
2. **val / test의 클래스 비율을 조작하지 않는다.** 샘플링은 train에만 적용한다.
   크기를 줄일 때도 층화 샘플링으로 원본 비율(정상 약 92.7%)을 유지한다.
3. **`label_map` 을 코드에 하드코딩하지 않는다.** `configs/base.yaml` 한 곳에만 둔다.
   매핑되지 않은 카테고리가 있으면 조용히 넘기지 말고 에러로 중단한다.
4. **Accuracy를 대표 지표로 쓰지 않는다.** 전부 정상으로 찍어도 92.7%가 나온다.
   대표 지표는 **macro F1**.
5. **`data/` 를 커밋하지 않는다.** `.gitignore` 에 `data/`, `runs/` 를 넣는다.
   원본 로그는 재배포 조건이 있으므로 README에 다운로드 경로만 적는다.
6. **`level` / `component` 컬럼을 기본 입력에 넣지 않는다.**
   `level`(FATAL/INFO)은 라벨과 거의 같은 정보라 성능이 부풀려진다.
   비교 실험(E5)에서만 쓰고, 대표 성능은 `msg_only` 로 보고한다.
7. **시드 42로 고정.** 모든 랜덤 연산에 적용한다.

---

## 데이터 스키마

**BGL.log** — 공백 구분, 헤더 9개 + 메시지

```
- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
```

| # | 컬럼 | 비고 |
|---|---|---|
| 0 | `label` | `-` 는 정상, 그 외 41종은 이상. **이것이 정답 라벨** |
| 1 | `unix_ts` | 정렬/분할 기준 |
| 2 | `date` | |
| 3 | `node` | |
| 4 | `time` | |
| 5 | `node_repeat` | |
| 6 | `type` | 항상 `RAS` |
| 7 | `component` | KERNEL / APP / LINKCARD 등 |
| 8 | `level` | INFO / FATAL 등 |
| 9+ | `message` | 공백 포함, 나머지 전부 |

```python
parts = line.split(maxsplit=9)   # 반드시 maxsplit=9
```

`split()` 을 그냥 쓰면 메시지의 공백 때문에 줄마다 컬럼 수가 달라진다.

**nasa.csv** — 발생량 감지기 검증용

`host, time(unix초), method, url, response, bytes`
`Unnamed: 0` 컬럼은 삭제. `response`/`bytes` 결측 행은 drop.

---

## 클래스 정의

BGL 전체 4,747,963줄 / 알럿 348,460줄(7.34%) 기준. 41개 원본 카테고리를 4클래스로 묶는다.

| 클래스 | 건수 | 알럿 대비 | 의미 | 운영자가 볼 곳 |
|---|---|---|---|---|
| `normal` | 4,399,503 | — | 정상 | — |
| `kernel_mem` | 218,075 | 62.6% | 메모리·주소변환 오류 | 하드웨어 |
| `kernel_ops` | 66,269 | 19.0% | 마운트·종료·복구 실패 | 시스템 설정 |
| `app` | 63,845 | 18.3% | 애플리케이션 오류 | 코드 |

`configs/base.yaml` 에 아래를 그대로 넣는다. **코드에 복사하지 말 것.**

```yaml
label_map:
  normal:     ["-"]
  kernel_mem: [KERNDTLB, KERNSTOR, KERNMICRO, KERNMC, KERNFLOAT, KERNBIT, KERNTLBE]
  kernel_ops: [KERNMNTF, KERNTERM, KERNREC, KERNRTSP, KERNMNT, KERNSOCK,
               KERNPOW, KERNSERV, KERNPAN, KERNCON, KERNNOETH, KERNPROG,
               KERNRTSA, KERNEXT]
  app:        [APPSEV, APPREAD, APPRES, APPUNAV, APPTO, APPOUT, APPBUSY,
               APPCHILD, APPALLOC, APPTORUS]

# 학습·평가에서 제외. unknown 메커니즘 검증용으로만 사용 (총 271건)
holdout_rare: [LINKIAP, LINKDISC, LINKPAP, LINKBLL,
               MASABNORM, MASNORM, MONPOW, MONNULL, MONILL, MMCS]
```

**이 그룹핑은 공식 분류가 아니라 이 프로젝트에서 정한 기준이다.**
BGL 알럿 코드에 대한 완전한 공식 문서가 없어 코드명을 근거로 판단했다.
경계가 애매한 항목이 있다(예: `KERNPOW`는 전원 관련이라 `kernel_mem` 으로 봐도 무방).
README에 이 사실을 명시할 것.

`holdout_rare` 는 LINK/MAS/MON/MMCS 계열로 총 271건뿐이라 별도 클래스로 학습이 불가능하다.
버리지 않고 **모델이 한 번도 보지 못한 카테고리**로 남겨, 추론 시 `unknown` 으로
분류되는지 확인하는 데 쓴다. macro F1 계산에는 포함하지 않는다.

### 매핑 재검토 조건

`evaluate` 결과의 혼동행렬에서 **`kernel_mem` ↔ `kernel_ops` 오분류가 양방향으로 심하면**
모델 성능 문제가 아니라 매핑 경계가 잘못 그어졌다는 신호일 수 있다.
이 경우 사람에게 보고하고 매핑 재검토를 제안한다. 임의로 바꾸지 말 것.

---

## 정규화 규칙 (순서 고정)

정규화하지 않으면 모델이 노드 ID를 통째로 외워 새 노드에서 동작하지 않는다.

| 순서 | 대상 | 정규식 | 치환 |
|---|---|---|---|
| 1 | 노드 ID | `R\d+-M\d+-N\d+-C:J\d+-U\d+` | `[NODE]` |
| 2 | IP | `\d+\.\d+\.\d+\.\d+` | `[IP]` |
| 3 | 16진수 | `0x[0-9a-fA-F]+` | `[HEX]` |
| 4 | 경로 | `(/[\w.\-]+)+` | `[PATH]` |
| 5 | 남은 숫자 | `\d+` | `[NUM]` |

**숫자를 먼저 치환하면 노드 ID가 부서진다. 순서를 바꾸지 말 것.**

---

## 모듈별 완료 조건

각 모듈은 아래를 만족해야 완료로 간주한다.

| 모듈 | 완료 조건 |
|---|---|
| `parse_bgl` | 파싱 실패율 < 0.1%, 총 줄 수와 이상 비율을 stdout에 출력 |
| `label_map` | 미매핑 카테고리 0개, 4클래스 분포 출력, holdout_rare 271건 분리 |
| `normalize` | 정규화 후 고유 메시지 수가 전보다 크게 감소 |
| `split` | 세 구간의 클래스 분포를 각각 출력, 시간 경계가 겹치지 않음 |
| `sample` | train만 변경됨, val/test 파일 해시 불변 |
| `train` | val macro F1 기준 best checkpoint 저장, 에폭별 로그 남김 |
| `evaluate` | `metrics.json`, `cm.png`, `errors.csv`(오분류 50건) 생성 |
| `infer` | `windows.jsonl` 생성, 각 구간에 alert level과 근거 로그 포함 |

---

## 테스트

`tests/fixtures/BGL_2k.log` (LogHub 제공 2천 줄 샘플)을 픽스처로 쓴다.

- `test_parse.py` — 알려진 3줄의 컬럼 분리 결과 검증
- `test_normalize.py` — 노드 ID가 `[NODE]` 로, 숫자가 `[NUM]` 로 바뀌는지. 순서 역전 시 실패하는 케이스 포함
- `test_split.py` — 분할 경계의 `unix_ts` 가 단조 증가하는지, 세 구간에 중복 행이 없는지
- `test_sample.py` — val/test가 변경되지 않았는지

전체 데이터로 돌리기 전에 픽스처로 파이프라인 전체를 한 번 통과시킬 것.

---

## 구현 순서

1. `parse_bgl` → 라벨 분포가 아래와 일치하는지 확인 (총 4,747,963줄 / 알럿 7.34%)
2. `label_map` → 4클래스 분포 출력, `split` 후 세 구간 분포 확인.
   **어느 구간에서든 한 클래스가 100건 미만이면 멈추고 보고할 것**
3. `normalize` → `sample`
4. `evaluate` 를 먼저 만든다 — 평가 코드 없이 학습하면 결과를 볼 수 없다
5. **E0**(다수 클래스 고정 예측), **E1**(TF-IDF + 로지스틱 회귀) 로 하한선 확보
6. **E2** BERT 베이스라인
7. `calibrate` → `infer`
8. 발생량 감지기 (독립적이라 언제든 병렬 진행 가능)

**E1을 건너뛰지 말 것.** 로그 이상탐지에서는 단순 기법이 딥러닝을 이기는 사례가 흔하다.
E1 없이는 E2의 수치가 좋은 건지 판단할 수 없다.

---

## 막혔을 때

추측해서 진행하지 말고 멈추고 물어본다. 특히:

- 시간순 분할 후 특정 구간에 한 클래스가 100건 미만일 때
- 혼동행렬에서 `kernel_mem` ↔ `kernel_ops` 오분류가 양방향으로 심할 때
- 파싱 실패율이 0.1%를 넘을 때
- 이상 비율이 7.3%에서 크게 벗어날 때 (파일이 잘렸을 가능성)
- 평시 오탐률이 목표(1일 5건)를 크게 넘을 때
