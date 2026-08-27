# 진행 상황

현재: **BERT 5판 전패. 대표 모델은 E1c(TF-IDF+로지스틱회귀)로 확정한다.**
미등장 macro F1 기준 E1c **0.7451** vs BERT 최고 E2w 0.7330.
샘플링 축(cap / none / ratio)은 세 지점을 다 찍고 닫았다 — 어느 것도 세 알럿 클래스를
동시에 살리지 못한다. 남은 것은 파이프라인 마무리(calibrate / infer / 트래픽 감지기)다.
상세 내용은 `docs/SPEC.md`, 실험 수치는 `experiments.md`, 위기 기록은 `docs/RISKS.md`.

- [x] 전처리 — parse_bgl / label_map / normalize
- [x] 데이터셋 — split / sample (dataset 남음)
- [x] 평가 — evaluate (+ 기등장/미등장 부분집합 채점)
- [x] 하한선 — E0 0.2415 / E1 0.8067 / E1c **0.9316**
- [x] 선택 지표 — train --inner-val-frac / --select-metric
- [x] 학습 — **E2 0.7760 (대표)**. E2c 0.6829 / E2i 0.6429 / E2w 0.8289
- [x] 실험 확장 — E3 0.6410. **샘플링 축 종료** (E4 는 예측 가능해 생략, E5~E7 미착수)
- [x] 추론 — calibrate / infer / predictor / adapters.bgl  (E2w 로 검증)
- [ ] 트래픽 감지기 — detect 는 infer 안에서 동작. `parse_nasa` 미착수 (nasa.csv 미확보)
- [ ] 공개 — README / HF 모델 카드

**판정 기준은 미등장 템플릿 macro F1** (전체 F1 은 암기분 24.4% 가 섞여 있다):
E0 0.2411 / E1 0.5192 / **E1c 0.7451** / E2c 0.4998 / E2i 0.5049 / E2w 0.7330 /
E2 0.5144 / E3 0.5011

**대표 모델은 E1c 다.** BERT 5판(E2c/E2i/E2w/E2/E3) 전부 졌다. 실패가 아니라 결과다 —
로그 이상탐지에서 단순 기법이 딥러닝을 이기는 건 흔하고, 그래서 E1 을 먼저 세웠다.

오탐 14,481 은 미등장 템플릿 **1종**(`MACHINE CHECK DCR read timeout`)의 전부 아니면
전무다. kernel_mem 의 test support 가 100건뿐이라 이 한 칸이 미등장 macro F1 을
0.73 ↔ 0.50 으로 흔든다. 발화 조건은 (1) 학습셋 kernel_mem 실효 비중이 높거나
(2) 2에폭 이상 돌리거나. 자세한 표는 `experiments.md` 2026-08-27 항목.

추론 파이프라인 검증 (E2w, 원본 로그 그대로 통과, 파싱 실패 0.00%):

| | 알림/일 | 탐지/일 | 오탐/일 | 재현율 |
|---|---|---|---|---|
| val (튜닝면) | 3.3 | 1.6 | **1.7** | 23.3% |
| test (1회 보고) | 5.7 | 3.4 | **2.3** | 53.8% |

오탐은 둘 다 목표 5건/일 안이다.

다음 (내가 만들 것):

- E1c 로 calibrate / infer 재검증 — `runs/E1c/model.joblib` 이 나온 뒤.
- `src/traffic/parse_nasa.py` — `nasa.csv` 미확보로 대기.
- HF 업로드 + 모델 카드 (올릴 것은 E2w).

다음 후보 (사용자가 직접 실행, 선택):

    # E7. 축을 바꾸는 유일하게 남은 시도. 로그 코퍼스 MLM 후 파인튜닝. 몇 시간.
    #    지금 진단상 문제는 표현력이 아니라 템플릿 1종의 사전확률이라 기대값은 낮다.
