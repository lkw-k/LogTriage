# 진행 상황

현재: E2i 로 **체크포인트 선택은 원인이 아님**이 확인됐다. 원인은 클래스 가중치가
"machine check" 다수를 뒤집는 것. 다음은 이 가설 검증.
상세 내용은 `docs/SPEC.md`, 실험 수치는 `experiments.md`, 위기 기록은 `docs/RISKS.md`.

- [x] 전처리 — parse_bgl / label_map / normalize
- [x] 데이터셋 — split / sample (dataset 남음)
- [x] 평가 — evaluate (+ 기등장/미등장 부분집합 채점)
- [x] 하한선 — E0 0.2415 / E1 0.8067 / E1c **0.9316**
- [x] 선택 지표 — train --inner-val-frac / --select-metric
- [ ] 학습 — E2c 0.6829 / E2i 0.6429 (둘 다 패배). **원인 검증 남음**
- [ ] 실험 확장 — E2(none) / E3 ~ E7
- [ ] 추론 — calibrate / infer  (위기 E: 신뢰도 포화로 불가)
- [ ] 트래픽 감지기 — parse_nasa / detect
- [ ] 공개 — README / HF 모델 카드

**판정 기준은 미등장 템플릿 macro F1** (전체 F1 은 암기분 24.4% 가 섞여 있다):
E0 0.2411 / E1 0.5192 / **E1c 0.7451** / E2c 0.4998 / E2i 0.5049

다음 후보 (사용자가 직접 실행):

    # A. 가설 검증, 30분, 변수 1개 (class_weight)
    uv run python -m src.train --exp-id E2w --input data/processed/train_cap.parquet --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --no-class-weight

    # B. SPEC 대표 성능, 7.5시간
    uv run python -m src.train --exp-id E2 --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1
