# 진행 상황

현재: 정직한 선택 지표(inner-val 미등장 F1)를 코드에 넣었다 → 다음은 **E2i 학습**.
상세 내용은 `docs/SPEC.md`, 실험 수치는 `experiments.md`, 위기 기록은 `docs/RISKS.md`.

- [x] 전처리 — parse_bgl / label_map / normalize
- [x] 데이터셋 — split / sample (dataset 남음)
- [x] 평가 — evaluate (+ 기등장/미등장 부분집합 채점)
- [x] 하한선 — E0 0.2415 / E1 0.8067 / E1c **0.9316**
- [ ] 학습 — E2c 0.6829 (패배). **E2i 남음** = inner-val 미등장으로 체크포인트 선택
- [ ] 실험 확장 — E2(none) / E3 ~ E7
- [ ] 추론 — calibrate / infer  (위기 E: E2c 는 신뢰도 포화로 불가)
- [ ] 트래픽 감지기 — parse_nasa / detect
- [ ] 공개 — README / HF 모델 카드

**판정 기준은 미등장 템플릿 macro F1** (전체 F1 은 암기분 24.4% 가 섞여 있다):
E0 0.2411 / E1 0.5192 / **E1c 0.7451** / E2c 0.4998

다음 명령 (사용자가 직접 실행, 약 30분):

    uv run python -m src.train --exp-id E2i       --input data/processed/train_cap.parquet       --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1
    uv run python -m src.evaluate --exp-id E2i
