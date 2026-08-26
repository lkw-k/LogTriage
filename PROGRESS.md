# 진행 상황

현재: E2c(BERT+cap) 학습·채점 완료 → **E1c 에 패배**. SPEC 대표 성능인 E2(none) 미실행.
상세 내용은 `docs/SPEC.md`, 실험 수치는 `experiments.md`, 위기 기록은 `docs/RISKS.md`.

- [x] 전처리 — parse_bgl / label_map / normalize
- [x] 데이터셋 — split / sample (dataset 남음)
- [x] 평가 — evaluate
- [x] 하한선 — E0 0.2415 / E1 0.8067 / E1c **0.9316** (현재 최고)
- [ ] 학습 — E2c 0.6829 완료. **E2(none) 남음 = 대표 성능**
- [ ] 실험 확장 — E3 ~ E7
- [ ] 추론 — calibrate / infer  (위기 E: E2c 는 신뢰도 포화로 불가)
- [ ] 트래픽 감지기 — parse_nasa / detect
- [ ] 공개 — README / HF 모델 카드

미등장 템플릿 macro F1 (판정 기준): E0 0.2411 / E1 0.5192 / **E1c 0.7451** / E2c 0.4998
