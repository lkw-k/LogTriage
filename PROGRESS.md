# 진행 상황

현재: E2w 로 가설이 확인됐다. 클래스 가중치를 끄니 오탐 14,484건이 0 이 되고
미등장 0.5049 → **0.7330** (E1c 0.7451 에 0.0121 차). 단 **에폭 1에서만** 그렇고
에폭 3 이면 오탐이 돌아온다. 다음은 E2(none) 대표 성능.
상세 내용은 `docs/SPEC.md`, 실험 수치는 `experiments.md`, 위기 기록은 `docs/RISKS.md`.

- [x] 전처리 — parse_bgl / label_map / normalize
- [x] 데이터셋 — split / sample (dataset 남음)
- [x] 평가 — evaluate (+ 기등장/미등장 부분집합 채점)
- [x] 하한선 — E0 0.2415 / E1 0.8067 / E1c **0.9316**
- [x] 선택 지표 — train --inner-val-frac / --select-metric
- [ ] 학습 — E2c 0.6829 / E2i 0.6429 / **E2w 0.8289**. 원인 확인 완료, 대표 성능 남음
- [ ] 실험 확장 — E2(none) / E3 ~ E7
- [ ] 추론 — calibrate / infer  (위기 E: 신뢰도 포화로 불가)
- [ ] 트래픽 감지기 — parse_nasa / detect
- [ ] 공개 — README / HF 모델 카드

**판정 기준은 미등장 템플릿 macro F1** (전체 F1 은 암기분 24.4% 가 섞여 있다):
E0 0.2411 / E1 0.5192 / **E1c 0.7451** / E2c 0.4998 / E2i 0.5049 / **E2w 0.7330**

다음 후보 (사용자가 직접 실행):

    # A. SPEC 대표 성능 E2. none 3,299,255행, 에폭당 103,102스텝(약 52분).
    #    에폭 1을 넘기면 오탐이 돌아오므로 max-epochs 2 로 묶는다 (약 2시간).
    uv run python -m src.train --exp-id E2 --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --no-class-weight --max-epochs 2
    uv run python -m src.evaluate --exp-id E2

    # B. (선택) 대조군. 가중치를 켠 채 에폭 1만. 6분. 두 원인의 기여를 분리한다.
    uv run python -m src.train --exp-id E2v --input data/processed/train_cap.parquet --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --max-epochs 1
    uv run python -m src.evaluate --exp-id E2v
