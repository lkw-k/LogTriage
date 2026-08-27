# 진행 상황

현재: **E2 대표 성능까지 끝났고, BERT 는 TF-IDF 에 졌다.** E2(none) 0.7760 / 미등장
0.5144, BERT 최고인 E2w(cap) 가 0.8289 / **0.7330**, E1c 가 **0.9316 / 0.7451**.
cap 은 알럿을 살리고 app 템플릿을 자르며, none 은 app 을 살리고 알럿 recall 을 0 으로
만든다. 그 사이 지점이 E3(ratio) / E4(balanced) 다.
상세 내용은 `docs/SPEC.md`, 실험 수치는 `experiments.md`, 위기 기록은 `docs/RISKS.md`.

- [x] 전처리 — parse_bgl / label_map / normalize
- [x] 데이터셋 — split / sample (dataset 남음)
- [x] 평가 — evaluate (+ 기등장/미등장 부분집합 채점)
- [x] 하한선 — E0 0.2415 / E1 0.8067 / E1c **0.9316**
- [x] 선택 지표 — train --inner-val-frac / --select-metric
- [x] 학습 — **E2 0.7760 (대표)**. E2c 0.6829 / E2i 0.6429 / E2w 0.8289
- [ ] 실험 확장 — E3 ~ E7
- [ ] 추론 — calibrate / infer  (**모듈 자체가 아직 없음**)
- [ ] 트래픽 감지기 — parse_nasa / detect
- [ ] 공개 — README / HF 모델 카드

**판정 기준은 미등장 템플릿 macro F1** (전체 F1 은 암기분 24.4% 가 섞여 있다):
E0 0.2411 / E1 0.5192 / **E1c 0.7451** / E2c 0.4998 / E2i 0.5049 / E2w 0.7330 / E2 0.5144

**대표 모델은 E1c 다.** BERT 4판 전부 졌다. 이건 실패가 아니라 결과다 —
로그 이상탐지에서 단순 기법이 딥러닝을 이기는 건 흔하고, 그래서 E1 을 먼저 세웠다.

다음 후보 (사용자가 직접 실행):

    # A. E3(ratio). cap 과 none 사이. 정상을 알럿의 3배로만 남겨
    #    app 템플릿을 자르지 않으면서 알럿 비중을 올린다. 약 1,020,000행
    #    = 에폭당 약 25,600스텝(약 25분), 조기 종료까지 약 1시간 15분.
    uv run python -m src.sample --input data/processed/train.parquet --output data/processed/train_ratio.parquet --strategy ratio
    uv run python -m src.train --exp-id E3 --input data/processed/train_ratio.parquet --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --no-class-weight
    uv run python -m src.evaluate --exp-id E3

    # B. (선택) 대조군. 가중치를 켠 채 에폭 1만. 6분. 두 원인의 기여를 분리한다.
    uv run python -m src.train --exp-id E2v --input data/processed/train_cap.parquet --inner-val-frac 0.20 --select-metric inner_val_unseen_macro_f1 --max-epochs 1
    uv run python -m src.evaluate --exp-id E2v
