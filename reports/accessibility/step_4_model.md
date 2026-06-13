# Step 4. 접근성 등급 분류 모델

## 문제 정의

- 문제 유형: 다중 분류
- 타겟 변수: `accessibility_grade`
- 등급: A~E
- 주 평가 지표: macro F1

## 학습 Feature

| feature |
| --- |
| booking_lead_days |
| genre_group |
| venue_type |
| is_weekend |
| duration_days |
| organizer_type |
| organizer |
| start_time |

## 학습 제외 컬럼

아래 컬럼은 좌석 수 또는 타겟 산출에 직접 연결되는 값이므로 모델 feature에서 제외했다.

| excluded_column |
| --- |
| wheelchair_booking_count |
| total_seats |
| general_seats |
| wheelchair_seats |
| wheelchair_seat_ratio |
| wheelchair_booking_rate |

## 등급 분포

|  | count |
| --- | --- |
| A | 58 |
| B | 110 |
| C | 110 |
| D | 64 |
| E | 158 |

## 데이터 분할

- 방식: stratified train/test split
- test_size: 0.2
- random_state: 42
- 학습 행 수: 400
- 검증 행 수: 100

데이터 수가 500행으로 작아 각 등급이 검증 세트에 포함되도록 stratified split을 사용했다.

## 모델 비교

| model | accuracy | f1_macro |
| --- | --- | --- |
| XGBoostClassifier | 0.55 | 0.457 |
| ExtraTreesClassifier | 0.52 | 0.4479 |
| RandomForestClassifier | 0.52 | 0.4479 |
| LightGBMClassifier | 0.46 | 0.4068 |
| DummyClassifier | 0.31 | 0.0947 |

## 후보 알고리즘 구현 상태

| algorithm | status |
| --- | --- |
| RandomForest | 구현 및 학습 완료 |
| ExtraTrees | 구현 및 학습 완료 |
| XGBoost | 구현 및 학습 완료 |
| LightGBM | 구현 및 학습 완료 |

## 최종 선택 모델

- 모델: `XGBoostClassifier`
- 저장 경로: `artifacts/accessibility_classifier.joblib`
- 메트릭 저장 경로: `reports/accessibility/accessibility_metrics.json`

## 상위 Feature Importance

| rank | feature | importance |
| --- | --- | --- |
| f01 | cat__venue_type_중형 | 0.17122198641300201 |
| f02 | cat__venue_type_대형 | 0.13418793678283691 |
| f03 | cat__venue_type_소극장 | 0.06697431951761246 |
| f04 | cat__genre_group_클래식 | 0.04212233051657677 |
| f05 | num__duration_days | 0.03146527335047722 |
| f06 | cat__organizer_영음예술기획 | 0.028804663568735123 |
| f07 | cat__organizer_크레디아뮤직앤아티스트 | 0.026065643876791 |
| f08 | cat__genre_group_발레 | 0.024142613634467125 |
| f09 | cat__organizer_주식회사 목프로덕션 | 0.023487398400902748 |
| f10 | cat__organizer_위드클래식 | 0.018390147015452385 |

## 산출물

- 혼동행렬: `reports/accessibility/accessibility_confusion_matrix.png`
- Feature importance CSV: `reports/accessibility/accessibility_feature_importance.csv`
- Feature importance 이미지: `reports/accessibility/accessibility_feature_importance.png`
