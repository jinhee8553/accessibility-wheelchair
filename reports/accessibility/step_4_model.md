# Step 4. 모델 학습

## 문제 정의

- 문제 유형: A~E 다중 분류
- 타겟: `accessibility_grade`
- 평가 지표: macro F1

## 학습 Feature

| feature |
| --- |
| genre_group |
| venue_type |
| is_weekend |
| duration_days |
| organizer_type |
| organizer |

## 데이터 분할

- 방식: stratified train/test split
- test_size: 0.2
- random_state: 42
- 학습 행 수: 388
- 검증 행 수: 97

## 모델 비교

| model | accuracy | f1_macro |
| --- | --- | --- |
| ExtraTreesClassifier | 0.6082 | 0.5349 |
| XGBoostClassifier | 0.6701 | 0.5303 |
| RandomForestClassifier | 0.5979 | 0.5189 |
| DummyClassifier | 0.3918 | 0.1126 |

## 최종 모델

- 모델: `ExtraTreesClassifier`
- accuracy: `0.6082`
- macro F1: `0.5349`
- 저장 경로: `artifacts/accessibility_classifier.joblib`

## 상위 Feature Importance

| rank | feature | importance |
| --- | --- | --- |
| f01 | cat__venue_type_대형 | 0.23562769702532488 |
| f02 | cat__venue_type_중형 | 0.1709758800765418 |
| f03 | cat__genre_group_클래식 | 0.11873026153645862 |
| f04 | cat__venue_type_소극장 | 0.0719787177038511 |
| f05 | num__duration_days | 0.050457121462584006 |
| f06 | cat__genre_group_연극 | 0.0409207372439337 |
| f07 | cat__genre_group_뮤지컬 | 0.036813060861364856 |
| f08 | cat__genre_group_발레 | 0.028628259364692293 |
| f09 | cat__organizer_지클레프 | 0.021312677009472373 |
| f10 | num__is_weekend | 0.021145647308623932 |
