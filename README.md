# accessibility-wheelchair

예술의전당 공연 데이터를 사용해 휠체어석 접근성 수요 등급을 예측하는 ML 프로젝트입니다.

이 저장소는 코드, 설정, 문서, 실험 스크립트, 접근성 모델 실험 산출물을 함께 관리합니다. 원본 대용량 데이터는 DVC로 추적하고, 재현과 발표에 필요한 모델/리포트/MLflow 결과는 Git에 포함했습니다.

## Git Tracking Policy

Git에 포함합니다.

- `README.md`
- `src/`
- `configs/`
- `reports/`
- `scripts/`
- `artifacts/`
- `mlruns/`
- `requirements.txt`
- `.dvc` 메타데이터
- 전처리, 학습, 분석, API 코드
- 접근성 등급 예측 실험 결과와 최종 모델

Git에 직접 포함하지 않습니다.

- 큰 원본 데이터
- `.env`
- API key
- 캐시 파일

## Project Flow

1. `data/raw.dvc`로 원본 데이터 위치를 관리합니다.
2. `src/data/prepare.py`가 원본 CSV를 표준화하고 `accessibility_grade` 라벨을 생성합니다.
3. `src/models/train.py`와 `scripts/`의 실험 스크립트가 baseline, K-Fold, 하이퍼파라미터 튜닝, 등급 오차 분석, 앙상블, 순서형 모델링을 수행합니다.
4. `reports/accessibility/`에는 지표, 혼동행렬, feature importance, SHAP 이미지, HTML 리포트를 저장합니다.
5. `artifacts/`에는 실험별 모델과 최종 서빙 후보 모델을 저장합니다.
6. `src/api/main.py`가 최종 순서형 모델을 우선 불러오고, 없으면 앙상블 또는 baseline 모델로 fallback하여 `/health`, `/predict` API를 제공합니다.

## Experiment Summary

실험은 단일 train/test baseline에서 시작해 Stratified K-Fold 검증, feature 정제, Randomized Search 기반 튜닝, 등급 거리 기반 오차 분석, 앙상블 비교, Frank & Hall 방식 순서형 분류 모델까지 확장했습니다. 상세한 실험별 지표와 의사결정 흐름은 `reports/accessibility/`의 metrics/HTML 리포트와 `mlruns/`에서 확인합니다.

현재 API는 `artifacts/accessibility_classifier_ordinal.joblib`를 최우선으로 사용합니다.

주요 재실행 스크립트:

```bash
python3 scripts/build_accessibility_kfold_v31.py
python3 scripts/build_accessibility_kfold_v32.py
python3 scripts/build_accessibility_kfold_v4_hyperparamater.py
python3 scripts/build_accessibility_v42_grade.py
python3 scripts/build_accessibility_ensemble.py
python3 scripts/build_accessibility_ordinal.py
```

## Run

```bash
python3 src/data/prepare.py
python3 src/models/train.py
uvicorn src.api.main:app --reload --port 8000
```

## API Example

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "genre_group": "클래식",
    "venue_type": "대형",
    "is_weekend": 1,
    "duration_days": 2.0,
    "organizer_type": "대관",
    "organizer": "(재)서울시립교향악단"
  }'
```

## Notes

Run 산출물은 순차 커밋으로 보존되어 있으며, README에서는 개별 Run을 반복 설명하지 않습니다. 실험별 세부 비교는 리포트 파일과 MLflow 로그를 기준으로 확인합니다.
