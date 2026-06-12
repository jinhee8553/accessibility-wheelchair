# accessibility-wheelchair

예술의전당 공연 데이터를 사용해 휠체어석 접근성 수요 등급을 예측하는 ML 프로젝트입니다.

이 저장소는 코드, 설정, 문서, 실험 스크립트를 Git으로 관리하고, 원본 데이터와 모델 산출물처럼 큰 파일은 Git에 직접 넣지 않는 구조를 기준으로 합니다.

## Git Tracking Policy

Git에 포함합니다.

- `README.md`
- `src/`
- `configs/`
- `reports/`
- `requirements.txt`
- `.dvc` 메타데이터
- 전처리, 학습, API 코드

Git에 직접 포함하지 않습니다.

- 큰 CSV/parquet 원본 데이터
- 모델 파일(`*.joblib`, `*.pkl`)
- `.env`
- API key
- 캐시 파일
- MLflow 실행 산출물

## Project Flow

1. `data/raw.dvc`로 원본 데이터 위치를 관리합니다.
2. `src/data/prepare.py`가 원본 CSV를 표준화하고 `accessibility_grade` 라벨을 생성합니다.
3. `src/models/train.py`가 후보 모델을 비교한 뒤 best pipeline을 `artifacts/accessibility_classifier.joblib`에 저장합니다.
4. `src/api/main.py`가 저장된 모델을 불러와 `/health`, `/predict` API를 제공합니다.

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

## Commit Strategy Used

- `chore: initialize accessibility project repository`
- `chore: define tracked project boundaries`
- `feat: rebuild accessibility data preparation workflow`
- `feat: add accessibility model training and API`
