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

