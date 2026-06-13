# ♿ accessibility-wheelchair

> 예술의전당 공연 데이터를 기반으로 **휠체어석 접근성 수요 등급**을 예측하는 ML/MLOps 프로젝트

이 프로젝트는 공연 정보와 좌석 데이터를 활용해 공연별 휠체어석 수요 압박도를 분석하고, 이를 바탕으로 접근성 수요 등급을 예측합니다.
단순한 모델 학습에 그치지 않고, **데이터 버전 관리, 실험 추적, 모델 산출물 관리, API 서빙**까지 포함한 재현 가능한 MLOps 흐름을 목표로 합니다.

---

## 1. Project Overview

### 목적

신규 공연 정보를 입력했을 때 해당 공연의 **휠체어석 접근성 수요 등급**을 사전에 예측합니다.

이를 통해 다음과 같은 의사결정을 지원합니다.

* 휠체어석 수요가 높을 가능성이 있는 공연 사전 탐지
* 공연장 배치 및 접근성 개선 우선순위 판단
* D/E 등 취약 등급 공연에 대한 정책적 대응 근거 제공
* 유사 공연과 비교한 접근성 리스크 분석

---

## 2. Key Features

| 구분     | 내용                                                                        |
| ------ | ------------------------------------------------------------------------- |
| 데이터 관리 | 원본 대용량 데이터는 DVC로 추적                                                       |
| 전처리    | 공연 데이터 표준화 및 접근성 등급 라벨 생성                                                 |
| 모델링    | Baseline, K-Fold, Hyperparameter Tuning, Ensemble, Ordinal Classification |
| 평가     | Macro F1, Accuracy, MAE, Severe Error, 등급별 Recall                         |
| 분석     | Confusion Matrix, Feature Importance, SHAP, HTML Report                   |
| 실험 추적  | MLflow 기반 실험 결과 기록                                                        |
| 서빙     | FastAPI 기반 `/health`, `/predict` API 제공                                   |

---

## 3. Project Pipeline

```text
Raw Data
   │
   ▼
Data Versioning with DVC
   │
   ▼
Preprocessing & Labeling
   │
   ▼
Model Training
   ├── Baseline
   ├── Stratified K-Fold
   ├── Hyperparameter Tuning
   ├── Ensemble
   └── Ordinal Classification
   │
   ▼
Evaluation & Error Analysis
   │
   ▼
Model Artifact Saving
   │
   ▼
FastAPI Serving
```

---

## 4. Repository Structure

```text
accessibility-wheelchair/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── experiment and model configuration files
│
├── data/
│   └── raw.dvc
│
├── src/
│   ├── data/
│   │   └── prepare.py
│   │
│   ├── models/
│   │   └── train.py
│   │
│   └── api/
│       └── main.py
│
├── scripts/
│   ├── build_accessibility_kfold_v31.py
│   ├── build_accessibility_kfold_v32.py
│   ├── build_accessibility_kfold_v4_hyperparamater.py
│   ├── build_accessibility_v42_grade.py
│   ├── build_accessibility_ensemble.py
│   └── build_accessibility_ordinal.py
│
├── reports/
│   └── accessibility/
│       ├── metrics
│       ├── confusion_matrix
│       ├── feature_importance
│       ├── shap
│       └── html reports
│
├── artifacts/
│   ├── accessibility_classifier_ordinal.joblib
│   └── model artifacts
│
├── mlruns/
│   └── MLflow experiment logs
│
└── .dvc/
    └── DVC metadata
```

---

## 5. Data & Labeling

원본 공연 데이터는 Git에 직접 포함하지 않고 DVC로 관리합니다.

전처리 스크립트는 다음 역할을 수행합니다.

* 원본 CSV 로드
* 컬럼명 및 데이터 타입 표준화
* 공연장, 장르, 기간, 주말 여부 등 feature 생성
* 휠체어석 수요 기반 접근성 등급 라벨 생성
* 모델 학습용 데이터셋 저장

```bash
python3 src/data/prepare.py
```

---

## 6. Modeling Experiments

본 프로젝트의 모델링은 단순 baseline에서 시작해 점진적으로 개선하는 방식으로 진행했습니다.

| 단계                     | 주요 내용                                  |
| ---------------------- | -------------------------------------- |
| Baseline               | 기본 분류 모델 성능 확인                         |
| K-Fold CV              | 데이터 분할 편향을 줄이기 위한 Stratified K-Fold 검증 |
| Feature Refinement     | 접근성 예측에 적합한 feature 정제                 |
| Hyperparameter Tuning  | Randomized Search 기반 모델 성능 개선          |
| Grade Error Analysis   | 등급 간 거리 기반 오차 분석                       |
| Ensemble               | 여러 모델 조합 성능 비교                         |
| Ordinal Classification | A~E 등급의 순서성을 반영한 Frank & Hall 방식 모델링   |

현재 API는 다음 모델을 우선 사용합니다.

```text
artifacts/accessibility_classifier_ordinal.joblib
```

해당 모델이 없을 경우 ensemble 또는 baseline 모델로 fallback합니다.

---

## 7. Evaluation

모델은 단순 정확도만으로 평가하지 않고, 접근성 등급 예측 문제에 맞는 지표를 함께 사용합니다.

| Metric            | 설명                          |
| ----------------- | --------------------------- |
| Macro F1          | 등급별 성능을 균등하게 반영             |
| Accuracy          | 전체 예측 정확도                   |
| MAE               | 실제 등급과 예측 등급 사이의 평균 거리      |
| Severe Error      | 3등급 이상 크게 잘못 예측한 비율         |
| Class-wise Recall | D/E 등 취약 등급을 얼마나 잘 탐지하는지 확인 |

특히 본 프로젝트에서는 **D/E 등급 recall**과 **Severe Error**를 중요하게 봅니다.
접근성 수요가 높은 공연을 낮은 등급으로 잘못 예측하면 실제 정책적 대응이 늦어질 수 있기 때문입니다.

---

## 8. Reports & Artifacts

실험 결과는 `reports/accessibility/`에 저장됩니다.

포함되는 산출물은 다음과 같습니다.

* 실험별 metrics
* confusion matrix
* feature importance
* SHAP visualization
* 등급별 오분류 분석
* HTML 기반 실험 리포트

모델 파일은 `artifacts/`에 저장됩니다.

---

## 9. Reproduce Experiments

주요 실험 스크립트는 다음과 같습니다.

```bash
python3 scripts/build_accessibility_kfold_v31.py
python3 scripts/build_accessibility_kfold_v32.py
python3 scripts/build_accessibility_kfold_v4_hyperparamater.py
python3 scripts/build_accessibility_v42_grade.py
python3 scripts/build_accessibility_ensemble.py
python3 scripts/build_accessibility_ordinal.py
```

MLflow UI는 다음 명령어로 확인할 수 있습니다.

```bash
mlflow ui
```

---

## 10. Run API Server

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
python3 src/data/prepare.py
```

### 3. Train model

```bash
python3 src/models/train.py
```

### 4. Start API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

---

## 11. API Example

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict

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

---

## 12. Git & DVC Tracking Policy

### Git에 포함하는 항목

```text
README.md
src/
configs/
reports/
scripts/
artifacts/
mlruns/
requirements.txt
.dvc metadata
```

Git에는 다음 항목을 포함합니다.

* 전처리 코드
* 학습 코드
* 분석 코드
* API 코드
* 실험 스크립트
* 접근성 등급 예측 실험 결과
* 최종 모델 및 리포트
* MLflow 실험 로그

### Git에 직접 포함하지 않는 항목

```text
large raw data
.env
API keys
cache files
temporary files
```

대용량 원본 데이터는 Git이 아니라 DVC로 관리합니다.

---

## 13. DVC Usage

원본 데이터는 다음과 같이 DVC 메타데이터로 추적합니다.

```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Add raw data with DVC"
```

원격 저장소가 설정되어 있다면 다음 명령어로 데이터를 업로드합니다.

```bash
dvc push
```

다른 환경에서 데이터를 복원할 때는 다음 명령어를 사용합니다.

```bash
dvc pull
```

---

## 14. Tech Stack

| Category            | Tools                           |
| ------------------- | ------------------------------- |
| Language            | Python                          |
| ML                  | scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLflow                          |
| Data Versioning     | DVC                             |
| API                 | FastAPI, Uvicorn                |
| Visualization       | Matplotlib, SHAP                |
| Artifact Format     | joblib                          |
| Version Control     | Git, GitHub                     |

---

## 15. Project Goal

이 프로젝트의 최종 목표는 단순히 높은 분류 성능을 얻는 것이 아닙니다.

핵심은 공연 접근성 데이터를 기반으로 **휠체어석 수요가 높은 공연을 사전에 탐지**하고,
접근성 취약 공연에 대해 데이터 기반 개선 근거를 제공하는 것입니다.

따라서 모델 성능뿐 아니라 다음 항목을 함께 고려합니다.

* D/E 등급 탐지 성능
* 등급 간 큰 오차 감소
* feature 해석 가능성
* 실험 재현성
* API 기반 활용 가능성
* MLOps 흐름의 문서화


