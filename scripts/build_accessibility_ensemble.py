from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# SHAP 패키지 자동 설치 및 로드
try:
    import shap
except ImportError:
    print("Installing SHAP package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.models.label_encoded import LabelEncodedClassifier

# MLflow
import mlflow

# Data paths
DATASET_PATH = ROOT / "data" / "processed" / "accessibility_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports" / "accessibility"
MODEL_PATH = ARTIFACT_DIR / "accessibility_classifier_ensemble.joblib"
METRICS_PATH = REPORT_DIR / "accessibility_metrics_ensemble.json"

FEATURES = ["genre_group", "venue_type", "is_weekend", "duration_days", "organizer_type", "organizer"]
CATEGORICAL_FEATURES = ["genre_group", "venue_type", "organizer_type", "organizer"]
NUMERIC_FEATURES = ["is_weekend", "duration_days"]
TARGET = "accessibility_grade"
LABEL_ORDER = ["A", "B", "C", "D", "E"]
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

def get_preprocessor():
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False))
    ])
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ("num", numeric_pipe, NUMERIC_FEATURES)
    ])

def compute_ordinal_metrics(y_true, y_pred):
    y_true_ord = np.array([GRADE_MAP[val] for val in y_true])
    y_pred_ord = np.array([GRADE_MAP[val] for val in y_pred])
    distances = np.abs(y_true_ord - y_pred_ord)
    
    exact_acc = float(np.mean(distances == 0))
    adjacent_acc = float(np.mean(distances <= 1))
    mae = float(np.mean(distances))
    
    # Severe error: distance >= 3
    severe_errors = int(np.sum(distances >= 3))
    severe_rate = float(severe_errors / len(y_true))
    
    # Class D/E specific (vulnerable classes)
    de_indices = np.isin(y_true, ["D", "E"])
    de_macro_recall = 0.0
    if np.sum(de_indices) > 0:
        recalls = []
        for label in ["D", "E"]:
            label_indices = (y_true == label)
            if np.sum(label_indices) > 0:
                recalls.append(accuracy_score(y_true[label_indices], y_pred[label_indices]))
        de_macro_recall = float(np.mean(recalls)) if recalls else 0.0
        
    return {
        "exact_accuracy": exact_acc,
        "adjacent_accuracy": adjacent_acc,
        "mae": mae,
        "severe_error_count": severe_errors,
        "severe_error_rate": severe_rate,
        "de_macro_recall": de_macro_recall
    }

def plot_confusion_matrix(cm, labels, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    X = df[FEATURES]
    y = df[TARGET]
    
    # Define Base Estimators with tuned hyperparameters
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.7, max_depth=12, class_weight='balanced', random_state=42, n_jobs=1)
    et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, max_features='sqrt', max_depth=8, class_weight='balanced', random_state=42, n_jobs=1)
    
    try:
        from xgboost import XGBClassifier
        xgb_base = LabelEncodedClassifier(XGBClassifier(subsample=0.9, reg_lambda=5, n_estimators=300, max_depth=3, learning_rate=0.05, colsample_bytree=0.75, objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=1))
    except ImportError:
        xgb_base = None
        print("XGBoost not available")
        
    try:
        from lightgbm import LGBMClassifier
        lgbm_base = LabelEncodedClassifier(LGBMClassifier(subsample=1.0, num_leaves=31, n_estimators=300, max_depth=3, learning_rate=0.03, colsample_bytree=0.9, class_weight='balanced', random_state=42, n_jobs=1, verbose=-1))
    except Exception as exc:
        lgbm_base = None
        print(f"LightGBM not available: {type(exc).__name__}: {exc}")

    # Pipeline bases
    estimators = []
    estimators.append(("RandomForest", rf))
    estimators.append(("ExtraTrees", et))
    if xgb_base:
        estimators.append(("XGBoost", xgb_base))
    if lgbm_base:
        estimators.append(("LightGBM", lgbm_base))
        
    # Ensemble Estimators
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=[0.2, 0.3, 0.3, 0.2] if len(estimators) == 4 else None)
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), cv=5, n_jobs=1)
    
    all_models = {}
    for name, clf in estimators:
        all_models[name] = Pipeline([("preprocessor", get_preprocessor()), ("model", clf)])
    all_models["SoftVotingEnsemble"] = Pipeline([("preprocessor", get_preprocessor()), ("model", voting_clf)])
    all_models["StackingEnsemble"] = Pipeline([("preprocessor", get_preprocessor()), ("model", stacking_clf)])

    # Setup MLflow Experiment
    mlflow.set_experiment("accessibility_ensemble_experiment")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    print("\nStarting Cross Validation and MLflow Logging...")
    for model_name, model in all_models.items():
        print(f"\nEvaluating: {model_name}...")
        oof_preds = np.empty(len(y), dtype=object)
        
        # 5-Fold Cross Validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict(X_val)
            
        # Metrics calculation
        acc = accuracy_score(y, oof_preds)
        f1_macro = f1_score(y, oof_preds, average="macro")
        f2_macro = fbeta_score(y, oof_preds, beta=2, average="macro")
        ord_metrics = compute_ordinal_metrics(y, oof_preds)
        
        print(f"[{model_name}] OOF Accuracy: {acc:.4f} | F1 Macro: {f1_macro:.4f} | MAE: {ord_metrics['mae']:.4f} | Severe Errors: {ord_metrics['severe_error_count']} ({ord_metrics['severe_error_rate']*100:.2f}%)")
        
        # Log to MLflow (Manual Logging)
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({
                "model_name": model_name,
                "dataset_rows": len(df),
                "features_count": len(FEATURES),
                "validation_method": "5-Fold CV",
                "random_state": 42
            })
            
            mlflow.log_metrics({
                "oof_accuracy": acc,
                "oof_f1_macro": f1_macro,
                "oof_f2_macro": f2_macro,
                "oof_grade_mae": ord_metrics["mae"],
                "oof_adjacent_accuracy": ord_metrics["adjacent_accuracy"],
                "oof_severe_error_count": ord_metrics["severe_error_count"],
                "oof_severe_error_rate": ord_metrics["severe_error_rate"],
                "oof_de_macro_recall": ord_metrics["de_macro_recall"]
            })
            
            # Confusion matrix
            cm = confusion_matrix(y, oof_preds, labels=LABEL_ORDER)
            cm_path = REPORT_DIR / f"ensemble_{model_name.lower()}_confusion_matrix.png"
            plot_confusion_matrix(cm, LABEL_ORDER, f"{model_name} Confusion Matrix", cm_path)
            mlflow.log_artifact(str(cm_path))
            
            results[model_name] = {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f2_macro": f2_macro,
                "ordinal_metrics": ord_metrics,
                "confusion_matrix": cm.tolist()
            }
            
            # SHAP Analysis (RF 또는 ET 모델에 대해서만 수행 - TreeExplainer 지원)
            if model_name in ["RandomForest", "ExtraTrees"]:
                try:
                    print(f"Generating SHAP plots for {model_name}...")
                    # 1회성 fit을 실행하여 ColumnTransformer를 fitting함
                    fit_model = clone(model)
                    fit_model.fit(X, y)
                    
                    # 전처리기와 모델 추출
                    preprocessor = fit_model.named_steps["preprocessor"]
                    tree_model = fit_model.named_steps["model"]
                    
                    # 훈련 데이터 전체에 대해 transform 수행
                    X_trans = preprocessor.transform(X)
                    feature_names = preprocessor.get_feature_names_out()
                    
                    # Explainer 및 SHAP value 계산
                    explainer = shap.TreeExplainer(tree_model)
                    shap_values = explainer.shap_values(X_trans)
                    
                    # 1. SHAP Summary Plot (주류 클래스인 'E' 등급 분석)
                    class_idx = LABEL_ORDER.index("E")
                    if isinstance(shap_values, np.ndarray):
                        shap_class_values = shap_values[..., class_idx]
                    else:
                        shap_class_values = shap_values[class_idx]
                        
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_class_values, X_trans, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Summary Plot (Class E) - {model_name}")
                    plt.tight_layout()
                    summary_path = REPORT_DIR / f"shap_summary_{model_name.lower()}.png"
                    plt.savefig(summary_path, dpi=160)
                    plt.close()
                    mlflow.log_artifact(str(summary_path))
                    
                    # 2. SHAP Force Plot for a Single Sample (index 0)
                    sample_idx = 0
                    actual_class = y.iloc[sample_idx]
                    class_idx_sample = LABEL_ORDER.index(actual_class)
                    if isinstance(shap_values, np.ndarray):
                        shap_class_values_sample = shap_values[..., class_idx_sample]
                    else:
                        shap_class_values_sample = shap_values[class_idx_sample]
                    
                    plt.figure(figsize=(12, 4))
                    shap.force_plot(
                        explainer.expected_value[class_idx_sample],
                        shap_class_values_sample[sample_idx],
                        X_trans[sample_idx],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    plt.title(f"SHAP Force Plot (Sample 0, Class {actual_class}) - {model_name}")
                    plt.tight_layout()
                    force_path = REPORT_DIR / f"shap_force_{model_name.lower()}_sample0.png"
                    plt.savefig(force_path, dpi=160)
                    plt.close()
                    mlflow.log_artifact(str(force_path))
                    
                    print(f"SHAP plots logged for {model_name}.")
                except Exception as e:
                    print(f"Failed to generate SHAP for {model_name}: {e}")
            
    # Find best model
    safe_models = {k: v for k, v in results.items() if v["ordinal_metrics"]["severe_error_count"] == 0}
    if safe_models:
        best_model_name = max(safe_models, key=lambda k: safe_models[k]["f1_macro"])
    else:
        best_model_name = max(results, key=lambda k: results[k]["f1_macro"])
        
    print(f"\nBest Model selected: {best_model_name}")
    
    # Train best model on whole dataset and save
    best_model = all_models[best_model_name]
    best_model.fit(X, y)
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": best_model,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "label_order": LABEL_ORDER,
        "metrics": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "best_model": best_model_name,
            "best_f1_macro": results[best_model_name]["f1_macro"],
            "best_accuracy": results[best_model_name]["accuracy"],
            "best_severe_error_count": results[best_model_name]["ordinal_metrics"]["severe_error_count"],
            "all_results": results
        }
    }, MODEL_PATH)
    
    # Save ensemble metrics report
    METRICS_PATH.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_model_name,
        "results": results
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
