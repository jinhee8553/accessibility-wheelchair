from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.label_encoded import LabelEncodedClassifier


DATASET_FILE = ROOT / "data" / "processed" / "accessibility_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports" / "accessibility"
MODEL_FILE = ARTIFACT_DIR / "accessibility_classifier.joblib"
METRICS_FILE = REPORT_DIR / "accessibility_metrics.json"

CATEGORICAL_FEATURES = ["genre_group", "venue_type", "organizer_type", "organizer"]
NUMERIC_FEATURES = ["is_weekend", "duration_days"]
FEATURES = [*CATEGORICAL_FEATURES[:2], *NUMERIC_FEATURES, *CATEGORICAL_FEATURES[2:]]
TARGET = "accessibility_grade"
LABEL_ORDER = ["A", "B", "C", "D", "E"]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse=False)


def build_preprocessor() -> ColumnTransformer:
    categorical_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    numeric_steps = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_steps, CATEGORICAL_FEATURES),
            ("num", numeric_steps, NUMERIC_FEATURES),
        ]
    )


def pipeline_for(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {
        "RandomForestClassifier": pipeline_for(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
        ),
        "ExtraTreesClassifier": pipeline_for(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoostClassifier"] = pipeline_for(
            LabelEncodedClassifier(
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=1,
                )
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBMClassifier"] = pipeline_for(
            LabelEncodedClassifier(
                LGBMClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                    verbose=-1,
                )
            )
        )
    except Exception:
        pass

    return candidates


def score_predictions(y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    labels = [label for label in LABEL_ORDER if label in set(y_true) or label in set(y_pred)]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_label_distribution(labels: pd.Series, output_path: Path) -> None:
    counts = labels.value_counts().reindex(LABEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color="#386cb0")
    ax.set_xlabel("Accessibility grade")
    ax.set_ylabel("Count")
    ax.set_title("Accessibility Grade Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: pd.Series, y_pred: Any, labels: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Accessibility Grade Confusion Matrix")
    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            ax.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance(pipeline: Pipeline, output_csv: Path, output_png: Path) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    frame = (
        pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )
    frame["rank"] = [f"f{i:02d}" for i in range(1, len(frame) + 1)]
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 6))
    frame.sort_values("importance").plot.barh(x="rank", y="importance", ax=ax, legend=False, color="#59a14f")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    table = frame.fillna("")
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_model_report(metrics: dict[str, Any], feature_importance: pd.DataFrame) -> None:
    rows = [
        {
            "model": name,
            "accuracy": round(values["accuracy"], 4),
            "f1_macro": round(values["f1_macro"], 4),
        }
        for name, values in metrics["candidates"].items()
    ]
    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    top_features = feature_importance[["rank", "feature", "importance"]].head(10)
    report = f"""# Step 4. 모델 학습

## 문제 정의

- 문제 유형: A~E 다중 분류
- 타겟: `{TARGET}`
- 평가 지표: macro F1

## 학습 Feature

{markdown_table(pd.DataFrame({"feature": FEATURES}))}

## 데이터 분할

- 방식: stratified train/test split
- test_size: {metrics["split"]["test_size"]}
- random_state: {metrics["split"]["random_state"]}
- 학습 행 수: {metrics["split"]["train_rows"]:,}
- 검증 행 수: {metrics["split"]["test_rows"]:,}

## 모델 비교

{markdown_table(comparison)}

## 최종 모델

- 모델: `{metrics["best_model"]["name"]}`
- accuracy: `{metrics["best_model"]["accuracy"]:.4f}`
- macro F1: `{metrics["best_model"]["f1_macro"]:.4f}`
- 저장 경로: `{MODEL_FILE.relative_to(ROOT)}`

## 상위 Feature Importance

{markdown_table(top_features)}
"""
    (REPORT_DIR / "step_4_model.md").write_text(report, encoding="utf-8")


def train() -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATASET_FILE, low_memory=False)
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    fitted: dict[str, Pipeline] = {}
    candidate_scores: dict[str, dict[str, Any]] = {}

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    candidate_scores["DummyClassifier"] = {
        "model": "most_frequent",
        **score_predictions(y_test, baseline.predict(x_test)),
    }

    for name, pipeline in candidate_models().items():
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        candidate_scores[name] = {"model": name, **score_predictions(y_test, pipeline.predict(x_test))}

    best_name = max(fitted, key=lambda name: candidate_scores[name]["f1_macro"])
    best_pipeline = fitted[best_name]
    best_prediction = best_pipeline.predict(x_test)
    present_labels = [label for label in LABEL_ORDER if label in set(y)]

    metrics: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_FILE.relative_to(ROOT)),
        "rows": int(len(data)),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "label_order": present_labels,
        "label_distribution": y.value_counts().reindex(present_labels, fill_value=0).astype(int).to_dict(),
        "split": {
            "method": "stratified_train_test_split",
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "candidates": candidate_scores,
        "best_model": {
            "name": best_name,
            "accuracy": float(candidate_scores[best_name]["accuracy"]),
            "f1_macro": float(candidate_scores[best_name]["f1_macro"]),
        },
    }

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": FEATURES,
            "target": TARGET,
            "label_order": present_labels,
            "metrics": metrics,
        },
        MODEL_FILE,
    )
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_label_distribution(y, REPORT_DIR / "accessibility_label_distribution.png")
    plot_confusion(y_test, best_prediction, present_labels, REPORT_DIR / "accessibility_confusion_matrix.png")
    importance = save_feature_importance(
        best_pipeline,
        REPORT_DIR / "accessibility_feature_importance.csv",
        REPORT_DIR / "accessibility_feature_importance.png",
    )
    write_model_report(metrics, importance)
    return metrics


def main() -> None:
    metrics = train()
    best = metrics["best_model"]
    print(f"best_model: {best['name']}")
    print(f"accuracy: {best['accuracy']:.4f}")
    print(f"f1_macro: {best['f1_macro']:.4f}")
    print(f"saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()
