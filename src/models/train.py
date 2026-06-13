from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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


DATASET_PATH = ROOT / "data" / "processed" / "accessibility_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports" / "accessibility"
MODEL_PATH = ARTIFACT_DIR / "accessibility_classifier.joblib"
METRICS_PATH = REPORT_DIR / "accessibility_metrics.json"
FINAL_HTML_PATH = REPORT_DIR / "accessibility_project_summary.html"

CATEGORICAL_FEATURES = ["genre_group", "venue_type", "organizer_type", "organizer"]
NUMERIC_FEATURES = ["is_weekend", "duration_days"]
FEATURES = [
    "genre_group",
    "venue_type",
    "is_weekend",
    "duration_days",
    "organizer_type",
    "organizer",
]
TARGET = "accessibility_grade"
LABEL_ORDER = ["A", "B", "C", "D", "E"]
EXCLUDED_CONSTANT_COLUMNS = [
    "booking_lead_days",
    "start_time",
]
EXCLUDED_LEAKAGE_COLUMNS = [
    "paid_audience_count",
    "wheelchair_booking_count",
    "total_seats",
    "general_seats",
    "wheelchair_seats",
    "wheelchair_seat_ratio",
    "wheelchair_booking_rate_raw",
    "overall_booking_rate",
    "wheelchair_booking_rate",
]


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse=False)


def build_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("num", numeric_pipe, NUMERIC_FEATURES),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    candidates = {
        "RandomForestClassifier": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "ExtraTreesClassifier": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoostClassifier"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
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
                    ),
                ),
            ]
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBMClassifier"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
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
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return candidates


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series | list[str]) -> dict[str, object]:
    present_labels = [label for label in LABEL_ORDER if label in set(y_true) or label in set(y_pred)]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=present_labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_label_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df[TARGET].value_counts().reindex(LABEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color="#4c78a8")
    ax.set_xlabel("Accessibility grade")
    ax.set_ylabel("Count")
    ax.set_title("Accessibility Grade Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Accessibility Grade Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance(pipeline: Pipeline, output_csv: Path, output_png: Path) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(25)
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


def markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    table = frame.copy()
    if index:
        index_name = table.index.name or ""
        table = table.reset_index().rename(columns={"index": index_name})
    table = table.fillna("")
    headers = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_step4_report(metrics: dict[str, object], best_name: str, feature_importance: pd.DataFrame) -> None:
    model_rows = []
    for name, values in metrics["candidates"].items():
        model_rows.append(
            {
                "model": name,
                "accuracy": round(values["accuracy"], 4),
                "f1_macro": round(values["f1_macro"], 4),
            }
        )
    model_table = pd.DataFrame(model_rows).sort_values("f1_macro", ascending=False)
    label_counts = pd.Series(metrics["label_distribution"], name="count").to_frame()
    excluded = pd.DataFrame(
        [
            *[
                {"excluded_column": column, "reason": "상수값이라 학습 정보 없음"}
                for column in EXCLUDED_CONSTANT_COLUMNS
            ],
            *[
                {"excluded_column": column, "reason": "좌석 수 또는 타겟 산출에 직접 연결"}
                for column in EXCLUDED_LEAKAGE_COLUMNS
            ],
        ]
    )
    candidate_status = pd.DataFrame(
        [
            {
                "algorithm": "RandomForest",
                "status": "구현 및 학습 완료"
                if "RandomForestClassifier" in metrics["candidates"]
                else "미실행",
            },
            {
                "algorithm": "ExtraTrees",
                "status": "구현 및 학습 완료"
                if "ExtraTreesClassifier" in metrics["candidates"]
                else "미실행",
            },
            {
                "algorithm": "XGBoost",
                "status": "구현 및 학습 완료"
                if "XGBoostClassifier" in metrics["candidates"]
                else "패키지 미설치 또는 import 실패로 제외",
            },
            {
                "algorithm": "LightGBM",
                "status": "구현 및 학습 완료"
                if "LightGBMClassifier" in metrics["candidates"]
                else "패키지 미설치 또는 import 실패로 제외",
            },
        ]
    )
    report = f"""# Step 4. 접근성 등급 분류 모델

## 문제 정의

- 문제 유형: 다중 분류
- 타겟 변수: `{TARGET}`
- 등급: A~E
- 주 평가 지표: macro F1

## 학습 Feature

{markdown_table(pd.DataFrame({"feature": FEATURES}), index=False)}

## 학습 제외 컬럼

아래 컬럼은 상수값이거나 좌석 수 또는 타겟 산출에 직접 연결되는 값이므로 모델 feature에서 제외했다.

{markdown_table(excluded, index=False)}

## 등급 분포

{markdown_table(label_counts)}

## 데이터 분할

- 방식: stratified train/test split
- test_size: {metrics["split"]["test_size"]}
- random_state: {metrics["split"]["random_state"]}
- 학습 행 수: {metrics["split"]["train_rows"]:,}
- 검증 행 수: {metrics["split"]["test_rows"]:,}

데이터 수가 500행으로 작아 각 등급이 검증 세트에 포함되도록 stratified split을 사용했다.

## 모델 비교

{markdown_table(model_table, index=False)}

## 후보 알고리즘 구현 상태

{markdown_table(candidate_status, index=False)}

## 최종 선택 모델

- 모델: `{best_name}`
- 저장 경로: `{MODEL_PATH.relative_to(ROOT)}`
- 메트릭 저장 경로: `{METRICS_PATH.relative_to(ROOT)}`

## 상위 Feature Importance

{markdown_table(feature_importance[["rank", "feature", "importance"]].head(10), index=False)}

## 산출물

- 혼동행렬: `reports/accessibility/accessibility_confusion_matrix.png`
- Feature importance CSV: `reports/accessibility/accessibility_feature_importance.csv`
- Feature importance 이미지: `reports/accessibility/accessibility_feature_importance.png`
"""
    (REPORT_DIR / "step_4_model.md").write_text(report, encoding="utf-8")


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    in_list = False
    in_table = False
    table_rows: list[str] = []

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        output.append("<table>")
        for row_index, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if row_index == 1 and all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            tag = "th" if row_index == 0 else "td"
            output.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
        output.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_list()
            in_table = True
            table_rows.append(stripped)
            continue
        flush_table()
        if not stripped:
            flush_list()
            continue
        if stripped.startswith("# "):
            flush_list()
            output.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            flush_list()
            output.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            flush_list()
            output.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{html.escape(stripped[2:])}</li>")
        else:
            flush_list()
            output.append(f"<p>{html.escape(stripped)}</p>")
    flush_table()
    flush_list()
    return "\n".join(output)


def write_final_html(metrics: dict[str, object]) -> None:
    md_files = [
        REPORT_DIR / "step_1_dataset.md",
        REPORT_DIR / "step_2_labeling.md",
        REPORT_DIR / "step_3_features.md",
        REPORT_DIR / "step_4_model.md",
    ]
    sections = []
    for path in md_files:
        if path.exists():
            sections.append(markdown_to_html(path.read_text(encoding="utf-8")))
    best = metrics["best_model"]
    body = "\n<hr>\n".join(sections)
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>장애인 접근성 등급 진단 모델 구현 결과</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.6; color: #222; }}
    h1, h2, h3 {{ color: #17324d; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef3f8; }}
    code {{ background: #f3f5f7; padding: 2px 4px; border-radius: 4px; }}
    .summary {{ background: #f7fafc; border-left: 4px solid #4c78a8; padding: 16px 20px; margin-bottom: 28px; }}
    img {{ max-width: 760px; width: 100%; border: 1px solid #d7dde5; margin: 12px 0 24px; }}
  </style>
</head>
<body>
  <div class="summary">
    <h1>장애인 접근성 등급 진단 모델 구현 결과</h1>
    <p>생성 시각: {html.escape(metrics["created_at"])}</p>
    <p>최종 모델: <strong>{html.escape(best["name"])}</strong> / macro F1: <strong>{best["f1_macro"]:.4f}</strong> / accuracy: <strong>{best["accuracy"]:.4f}</strong></p>
    <p>이번 구현 범위는 서비스 계획의 1~4번이며, FastAPI, Streamlit, DVC, Evidently 단계는 진행하지 않았다.</p>
  </div>
  {body}
  <h2>시각화 산출물</h2>
  <h3>등급 분포</h3>
  <img src="accessibility_label_distribution.png" alt="Accessibility label distribution">
  <h3>혼동행렬</h3>
  <img src="accessibility_confusion_matrix.png" alt="Accessibility confusion matrix">
  <h3>Feature Importance</h3>
  <img src="accessibility_feature_importance.png" alt="Accessibility feature importance">
</body>
</html>
"""
    FINAL_HTML_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    candidate_metrics: dict[str, dict[str, object]] = {
        "DummyClassifier": {
            "model": "DummyClassifier(strategy='most_frequent')",
            **evaluate_predictions(y_test, baseline_pred),
        }
    }
    fitted_candidates: dict[str, Pipeline] = {}
    for name, pipeline in build_candidates().items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        candidate_metrics[name] = {"model": name, **evaluate_predictions(y_test, pred)}
        fitted_candidates[name] = pipeline

    best_name = max(fitted_candidates, key=lambda name: candidate_metrics[name]["f1_macro"])
    best_pipeline = fitted_candidates[best_name]
    best_pred = best_pipeline.predict(X_test)
    labels = [label for label in LABEL_ORDER if label in set(y)]

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "rows": int(len(df)),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "excluded_constant_columns": EXCLUDED_CONSTANT_COLUMNS,
        "excluded_leakage_columns": EXCLUDED_LEAKAGE_COLUMNS,
        "target": TARGET,
        "label_order": labels,
        "label_distribution": df[TARGET].value_counts().reindex(labels, fill_value=0).astype(int).to_dict(),
        "split": {
            "method": "stratified_train_test_split",
            "test_size": 0.2,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "candidates": candidate_metrics,
        "best_model": {
            "name": best_name,
            "accuracy": float(candidate_metrics[best_name]["accuracy"]),
            "f1_macro": float(candidate_metrics[best_name]["f1_macro"]),
        },
    }

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "excluded_constant_columns": EXCLUDED_CONSTANT_COLUMNS,
            "target": TARGET,
            "label_order": labels,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_label_distribution(df, REPORT_DIR / "accessibility_label_distribution.png")
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plot_confusion_matrix(cm, labels, REPORT_DIR / "accessibility_confusion_matrix.png")
    feature_importance = save_feature_importance(
        best_pipeline,
        REPORT_DIR / "accessibility_feature_importance.csv",
        REPORT_DIR / "accessibility_feature_importance.png",
    )
    write_step4_report(metrics, best_name, feature_importance)
    write_final_html(metrics)

    print(f"saved_model={MODEL_PATH}")
    print(f"saved_metrics={METRICS_PATH}")
    print(f"best_model={best_name}")
    print(f"best_accuracy={metrics['best_model']['accuracy']:.4f}")
    print(f"best_f1_macro={metrics['best_model']['f1_macro']:.4f}")
    print(f"final_html={FINAL_HTML_PATH}")


if __name__ == "__main__":
    main()
