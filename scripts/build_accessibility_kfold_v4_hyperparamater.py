from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime
from numbers import Number
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.label_encoded import LabelEncodedClassifier  # noqa: E402
from src.models.train import (  # noqa: E402
    CATEGORICAL_FEATURES,
    EXCLUDED_CONSTANT_COLUMNS,
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURES,
    LABEL_ORDER,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    plot_confusion_matrix,
    plot_label_distribution,
    save_feature_importance,
)

DATASET_PATH = ROOT / "data/processed/accessibility_dataset.csv"
REPORT_DIR = ROOT / "reports/accessibility"
ARTIFACT_DIR = ROOT / "artifacts"

METRICS_PATH = REPORT_DIR / "accessibility_metrics_kfold_v4_hyperparamater.json"
MODEL_PATH = ARTIFACT_DIR / "accessibility_classifier_kfold_v4_hyperparamater.joblib"
FINAL_HTML_PATH = REPORT_DIR / "accessibility_project_summary_v4_hyperparamater.html"
LABEL_PNG = REPORT_DIR / "accessibility_label_distribution_kfold_v4_hyperparamater.png"
CM_PNG = REPORT_DIR / "accessibility_confusion_matrix_kfold_v4_hyperparamater.png"
FI_CSV = REPORT_DIR / "accessibility_feature_importance_kfold_v4_hyperparamater.csv"
FI_PNG = REPORT_DIR / "accessibility_feature_importance_kfold_v4_hyperparamater.png"
LIGHTGBM_MODEL_PATH = ARTIFACT_DIR / "accessibility_lightgbm_classifier_kfold_v4_hyperparamater.joblib"
LIGHTGBM_INFERENCE_PATH = REPORT_DIR / "accessibility_lightgbm_inference_v4_hyperparamater.csv"
DE_MISCLASSIFICATION_PATH = REPORT_DIR / "accessibility_de_misclassifications_kfold_v4_hyperparamater.csv"

N_SPLITS = 5
RANDOM_STATE = 42
N_ITER = 18


def format_table_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, Number):
        numeric = float(value)
        if numeric.is_integer():
            return f"{int(numeric):,}"
        return f"{numeric:.4f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(frame: pd.DataFrame, index: bool = True) -> str:
    table = frame.copy()
    if index:
        index_name = table.index.name or ""
        table = table.reset_index().rename(columns={"index": index_name})
    headers = [str(column) for column in table.columns]
    rows = [[format_table_value(value) for value in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_inline(markdown_text: str) -> str:
    parts = re.split(r"(`[^`]+`)", markdown_text)
    rendered = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            rendered.append(f"<code>{html.escape(part[1:-1])}</code>")
        else:
            rendered.append(html.escape(part))
    return "".join(rendered)


def markdown_to_html(markdown_text: str, h1_tag: str = "h2") -> str:
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
        output.append('<div class="table-wrap"><table>')
        for row_index, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if row_index == 1 and all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            tag = "th" if row_index == 0 else "td"
            output.append("<tr>" + "".join(f"<{tag}>{render_inline(cell)}</{tag}>" for cell in cells) + "</tr>")
        output.append("</table></div>")
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
            output.append(f"<{h1_tag}>{render_inline(stripped[2:])}</{h1_tag}>")
        elif stripped.startswith("## "):
            flush_list()
            output.append(f"<h3>{render_inline(stripped[3:])}</h3>")
        elif stripped.startswith("### "):
            flush_list()
            output.append(f"<h4>{render_inline(stripped[4:])}</h4>")
        elif stripped.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{render_inline(stripped[2:])}</li>")
        else:
            flush_list()
            output.append(f"<p>{render_inline(stripped)}</p>")
    flush_table()
    flush_list()
    return "\n".join(output)


def report_section(markdown_text: str) -> str:
    return f'<section class="report-section">\n{markdown_to_html(markdown_text)}\n</section>'


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def build_tuning_spaces() -> tuple[dict[str, tuple[Pipeline, dict[str, list[object]]]], dict[str, str]]:
    spaces: dict[str, tuple[Pipeline, dict[str, list[object]]]] = {
        "RandomForestClassifier": (
            make_pipeline(
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.7, None],
            },
        ),
        "ExtraTreesClassifier": (
            make_pipeline(
                ExtraTreesClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.7, None],
            },
        ),
    }
    unavailable: dict[str, str] = {}

    try:
        from xgboost import XGBClassifier

        spaces["XGBoostClassifier"] = (
            make_pipeline(
                LabelEncodedClassifier(
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    )
                )
            ),
            {
                "model__estimator__n_estimators": [100, 200, 300],
                "model__estimator__max_depth": [2, 3, 4],
                "model__estimator__learning_rate": [0.03, 0.05, 0.1],
                "model__estimator__subsample": [0.75, 0.9, 1.0],
                "model__estimator__colsample_bytree": [0.75, 0.9, 1.0],
                "model__estimator__reg_lambda": [1, 3, 5],
            },
        )
    except Exception as exc:
        unavailable["XGBoostClassifier"] = f"{type(exc).__name__}: {exc}"

    try:
        from lightgbm import LGBMClassifier

        spaces["LightGBMClassifier"] = (
            make_pipeline(
                LabelEncodedClassifier(
                    LGBMClassifier(
                        objective="multiclass",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        verbose=-1,
                    )
                )
            ),
            {
                "model__estimator__n_estimators": [100, 200, 300],
                "model__estimator__max_depth": [3, 4, -1],
                "model__estimator__learning_rate": [0.03, 0.05, 0.1],
                "model__estimator__num_leaves": [15, 31, 63],
                "model__estimator__subsample": [0.75, 0.9, 1.0],
                "model__estimator__colsample_bytree": [0.75, 0.9, 1.0],
            },
        )
    except Exception as exc:
        unavailable["LightGBMClassifier"] = f"{type(exc).__name__}: {exc}"

    return spaces, unavailable


def compact_params(params: dict[str, object]) -> dict[str, object]:
    compacted = {}
    for key, value in params.items():
        compacted[key.replace("model__", "").replace("estimator__", "")] = value
    return compacted


def recall_by_label(y_true: list[str] | pd.Series, y_pred: list[str] | pd.Series, labels: list[str]) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: float(value) for label, value in zip(labels, recalls)}


def f2_by_label(y_true: list[str] | pd.Series, y_pred: list[str] | pd.Series, labels: list[str]) -> dict[str, float]:
    scores = fbeta_score(y_true, y_pred, beta=2, labels=labels, average=None, zero_division=0)
    return {label: float(value) for label, value in zip(labels, scores)}


def vulnerable_group_recall(y_true: list[str], y_pred: list[str]) -> float:
    vulnerable = {"D", "E"}
    true_vulnerable = [actual in vulnerable for actual in y_true]
    denominator = sum(true_vulnerable)
    if denominator == 0:
        return 0.0
    numerator = sum(is_vulnerable and predicted in vulnerable for is_vulnerable, predicted in zip(true_vulnerable, y_pred))
    return float(numerator / denominator)


def de_macro_recall(recalls: dict[str, float]) -> float:
    values = [recalls[label] for label in ("D", "E") if label in recalls]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def de_macro_f2(scores: dict[str, float]) -> float:
    values = [scores[label] for label in ("D", "E") if label in scores]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_cv_oof(
    estimator: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    labels: list[str],
    splitter: StratifiedKFold,
) -> tuple[dict[str, object], list[str], list[str], list[int]]:
    folds = []
    all_true: list[str] = []
    all_pred: list[str] = []
    all_index: list[int] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = clone(estimator)
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        fold_recalls = recall_by_label(y_valid, pred, labels)
        fold_f2 = f2_by_label(y_valid, pred, labels)
        fold_record = {
            "fold": fold_idx,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "accuracy": float(accuracy_score(y_valid, pred)),
            "f1_macro": float(f1_score(y_valid, pred, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_valid, pred, labels=labels, average="weighted", zero_division=0)),
            "f2_macro": float(fbeta_score(y_valid, pred, beta=2, labels=labels, average="macro", zero_division=0)),
            "f2_weighted": float(
                fbeta_score(y_valid, pred, beta=2, labels=labels, average="weighted", zero_division=0)
            ),
            "recall_d": fold_recalls.get("D", 0.0),
            "recall_e": fold_recalls.get("E", 0.0),
            "de_macro_recall": de_macro_recall(fold_recalls),
            "de_group_recall": vulnerable_group_recall(y_valid.tolist(), list(pred)),
            "f2_d": fold_f2.get("D", 0.0),
            "f2_e": fold_f2.get("E", 0.0),
            "de_macro_f2": de_macro_f2(fold_f2),
        }
        for label in labels:
            suffix = label.lower()
            fold_record[f"recall_{suffix}"] = fold_recalls.get(label, 0.0)
            fold_record[f"f2_{suffix}"] = fold_f2.get(label, 0.0)
        folds.append(fold_record)
        all_true.extend(y_valid.tolist())
        all_pred.extend(list(pred))
        all_index.extend([int(idx) for idx in valid_idx])

    fold_df = pd.DataFrame(folds)
    oof_recalls = recall_by_label(all_true, all_pred, labels)
    oof_f2 = f2_by_label(all_true, all_pred, labels)
    metrics = {
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=1)),
        "f1_macro_mean": float(fold_df["f1_macro"].mean()),
        "f1_macro_std": float(fold_df["f1_macro"].std(ddof=1)),
        "f1_weighted_mean": float(fold_df["f1_weighted"].mean()),
        "f1_weighted_std": float(fold_df["f1_weighted"].std(ddof=1)),
        "f2_macro_mean": float(fold_df["f2_macro"].mean()),
        "f2_macro_std": float(fold_df["f2_macro"].std(ddof=1)),
        "f2_weighted_mean": float(fold_df["f2_weighted"].mean()),
        "f2_weighted_std": float(fold_df["f2_weighted"].std(ddof=1)),
        "recall_d_mean": float(fold_df["recall_d"].mean()),
        "recall_e_mean": float(fold_df["recall_e"].mean()),
        "de_macro_recall_mean": float(fold_df["de_macro_recall"].mean()),
        "de_group_recall_mean": float(fold_df["de_group_recall"].mean()),
        "f2_d_mean": float(fold_df["f2_d"].mean()),
        "f2_e_mean": float(fold_df["f2_e"].mean()),
        "de_macro_f2_mean": float(fold_df["de_macro_f2"].mean()),
        "oof_accuracy": float(accuracy_score(all_true, all_pred)),
        "oof_f1_macro": float(f1_score(all_true, all_pred, labels=labels, average="macro", zero_division=0)),
        "oof_f1_weighted": float(f1_score(all_true, all_pred, labels=labels, average="weighted", zero_division=0)),
        "oof_f2_macro": float(fbeta_score(all_true, all_pred, beta=2, labels=labels, average="macro", zero_division=0)),
        "oof_f2_weighted": float(
            fbeta_score(all_true, all_pred, beta=2, labels=labels, average="weighted", zero_division=0)
        ),
        "oof_recall_by_label": oof_recalls,
        "oof_recall_d": oof_recalls.get("D", 0.0),
        "oof_recall_e": oof_recalls.get("E", 0.0),
        "oof_de_macro_recall": de_macro_recall(oof_recalls),
        "oof_de_group_recall": vulnerable_group_recall(all_true, all_pred),
        "oof_f2_by_label": oof_f2,
        "oof_f2_d": oof_f2.get("D", 0.0),
        "oof_f2_e": oof_f2.get("E", 0.0),
        "oof_de_macro_f2": de_macro_f2(oof_f2),
        "folds": folds,
        "classification_report": classification_report(
            all_true,
            all_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, all_true, all_pred, all_index


def macro_f1_scorer(estimator: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> float:
    labels = [label for label in LABEL_ORDER if label in set(y_valid)]
    pred = estimator.predict(x_valid)
    return float(f1_score(y_valid, pred, labels=labels, average="macro", zero_division=0))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    x = df[FEATURES]
    y = df[TARGET]
    labels = [label for label in LABEL_ORDER if label in set(y)]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    spaces, unavailable = build_tuning_spaces()

    candidate_metrics: dict[str, dict[str, object]] = {}
    candidate_predictions: dict[str, tuple[list[str], list[str], list[int]]] = {}
    tuned_estimators: dict[str, Pipeline] = {}

    baseline = DummyClassifier(strategy="most_frequent")
    baseline_metrics, baseline_true, baseline_pred, baseline_index = evaluate_cv_oof(
        baseline,
        x,
        y,
        labels,
        splitter,
    )
    candidate_metrics["DummyClassifier"] = {
        "model": "DummyClassifier(strategy='most_frequent')",
        "tuned": False,
        "search_status": "baseline_only",
        "best_params": {},
        "searched_candidates": 1,
        **baseline_metrics,
    }
    candidate_predictions["DummyClassifier"] = (baseline_true, baseline_pred, baseline_index)

    for name, (pipeline, param_distributions) in spaces.items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=N_ITER,
            scoring=macro_f1_scorer,
            cv=splitter,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(x, y)
        best_estimator = search.best_estimator_
        tuned_estimators[name] = best_estimator
        cv_metrics, all_true, all_pred, all_index = evaluate_cv_oof(best_estimator, x, y, labels, splitter)
        candidate_metrics[name] = {
            "model": name,
            "tuned": True,
            "search_status": "completed",
            "searched_candidates": int(len(search.cv_results_["params"])),
            "best_search_score": float(search.best_score_),
            "best_params": compact_params(search.best_params_),
            **cv_metrics,
        }
        candidate_predictions[name] = (all_true, all_pred, all_index)

    for name, reason in unavailable.items():
        candidate_metrics[name] = {
            "model": name,
            "tuned": False,
            "search_status": "skipped_import_error",
            "skip_reason": reason,
        }

    eligible = [
        name
        for name, values in candidate_metrics.items()
        if name != "DummyClassifier" and values.get("search_status") == "completed"
    ]
    if not eligible:
        raise RuntimeError("No tunable model candidates were available.")

    best_name = max(eligible, key=lambda model_name: candidate_metrics[model_name]["f1_macro_mean"])
    best_pipeline = tuned_estimators[best_name]
    best_pipeline.fit(x, y)
    best_true, best_pred, best_index = candidate_predictions[best_name]

    metrics = {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "conda_env": "pronii",
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
            "method": "randomized_search_cv_with_stratified_k_fold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "n_iter_per_model": N_ITER,
            "scoring": "macro_f1",
            "fold_train_rows": int(len(df) * (N_SPLITS - 1) / N_SPLITS),
            "fold_validation_rows": int(len(df) / N_SPLITS),
        },
        "candidates": candidate_metrics,
        "best_model": {
            "name": best_name,
            "accuracy_mean": candidate_metrics[best_name]["accuracy_mean"],
            "accuracy_std": candidate_metrics[best_name]["accuracy_std"],
            "f1_macro_mean": candidate_metrics[best_name]["f1_macro_mean"],
            "f1_macro_std": candidate_metrics[best_name]["f1_macro_std"],
            "f1_weighted_mean": candidate_metrics[best_name]["f1_weighted_mean"],
            "f1_weighted_std": candidate_metrics[best_name]["f1_weighted_std"],
            "f2_macro_mean": candidate_metrics[best_name]["f2_macro_mean"],
            "f2_macro_std": candidate_metrics[best_name]["f2_macro_std"],
            "f2_weighted_mean": candidate_metrics[best_name]["f2_weighted_mean"],
            "f2_weighted_std": candidate_metrics[best_name]["f2_weighted_std"],
            "recall_d_mean": candidate_metrics[best_name]["recall_d_mean"],
            "recall_e_mean": candidate_metrics[best_name]["recall_e_mean"],
            "de_macro_recall_mean": candidate_metrics[best_name]["de_macro_recall_mean"],
            "de_group_recall_mean": candidate_metrics[best_name]["de_group_recall_mean"],
            "f2_d_mean": candidate_metrics[best_name]["f2_d_mean"],
            "f2_e_mean": candidate_metrics[best_name]["f2_e_mean"],
            "de_macro_f2_mean": candidate_metrics[best_name]["de_macro_f2_mean"],
            "oof_accuracy": candidate_metrics[best_name]["oof_accuracy"],
            "oof_f1_macro": candidate_metrics[best_name]["oof_f1_macro"],
            "oof_f1_weighted": candidate_metrics[best_name]["oof_f1_weighted"],
            "oof_f2_macro": candidate_metrics[best_name]["oof_f2_macro"],
            "oof_f2_weighted": candidate_metrics[best_name]["oof_f2_weighted"],
            "oof_recall_d": candidate_metrics[best_name]["oof_recall_d"],
            "oof_recall_e": candidate_metrics[best_name]["oof_recall_e"],
            "oof_de_macro_recall": candidate_metrics[best_name]["oof_de_macro_recall"],
            "oof_de_group_recall": candidate_metrics[best_name]["oof_de_group_recall"],
            "oof_f2_d": candidate_metrics[best_name]["oof_f2_d"],
            "oof_f2_e": candidate_metrics[best_name]["oof_f2_e"],
            "oof_de_macro_f2": candidate_metrics[best_name]["oof_de_macro_f2"],
            "best_params": candidate_metrics[best_name]["best_params"],
        },
    }

    oof_frame = df.iloc[best_index][
        ["title", "start_date", "genre_group", "venue", "venue_type", "organizer_type", "organizer"]
    ].copy()
    oof_frame.insert(0, "row_index", best_index)
    oof_frame["actual_grade"] = best_true
    oof_frame["predicted_grade"] = best_pred
    vulnerable_mask = oof_frame["actual_grade"].isin(["D", "E"])
    de_misclassified = oof_frame[vulnerable_mask & (oof_frame["actual_grade"] != oof_frame["predicted_grade"])].copy()
    de_misclassified["failure_type"] = de_misclassified["predicted_grade"].isin(["D", "E"]).map(
        {True: "D/E 내부 혼동", False: "취약 등급 누락"}
    )
    de_misclassified = de_misclassified[
        [
            "row_index",
            "failure_type",
            "actual_grade",
            "predicted_grade",
            "title",
            "start_date",
            "genre_group",
            "venue",
            "venue_type",
            "organizer_type",
            "organizer",
        ]
    ].sort_values(["failure_type", "actual_grade", "predicted_grade", "start_date", "row_index"])
    de_misclassified.to_csv(DE_MISCLASSIFICATION_PATH, index=False, encoding="utf-8-sig")
    metrics["best_model"]["de_misclassification_count"] = int(len(de_misclassified))
    metrics["best_model"]["de_misclassification_path"] = str(DE_MISCLASSIFICATION_PATH.relative_to(ROOT))

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "excluded_constant_columns": EXCLUDED_CONSTANT_COLUMNS,
            "excluded_leakage_columns": EXCLUDED_LEAKAGE_COLUMNS,
            "target": TARGET,
            "label_order": labels,
            "metrics": metrics,
            "validation_method": "randomized_search_cv_with_stratified_k_fold",
            "conda_env": "pronii",
        },
        MODEL_PATH,
    )

    plot_label_distribution(df, LABEL_PNG)
    plot_confusion_matrix(confusion_matrix(best_true, best_pred, labels=labels), labels, CM_PNG)
    feature_importance = save_feature_importance(best_pipeline, FI_CSV, FI_PNG)

    model_rows = []
    for name, values in candidate_metrics.items():
        if values.get("search_status") == "skipped_import_error":
            model_rows.append(
                {
                    "model": name,
                    "status": "skipped",
                    "searched": "",
                    "search_best_f1": "",
                    "cv_f1_mean": "",
                    "cv_f1_std": "",
                    "oof_f1": "",
                    "d_recall": "",
                    "e_recall": "",
                    "de_macro_recall": "",
                    "d_f2": "",
                    "e_f2": "",
                    "de_macro_f2": "",
                    "weighted_f1": "",
                    "weighted_f2": "",
                    "accuracy_mean": "",
                }
            )
            continue
        model_rows.append(
            {
                "model": name,
                "status": values["search_status"],
                "searched": values["searched_candidates"],
                "search_best_f1": round(values.get("best_search_score", values["f1_macro_mean"]), 4),
                "cv_f1_mean": round(values["f1_macro_mean"], 4),
                "cv_f1_std": round(values["f1_macro_std"], 4),
                "oof_f1": round(values["oof_f1_macro"], 4),
                "d_recall": round(values["oof_recall_d"], 4),
                "e_recall": round(values["oof_recall_e"], 4),
                "de_macro_recall": round(values["oof_de_macro_recall"], 4),
                "d_f2": round(values["oof_f2_d"], 4),
                "e_f2": round(values["oof_f2_e"], 4),
                "de_macro_f2": round(values["oof_de_macro_f2"], 4),
                "weighted_f1": round(values["oof_f1_weighted"], 4),
                "weighted_f2": round(values["oof_f2_weighted"], 4),
                "accuracy_mean": round(values["accuracy_mean"], 4),
            }
        )
    model_table = pd.DataFrame(model_rows)
    model_table["_sort"] = pd.to_numeric(model_table["cv_f1_mean"], errors="coerce").fillna(-1)
    model_table = model_table.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    param_rows = []
    for name, values in candidate_metrics.items():
        params = values.get("best_params") or {}
        if not params:
            continue
        for param, value in params.items():
            param_rows.append({"model": name, "parameter": param, "best_value": value})
    param_table = pd.DataFrame(param_rows)

    focus_models = [best_name]
    if "XGBoostClassifier" in candidate_metrics and candidate_metrics["XGBoostClassifier"].get("search_status") == "completed":
        if "XGBoostClassifier" not in focus_models:
            focus_models.append("XGBoostClassifier")

    fold_rows = []
    for model_name in focus_models:
        for fold in candidate_metrics[model_name]["folds"]:
            fold_rows.append({"model": model_name, **fold})
    fold_table = pd.DataFrame(fold_rows)
    fold_table[["fold", "train_rows", "validation_rows"]] = fold_table[
        ["fold", "train_rows", "validation_rows"]
    ].astype(int)
    fold_columns = [
        "model",
        "fold",
        "train_rows",
        "validation_rows",
        "accuracy",
        "f1_macro",
        "f2_macro",
        "f1_weighted",
        "f2_weighted",
        *[f"recall_{label.lower()}" for label in labels],
        *[f"f2_{label.lower()}" for label in labels],
        "de_macro_recall",
        "de_macro_f2",
        "de_group_recall",
    ]
    fold_table = fold_table[[column for column in fold_columns if column in fold_table.columns]]
    for column in fold_table.columns:
        if column not in {"model", "fold", "train_rows", "validation_rows"}:
            fold_table[column] = fold_table[column].round(4)

    class_metric_rows = []
    for model_name in focus_models:
        report = candidate_metrics[model_name]["classification_report"]
        f2_scores = candidate_metrics[model_name]["oof_f2_by_label"]
        for label in labels:
            label_report = report.get(label, {})
            class_metric_rows.append(
                {
                    "model": model_name,
                    "grade": label,
                    "precision": round(label_report.get("precision", 0.0), 4),
                    "recall": round(label_report.get("recall", 0.0), 4),
                    "f1": round(label_report.get("f1-score", 0.0), 4),
                    "f2": round(f2_scores.get(label, 0.0), 4),
                    "support": int(label_report.get("support", 0)),
                }
            )
    class_metric_table = pd.DataFrame(class_metric_rows)

    unavailable_table = pd.DataFrame(
        [
            {"model": name, "reason": values["skip_reason"]}
            for name, values in candidate_metrics.items()
            if values.get("search_status") == "skipped_import_error"
        ]
    )
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
    policy_metric_table = pd.DataFrame(
        [
            {
                "metric": "Macro F1",
                "value": round(metrics["best_model"]["oof_f1_macro"], 4),
                "interpretation": "A~E 각 등급을 동일한 비중으로 평가하는 메인 지표",
            },
            {
                "metric": "D recall",
                "value": round(metrics["best_model"]["oof_recall_d"], 4),
                "interpretation": "실제 D등급 공연을 D로 맞힌 비율",
            },
            {
                "metric": "E recall",
                "value": round(metrics["best_model"]["oof_recall_e"], 4),
                "interpretation": "실제 E등급 공연을 E로 맞힌 비율",
            },
            {
                "metric": "D/E macro recall",
                "value": round(metrics["best_model"]["oof_de_macro_recall"], 4),
                "interpretation": "D recall과 E recall의 평균",
            },
            {
                "metric": "D/E group recall",
                "value": round(metrics["best_model"]["oof_de_group_recall"], 4),
                "interpretation": "실제 D/E 공연을 취약 등급 범주 안으로 잡아낸 비율",
            },
            {
                "metric": "Macro F2",
                "value": round(metrics["best_model"]["oof_f2_macro"], 4),
                "interpretation": "recall을 precision보다 더 크게 반영한 A~E 균형 지표",
            },
            {
                "metric": "D F2",
                "value": round(metrics["best_model"]["oof_f2_d"], 4),
                "interpretation": "D등급 탐지에서 recall을 더 중시한 F-beta(beta=2) 점수",
            },
            {
                "metric": "E F2",
                "value": round(metrics["best_model"]["oof_f2_e"], 4),
                "interpretation": "E등급 탐지에서 recall을 더 중시한 F-beta(beta=2) 점수",
            },
            {
                "metric": "D/E macro F2",
                "value": round(metrics["best_model"]["oof_de_macro_f2"], 4),
                "interpretation": "D F2와 E F2의 평균",
            },
            {
                "metric": "Weighted F1",
                "value": round(metrics["best_model"]["oof_f1_weighted"], 4),
                "interpretation": "실제 등급 분포를 반영한 전체 성능",
            },
            {
                "metric": "D/E misclassified cases",
                "value": metrics["best_model"]["de_misclassification_count"],
                "interpretation": "실제 D/E 중 예측 등급이 달랐던 OOF 사례 수",
            },
        ]
    )
    policy_rank_rows = []
    for name, values in candidate_metrics.items():
        if name == "DummyClassifier" or values.get("search_status") != "completed":
            continue
        policy_rank_rows.append(
            {
                "model": name,
                "macro_f1": round(values["oof_f1_macro"], 4),
                "d_recall": round(values["oof_recall_d"], 4),
                "e_recall": round(values["oof_recall_e"], 4),
                "de_macro_recall": round(values["oof_de_macro_recall"], 4),
                "de_macro_f2": round(values["oof_de_macro_f2"], 4),
                "weighted_f1": round(values["oof_f1_weighted"], 4),
                "weighted_f2": round(values["oof_f2_weighted"], 4),
            }
        )
    policy_rank_table = (
        pd.DataFrame(policy_rank_rows)
        .sort_values(["de_macro_f2", "de_macro_recall", "macro_f1"], ascending=False)
        .reset_index(drop=True)
    )
    policy_screening_name = str(policy_rank_table.iloc[0]["model"]) if not policy_rank_table.empty else best_name
    policy_screening_de_f2 = (
        float(policy_rank_table.iloc[0]["de_macro_f2"]) if not policy_rank_table.empty else 0.0
    )
    de_failure_summary = (
        de_misclassified.groupby(["failure_type", "actual_grade", "predicted_grade"])
        .size()
        .reset_index(name="count")
        .sort_values(["failure_type", "actual_grade", "predicted_grade"])
    )
    if de_failure_summary.empty:
        de_failure_summary = pd.DataFrame(
            [{"failure_type": "없음", "actual_grade": "", "predicted_grade": "", "count": 0}]
        )
    de_case_preview = de_misclassified[
        ["failure_type", "actual_grade", "predicted_grade", "title", "start_date", "genre_group", "venue"]
    ].head(12)
    if de_case_preview.empty:
        de_case_preview = pd.DataFrame(
            [
                {
                    "failure_type": "없음",
                    "actual_grade": "",
                    "predicted_grade": "",
                    "title": "",
                    "start_date": "",
                    "genre_group": "",
                    "venue": "",
                }
            ]
        )

    step4 = f"""# Step 4. 접근성 등급 분류 모델 v4: 모델별 하이퍼파라미터 튜닝

## 문제 정의

- 문제 유형: 다중 분류
- 타겟 변수: `{TARGET}`
- 등급: A~E
- 주 평가 지표: macro F1
- 보조 지표: D recall, E recall, D/E macro recall, F2 score, weighted F1
- 정책 관점: 신규 공연의 장애인석 상대 수요 취약 등급(D/E)을 사전에 놓치지 않는 것이 중요

## v4 변경점

v3.2까지는 모델별 기본 설정을 고정해 K-Fold 비교를 수행했다. v4에서는 RandomForest, ExtraTrees, XGBoost, LightGBM 후보별로 하이퍼파라미터 탐색 공간을 별도로 정의하고 `RandomizedSearchCV`로 각 모델의 최적 조합을 찾았다.

## 학습 Feature

{markdown_table(pd.DataFrame({"feature": FEATURES}), index=False)}

## 학습 제외 컬럼

아래 컬럼은 상수값이거나 좌석 수 또는 타겟 산출에 직접 연결되는 값이므로 모델 feature에서 제외했다.

{markdown_table(excluded, index=False)}

## 등급 분포

{markdown_table(label_counts)}

## 튜닝 방식

- 실행 환경: conda env `pronii`
- 탐색 방식: `RandomizedSearchCV`
- 검증 방식: stratified 5-fold cross validation
- n_splits: {N_SPLITS}
- 모델별 탐색 후보 수: {N_ITER}
- scoring: macro F1
- shuffle: true
- random_state: {RANDOM_STATE}
- fold별 학습 행 수: {metrics["split"]["fold_train_rows"]:,}
- fold별 검증 행 수: {metrics["split"]["fold_validation_rows"]:,}

## 튜닝 후 모델 비교

{markdown_table(model_table, index=False)}

## 정책 중요 지표 분석

이 프로젝트는 신규 공연의 장애인석 상대 수요 등급을 A~E로 사전 예측하는 문제다. 데이터 분포가 균형적이지 않고 D/E 같은 취약 등급을 놓치면 운영 대응 기회를 잃을 수 있으므로, 최종 판단은 accuracy보다 `Macro F1`과 D/E recall 계열 지표를 중심으로 본다.

{markdown_table(policy_metric_table, index=False)}

`D/E macro recall`은 D와 E를 각각 정확히 맞히는 능력을 요약한다. `D/E group recall`은 실제 D/E 공연이 D 또는 E 범주 안에라도 들어왔는지 보는 지표라서, 취약 공연을 우선 검토 대상으로 올리는 정책 목적에 함께 참고할 수 있다. `F2 score`는 F-beta에서 beta=2를 적용한 값으로, precision보다 recall을 더 강하게 반영한다. 따라서 취약 등급을 놓치는 비용이 큰 이 프로젝트에서는 F1보다 운영 리스크에 민감한 보조 지표로 해석한다.

### D/E 탐지 관점 모델 순위

{markdown_table(policy_rank_table, index=False)}

Macro F1 기준 최종 선택 모델은 `{best_name}`이지만, D/E 탐지력을 우선하는 운영 스크리닝 관점에서는 `{policy_screening_name}`의 D/E macro F2가 `{policy_screening_de_f2:.4f}`로 가장 높다. 따라서 리포트의 기본 최종 모델은 균형 성능 기준으로 유지하되, 취약 등급 누락 비용이 더 크다면 `{policy_screening_name}`을 보조 후보로 함께 검토하는 것이 타당하다.

## 모델별 최적 하이퍼파라미터

{markdown_table(param_table, index=False)}
"""
    if not unavailable_table.empty:
        step4 += f"""
## 미실행 모델

아래 모델은 현재 환경에서 import 단계 오류가 발생해 튜닝 대상에서 제외했다.

{markdown_table(unavailable_table, index=False)}
"""

    step4 += f"""
## 최종 선택 모델

- 모델: `{best_name}`
- 선택 기준: 튜닝 완료 후보 중 fold별 macro F1 평균이 가장 높은 모델
- macro F1 평균: `{metrics["best_model"]["f1_macro_mean"]:.4f}`
- macro F1 표준편차: `{metrics["best_model"]["f1_macro_std"]:.4f}`
- OOF macro F1: `{metrics["best_model"]["oof_f1_macro"]:.4f}`
- OOF weighted F1: `{metrics["best_model"]["oof_f1_weighted"]:.4f}`
- OOF macro F2: `{metrics["best_model"]["oof_f2_macro"]:.4f}`
- OOF weighted F2: `{metrics["best_model"]["oof_f2_weighted"]:.4f}`
- OOF D recall: `{metrics["best_model"]["oof_recall_d"]:.4f}`
- OOF E recall: `{metrics["best_model"]["oof_recall_e"]:.4f}`
- OOF D/E macro recall: `{metrics["best_model"]["oof_de_macro_recall"]:.4f}`
- OOF D/E group recall: `{metrics["best_model"]["oof_de_group_recall"]:.4f}`
- OOF D F2: `{metrics["best_model"]["oof_f2_d"]:.4f}`
- OOF E F2: `{metrics["best_model"]["oof_f2_e"]:.4f}`
- OOF D/E macro F2: `{metrics["best_model"]["oof_de_macro_f2"]:.4f}`
- 저장 경로: `{MODEL_PATH.relative_to(ROOT)}`
- 메트릭 저장 경로: `{METRICS_PATH.relative_to(ROOT)}`
"""

    lightgbm_metrics = candidate_metrics.get("LightGBMClassifier")
    if lightgbm_metrics and lightgbm_metrics.get("search_status") == "completed":
        lightgbm_summary = pd.DataFrame(
            [
                {"item": "status", "value": lightgbm_metrics["search_status"]},
                {"item": "searched_candidates", "value": lightgbm_metrics["searched_candidates"]},
                {"item": "cv_f1_mean", "value": round(lightgbm_metrics["f1_macro_mean"], 4)},
                {"item": "cv_f1_std", "value": round(lightgbm_metrics["f1_macro_std"], 4)},
                {"item": "oof_f1", "value": round(lightgbm_metrics["oof_f1_macro"], 4)},
                {"item": "accuracy_mean", "value": round(lightgbm_metrics["accuracy_mean"], 4)},
            ]
        )
        lightgbm_params = pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in lightgbm_metrics["best_params"].items()]
        )
        output_rows = []
        if LIGHTGBM_MODEL_PATH.exists():
            output_rows.append(
                {"artifact": "LightGBM 전용 모델", "path": str(LIGHTGBM_MODEL_PATH.relative_to(ROOT))}
            )
        if LIGHTGBM_INFERENCE_PATH.exists():
            output_rows.append(
                {"artifact": "LightGBM 인퍼런스 결과 CSV", "path": str(LIGHTGBM_INFERENCE_PATH.relative_to(ROOT))}
            )
        lightgbm_outputs = (
            markdown_table(pd.DataFrame(output_rows), index=False)
            if output_rows
            else "- LightGBM 전용 인퍼런스 산출물은 별도 실행 후 저장한다."
        )

        step4 += f"""
## LightGBMClassifier 해결 및 인퍼런스 확인

기존 v4 리포트에서는 `LightGBMClassifier`가 `pandas.core.strings.StringMethods` 관련 import 오류로 제외되었지만, `pronii` 환경에서 재실행한 결과 LightGBM 튜닝이 정상 완료되었다.

{markdown_table(lightgbm_summary, index=False)}

### LightGBM 최적 하이퍼파라미터

{markdown_table(lightgbm_params, index=False)}

### LightGBM 인퍼런스 산출물

{lightgbm_outputs}
"""

    step4 += f"""
## 최종 모델 및 XGBoost Fold별 결과

아래 표는 균형 성능 기준 최종 모델인 `{best_name}`과 D/E 탐지 보조 후보인 `XGBoostClassifier`의 fold별 결과를 함께 비교한 것이다. 각 fold에서 A~E 등급별 recall과 F2를 같이 제공해 특정 등급에서 성능이 흔들리는지 확인할 수 있다.

{markdown_table(fold_table, index=False)}

## A~E 등급별 OOF 세부 지표

아래 표는 전체 OOF 예측 기준으로 A~E 각 등급의 precision, recall, F1, F2, support를 정리한 것이다. `F2`는 recall을 더 중시하므로, 실제 취약 등급을 놓치는 비용이 큰 운영 판단에서 함께 확인한다.

{markdown_table(class_metric_table, index=False)}

## D/E 오분류 사례 분석

아래 표는 최종 모델의 OOF 예측 기준으로 실제 등급이 D/E였지만 예측 등급이 달랐던 사례를 요약한 것이다. `취약 등급 누락`은 실제 D/E가 A/B/C로 예측된 경우이고, `D/E 내부 혼동`은 D와 E 사이에서 서로 바뀐 경우다.

{markdown_table(de_failure_summary, index=False)}

### D/E 오분류 사례 일부

전체 사례는 `{DE_MISCLASSIFICATION_PATH.relative_to(ROOT)}`에 저장했다.

{markdown_table(de_case_preview, index=False)}

## 상위 Feature Importance

{markdown_table(feature_importance[["rank", "feature", "importance"]].head(10), index=False)}

## 해석상 주의

이 리포트는 작은 데이터셋에서 모델별 튜닝 효과를 빠르게 비교하기 위한 버전이다. 하이퍼파라미터 선택과 성능 추정에 같은 5-fold 구조를 사용하므로, 엄밀한 일반화 성능 추정이 필요하면 추후 nested cross validation으로 한 번 더 검증하는 것이 좋다.

## 산출물

- 혼동행렬: `reports/accessibility/{CM_PNG.name}`
- Feature importance CSV: `reports/accessibility/{FI_CSV.name}`
- Feature importance 이미지: `reports/accessibility/{FI_PNG.name}`
- D/E 오분류 사례 CSV: `{DE_MISCLASSIFICATION_PATH.relative_to(ROOT)}`
"""

    md_files = [
        REPORT_DIR / "step_1_dataset.md",
        REPORT_DIR / "step_2_labeling.md",
        REPORT_DIR / "step_3_features.md",
    ]
    sections = [report_section(path.read_text(encoding="utf-8")) for path in md_files if path.exists()]
    sections.append(report_section(step4))
    body = "\n".join(sections)

    best = metrics["best_model"]
    generated_at = datetime.fromisoformat(metrics["created_at"]).strftime("%Y-%m-%d %H:%M:%S %Z")
    best_params = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in best["best_params"].items()]
    )
    best_params_html = markdown_to_html(
        "# 최종 모델 하이퍼파라미터\n\n" + markdown_table(best_params, index=False)
    )

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>장애인 접근성 등급 진단 모델 구현 결과 v4 - 하이퍼파라미터 튜닝</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2733;
      --muted: #657386;
      --line: #d9e1ea;
      --soft: #f6f8fb;
      --panel: #ffffff;
      --accent: #2f6f9f;
      --accent-2: #7a6b2f;
      --code: #eef2f6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #eef2f6;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.62;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 28px 56px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 5px solid var(--accent);
      padding: 30px 32px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h1, h2, h3, h4 {{
      color: #17324d;
      letter-spacing: 0;
      line-height: 1.28;
    }}
    h1 {{ margin: 0 0 12px; font-size: 30px; }}
    h2 {{ margin: 0 0 18px; font-size: 23px; }}
    h3 {{ margin: 26px 0 10px; font-size: 18px; }}
    h4 {{ margin: 20px 0 8px; font-size: 15px; color: var(--muted); }}
    p {{ margin: 8px 0; }}
    .summary-text {{
      max-width: 900px;
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--soft);
      padding: 14px 16px;
      min-height: 86px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .metric strong {{
      display: block;
      margin-top: 7px;
      font-size: 21px;
      color: var(--ink);
      word-break: break-word;
    }}
    .note {{
      background: #fbfaf3;
      border: 1px solid #e4dcc2;
      border-left: 5px solid var(--accent-2);
      padding: 16px 18px;
      margin: 0 0 18px;
    }}
    .report-section, .best-params, .visuals {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 26px 30px;
      margin: 18px 0;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 14px 0 20px;
      border: 1px solid var(--line);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 560px;
      font-size: 14px;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 11px;
      text-align: left;
      vertical-align: top;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{
      background: #f1f5f9;
      color: #2d3d50;
      font-weight: 700;
      white-space: nowrap;
    }}
    code {{
      background: var(--code);
      padding: 2px 5px;
      border-radius: 4px;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.92em;
    }}
    ul {{ padding-left: 20px; margin: 8px 0 16px; }}
    li + li {{ margin-top: 5px; }}
    .visual-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #fff;
      padding: 12px;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    @media (max-width: 860px) {{
      .page {{ padding: 22px 14px 36px; }}
      .hero, .report-section, .best-params, .visuals {{ padding: 20px; }}
      .metric-grid, .visual-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
      h2 {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <p class="eyebrow">Accessibility Model Report v4 Hyperparamater</p>
      <h1>장애인 접근성 등급 진단 모델 구현 결과</h1>
      <p class="summary-text">모델별 하이퍼파라미터 탐색 공간을 정의하고, stratified 5-fold 기반 macro F1로 최적 조합을 비교했다.</p>
      <div class="metric-grid">
        <div class="metric"><span>최종 모델</span><strong>{html.escape(best_name)}</strong></div>
        <div class="metric"><span>OOF Macro F1</span><strong>{best["oof_f1_macro"]:.4f}</strong></div>
        <div class="metric"><span>D/E Macro Recall</span><strong>{best["oof_de_macro_recall"]:.4f}</strong></div>
        <div class="metric"><span>D/E Macro F2</span><strong>{best["oof_de_macro_f2"]:.4f}</strong></div>
      </div>
    </header>
    <section class="note">
      <strong>핵심 변경점</strong>
      <p>v4에서는 기본 파라미터 비교를 넘어 각 모델 후보별로 별도 탐색 공간을 두고 최적 하이퍼파라미터를 선택했다.</p>
      <p>생성 시각: {html.escape(generated_at)} / 실행 환경: <code>conda env pronii</code></p>
    </section>
    <section class="best-params">
      {best_params_html}
    </section>
    {body}
    <section class="visuals">
      <h2>시각화 산출물</h2>
      <div class="visual-grid">
        <figure>
          <figcaption>등급 분포</figcaption>
          <img src="{LABEL_PNG.name}" alt="Accessibility label distribution">
        </figure>
        <figure>
          <figcaption>튜닝 최종 모델 OOF 혼동행렬</figcaption>
          <img src="{CM_PNG.name}" alt="Accessibility tuned model confusion matrix">
        </figure>
        <figure>
          <figcaption>Feature Importance</figcaption>
          <img src="{FI_PNG.name}" alt="Accessibility tuned model feature importance">
        </figure>
      </div>
    </section>
  </main>
</body>
</html>
"""
    FINAL_HTML_PATH.write_text(html_text, encoding="utf-8")

    print(json.dumps(metrics["best_model"], ensure_ascii=False, indent=2))
    print(f"HTML report written to {FINAL_HTML_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
