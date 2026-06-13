from __future__ import annotations

import html
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train import (  # noqa: E402
    CATEGORICAL_FEATURES,
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURES,
    LABEL_ORDER,
    NUMERIC_FEATURES,
    TARGET,
    build_candidates,
    markdown_table,
    markdown_to_html,
    plot_confusion_matrix,
    plot_label_distribution,
    save_feature_importance,
)

DATASET_PATH = ROOT / "data/processed/accessibility_dataset.csv"
REPORT_DIR = ROOT / "reports/accessibility"
ARTIFACT_DIR = ROOT / "artifacts"

METRICS_PATH = REPORT_DIR / "accessibility_metrics_kfold_v31.json"
MODEL_PATH = ARTIFACT_DIR / "accessibility_classifier_kfold_v31.joblib"
FINAL_HTML_PATH = REPORT_DIR / "accessibility_project_summary_v3_1_kfold.html"
LABEL_PNG = REPORT_DIR / "accessibility_label_distribution_kfold_v31.png"
CM_PNG = REPORT_DIR / "accessibility_confusion_matrix_kfold_v31.png"
FI_CSV = REPORT_DIR / "accessibility_feature_importance_kfold_v31.csv"
FI_PNG = REPORT_DIR / "accessibility_feature_importance_kfold_v31.png"

N_SPLITS = 5
RANDOM_STATE = 42


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    x = df[FEATURES]
    y = df[TARGET]
    labels = [label for label in LABEL_ORDER if label in set(y)]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    candidate_builders = {"DummyClassifier": lambda: DummyClassifier(strategy="most_frequent")}
    for name, pipeline in build_candidates().items():
        candidate_builders[name] = lambda p=pipeline: clone(p)

    candidate_metrics: dict[str, dict[str, object]] = {}
    candidate_predictions: dict[str, tuple[list[str], list[str]]] = {}

    for name, builder in candidate_builders.items():
        folds = []
        all_true: list[str] = []
        all_pred: list[str] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
            model = builder()
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            model.fit(x_train, y_train)
            pred = model.predict(x_valid)
            folds.append(
                {
                    "fold": fold_idx,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                    "accuracy": float(accuracy_score(y_valid, pred)),
                    "f1_macro": float(
                        f1_score(y_valid, pred, labels=labels, average="macro", zero_division=0)
                    ),
                }
            )
            all_true.extend(y_valid.tolist())
            all_pred.extend(list(pred))

        fold_df = pd.DataFrame(folds)
        candidate_metrics[name] = {
            "model": name if name != "DummyClassifier" else "DummyClassifier(strategy='most_frequent')",
            "accuracy_mean": float(fold_df["accuracy"].mean()),
            "accuracy_std": float(fold_df["accuracy"].std(ddof=1)),
            "f1_macro_mean": float(fold_df["f1_macro"].mean()),
            "f1_macro_std": float(fold_df["f1_macro"].std(ddof=1)),
            "oof_accuracy": float(accuracy_score(all_true, all_pred)),
            "oof_f1_macro": float(
                f1_score(all_true, all_pred, labels=labels, average="macro", zero_division=0)
            ),
            "folds": folds,
            "classification_report": classification_report(
                all_true,
                all_pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }
        candidate_predictions[name] = (all_true, all_pred)

    non_dummy = [name for name in candidate_metrics if name != "DummyClassifier"]
    best_name = max(non_dummy, key=lambda name: candidate_metrics[name]["f1_macro_mean"])
    best_pipeline = candidate_builders[best_name]()
    best_pipeline.fit(x, y)
    best_true, best_pred = candidate_predictions[best_name]

    metrics = {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "conda_env": "pronii",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "rows": int(len(df)),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "excluded_leakage_columns": EXCLUDED_LEAKAGE_COLUMNS,
        "target": TARGET,
        "label_order": labels,
        "label_distribution": df[TARGET].value_counts().reindex(labels, fill_value=0).astype(int).to_dict(),
        "split": {
            "method": "stratified_k_fold_cross_validation",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
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
            "oof_accuracy": candidate_metrics[best_name]["oof_accuracy"],
            "oof_f1_macro": candidate_metrics[best_name]["oof_f1_macro"],
        },
    }

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "target": TARGET,
            "label_order": labels,
            "metrics": metrics,
            "validation_method": "stratified_k_fold_cross_validation",
            "conda_env": "pronii",
        },
        MODEL_PATH,
    )

    plot_label_distribution(df, LABEL_PNG)
    plot_confusion_matrix(confusion_matrix(best_true, best_pred, labels=labels), labels, CM_PNG)
    feature_importance = save_feature_importance(best_pipeline, FI_CSV, FI_PNG)

    model_rows = []
    for name, values in candidate_metrics.items():
        model_rows.append(
            {
                "model": name,
                "accuracy_mean": round(values["accuracy_mean"], 4),
                "accuracy_std": round(values["accuracy_std"], 4),
                "f1_macro_mean": round(values["f1_macro_mean"], 4),
                "f1_macro_std": round(values["f1_macro_std"], 4),
                "oof_f1_macro": round(values["oof_f1_macro"], 4),
            }
        )
    model_table = pd.DataFrame(model_rows).sort_values("f1_macro_mean", ascending=False)

    fold_table = pd.DataFrame(candidate_metrics[best_name]["folds"]).copy()
    fold_table["accuracy"] = fold_table["accuracy"].round(4)
    fold_table["f1_macro"] = fold_table["f1_macro"].round(4)

    venue_counts = Counter(df["venue_type"])
    venue_map = defaultdict(Counter)
    for _, row in df.iterrows():
        venue_map[row["venue"]][row["venue_type"]] += 1
    scale_rows = []
    for scale in ["대형", "중형", "소극장", "소형", "기타"]:
        venues = []
        total = 0
        for venue, counts in sorted(venue_map.items()):
            if counts.most_common(1)[0][0] == scale:
                count = sum(counts.values())
                venues.append(f"{venue} ({count:,}건)")
                total += count
        if venues:
            scale_rows.append(
                {
                    "규모 라벨": "소극장/소형" if scale == "소극장" else scale,
                    "공연장": ", ".join(venues),
                    "행 수": total,
                }
            )
    scale_table = pd.DataFrame(scale_rows)

    excluded = pd.DataFrame({"excluded_column": EXCLUDED_LEAKAGE_COLUMNS})
    label_counts = pd.Series(metrics["label_distribution"], name="count").to_frame()
    step4 = f"""# Step 4. 접근성 등급 분류 모델 v3.1: K-Fold 검증

## 문제 정의

- 문제 유형: 다중 분류
- 타겟 변수: `{TARGET}`
- 등급: A~E
- 주 평가 지표: macro F1

## K-Fold 적용 이유

전체 데이터가 500행으로 많지 않기 때문에 단일 train/test split만 사용하면 특정 검증 세트 구성에 결과가 흔들릴 수 있다. v3.1에서는 `pronii` conda 환경에서 `StratifiedKFold(n_splits=5)`를 적용했고, XGBoost와 LightGBM 후보까지 포함했다.

## 학습 Feature

{markdown_table(pd.DataFrame({"feature": FEATURES}), index=False)}

## 학습 제외 컬럼

아래 컬럼은 좌석 수 또는 타겟 산출에 직접 연결되는 값이므로 모델 feature에서 제외했다.

{markdown_table(excluded, index=False)}

## 등급 분포

{markdown_table(label_counts)}

## 검증 방식

- 실행 환경: conda env `pronii`
- 방식: stratified 5-fold cross validation
- n_splits: {N_SPLITS}
- shuffle: true
- random_state: {RANDOM_STATE}
- fold별 학습 행 수: 400
- fold별 검증 행 수: 100

## 모델 비교

{markdown_table(model_table, index=False)}

## 최종 선택 모델

- 모델: `{best_name}`
- 선택 기준: fold별 macro F1 평균이 가장 높은 후보
- macro F1 평균: `{metrics["best_model"]["f1_macro_mean"]:.4f}`
- macro F1 표준편차: `{metrics["best_model"]["f1_macro_std"]:.4f}`
- OOF macro F1: `{metrics["best_model"]["oof_f1_macro"]:.4f}`
- 저장 경로: `{MODEL_PATH.relative_to(ROOT)}`
- 메트릭 저장 경로: `{METRICS_PATH.relative_to(ROOT)}`

## 최종 모델 Fold별 결과

{markdown_table(fold_table, index=False)}

## 상위 Feature Importance

{markdown_table(feature_importance[["rank", "feature", "importance"]].head(10), index=False)}

## 산출물

- 혼동행렬: `reports/accessibility/{CM_PNG.name}`
- Feature importance CSV: `reports/accessibility/{FI_CSV.name}`
- Feature importance 이미지: `reports/accessibility/{FI_PNG.name}`
"""

    md_files = [
        REPORT_DIR / "step_1_dataset.md",
        REPORT_DIR / "step_2_labeling.md",
        REPORT_DIR / "step_3_features.md",
    ]
    sections = [markdown_to_html(path.read_text(encoding="utf-8")) for path in md_files if path.exists()]
    sections.append(markdown_to_html(step4))
    body = "\n<hr>\n".join(sections)
    venue_table_html = markdown_to_html(
        "# v3.1 핵심 기준. 공연장 규모 분류\n\n"
        "`venue_type`은 공연장명을 기준으로 확정한 규모 라벨이다. K-fold 모델에서도 이 값을 범주형 feature로 사용한다.\n\n"
        + markdown_table(scale_table, index=False)
    )

    best = metrics["best_model"]
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>장애인 접근성 등급 진단 모델 구현 결과 v3.1 - K-Fold</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.6; color: #222; }}
    h1, h2, h3 {{ color: #17324d; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    code {{ background: #f3f5f7; padding: 2px 4px; border-radius: 4px; }}
    .summary {{ background: #f7fafc; border-left: 4px solid #805ad5; padding: 16px 20px; margin-bottom: 28px; }}
    .scale-definition {{ background: #fbfcfd; border: 1px solid #d7dde5; padding: 18px 20px; margin: 0 0 28px; }}
    img {{ max-width: 760px; width: 100%; border: 1px solid #d7dde5; margin: 12px 0 24px; }}
  </style>
</head>
<body>
  <div class="summary">
    <h1>장애인 접근성 등급 진단 모델 구현 결과 v3.1</h1>
    <p>생성 시각: {html.escape(metrics["created_at"])}</p>
    <p>실행 환경: <strong>conda env pronii</strong></p>
    <p>검증 방식: <strong>Stratified 5-Fold Cross Validation</strong></p>
    <p>최종 모델: <strong>{html.escape(best_name)}</strong> / macro F1 평균: <strong>{best["f1_macro_mean"]:.4f}</strong> / 표준편차: <strong>{best["f1_macro_std"]:.4f}</strong> / OOF macro F1: <strong>{best["oof_f1_macro"]:.4f}</strong></p>
    <p>v3.1 핵심 변경점: `pronii` 환경에서 XGBoost와 LightGBM까지 포함해 5-fold 검증을 다시 수행했다.</p>
  </div>
  <section class="scale-definition">
    {venue_table_html}
  </section>
  {body}
  <h2>시각화 산출물</h2>
  <h3>등급 분포</h3>
  <img src="{LABEL_PNG.name}" alt="Accessibility label distribution">
  <h3>K-Fold OOF 혼동행렬</h3>
  <img src="{CM_PNG.name}" alt="Accessibility k-fold confusion matrix">
  <h3>Feature Importance</h3>
  <img src="{FI_PNG.name}" alt="Accessibility k-fold feature importance">
</body>
</html>
"""
    FINAL_HTML_PATH.write_text(html_text, encoding="utf-8")

    print(f"final_html={FINAL_HTML_PATH}")
    print(f"metrics={METRICS_PATH}")
    print(f"model={MODEL_PATH}")
    print(f"best_model={best_name}")
    print(f"best_f1_mean={best['f1_macro_mean']:.4f}")
    print(f"best_f1_std={best['f1_macro_std']:.4f}")
    print(f"best_oof_f1={best['oof_f1_macro']:.4f}")
    print(model_table.to_string(index=False))


if __name__ == "__main__":
    main()
