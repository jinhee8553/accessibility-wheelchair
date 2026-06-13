from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from numbers import Number
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
    EXCLUDED_CONSTANT_COLUMNS,
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURES,
    LABEL_ORDER,
    NUMERIC_FEATURES,
    TARGET,
    build_candidates,
    plot_confusion_matrix,
    plot_label_distribution,
    save_feature_importance,
)

DATASET_PATH = ROOT / "data/processed/accessibility_dataset.csv"
REPORT_DIR = ROOT / "reports/accessibility"
ARTIFACT_DIR = ROOT / "artifacts"

METRICS_PATH = REPORT_DIR / "accessibility_metrics_kfold_v32.json"
MODEL_PATH = ARTIFACT_DIR / "accessibility_classifier_kfold_v32.joblib"
FINAL_HTML_PATH = REPORT_DIR / "accessibility_project_summary_v3_2_relative_rate.html"
LABEL_PNG = REPORT_DIR / "accessibility_label_distribution_kfold_v32.png"
CM_PNG = REPORT_DIR / "accessibility_confusion_matrix_kfold_v32.png"
FI_CSV = REPORT_DIR / "accessibility_feature_importance_kfold_v32.csv"
FI_PNG = REPORT_DIR / "accessibility_feature_importance_kfold_v32.png"

N_SPLITS = 5
RANDOM_STATE = 42


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
        "excluded_constant_columns": EXCLUDED_CONSTANT_COLUMNS,
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
            "excluded_constant_columns": EXCLUDED_CONSTANT_COLUMNS,
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
    fold_table[["fold", "train_rows", "validation_rows"]] = fold_table[
        ["fold", "train_rows", "validation_rows"]
    ].astype(int)
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
    label_counts = pd.Series(metrics["label_distribution"], name="count").to_frame()
    step4 = f"""# Step 4. 접근성 등급 분류 모델 v3.2: 상대 예매율 + K-Fold 검증

## 문제 정의

- 문제 유형: 다중 분류
- 타겟 변수: `{TARGET}`
- 등급: A~E
- 주 평가 지표: macro F1

## K-Fold 적용 이유

전체 데이터가 {len(df):,}행으로 많지 않기 때문에 단일 train/test split만 사용하면 특정 검증 세트 구성에 결과가 흔들릴 수 있다. v3.2에서는 `pronii` conda 환경에서 `StratifiedKFold(n_splits=5)`를 적용했고, 사용 가능한 경우 XGBoost와 LightGBM 후보까지 포함한다.

## 학습 Feature

{markdown_table(pd.DataFrame({"feature": FEATURES}), index=False)}

## 학습 제외 컬럼

아래 컬럼은 상수값이거나 좌석 수 또는 타겟 산출에 직접 연결되는 값이므로 모델 feature에서 제외했다.

{markdown_table(excluded, index=False)}

## 등급 분포

{markdown_table(label_counts)}

## 검증 방식

- 실행 환경: conda env `pronii`
- 방식: stratified 5-fold cross validation
- n_splits: {N_SPLITS}
- shuffle: true
- random_state: {RANDOM_STATE}
- fold별 학습 행 수: {metrics["split"]["fold_train_rows"]:,}
- fold별 검증 행 수: {metrics["split"]["fold_validation_rows"]:,}

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
    sections = [report_section(path.read_text(encoding="utf-8")) for path in md_files if path.exists()]
    sections.append(report_section(step4))
    body = "\n".join(sections)
    venue_table_html = markdown_to_html(
        "# v3.2 핵심 기준. 공연장 규모 분류\n\n"
        "`venue_type`은 공연장명을 기준으로 확정한 규모 라벨이다. K-fold 모델에서도 이 값을 범주형 feature로 사용한다.\n\n"
        + markdown_table(scale_table, index=False)
    )

    best = metrics["best_model"]
    generated_at = datetime.fromisoformat(metrics["created_at"]).strftime("%Y-%m-%d %H:%M:%S %Z")
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>장애인 접근성 등급 진단 모델 구현 결과 v3.2 - 상대 예매율 K-Fold</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2733;
      --muted: #657386;
      --line: #d9e1ea;
      --soft: #f6f8fb;
      --panel: #ffffff;
      --accent: #2f6f9f;
      --accent-2: #6a7d38;
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
      background: #f8faf4;
      border: 1px solid #dfe8c7;
      border-left: 5px solid var(--accent-2);
      padding: 16px 18px;
      margin: 0 0 18px;
    }}
    .report-section, .scale-definition, .visuals {{
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
      .hero, .report-section, .scale-definition, .visuals {{ padding: 20px; }}
      .metric-grid, .visual-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
      h2 {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <p class="eyebrow">Accessibility Model Report v3.2</p>
      <h1>장애인 접근성 등급 진단 모델 구현 결과</h1>
      <p class="summary-text">전체 예매율 대비 휠체어석 예매율로 타겟 라벨을 보정하고, stratified 5-fold cross validation으로 후보 모델을 비교했다.</p>
      <div class="metric-grid">
        <div class="metric"><span>최종 모델</span><strong>{html.escape(best_name)}</strong></div>
        <div class="metric"><span>Macro F1 평균</span><strong>{best["f1_macro_mean"]:.4f}</strong></div>
        <div class="metric"><span>OOF Macro F1</span><strong>{best["oof_f1_macro"]:.4f}</strong></div>
        <div class="metric"><span>데이터 행 수</span><strong>{len(df):,}</strong></div>
      </div>
    </header>
    <section class="note">
      <strong>핵심 변경점</strong>
      <p>타겟 라벨 기준을 <code>wheelchair_booking_count / wheelchair_seats</code>에서 <code>(wheelchair_booking_count / wheelchair_seats) / (paid_audience_count / total_seats)</code>로 보정했다.</p>
      <p>생성 시각: {html.escape(generated_at)} / 실행 환경: <code>conda env pronii</code></p>
    </section>
    <section class="scale-definition">
      {venue_table_html}
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
          <figcaption>K-Fold OOF 혼동행렬</figcaption>
          <img src="{CM_PNG.name}" alt="Accessibility k-fold confusion matrix">
        </figure>
        <figure>
          <figcaption>Feature Importance</figcaption>
          <img src="{FI_PNG.name}" alt="Accessibility k-fold feature importance">
        </figure>
      </div>
    </section>
  </main>
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
