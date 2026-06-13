from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_accessibility_kfold_v4_hyperparamater import (  # noqa: E402
    DATASET_PATH,
    FEATURES,
    LABEL_ORDER,
    METRICS_PATH as V4_METRICS_PATH,
    N_SPLITS,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
    build_tuning_spaces,
    evaluate_cv_oof,
    markdown_table,
    markdown_to_html,
)

OUTPUT_HTML_PATH = REPORT_DIR / "accessibility_project_summary_v4_2_grade.html"
OUTPUT_METRICS_PATH = REPORT_DIR / "accessibility_metrics_v4_2_grade.json"
OOF_PREDICTIONS_PATH = REPORT_DIR / "accessibility_ordinal_oof_predictions_v4_2_grade.csv"
SEVERE_ERRORS_PATH = REPORT_DIR / "accessibility_ordinal_severe_errors_v4_2_grade.csv"

GRADE_TO_SCORE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
SCORE_TO_GRADE = {value: key for key, value in GRADE_TO_SCORE.items()}


def set_best_params(pipeline, params: dict[str, object]):
    model = pipeline.named_steps["model"]
    if model.__class__.__name__ == "LabelEncodedClassifier":
        prefixed = {f"model__estimator__{key}": value for key, value in params.items()}
    else:
        prefixed = {f"model__{key}": value for key, value in params.items()}
    return pipeline.set_params(**prefixed)


def grade_distance_frame(model_name: str, true_labels: list[str], pred_labels: list[str]) -> pd.DataFrame:
    rows = []
    for actual, predicted in zip(true_labels, pred_labels):
        actual_score = GRADE_TO_SCORE[actual]
        predicted_score = GRADE_TO_SCORE[predicted]
        signed_error = predicted_score - actual_score
        rows.append(
            {
                "model": model_name,
                "actual_grade": actual,
                "predicted_grade": predicted,
                "actual_score": actual_score,
                "predicted_score": predicted_score,
                "signed_error": signed_error,
                "abs_error": abs(signed_error),
                "is_adjacent_or_exact": int(abs(signed_error) <= 1),
                "is_exact": int(signed_error == 0),
                "is_severe_error": int(actual in {"D", "E"} and predicted in {"A", "B"}),
            }
        )
    return pd.DataFrame(rows)


def ordinal_metrics(frame: pd.DataFrame) -> dict[str, object]:
    vulnerable = frame["actual_grade"].isin(["D", "E"])
    severe_count = int(frame["is_severe_error"].sum())
    vulnerable_count = int(vulnerable.sum())
    return {
        "grade_mae": float(frame["abs_error"].mean()),
        "adjacent_accuracy": float(frame["is_adjacent_or_exact"].mean()),
        "exact_accuracy": float(frame["is_exact"].mean()),
        "mean_signed_error": float(frame["signed_error"].mean()),
        "severe_error_count": severe_count,
        "vulnerable_support": vulnerable_count,
        "severe_error_rate": float(severe_count / vulnerable_count) if vulnerable_count else 0.0,
        "distance_distribution": frame["abs_error"].value_counts().sort_index().astype(int).to_dict(),
    }


def format_metric_table(metrics_by_model: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model_name, values in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "grade_mae": round(values["grade_mae"], 4),
                "adjacent_accuracy": round(values["adjacent_accuracy"], 4),
                "exact_accuracy": round(values["exact_accuracy"], 4),
                "severe_error_rate": round(values["severe_error_rate"], 4),
                "severe_error_count": values["severe_error_count"],
                "vulnerable_support": values["vulnerable_support"],
                "mean_signed_error": round(values["mean_signed_error"], 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["severe_error_rate", "grade_mae", "adjacent_accuracy"], ascending=[True, True, False])


def distance_distribution_table(metrics_by_model: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model_name, values in metrics_by_model.items():
        distribution = values["distance_distribution"]
        rows.append(
            {
                "model": model_name,
                "distance_0_exact": distribution.get(0, 0),
                "distance_1_adjacent": distribution.get(1, 0),
                "distance_2": distribution.get(2, 0),
                "distance_3": distribution.get(3, 0),
                "distance_4": distribution.get(4, 0),
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v4_metrics = json.loads(V4_METRICS_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    x = df[FEATURES]
    y = df[TARGET]
    labels = [label for label in LABEL_ORDER if label in set(y)]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    spaces, unavailable = build_tuning_spaces()

    predictions = []
    metrics_by_model: dict[str, dict[str, object]] = {}
    cv_metrics_by_model: dict[str, dict[str, object]] = {}

    for model_name, candidate in v4_metrics["candidates"].items():
        if candidate.get("search_status") != "completed" or model_name not in spaces:
            continue
        pipeline, _ = spaces[model_name]
        pipeline = set_best_params(pipeline, candidate.get("best_params", {}))
        cv_metrics, true_labels, pred_labels, row_indices = evaluate_cv_oof(pipeline, x, y, labels, splitter)
        cv_metrics_by_model[model_name] = cv_metrics

        distance_frame = grade_distance_frame(model_name, true_labels, pred_labels)
        distance_frame.insert(1, "row_index", row_indices)
        distance_frame["title"] = df.iloc[row_indices]["title"].to_list()
        distance_frame["start_date"] = df.iloc[row_indices]["start_date"].to_list()
        distance_frame["genre_group"] = df.iloc[row_indices]["genre_group"].to_list()
        distance_frame["venue"] = df.iloc[row_indices]["venue"].to_list()
        distance_frame["venue_type"] = df.iloc[row_indices]["venue_type"].to_list()
        distance_frame["organizer"] = df.iloc[row_indices]["organizer"].to_list()
        predictions.append(distance_frame)
        metrics_by_model[model_name] = {
            **ordinal_metrics(distance_frame),
            "macro_f1": cv_metrics["oof_f1_macro"],
            "de_macro_f2": cv_metrics["oof_de_macro_f2"],
            "de_macro_recall": cv_metrics["oof_de_macro_recall"],
        }

    if not predictions:
        raise RuntimeError("No completed v4 model candidates were available for v4.2 grade analysis.")

    oof_predictions = pd.concat(predictions, ignore_index=True)
    oof_predictions.to_csv(OOF_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    severe_errors = oof_predictions[oof_predictions["is_severe_error"] == 1].copy()
    severe_errors.to_csv(SEVERE_ERRORS_PATH, index=False, encoding="utf-8-sig")

    metric_table = format_metric_table(metrics_by_model)
    distance_table = distance_distribution_table(metrics_by_model)
    focus_models = [name for name in ["RandomForestClassifier", "XGBoostClassifier"] if name in metrics_by_model]
    focus_metric_table = metric_table[metric_table["model"].isin(focus_models)].copy()
    severe_preview = severe_errors[
        [
            "model",
            "actual_grade",
            "predicted_grade",
            "abs_error",
            "title",
            "start_date",
            "genre_group",
            "venue",
        ]
    ].head(20)
    if severe_preview.empty:
        severe_preview = pd.DataFrame(
            [
                {
                    "model": "",
                    "actual_grade": "",
                    "predicted_grade": "",
                    "abs_error": "",
                    "title": "Severe error 없음",
                    "start_date": "",
                    "genre_group": "",
                    "venue": "",
                }
            ]
        )

    best_by_macro = v4_metrics["best_model"]["name"]
    best_by_adjacent = str(metric_table.sort_values("adjacent_accuracy", ascending=False).iloc[0]["model"])
    best_by_mae = str(metric_table.sort_values("grade_mae", ascending=True).iloc[0]["model"])
    best_by_severe = str(metric_table.sort_values(["severe_error_rate", "grade_mae"], ascending=[True, True]).iloc[0]["model"])

    metrics_output = {
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "source_metrics_path": str(V4_METRICS_PATH.relative_to(ROOT)),
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "grade_mapping": GRADE_TO_SCORE,
        "unavailable_models": unavailable,
        "models": metrics_by_model,
        "best_by_macro_f1": best_by_macro,
        "best_by_grade_mae": best_by_mae,
        "best_by_adjacent_accuracy": best_by_adjacent,
        "best_by_severe_error_rate": best_by_severe,
        "oof_predictions_path": str(OOF_PREDICTIONS_PATH.relative_to(ROOT)),
        "severe_errors_path": str(SEVERE_ERRORS_PATH.relative_to(ROOT)),
    }
    OUTPUT_METRICS_PATH.write_text(json.dumps(metrics_output, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# Step 4.2. 순서형 등급 평가 리포트

## 분석 목적

A~E는 단순 카테고리가 아니라 `A > B > C > D > E` 순서를 가진 등급이다. 따라서 실제 D를 C로 예측한 경우와 실제 E를 A로 예측한 경우는 같은 오분류가 아니다. v4.2_grade에서는 기존 v4 모델의 최적 하이퍼파라미터를 재사용하고, OOF 예측을 기준으로 등급 오차의 크기를 평가했다.

## 순서형 변환

{markdown_table(pd.DataFrame([{"grade": grade, "score": score} for grade, score in GRADE_TO_SCORE.items()]), index=False)}

## 추가 평가 지표

{markdown_table(pd.DataFrame([
    {"metric": "Grade MAE", "description": "A=1, B=2, C=3, D=4, E=5로 변환한 뒤 평균 절대 등급 오차를 계산한다. 낮을수록 좋다."},
    {"metric": "Adjacent Accuracy", "description": "실제 등급과 예측 등급 차이가 0 또는 1이면 맞은 것으로 본다. 높을수록 좋다."},
    {"metric": "Severe Error Rate", "description": "실제 D/E를 A/B로 예측한 비율이다. 취약 등급을 과소평가하는 심각 오류를 본다. 낮을수록 좋다."},
    {"metric": "Mean Signed Error", "description": "예측 점수 - 실제 점수의 평균이다. 음수면 실제보다 좋은 등급으로 예측하는 경향, 양수면 더 나쁜 등급으로 예측하는 경향이다."},
]), index=False)}

## 모델별 순서형 지표

{markdown_table(metric_table, index=False)}

## RandomForest와 XGBoost 비교

{markdown_table(focus_metric_table, index=False)}

기존 v4의 Macro F1 기준 최종 모델은 `{best_by_macro}`이다. 그러나 순서형 관점에서는 `Grade MAE`가 낮고 `Adjacent Accuracy`가 높은 모델이 등급 경향성을 더 잘 학습했다고 볼 수 있다. Severe Error Rate는 실제 D/E 공연을 A/B로 예측하는 정책상 위험한 오류를 따로 분리해 보여준다.

## 등급 오차 거리 분포

{markdown_table(distance_table, index=False)}

정확히 맞춘 비율이 낮더라도 `distance_1_adjacent`가 크고 `distance_2` 이상이 작다면, 모델이 등급의 방향성은 어느 정도 학습했다고 해석할 수 있다. 반대로 distance가 큰 오류가 많으면 등급 순서 정보를 충분히 반영하지 못한 것이다.

## Severe Error 사례

아래는 실제 D/E를 A/B로 예측한 심각 오류 사례 일부다. 전체 파일은 `{SEVERE_ERRORS_PATH.relative_to(ROOT)}`에 저장했다.

{markdown_table(severe_preview, index=False)}

## 산출물

- HTML 리포트: `{OUTPUT_HTML_PATH.relative_to(ROOT)}`
- 순서형 메트릭 JSON: `{OUTPUT_METRICS_PATH.relative_to(ROOT)}`
- OOF 예측 CSV: `{OOF_PREDICTIONS_PATH.relative_to(ROOT)}`
- Severe error CSV: `{SEVERE_ERRORS_PATH.relative_to(ROOT)}`
"""

    generated_at = datetime.fromisoformat(metrics_output["created_at"]).strftime("%Y-%m-%d %H:%M:%S %Z")
    hero = {
        "best_by_mae": metrics_by_model[best_by_mae]["grade_mae"],
        "best_by_adjacent": metrics_by_model[best_by_adjacent]["adjacent_accuracy"],
        "best_by_severe": metrics_by_model[best_by_severe]["severe_error_rate"],
    }
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>장애인 접근성 등급 진단 모델 v4.2_grade - 순서형 평가</title>
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
    .page {{ max-width: 1180px; margin: 0 auto; padding: 40px 28px 56px; }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-top: 5px solid var(--accent);
      padding: 30px 32px;
      margin-bottom: 18px;
    }}
    .eyebrow {{ margin: 0 0 8px; color: var(--accent); font-size: 13px; font-weight: 700; }}
    h1, h2, h3, h4 {{ color: #17324d; line-height: 1.28; }}
    h1 {{ margin: 0 0 12px; font-size: 30px; }}
    h2 {{ margin: 0 0 18px; font-size: 23px; }}
    h3 {{ margin: 26px 0 10px; font-size: 18px; }}
    h4 {{ margin: 20px 0 8px; font-size: 15px; color: var(--muted); }}
    p {{ margin: 8px 0; }}
    .summary-text {{ max-width: 900px; margin: 0; color: var(--muted); font-size: 15px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .metric {{ border: 1px solid var(--line); background: var(--soft); padding: 14px 16px; min-height: 86px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .metric strong {{ display: block; margin-top: 7px; font-size: 21px; color: var(--ink); word-break: break-word; }}
    .note, .report-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 26px 30px;
      margin: 18px 0;
    }}
    .note {{ border-left: 5px solid var(--accent-2); background: #fbfaf3; }}
    .table-wrap {{ width: 100%; overflow-x: auto; margin: 14px 0 20px; border: 1px solid var(--line); }}
    table {{ border-collapse: collapse; width: 100%; min-width: 640px; font-size: 14px; background: #fff; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 11px; text-align: left; vertical-align: top; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ background: #f1f5f9; color: #2d3d50; font-weight: 700; white-space: nowrap; }}
    code {{ background: var(--code); padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 860px) {{
      .page {{ padding: 22px 14px 36px; }}
      .hero, .note, .report-section {{ padding: 20px; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <p class="eyebrow">Accessibility Model Report v4.2_grade</p>
      <h1>장애인 접근성 등급 진단 모델 순서형 평가</h1>
      <p class="summary-text">A~E 등급의 순서를 반영해 평균 등급 오차, 인접 등급 허용 정확도, 심각 오류율을 추가 분석했다.</p>
      <div class="metric-grid">
        <div class="metric"><span>Grade MAE 최저</span><strong>{html.escape(best_by_mae)} ({hero["best_by_mae"]:.4f})</strong></div>
        <div class="metric"><span>Adjacent Accuracy 최고</span><strong>{html.escape(best_by_adjacent)} ({hero["best_by_adjacent"]:.4f})</strong></div>
        <div class="metric"><span>Severe Error Rate 최저</span><strong>{html.escape(best_by_severe)} ({hero["best_by_severe"]:.4f})</strong></div>
        <div class="metric"><span>기존 Macro F1 최종</span><strong>{html.escape(best_by_macro)}</strong></div>
      </div>
    </header>
    <section class="note">
      <strong>핵심 해석</strong>
      <p>정확히 맞춘 비율만 보면 등급 오차의 심각도를 놓칠 수 있다. v4.2_grade는 오차가 인접 등급 안에 머무는지, 실제 D/E를 A/B로 과소평가하는 심각 오류가 있는지를 별도로 확인한다.</p>
      <p>생성 시각: {html.escape(generated_at)} / 실행 환경: <code>conda env pronii</code></p>
    </section>
    <section class="report-section">
      {markdown_to_html(md)}
    </section>
  </main>
</body>
</html>
"""
    OUTPUT_HTML_PATH.write_text(html_text, encoding="utf-8")
    print(json.dumps({key: metrics_output[key] for key in ["best_by_grade_mae", "best_by_adjacent_accuracy", "best_by_severe_error_rate"]}, ensure_ascii=False, indent=2))
    print(f"HTML report written to {OUTPUT_HTML_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
