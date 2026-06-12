from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "ap_whellchair_hall_total.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "accessibility"
DATASET_FILE = PROCESSED_DIR / "accessibility_dataset.csv"

GRADE_BINS = [-float("inf"), 0.10, 0.25, 0.50, 1.00, float("inf")]
GRADE_LABELS = ["E", "D", "C", "B", "A"]

GENRE_GROUPS = {
    "교향곡": "클래식",
    "관현악": "클래식",
    "독주": "클래식",
    "성악": "클래식",
    "실내악": "클래식",
    "합창": "클래식",
    "오페라": "오페라",
}

VENUE_SCALE = {
    "콘서트홀": "대형",
    "오페라극장": "대형",
    "CJ 토월극장": "대형",
    "IBK기업은행챔버홀": "중형",
    "IBK챔버홀": "중형",
    "리사이틀홀": "중형",
    "인춘아트홀": "소극장",
    "자유소극장": "소극장",
}

STANDARD_COLUMNS = {
    "제목": "title",
    "공연시작일": "start_date",
    "공연종료일": "end_date",
    "집계최초일자": "first_booking_date",
    "장르": "genre",
    "구분": "organizer_type",
    "공연장": "venue",
    "대관 기업명": "organizer",
    "유료입장객수": "paid_audience_count",
    "휠체어석예매수": "wheelchair_booking_count",
    "일반석": "general_seats",
    "장애인석": "wheelchair_seats",
}

REQUIRED_COLUMNS = [
    "title",
    "start_date",
    "end_date",
    "first_booking_date",
    "genre",
    "organizer_type",
    "venue",
    "organizer",
    "paid_audience_count",
    "wheelchair_booking_count",
    "total_seats",
    "general_seats",
    "wheelchair_seats",
]

OUTPUT_COLUMNS = [
    "title",
    "start_date",
    "end_date",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "duration_days",
    "booking_lead_days",
    "genre",
    "genre_group",
    "venue",
    "venue_type",
    "organizer_type",
    "organizer",
    "start_time",
    "paid_audience_count",
    "wheelchair_booking_count",
    "total_seats",
    "general_seats",
    "wheelchair_seats",
    "wheelchair_seat_ratio",
    "wheelchair_booking_rate_raw",
    "overall_booking_rate",
    "wheelchair_booking_rate",
    "accessibility_grade",
]


@dataclass(frozen=True)
class DatasetSummary:
    raw_rows: int
    processed_rows: int
    processed_columns: int
    excluded_rows: int


def find_total_seat_column(columns: pd.Index) -> str:
    for name in ("total", "총좌석수", "총 좌석"):
        if name in columns:
            return name
    raise KeyError("총 좌석 수 컬럼을 찾을 수 없습니다.")


def normalize_text(values: pd.Series) -> pd.Series:
    cleaned = values.fillna("unknown").astype(str).str.strip()
    return cleaned.mask(cleaned.eq(""), "unknown")


def load_raw_dataset(path: Path = RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {path}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    total_column = find_total_seat_column(raw.columns)
    return raw.rename(columns={**STANDARD_COLUMNS, total_column: "total_seats"})


def assert_required_columns(frame: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")


def assign_accessibility_grade(rate: pd.Series) -> pd.Series:
    return pd.cut(
        rate,
        bins=GRADE_BINS,
        labels=GRADE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    typed = frame.copy()
    typed["start_date"] = pd.to_datetime(typed["start_date"], errors="coerce")
    typed["end_date"] = pd.to_datetime(typed["end_date"], errors="coerce")
    typed["first_booking_date"] = pd.to_datetime(
        typed["first_booking_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    numeric_columns = [
        "paid_audience_count",
        "wheelchair_booking_count",
        "total_seats",
        "general_seats",
        "wheelchair_seats",
    ]
    for column in numeric_columns:
        typed[column] = pd.to_numeric(typed[column], errors="coerce")
    return typed


def keep_modelable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required_non_null = [
        "start_date",
        "end_date",
        "paid_audience_count",
        "wheelchair_booking_count",
        "total_seats",
        "wheelchair_seats",
    ]
    filtered = frame.dropna(subset=required_non_null).copy()
    return filtered[
        (filtered["wheelchair_seats"] > 0)
        & (filtered["total_seats"] > 0)
        & (filtered["paid_audience_count"] > 0)
    ].copy()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = frame.copy()
    for column in ("genre", "organizer_type", "organizer", "venue"):
        featured[column] = normalize_text(featured[column])

    featured["genre_group"] = featured["genre"].replace(GENRE_GROUPS)
    featured["venue_type"] = featured["venue"].map(VENUE_SCALE).fillna("기타")
    featured["duration_days"] = (
        (featured["end_date"] - featured["start_date"]).dt.days.add(1).clip(lower=1)
    )
    featured["booking_lead_days"] = (
        (featured["start_date"] - featured["first_booking_date"]).dt.days.fillna(0).clip(lower=0)
    )
    featured["day_of_week"] = featured["start_date"].dt.dayofweek
    featured["is_weekend"] = featured["day_of_week"].isin([5, 6]).astype(int)
    featured["month"] = featured["start_date"].dt.month
    featured["year"] = featured["start_date"].dt.year
    featured["start_time"] = 0
    featured["wheelchair_booking_rate_raw"] = (
        featured["wheelchair_booking_count"] / featured["wheelchair_seats"]
    )
    featured["overall_booking_rate"] = featured["paid_audience_count"] / featured["total_seats"]
    featured["wheelchair_booking_rate"] = (
        featured["wheelchair_booking_rate_raw"] / featured["overall_booking_rate"]
    )
    featured["wheelchair_seat_ratio"] = featured["wheelchair_seats"] / featured["total_seats"]
    featured["accessibility_grade"] = assign_accessibility_grade(featured["wheelchair_booking_rate"])
    return featured


def build_accessibility_dataset(raw_path: Path = RAW_FILE) -> pd.DataFrame:
    raw = load_raw_dataset(raw_path)
    assert_required_columns(raw)
    typed = coerce_types(raw)
    modelable = keep_modelable_rows(typed)
    featured = add_features(modelable)
    return featured[OUTPUT_COLUMNS].sort_values(["start_date", "title"]).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    table = frame.copy()
    if index:
        table = table.reset_index()
    table = table.fillna("")
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
    return "\n".join([header, separator, *rows])


def build_summary(raw: pd.DataFrame, processed: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        raw_rows=len(raw),
        processed_rows=len(processed),
        processed_columns=len(processed.columns),
        excluded_rows=len(raw) - len(processed),
    )


def write_reports(raw: pd.DataFrame, processed: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary(raw, processed)
    grade_counts = (
        processed["accessibility_grade"].value_counts().reindex(["A", "B", "C", "D", "E"], fill_value=0)
    )
    feature_rows = pd.DataFrame(
        [
            ["genre_group", "장르를 클래식/오페라 등으로 정규화한 범주", "학습"],
            ["venue_type", "공연장을 대형/중형/소극장/기타로 묶은 범주", "학습"],
            ["is_weekend", "공연 시작일 기준 주말 여부", "학습"],
            ["duration_days", "공연 기간 일수", "학습"],
            ["organizer_type", "대관/기획 등 운영 구분", "학습"],
            ["organizer", "대관 기업명 또는 주최 정보", "학습"],
            ["paid_audience_count", "전체 예매율 산출에 직접 연결", "제외"],
            ["wheelchair_booking_count", "타겟 산출에 직접 연결", "제외"],
            ["wheelchair_booking_rate", "타겟 라벨 산출값", "제외"],
        ],
        columns=["feature", "설명", "사용 여부"],
    )

    dataset_report = f"""# Step 1. 데이터셋 구성

| 항목 | 값 |
| --- | ---: |
| 원본 행 수 | {summary.raw_rows:,} |
| 전처리 후 행 수 | {summary.processed_rows:,} |
| 전처리 후 컬럼 수 | {summary.processed_columns:,} |
| 제외 행 수 | {summary.excluded_rows:,} |

## 입력과 출력

- 원본: `{RAW_FILE.relative_to(ROOT)}`
- 전처리 결과: `{DATASET_FILE.relative_to(ROOT)}`

## 전처리 기준

- 날짜와 좌석/예매 수를 명시적 타입으로 변환했다.
- 장애인석, 총좌석, 유료입장객 수가 0 이하인 행은 보정 예매율을 안정적으로 계산할 수 없어 제외했다.
- 텍스트 결측값은 `unknown`으로 채우고 앞뒤 공백을 제거했다.
"""
    label_report = f"""# Step 2. 접근성 라벨링

## 타겟 정의

`accessibility_grade`는 전체 좌석 예매율 대비 휠체어석 예매율입니다.

`wheelchair_booking_rate = (wheelchair_booking_count / wheelchair_seats) / (paid_audience_count / total_seats)`

| 등급 | 기준 |
| --- | --- |
| A | `1.00 <= rate` |
| B | `0.50 <= rate < 1.00` |
| C | `0.25 <= rate < 0.50` |
| D | `0.10 <= rate < 0.25` |
| E | `rate < 0.10` |

## 등급별 행 수

{markdown_table(grade_counts.rename("count").to_frame(), index=True)}
"""
    feature_report = f"""# Step 3. Feature Engineering

## 학습 후보 Feature

{markdown_table(feature_rows)}

## 제외 기준

좌석 수, 입장객 수, 휠체어석 예매 수, 보정 예매율은 타겟과 직접 연결되어 있어 학습 feature에서 제외한다.
"""

    (REPORT_DIR / "step_1_dataset.md").write_text(dataset_report, encoding="utf-8")
    (REPORT_DIR / "step_2_labeling.md").write_text(label_report, encoding="utf-8")
    (REPORT_DIR / "step_3_features.md").write_text(feature_report, encoding="utf-8")


def main() -> None:
    raw = load_raw_dataset()
    dataset = build_accessibility_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")
    write_reports(raw, dataset)
    print(f"saved: {DATASET_FILE}")
    print(f"rows: {len(dataset)}")
    print(dataset["accessibility_grade"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

