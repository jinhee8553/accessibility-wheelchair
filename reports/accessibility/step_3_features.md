# Step 3. Feature Engineering

## Feature 목록

| feature | 설명 | 사용 |
| --- | --- | --- |
| booking_lead_days | 공연시작일 - 집계최초일자. 현재 데이터에서는 전체 값이 0인 상수 컬럼 | 학습 제외 |
| genre_group | 장르 정규화 결과 | 학습 |
| venue_type | 공연장명 기준 대형/중형/소극장 매핑 | 학습 |
| is_weekend | 공연시작일 기준 주말 여부 | 학습 |
| duration_days | 공연종료일 - 공연시작일 + 1 | 학습 |
| organizer_type | 기획 주체. 대관/기획 등 | 학습 |
| organizer | 대관 기업 또는 기획 주체명 | 학습 |
| start_time | 현재 원천 파일에 시간이 없어 0으로 고정한 상수 placeholder | 학습 제외 |
| paid_audience_count | 전체 유료입장객 수. 전체 예매율 산출에 직접 사용 | 학습 제외 |
| wheelchair_booking_count | 휠체어석 예매 수. 타겟 산출에 직접 사용 | 학습 제외 |
| total_seats | 공연장 전체 좌석 수 | 분석/대시보드용. 학습 제외 |
| general_seats | 일반 판매석 수 | 분석/대시보드용. 학습 제외 |
| wheelchair_seats | 장애인석 수. 휠체어석 예매율 산출의 분모 | 분석/대시보드용. 학습 제외 |
| wheelchair_seat_ratio | 전체 좌석 중 휠체어석 비율 | 분석/대시보드용. 학습 제외 |
| wheelchair_booking_rate_raw | 휠체어석예매수 / 장애인석. 보정 전 예매율 | 학습 제외 |
| overall_booking_rate | 유료입장객수 / 총좌석수. 전체 예매율 | 학습 제외 |
| wheelchair_booking_rate | 전체 예매율 대비 휠체어석 예매율. 타겟 라벨 산출값 | 학습 제외 |

## 학습에서 제외한 좌석 정보

`booking_lead_days`, `start_time`은 현재 데이터에서 모든 값이 0인 상수 컬럼이라 학습 feature에서 제외한다.
`paid_audience_count`, `wheelchair_booking_count`, `total_seats`, `general_seats`, `wheelchair_seats`, `wheelchair_seat_ratio`, `wheelchair_booking_rate_raw`, `overall_booking_rate`, `wheelchair_booking_rate`는 접근성 등급 산출에 직접 연결되거나 좌석 수 신호가 강한 값이라 모델이 공연 특성 대신 사후 예매/좌석 정보를 보고 맞추는 문제가 생길 수 있어 제외한다.

## 주요 범주 분포

### 장르 그룹

| genre_group | count |
| --- | --- |
| 클래식 | 397 |
| 오페라 | 21 |
| 발레 | 16 |
| 뮤지컬 | 15 |
| 연극 | 14 |
| 기타 | 9 |
| 기타(복합) | 6 |
| 이벤트콘서트 | 5 |
| 무용 | 2 |

### 공연장 규모

| venue_type | count |
| --- | --- |
| 대형 | 277 |
| 중형 | 123 |
| 소극장 | 85 |
