# Step 1. 데이터셋 기준 확정

## 입력 파일

- 원본: `data/raw/ap_whellchair_hall_total.csv`
- 전처리 결과: `data/processed/accessibility_dataset.csv`

## 데이터 크기

| 항목 | 값 |
| --- | ---: |
| 원본 행 수 | 500 |
| 전처리 후 행 수 | 485 |
| 전처리 후 컬럼 수 | 26 |

## 제외 행 요약

- 제외 행 수: `15`행
- 제외 이유: 새 metric의 분모인 `overall_booking_rate = paid_audience_count / total_seats`가 0이 되면 `wheelchair_booking_rate = wheelchair_booking_rate_raw / overall_booking_rate`를 정의할 수 없다.
- 이번 데이터에서 제외된 15개 행은 모두 `paid_audience_count <= 0`, 즉 `유료입장객수 = 0`인 공연이다.

### 제외 사유별 건수

| 제외 사유 | count |
| --- | --- |
| paid_audience_count <= 0 | 15 |

### 제외된 공연 목록

| raw_index | title | start_date | venue | genre | paid_audience_count | wheelchair_booking_count | total_seats | wheelchair_seats | 제외 사유 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2023 대원문화재단 신년음악회 | 2023-01-07 | 콘서트홀 | 교향곡 | 0 | 25 | 2505 | 24 | paid_audience_count <= 0 |
| 31 | 2024 IBK음악회 | 2024-10-16 | 콘서트홀 | 클래식 | 0 | 24 | 2505 | 24 | paid_audience_count <= 0 |
| 40 | 2024 서울시향 크리스티안 테츨라프의 브람스 바이올린 협주곡 ① | 2024-09-05 | 콘서트홀 | 교향곡 | 0 | 4 | 2505 | 24 | paid_audience_count <= 0 |
| 42 | 2024 신년음악회 | 2024-01-09 | 콘서트홀 | 클래식 | 0 | 20 | 2505 | 24 | paid_audience_count <= 0 |
| 50 | 2024 예술의전당 회원음악회 | 2024-09-07 | 콘서트홀 | 클래식 | 0 | 1 | 2505 | 24 | paid_audience_count <= 0 |
| 51 | 2024 장애인식개선을 위한 하트 투 하트 콘서트 | 2024-11-19 | 콘서트홀 | 클래식 | 0 | 48 | 2505 | 24 | paid_audience_count <= 0 |
| 60 | IBK 음악회  with 조수미 & 서혜경 월드 클래식 | 2023-09-08 | 콘서트홀 | 크로스오버 | 0 | 24 | 2505 | 24 | paid_audience_count <= 0 |
| 217 | 못 말리는 프랑켄슈타인 | 2024-04-24 | CJ 토월극장 | 연극 | 0 | 20 | 1004 | 10 | paid_audience_count <= 0 |
| 264 | 서강은 피아노 독주회 | 2023-11-10 | 인춘아트홀 | 독주 | 0 | 2 | 100 | 2 | paid_audience_count <= 0 |
| 270 | 소프라노 노정애 독창회 | 2024-04-12 | 인춘아트홀 | 클래식 | 0 | 2 | 100 | 2 | paid_audience_count <= 0 |
| 378 | 장애인식개선을 위한 <하트 투 하트 콘서트> | 2023-08-29 | 콘서트홀 | 클래식 | 0 | 24 | 2505 | 24 | paid_audience_count <= 0 |
| 392 | 제14회ARKO한국창작음악제(양악부문) | 2023-02-01 | 콘서트홀 | 교향곡 | 0 | 4 | 2505 | 24 | paid_audience_count <= 0 |
| 393 | 제15회ARKO한국창작음악제(양악부문) | 2024-02-06 | 콘서트홀 | 교향곡 | 0 | 8 | 2505 | 24 | paid_audience_count <= 0 |
| 399 | 조성진 삼성호암상 수상기념 리사이틀 | 2024-06-18 | 콘서트홀 | 클래식 | 0 | 24 | 2505 | 24 | paid_audience_count <= 0 |
| 467 | 한국예술종합학교 음악원 개원30주년 기념음악회 | 2023-04-12 | 콘서트홀 | 교향곡 | 0 | 48 | 2505 | 24 | paid_audience_count <= 0 |

## 사용 원천 컬럼

- 공연 정보: `제목`, `공연시작일`, `공연종료일`, `장르`, `구분`, `공연장`, `대관 기업명`
- 좌석/예매 정보: `유료입장객수`, `휠체어석예매수`, `total/총좌석수`, `일반석`, `장애인석`

## 전처리 수행 내용

| 전처리 항목 | 내용 |
| --- | --- |
| 컬럼 표준화 | `제목`, `공연시작일`, `공연종료일`, `집계최초일자`, `장르`, `구분`, `공연장`, `대관 기업명`, `유료입장객수`, `휠체어석예매수`, `총좌석수`, `일반석`, `장애인석`을 영문 컬럼명으로 변경 |
| 타입 변환 | `start_date`, `end_date`, `first_booking_date`는 날짜형으로 변환하고, 좌석/예매 수 컬럼은 숫자형으로 변환 |
| 유효 행 필터링 | 날짜, 유료입장객수, 휠체어석예매수, 총좌석수, 장애인석이 결측인 행을 제외하고 `wheelchair_seats > 0`, `total_seats > 0`, `paid_audience_count > 0` 조건만 유지 |
| 텍스트 정리 | `genre`, `organizer_type`, `organizer`, `venue`의 결측/빈 문자열을 `unknown`으로 채우고 앞뒤 공백 제거 |
| 범주 매핑 | 세부 장르를 `genre_group`으로 정규화하고, 공연장명을 기준으로 `venue_type`을 대형/중형/소극장/기타로 매핑 |
| 날짜 파생변수 | `year`, `month`, `day_of_week`, `is_weekend`, `duration_days`, `booking_lead_days` 생성 |
| 예매율 파생변수 | `wheelchair_booking_rate_raw = wheelchair_booking_count / wheelchair_seats`, `overall_booking_rate = paid_audience_count / total_seats`, `wheelchair_booking_rate = wheelchair_booking_rate_raw / overall_booking_rate` 생성 |
| 좌석 비율 파생변수 | `wheelchair_seat_ratio = wheelchair_seats / total_seats` 생성 |
| 타겟 라벨링 | 보정 예매율 `wheelchair_booking_rate`를 A~E 구간으로 나누어 `accessibility_grade` 생성 |
| 출력 정렬 | `start_date`, `title` 기준으로 정렬한 뒤 `data/processed/accessibility_dataset.csv`로 저장 |

## 결측치 점검

| column | missing |
| --- | --- |
| title | 0 |
| start_date | 0 |
| end_date | 0 |
| year | 0 |
| month | 0 |
| day_of_week | 0 |
| is_weekend | 0 |
| duration_days | 0 |
| booking_lead_days | 0 |
| genre | 0 |
| genre_group | 0 |
| venue | 0 |
| venue_type | 0 |
| organizer_type | 0 |
| organizer | 0 |
| start_time | 0 |
| paid_audience_count | 0 |
| wheelchair_booking_count | 0 |
| total_seats | 0 |
| general_seats | 0 |
| wheelchair_seats | 0 |
| wheelchair_seat_ratio | 0 |
| wheelchair_booking_rate_raw | 0 |
| overall_booking_rate | 0 |
| wheelchair_booking_rate | 0 |
| accessibility_grade | 0 |
