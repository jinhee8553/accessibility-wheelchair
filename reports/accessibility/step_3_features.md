# Step 3. Feature Engineering

## 학습 후보 Feature

| feature | 설명 | 사용 여부 |
| --- | --- | --- |
| genre_group | 장르를 클래식/오페라 등으로 정규화한 범주 | 학습 |
| venue_type | 공연장을 대형/중형/소극장/기타로 묶은 범주 | 학습 |
| is_weekend | 공연 시작일 기준 주말 여부 | 학습 |
| duration_days | 공연 기간 일수 | 학습 |
| organizer_type | 대관/기획 등 운영 구분 | 학습 |
| organizer | 대관 기업명 또는 주최 정보 | 학습 |
| paid_audience_count | 전체 예매율 산출에 직접 연결 | 제외 |
| wheelchair_booking_count | 타겟 산출에 직접 연결 | 제외 |
| wheelchair_booking_rate | 타겟 라벨 산출값 | 제외 |

## 제외 기준

좌석 수, 입장객 수, 휠체어석 예매 수, 보정 예매율은 타겟과 직접 연결되어 있어 학습 feature에서 제외한다.
