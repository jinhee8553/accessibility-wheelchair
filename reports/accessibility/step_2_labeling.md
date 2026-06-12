# Step 2. 접근성 라벨링

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

| accessibility_grade | count |
| --- | --- |
| A | 189 |
| B | 54 |
| C | 59 |
| D | 119 |
| E | 64 |
