# Step 2. 타겟 라벨 설계

## 타겟 변수

- 타겟: `accessibility_grade`
- 기준값: `wheelchair_booking_rate = (wheelchair_booking_count / wheelchair_seats) / (paid_audience_count / total_seats)`
- 해석: 전체 예매율 대비 휠체어석 예매율. `1.0`이면 전체 좌석 예매율과 휠체어석 예매율이 같은 수준이다.

## 등급 기준

| 등급 | 기준 |
| --- | --- |
| A | `wheelchair_booking_rate >= 1.00` |
| B | `0.50 <= wheelchair_booking_rate < 1.00` |
| C | `0.25 <= wheelchair_booking_rate < 0.50` |
| D | `0.10 <= wheelchair_booking_rate < 0.25` |
| E | `wheelchair_booking_rate < 0.10` |

기존 `wheelchair_booking_count / wheelchair_seats` 방식은 장기 공연에서 공연 회차/기간 효과가 크게 반영되어 값이 과도하게 커질 수 있었다.
따라서 전체 예매율로 한 번 더 나눈 상대 예매율을 기준으로 라벨링했다.

## 등급별 샘플 수

| accessibility_grade | count |
| --- | --- |
| A | 189 |
| B | 54 |
| C | 59 |
| D | 119 |
| E | 64 |

## 예매율 요약

|  | value |
| --- | --- |
| count | 485.0 |
| mean | 1.8491404516428849 |
| std | 5.7195291368038 |
| min | 0.010128773749074131 |
| 25% | 0.15654145995747698 |
| 50% | 0.5034973468403281 |
| 75% | 2.058139534883721 |
| max | 104.375 |
