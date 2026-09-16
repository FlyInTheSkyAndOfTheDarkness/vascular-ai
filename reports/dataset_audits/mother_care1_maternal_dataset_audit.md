# Dataset audit: mother_care1_maternal_dataset

## Summary

- Source path: `C:\Users\Lenovo\Desktop\code\VascularAI\data\external\mother_care1_maternal_dataset.csv`
- Header row: 0
- Rows: 609
- Columns: 13

## Warnings

- Potential leakage columns detected; review before modeling.
- Column `delivery_mode` has more than 5% sentinel text values.
- Column `baby_condition` has more than 5% sentinel text values.
- Column `baby_sex` has more than 5% sentinel text values.
- Column `birth_weight_grams` has more than 20% missing values.
- Column `apgar_score` has more than 20% missing values.

## Target candidates

- `high_risk`: 2 distinct values
  - `Yes`: 379
  - `No`: 230

## Duplicate profile

- Exact duplicate rows: 0
- Exact duplicate rate: 0.00%
- `patient_id` duplicated rows: 0 (0.00%)

## Potential leakage columns

- `delivery_mode`
- `birth_weight_grams`
- `apgar_score`

## Numeric ranges

| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `mother_age_years` | 16.0 | 23.0 | 48.0 | 2 | (10, 55) | 0 |
| `parity` | 0.0 | 1.0 | 4.0 | 0 |  |  |
| `gravida` | 0.0 | 2.0 | 5.0 | 48 |  |  |
| `anc_visits` | 0.0 | 3.0 | 5.0 | 4 |  |  |
| `gestational_age_weeks` | 22.0 | 39.0 | 40.0 | 14 | (4, 45) | 0 |
| `birth_weight_grams` | 1900.0 | 3000.0 | 4200.0 | 3 |  |  |
| `apgar_score` | 2.0 | 9.0 | 10.0 | 3 |  |  |

## Column profile

| Column | Type | Missing | Sentinel text | Distinct |
| --- | --- | ---: | ---: | ---: |
| `patient_id` | `str` | 0.00% | 0.00% | 609 |
| `mother_age_years` | `int64` | 0.00% | 0.00% | 24 |
| `parity` | `int64` | 0.00% | 0.00% | 5 |
| `gravida` | `int64` | 0.00% | 0.00% | 6 |
| `anc_visits` | `int64` | 0.00% | 0.00% | 6 |
| `admission_reason` | `str` | 0.00% | 1.97% | 6 |
| `gestational_age_weeks` | `int64` | 0.00% | 0.00% | 10 |
| `delivery_mode` | `str` | 0.00% | 25.94% | 3 |
| `baby_condition` | `str` | 0.00% | 30.05% | 4 |
| `baby_sex` | `str` | 0.00% | 31.20% | 5 |
| `birth_weight_grams` | `float64` | 34.81% | 0.00% | 26 |
| `apgar_score` | `float64` | 36.45% | 0.00% | 8 |
| `high_risk` | `str` | 0.00% | 0.00% | 2 |

## Manual review before inclusion

- Confirm license and permitted use.
- Confirm one-row grain and whether repeat patient visits exist.
- Confirm target definition and label source.
- Confirm units, especially glucose and body temperature.
- Remove post-outcome, derived-flag, and direct-label leakage columns.
- Decide whether this dataset joins the main model or needs a separate model.
