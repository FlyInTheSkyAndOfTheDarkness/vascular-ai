# Dataset audit: mother_care2_clinical_maternal_dataset

## Summary

- Source path: `C:\Users\Lenovo\Desktop\code\VascularAI\data\external\mother_care2_clinical_maternal_dataset.csv`
- Header row: 0
- Rows: 111
- Columns: 21

## Warnings

- Potential leakage columns detected; review before modeling.
- Column `parity_raw` has more than 20% missing values.

## Target candidates

- `high_risk`: 2 distinct values
  - `Yes`: 89
  - `No`: 22

## Duplicate profile

- Exact duplicate rows: 0
- Exact duplicate rate: 0.00%
- `patient_id` duplicated rows: 0 (0.00%)

## Potential leakage columns

- `age_risk`
- `anemia_flag`
- `hypertension_flag`
- `preterm_risk`
- `low_birth_weight`
- `delivery_mode`
- `birth_weight_grams`
- `apgar_1min`
- `apgar_5min`

## Numeric ranges

| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `mother_age_years` | 17.0 | 24.0 | 44.0 | 2 | (10, 55) | 0 |
| `gravida_raw` | 1.0 | 2.0 | 5.0 | 4 |  |  |
| `gravida_filled` | 1.0 | 2.0 | 5.0 | 4 |  |  |
| `parity_raw` | 0.0 | 1.0 | 3.0 | 1 |  |  |
| `parity_filled` | 0.0 | 0.0 | 3.0 | 1 |  |  |
| `gestational_age_weeks` | 17.0 | 38.0 | 44.0 | 15 | (4, 45) | 0 |
| `systolic_bp_mmhg` | 80.0 | 110.0 | 160.0 | 1 | (70, 220) | 0 |
| `diastolic_bp_mmhg` | 50.0 | 70.0 | 110.0 | 7 | (40, 140) | 0 |
| `hemoglobin_g_dl` | 6.4 | 10.6 | 13.0 | 2 | (4, 20) | 0 |
| `birth_weight_grams` | 500.0 | 2800.0 | 4500.0 | 3 |  |  |
| `apgar_1min` | 0.0 | 8.0 | 9.0 | 11 |  |  |
| `apgar_5min` | 6.0 | 10.0 | 10.0 | 11 |  |  |

## Column profile

| Column | Type | Missing | Sentinel text | Distinct |
| --- | --- | ---: | ---: | ---: |
| `patient_id` | `str` | 0.00% | 0.00% | 111 |
| `mother_age_years` | `float64` | 4.50% | 0.00% | 23 |
| `gravida_raw` | `float64` | 4.50% | 0.00% | 5 |
| `gravida_filled` | `int64` | 0.00% | 0.00% | 5 |
| `parity_raw` | `float64` | 36.04% | 0.00% | 4 |
| `parity_filled` | `float64` | 13.51% | 0.00% | 4 |
| `gestational_age_weeks` | `float64` | 13.51% | 0.00% | 19 |
| `systolic_bp_mmhg` | `float64` | 17.12% | 0.00% | 9 |
| `diastolic_bp_mmhg` | `float64` | 17.12% | 0.00% | 8 |
| `hemoglobin_g_dl` | `float64` | 14.41% | 0.00% | 40 |
| `age_risk` | `str` | 3.60% | 0.00% | 2 |
| `anemia_flag` | `str` | 14.41% | 0.00% | 2 |
| `hypertension_flag` | `str` | 17.12% | 0.00% | 2 |
| `preterm_risk` | `str` | 13.51% | 0.00% | 2 |
| `low_birth_weight` | `str` | 16.22% | 0.00% | 2 |
| `delivery_mode` | `str` | 17.12% | 0.00% | 3 |
| `baby_sex` | `str` | 18.02% | 1.80% | 3 |
| `birth_weight_grams` | `float64` | 16.22% | 0.00% | 26 |
| `apgar_1min` | `float64` | 16.22% | 0.00% | 7 |
| `apgar_5min` | `float64` | 17.12% | 0.00% | 5 |
| `high_risk` | `str` | 0.00% | 0.00% | 2 |

## Manual review before inclusion

- Confirm license and permitted use.
- Confirm one-row grain and whether repeat patient visits exist.
- Confirm target definition and label source.
- Confirm units, especially glucose and body temperature.
- Remove post-outcome, derived-flag, and direct-label leakage columns.
- Decide whether this dataset joins the main model or needs a separate model.
