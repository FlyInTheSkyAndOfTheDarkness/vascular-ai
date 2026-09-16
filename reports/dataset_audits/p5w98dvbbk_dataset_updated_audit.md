# Dataset audit: p5w98dvbbk_dataset_updated

## Summary

- Source path: `C:\Users\Lenovo\Desktop\code\VascularAI\data\external\p5w98dvbbk_dataset_updated.csv`
- Header row: 0
- Rows: 1205
- Columns: 12

## Warnings

- Exact duplicate rate is above 1%; verify dataset grain before training.
- Column `Age` has 18 values outside expected clinical range.
- Column `BMI` has 1 values outside expected clinical range.

## Target candidates

- `Risk Level`: 3 distinct values
  - `Low`: 713
  - `High`: 474
  - `nan`: 18

## Duplicate profile

- Exact duplicate rows: 18
- Exact duplicate rate: 1.49%

## Potential leakage columns

- None detected by keyword scan.

## Numeric ranges

| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `Age` | 10.0 | 25.0 | 325.0 | 58 | (10, 55) | 18 |
| `Systolic BP` | 70.0 | 120.0 | 200.0 | 8 | (70, 220) | 0 |
| `Diastolic` | 40.0 | 80.0 | 140.0 | 2 | (40, 140) | 0 |
| `BS` | 3.0 | 6.9 | 19.0 | 181 | (2, 35) | 0 |
| `Body Temp` | 97.0 | 98.0 | 103.0 | 175 | (94, 106) | 0 |
| `BMI` | 0.0 | 23.0 | 37.0 | 37 | (10, 70) | 1 |
| `Previous Complications` | 0.0 | 0.0 | 1.0 | 211 |  |  |
| `Preexisting Diabetes` | 0.0 | 0.0 | 1.0 | 0 |  |  |
| `Gestational Diabetes` | 0.0 | 0.0 | 1.0 | 142 |  |  |
| `Mental Health` | 0.0 | 0.0 | 1.0 | 0 |  |  |
| `Heart Rate` | 58.0 | 76.0 | 92.0 | 0 | (40, 180) | 0 |

## Column profile

| Column | Type | Missing | Sentinel text | Distinct |
| --- | --- | ---: | ---: | ---: |
| `Age` | `int64` | 0.00% | 0.00% | 43 |
| `Systolic BP` | `float64` | 0.41% | 0.00% | 24 |
| `Diastolic` | `float64` | 0.33% | 0.00% | 21 |
| `BS` | `float64` | 0.17% | 0.00% | 84 |
| `Body Temp` | `int64` | 0.00% | 0.00% | 7 |
| `BMI` | `float64` | 1.49% | 0.00% | 157 |
| `Previous Complications` | `float64` | 0.17% | 0.00% | 2 |
| `Preexisting Diabetes` | `float64` | 0.17% | 0.00% | 2 |
| `Gestational Diabetes` | `int64` | 0.00% | 0.00% | 2 |
| `Mental Health` | `int64` | 0.00% | 0.00% | 2 |
| `Heart Rate` | `float64` | 0.17% | 0.00% | 31 |
| `Risk Level` | `str` | 1.49% | 0.00% | 2 |

## Manual review before inclusion

- Confirm license and permitted use.
- Confirm one-row grain and whether repeat patient visits exist.
- Confirm target definition and label source.
- Confirm units, especially glucose and body temperature.
- Remove post-outcome, derived-flag, and direct-label leakage columns.
- Decide whether this dataset joins the main model or needs a separate model.
