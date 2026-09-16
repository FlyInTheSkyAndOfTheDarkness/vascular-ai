# Dataset audit: uci_maternal_health_risk

## Summary

- Source path: `C:\Users\Lenovo\Desktop\code\VascularAI\data\raw\Maternal Health Risk Data Set.csv`
- Header row: 0
- Rows: 1014
- Columns: 7

## Warnings

- Exact duplicate rate is above 1%; verify dataset grain before training.
- Column `Age` has 45 values outside expected clinical range.
- Column `HeartRate` has 2 values outside expected clinical range.

## Target candidates

- `RiskLevel`: 3 distinct values
  - `low risk`: 406
  - `mid risk`: 336
  - `high risk`: 272

## Duplicate profile

- Exact duplicate rows: 562
- Exact duplicate rate: 55.42%

## Potential leakage columns

- None detected by keyword scan.

## Numeric ranges

| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `Age` | 10.0 | 26.0 | 70.0 | 1 | (10, 55) | 45 |
| `SystolicBP` | 70.0 | 120.0 | 160.0 | 10 | (70, 220) | 0 |
| `DiastolicBP` | 49.0 | 80.0 | 100.0 | 0 | (40, 140) | 0 |
| `BS` | 6.0 | 7.5 | 19.0 | 210 | (2, 35) | 0 |
| `BodyTemp` | 98.0 | 98.0 | 103.0 | 210 | (94, 106) | 0 |
| `HeartRate` | 7.0 | 76.0 | 90.0 | 2 | (40, 180) | 2 |

## Column profile

| Column | Type | Missing | Sentinel text | Distinct |
| --- | --- | ---: | ---: | ---: |
| `Age` | `int64` | 0.00% | 0.00% | 50 |
| `SystolicBP` | `int64` | 0.00% | 0.00% | 19 |
| `DiastolicBP` | `int64` | 0.00% | 0.00% | 16 |
| `BS` | `float64` | 0.00% | 0.00% | 29 |
| `BodyTemp` | `float64` | 0.00% | 0.00% | 8 |
| `HeartRate` | `int64` | 0.00% | 0.00% | 16 |
| `RiskLevel` | `str` | 0.00% | 0.00% | 3 |

## Manual review before inclusion

- Confirm license and permitted use.
- Confirm one-row grain and whether repeat patient visits exist.
- Confirm target definition and label source.
- Confirm units, especially glucose and body temperature.
- Remove post-outcome, derived-flag, and direct-label leakage columns.
- Decide whether this dataset joins the main model or needs a separate model.
