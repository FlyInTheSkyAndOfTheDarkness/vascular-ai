# Dataset audit: ckhgtgdg4f_springer

## Summary

- Source path: `C:\Users\Lenovo\Desktop\code\VascularAI\data\external\ckhgtgdg4f_springer.csv`
- Header row: 0
- Rows: 6103
- Columns: 11

## Warnings

- Column `Age` has 1 values outside expected clinical range.
- Column `Body Temperature(F) ` has 4 values outside expected clinical range.
- Column `Diastolic Blood Pressure(mm Hg)` has 2 values outside expected clinical range.
- Column `Blood Glucose(HbA1c)` looks like HbA1c in IFCC mmol/mol, not percent; confirm units before modeling.
- Column `Blood Glucose(Fasting hour-mg/dl)` is named mg/dL but values look like mmol/L; confirm units before mapping to `BS`.

## Target candidates

- `Status`: 3 distinct values
  - `high risk`: 2059
  - `mid risk`: 2043
  - `low risk`: 2001

## Duplicate profile

- Exact duplicate rows: 0
- Exact duplicate rate: 0.00%
- `Patient ID` duplicated rows: 0 (0.00%)
- `Name` duplicated rows: 609 (9.98%)

## Potential leakage columns

- None detected by keyword scan.

## Numeric ranges

| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `Patient ID` | 101.0 | 3152.0 | 6203.0 | 0 |  |  |
| `Age` | 15.0 | 25.0 | 250.0 | 86 | (10, 55) | 1 |
| `Body Temperature(F) ` | 39.6 | 98.6 | 104.0 | 1763 | (94, 106) | 4 |
| `Heart rate(bpm)` | 45.0 | 80.0 | 150.0 | 742 | (40, 180) | 0 |
| `Systolic Blood Pressure(mm Hg)` | 90.0 | 128.0 | 169.0 | 0 | (70, 220) | 0 |
| `Diastolic Blood Pressure(mm Hg)` | 9.0 | 87.0 | 142.0 | 149 | (40, 140) | 2 |
| `BMI(kg/m 2)` | 14.9 | 21.3 | 27.9 | 0 | (10, 70) | 0 |
| `Blood Glucose(HbA1c)` | 30.0 | 38.0 | 50.0 | 0 | (20, 140) | 0 |
| `Blood Glucose(Fasting hour-mg/dl)` | 3.5 | 5.7 | 8.9 | 27 | (2, 35) | 0 |

## Column profile

| Column | Type | Missing | Sentinel text | Distinct |
| --- | --- | ---: | ---: | ---: |
| `Patient ID` | `int64` | 0.00% | 0.00% | 6103 |
| `Name` | `str` | 0.00% | 0.00% | 5794 |
| `Age` | `int64` | 0.00% | 0.00% | 35 |
| `Body Temperature(F) ` | `float64` | 0.00% | 0.00% | 104 |
| `Heart rate(bpm)` | `int64` | 0.00% | 0.00% | 106 |
| `Systolic Blood Pressure(mm Hg)` | `int64` | 0.00% | 0.00% | 79 |
| `Diastolic Blood Pressure(mm Hg)` | `int64` | 0.00% | 0.00% | 65 |
| `BMI(kg/m 2)` | `float64` | 0.00% | 0.00% | 127 |
| `Blood Glucose(HbA1c)` | `int64` | 0.00% | 0.00% | 21 |
| `Blood Glucose(Fasting hour-mg/dl)` | `float64` | 0.00% | 0.00% | 49 |
| `Status` | `str` | 0.00% | 0.00% | 3 |

## Manual review before inclusion

- Confirm license and permitted use.
- Confirm one-row grain and whether repeat patient visits exist.
- Confirm target definition and label source.
- Confirm units, especially glucose and body temperature.
- Remove post-outcome, derived-flag, and direct-label leakage columns.
- Decide whether this dataset joins the main model or needs a separate model.
