# Dataset audit: high_risk_pregnancy_book2_header1

## Summary

- Source path: `C:\Users\Lenovo\Desktop\code\VascularAI\data\external\high_risk_pregnancy_book2.xlsx`
- Header row: 1
- Rows: 998
- Columns: 18

## Warnings

- Column `রক্তস্বল্পতা` has more than 20% missing values.
- Column `জন্ডিস` has more than 20% missing values.
- Column `প্রসাব পরিক্ষা এলবুমিন` has more than 20% missing values.

## Target candidates

- `ঝুকিপূর্ণ গর্ভ`: 2 distinct values
  - `Yes`: 666
  - `No`: 332

## Duplicate profile

- Exact duplicate rows: 0
- Exact duplicate rate: 0.00%
- `Name` duplicated rows: 848 (84.97%)

## Potential leakage columns

- None detected by keyword scan.

## Numeric ranges

| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `Age` | 18.0 | 22.0 | 32.0 | 0 | (10, 55) | 0 |

## Column profile

| Column | Type | Missing | Sentinel text | Distinct |
| --- | --- | ---: | ---: | ---: |
| `Name` | `str` | 0.00% | 0.00% | 350 |
| `Age` | `int64` | 0.00% | 0.00% | 13 |
| `Gravida` | `str` | 0.00% | 0.00% | 3 |
| `TiTi Tika` | `str` | 0.00% | 0.00% | 3 |
| `গর্ভকাল` | `str` | 0.00% | 0.00% | 11 |
| `ওজন` | `str` | 0.00% | 0.00% | 14 |
| `উচ্চতা` | `str` | 0.00% | 0.00% | 7 |
| `রক্ত চাপ` | `str` | 0.00% | 0.00% | 11 |
| `রক্তস্বল্পতা` | `str` | 87.68% | 0.00% | 2 |
| `জন্ডিস` | `str` | 98.80% | 0.00% | 2 |
| `গর্ভস্হ শিশু অবস্থান` | `str` | 0.00% | 0.00% | 2 |
| `গর্ভস্হ শিশু নাড়াচাড়া` | `str` | 0.00% | 0.00% | 1 |
| `গর্ভস্হ শিশু হৃৎস্পন্দন` | `str` | 0.00% | 0.00% | 5 |
| `প্রসাব পরিক্ষা এলবুমিন` | `str` | 86.57% | 0.00% | 3 |
| `প্রসাব পরিক্ষা সুগার` | `str` | 0.00% | 0.00% | 2 |
| `VDRL` | `str` | 0.00% | 0.00% | 2 |
| `HRsAG` | `str` | 0.00% | 0.00% | 2 |
| `ঝুকিপূর্ণ গর্ভ` | `str` | 0.00% | 0.00% | 2 |

## Manual review before inclusion

- Confirm license and permitted use.
- Confirm one-row grain and whether repeat patient visits exist.
- Confirm target definition and label source.
- Confirm units, especially glucose and body temperature.
- Remove post-outcome, derived-flag, and direct-label leakage columns.
- Decide whether this dataset joins the main model or needs a separate model.
