# External Dataset Audit Decision Matrix

Generated: 2026-09-14

## Intended use

Current VascularAI model predicts 3 maternal risk classes from six intake fields:
`Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, and `HeartRate`.

The useful external dataset must therefore have:

- compatible pre-delivery / intake-time features
- a compatible 3-class maternal risk target, or a clear reason to build a separate model
- clear units for glucose and body temperature
- no direct identifiers or post-outcome leakage features in the predictor set
- acceptable license for the intended product use

## Decision

| Dataset | Local file | Rows | Target | Main quality notes | Decision |
| --- | --- | ---: | --- | --- | --- |
| UCI Maternal Health Risk | `data/raw/Maternal Health Risk Data Set.csv` | 1014 | low / mid / high | No missing values, but 562 exact duplicates (55.42%), 45 ages outside 10-55, 2 heart rates outside 40-180. | Keep as baseline, but document duplicate-heavy grain. |
| Mendeley `ckhgtgdg4f` | `data/external/ckhgtgdg4f_springer.csv` | 6103 raw, 6058 after documented outlier rules | low / mid / high | No duplicates or missing values. Contains 45 documented outlier rows before cleaning. Drop `Patient ID` and `Name`. `Blood Glucose(Fasting hour-mg/dl)` values look like mmol/L despite mg/dL label. `Blood Glucose(HbA1c)` looks like IFCC mmol/mol, not percent. | Best next candidate: include with cleaning and column mapping. |
| Mendeley `p5w98dvbbk` | `data/external/p5w98dvbbk_dataset_updated.csv` | 1205 | Low / High plus 18 missing labels | Same six core fields plus BMI/history factors. Has 18 exact duplicates, 53 missing cells, 18 ages outside 10-55 including one 325, one BMI = 0. Binary target, no mid class. | Separate binary model, or use only if we redesign target. Do not merge directly into 3-class model. |
| MOTHER_CARE1 `fvdt76zwhn` | `data/external/mother_care1_maternal_dataset.csv` | 609 | high_risk Yes / No | Binary target. README warns `anc_visits` is near-deterministic with label. Includes post-delivery/neonatal fields with large missingness. | Research only / separate antenatal model after strict leakage removal. |
| MOTHER_CARE2 `fcbzdg5gjm` | `data/external/mother_care2_clinical_maternal_dataset.csv` | 111 | high_risk Yes / No | Small and highly imbalanced. README says derived flags almost compose the target; post-delivery variables must be excluded for antenatal prediction. | Research only. Not suitable for current model. |
| High-Risk Pregnancy `8k9pvpmykk` / Kaggle mirror | `data/external/high_risk_pregnancy_book2.xlsx` | 998 | Yes / No | Excel has a title row, mixed English/Bengali columns, name column, many string-with-unit fields, 2725 missing cells. License is CC BY-NC-SA 4.0. | Research only unless license and schema are acceptable. Not for product training now. |

## Recommended next step

Use `ckhgtgdg4f` first, but only after a deterministic preparation step:

1. Remove the documented 45 raw outliers:
   - `Age > 100`
   - `Body Temperature(F) < 95` or `> 105`
   - `Diastolic Blood Pressure(mm Hg) < 50`
2. Drop `Patient ID` and `Name`.
3. Map columns:
   - `Age` -> `Age`
   - `Systolic Blood Pressure(mm Hg)` -> `SystolicBP`
   - `Diastolic Blood Pressure(mm Hg)` -> `DiastolicBP`
   - `Blood Glucose(Fasting hour-mg/dl)` -> `BS`
   - `Body Temperature(F)` -> `BodyTemp`
   - `Heart rate(bpm)` -> `HeartRate`
   - `Status` -> `RiskLevel`
4. Keep `BMI` and `Blood Glucose(HbA1c)` out of the current Streamlit form until we add new fields intentionally.
5. Train a combined model and evaluate against:
   - holdout from original UCI
   - holdout from cleaned `ckhgtgdg4f`
   - source-level split, so the model cannot hide dataset shift.

## Source URLs

- UCI Maternal Health Risk: https://uci-ics-mlr-prod.aws.uci.edu/dataset/863/maternal%2Bhealth%2Brisk
- Mendeley `ckhgtgdg4f`: https://data.mendeley.com/datasets/ckhgtgdg4f/1
- Mendeley `p5w98dvbbk`: https://data.mendeley.com/datasets/p5w98dvbbk/1
- MOTHER_CARE1: https://data.mendeley.com/datasets/fvdt76zwhn/2
- MOTHER_CARE2: https://data.mendeley.com/datasets/fcbzdg5gjm/2
- High-Risk Pregnancy `8k9pvpmykk`: https://data.mendeley.com/datasets/8k9pvpmykk/1

## Audit reports

- `reports/dataset_audits/uci_maternal_health_risk_audit.md`
- `reports/dataset_audits/ckhgtgdg4f_springer_audit.md`
- `reports/dataset_audits/p5w98dvbbk_dataset_updated_audit.md`
- `reports/dataset_audits/mother_care1_maternal_dataset_audit.md`
- `reports/dataset_audits/mother_care2_clinical_maternal_dataset_audit.md`
- `reports/dataset_audits/high_risk_pregnancy_book2_header1_audit.md`
