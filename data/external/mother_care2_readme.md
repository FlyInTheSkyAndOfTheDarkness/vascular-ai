# Clinical Maternal and Neonatal Dataset for High-Risk Pregnancy Assessment in Bangladesh

## Overview

This dataset contains **111 de-identified maternal and neonatal clinical records** retrospectively compiled from routine antenatal, intrapartum, and postnatal hospital documentation maintained at a healthcare facility in Bangladesh.

The dataset was created to support research on maternal and pregnancy risk assessment, obstetric epidemiology, clinical decision-support research, and machine learning applications. It includes maternal demographic information, obstetric history, clinical measurements, delivery information, neonatal characteristics and outcomes, and clinically derived pregnancy risk indicators.

> **Note on scope:** The published composite `high_risk` label includes `low_birth_weight`, which is determined only after delivery. Therefore, `high_risk` should be interpreted as a **retrospective composite pregnancy-risk classification** rather than a strictly antenatal (pre-delivery) prediction outcome. See "Label Derivation" and "Data Quality Notes" below for details and guidance on defining antenatal prediction studies.

---

## Dataset Information

| Item | Value |
|------|-------|
| File | clinical_maternal_dataset.csv |
| Records | 111 |
| Variables | 21 |
| Format | CSV |
| Language | English |
| Country | Bangladesh |

---

## Data Source

The dataset was retrospectively compiled from routine antenatal, intrapartum, and postnatal hospital records maintained at a hospital in Bangladesh. Data were manually transcribed into a structured research dataset.

Direct personally identifiable information (PII), including patient names, addresses, and contact information, was removed before dataset preparation. The published `patient_id` values are study-specific identifiers created for this dataset and do not contain direct identifying information.

Original recorded values were preserved in the `_raw` variables. No source values were intentionally altered during transcription; two exceptions in `gravida_filled` are documented under "Data Quality Notes."

---

## Variables

The dataset contains variables describing:

- Maternal demographics
- Obstetric history
- Clinical measurements
- Delivery information
- Neonatal characteristics and outcomes
- Derived clinical risk indicators
- Overall composite high-risk status

A detailed description of every variable, including data type, unit, observed range, and missing-value count, is provided in the accompanying **Data Dictionary**.

---

## Summary Statistics

### Dataset Composition

- Total records: **111**
- High-risk pregnancies: **89 (80.2%)**
- Low-risk pregnancies: **22 (19.8%)**

### Individual Risk Flag Prevalence

| Risk Indicator | Positive Cases |
|---------------|---------------:|
| Anemia | 59 |
| Preterm pregnancy | 29 |
| Maternal age risk | 26 |
| Low birth weight | 21 |
| Hypertension | 12 |

---

## Label Derivation (`high_risk`)

The variable `high_risk` is a binary outcome (`Yes` / `No`) derived during dataset preparation from five clinically derived risk indicators available in the dataset:

- `age_risk`
- `anemia_flag`
- `hypertension_flag`
- `preterm_risk`
- `low_birth_weight`

These derived indicators were generated during dataset preparation and were not directly recorded in the original hospital records.

The original documentation describing the exact clinical thresholds and derivation procedure used to generate these indicators was not available at the time of dataset publication. Consequently, the precise derivation rules cannot be fully reproduced from the published documentation alone.

Users should therefore treat these variables as published derived clinical indicators rather than attempt to reconstruct them solely from the corresponding raw clinical measurements. A small number of records contain derived values that cannot be fully explained by the available published variables; these cases are described in the Data Quality Notes section.

### Important Note for Machine Learning Users

A post-hoc statistical review showed that the `high_risk` label is strongly associated with these five derived risk indicators:

- **88 of 89** high-risk records (98.9%) contain at least one risk flag marked **"Yes"**.
- **22 of 22** low-risk records (100%) contain no positive risk flags.

One exception (patient **P092**) was labeled as high-risk despite having no explicitly positive risk flag; several risk-flag values for this record were also missing. The available published variables do not fully explain the `high_risk` label for P092, and the reason for this discrepancy could not be confirmed from the structured dataset.

Because of this strong association, researchers developing predictive models for `high_risk` should **exclude the five derived risk-flag variables from the predictor set**, as including them would introduce **target leakage** rather than genuine predictive learning.

The risk-flag variables remain useful for descriptive analysis and studies investigating the composition of the overall risk label.

### Guidance for Antenatal Prediction Studies

Because `low_birth_weight`, `delivery_mode`, `baby_sex`, `birth_weight_grams`, `apgar_1min`, and `apgar_5min` are only known after delivery, researchers aiming to build a strictly **antenatal** prediction model should:

1. Define a clear prediction time point (e.g., at admission or at a specific gestational week).
2. Exclude all post-delivery variables listed above from the predictor set.
3. Optionally redefine the outcome using only pre-delivery indicators (`age_risk`, `anemia_flag`, `hypertension_flag`, `preterm_risk`) if a purely antenatal outcome is required.

---

## Data Quality Notes

### 1. Gravida and Parity

Both original (`*_raw`) and cleaned (`*_filled`) versions are provided.

- Original values preserve the hospital records.
- Filled values were generated during dataset preparation, primarily by imputing missing raw values.

Differences between raw and filled values exist in:

- **Gravida: 7 records** — 5 involved imputation of missing raw values; **2 involved modification of non-missing raw values** (patients P071 and P099, both changed from 3 to 2). The original rationale for these two corrections could not be reconstructed from the available documentation.
- **Parity: 25 records** — all 25 involved imputation of missing raw values; no non-missing raw values were modified.

Researchers requiring the original recorded values should use the `_raw` variables. Note that `parity_filled` still contains **15 missing observations** and should not be treated as fully complete.

---

### 2. Logical Consistency

Under the assumption that parity represents births occurring **before** the current pregnancy, standard obstetric convention expects:

> Gravida ≥ Parity + 1

Among **96** records with valid filled values:

- 94 (97.9%) satisfy this relationship.
- 2 (2.1%) have Gravida = Parity (patients **P078** and **P107**).

No records have Gravida < Parity. If parity was recorded or updated to include the current pregnancy/delivery in some cases, Gravida = Parity may be clinically valid rather than an error. These two records were retained as originally documented rather than reclassified.

---

### 3. Delivery Mode

One record (P095) contains the value **"Preterm"** in `delivery_mode`.

This is not a valid delivery mode and appears to be an original recording error.

The value has been intentionally preserved rather than corrected. Users are advised to treat this value as missing or invalid during analysis.

---

### 4. Baby Sex

Two records (P068 and P086) contain the value **"Unknown"**.

This indicates that neonatal sex was explicitly recorded as unknown rather than left blank.

---

### 5. APGAR Scores

Approximately **16–17%** of APGAR measurements are missing.

One record (P074) has an APGAR score of **0** at one minute with a missing five-minute score. The clinical meaning of this combination cannot be confirmed from the available variables and should not be interpreted as neonatal death or any other specific outcome without supporting source documentation.

---

### 6. Missing Clinical Measurements

Several routinely collected measurements contain missing values, including:

- systolic_bp_mmhg
- diastolic_bp_mmhg
- hemoglobin_g_dl
- gestational_age_weeks

These missing values arose from unavailable or unrecorded information in routine hospital documentation. No assumption should be made that this missingness is completely random; users performing statistical analysis are encouraged to assess the missingness pattern independently.

---

### 7. Unexplained Derived-Flag Values

Two records show derived risk-flag values that do not appear fully reproducible from the corresponding published raw measurements using common clinical thresholds:

- **P055**: `mother_age_years` is missing, yet `age_risk = Yes`.
- **P080**: blood pressure is recorded as 100/70 mmHg, yet `hypertension_flag = Yes`.

These values were retained from the originally prepared dataset and should be interpreted cautiously. Users performing sensitivity analyses may wish to exclude or separately flag these records.

---

## Missing Values

Blank cells indicate that the corresponding information was unavailable or not recorded in the original hospital records.

The value **"Unknown"** in `baby_sex` represents an explicitly documented unknown value and should not be interpreted as a missing observation.

---

## Ethical Statement

This dataset was retrospectively compiled from routine hospital records with prior administrative permission from the concerned hospital authority (Civil Surgeon).

All direct personal identifiers (names, addresses, telephone numbers, and other direct identifiers) were removed before publication. The published `patient_id` values are study-specific identifiers and do not contain direct identifying information; the dataset is therefore described as **de-identified**.

---

## Intended Use

This dataset is intended for research and educational purposes, including:

- Maternal and neonatal health research
- Pregnancy risk assessment (retrospective and, with appropriate variable restriction, antenatal prediction studies)
- Clinical epidemiology
- Obstetric risk factor analysis
- Machine learning
- Explainable artificial intelligence (XAI)
- Statistical modeling
- Healthcare data science education

This dataset is **not intended for direct clinical decision-making** and has not undergone external clinical validation.

---

## Repository Contents

- clinical_maternal_dataset.csv
- README.md
- data_dictionary.pdf

---

## License

This dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

---

## Citation

Please cite this dataset if used in research or publications.

Citation details will be available after publication on the data repository.

---

## Contact

| Name | Department | Email |
|------|------------|-------|
| Abdullah Al Rifat | Dept. of CSE, Daffodil International University, Bangladesh | al-rifat2305101694@diu.edu.bd |
| Abdul Momin Rahat | Dept. of CSE, Daffodil International University, Bangladesh | rahat2305101544@diu.edu.bd |
| Obaidul Haque Buyan | Dept. of CSE, Daffodil International University, Bangladesh | obaidul2305101350@diu.edu.bd |
