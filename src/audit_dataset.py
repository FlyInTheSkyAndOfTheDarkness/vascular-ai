from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "dataset_audits"

SENTINELS = {"", "unknown", "not recorded", "not_recorded", "n/a", "na", "none", "null", "-", "--", "?"}

TARGET_CANDIDATES = {
    "risklevel",
    "risk_level",
    "risk",
    "high_risk",
    "risk_status",
    "risk_category",
    "fetal_health",
    "nsp",
    "status",
}

TARGET_TEXT_MARKERS = [
    "ঝুকিপূর্ণ",
]

LEAKAGE_PATTERNS = [
    "risk",
    "flag",
    "outcome",
    "label",
    "delivery",
    "birth",
    "apgar",
    "neonatal",
    "postnatal",
    "low_birth_weight",
    "mortality",
]

CLINICAL_RANGE_RULES = [
    (("gestational_age", "gestational_age_weeks"), (4, 45)),
    (("mother_age", "patient_age", "age"), (10, 55)),
    (("systolicbp", "systolic_bp", "systolic_blood_pressure", "sbp"), (70, 220)),
    (("diastolicbp", "diastolic_bp", "diastolic_blood_pressure", "diastolic", "dbp"), (40, 140)),
    (("bodytemp", "body_temp", "body_temperature", "temperature"), (94, 106)),
    (("heartrate", "heart_rate", "heart_rate_bpm"), (40, 180)),
    (("bmi",), (10, 70)),
    (("hemoglobin", "haemoglobin"), (4, 20)),
    (("blood_sugar", "fasting_glucose", "glucose", "bs"), (2, 35)),
]


@dataclass
class AuditResult:
    summary: dict[str, Any]
    columns: list[dict[str, Any]]
    numeric_ranges: list[dict[str, Any]]
    target_profiles: list[dict[str, Any]]
    duplicate_profile: dict[str, Any]
    leakage_candidates: list[str]
    warnings: list[str]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9а-яё]+", "_", value, flags=re.IGNORECASE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "dataset"


def normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def clinical_range_for(column: str, numeric: pd.Series) -> tuple[float, float] | None:
    normalized = normalize_column(column)
    if "hba1c" in normalized:
        median = numeric.median()
        if pd.notna(median) and median > 20:
            return (20, 140)
        return (3, 16)

    for aliases, value_range in CLINICAL_RANGE_RULES:
        if any(normalized == alias or alias in normalized for alias in aliases):
            return value_range
    return None


def is_identifier_like(column: str) -> bool:
    normalized = normalize_column(column)
    tokens = set(normalized.split("_"))
    if normalized in {"id", "name", "full_name", "patient_name"}:
        return True
    if normalized.endswith("_id"):
        return True
    return bool(tokens.intersection({"patient", "record", "subject", "participant"}))


def read_dataset(path: Path, header_row: int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header_row)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, header=header_row)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV, XLSX, XLS, or JSON.")


def sentinel_rate(series: pd.Series) -> float:
    if series.dtype.kind in {"i", "u", "f", "b"}:
        return 0.0
    normalized = series.astype(str).str.strip().str.lower()
    return float(normalized.isin(SENTINELS).mean())


def column_profiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    profiles = []
    for column in df.columns:
        series = df[column]
        profiles.append(
            {
                "column": column,
                "normalized": normalize_column(column),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": round(float(series.isna().mean()), 4),
                "sentinel_rate": round(sentinel_rate(series), 4),
                "distinct_count": int(series.nunique(dropna=True)),
                "sample_values": [str(value) for value in series.dropna().head(5).tolist()],
            }
        )
    return profiles


def numeric_profiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    profiles = []
    normalized_columns = {column: normalize_column(column) for column in df.columns}

    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        iqr_low = q1 - 1.5 * iqr
        iqr_high = q3 + 1.5 * iqr
        normalized = normalized_columns[column]

        matching_range = clinical_range_for(column, numeric)

        outside_clinical_range = None
        if matching_range:
            low, high = matching_range
            outside_clinical_range = int(((numeric < low) | (numeric > high)).sum())

        profiles.append(
            {
                "column": column,
                "valid_numeric_count": int(numeric.notna().sum()),
                "min": round(float(numeric.min()), 4),
                "p25": round(float(q1), 4),
                "median": round(float(numeric.median()), 4),
                "p75": round(float(q3), 4),
                "max": round(float(numeric.max()), 4),
                "iqr_outliers": int(((numeric < iqr_low) | (numeric > iqr_high)).sum()),
                "clinical_range": matching_range,
                "outside_clinical_range": outside_clinical_range,
            }
        )

    return profiles


def target_profiles(df: pd.DataFrame) -> list[dict[str, Any]]:
    profiles = []
    for column in df.columns:
        normalized = normalize_column(column)
        lower_column = str(column).strip().lower()
        if normalized not in TARGET_CANDIDATES and not any(marker in lower_column for marker in TARGET_TEXT_MARKERS):
            continue
        counts = df[column].astype(str).str.strip().value_counts(dropna=False).head(20)
        profiles.append(
            {
                "column": column,
                "distinct_count": int(df[column].nunique(dropna=False)),
                "top_values": {str(key): int(value) for key, value in counts.items()},
            }
        )
    return profiles


def duplicate_profile(df: pd.DataFrame) -> dict[str, Any]:
    exact_duplicates = int(df.duplicated().sum())
    profile: dict[str, Any] = {
        "exact_duplicate_rows": exact_duplicates,
        "exact_duplicate_rate": round(exact_duplicates / len(df), 4) if len(df) else 0,
    }

    id_like_columns = [
        column
        for column in df.columns
        if is_identifier_like(column)
    ]
    key_checks = {}
    for column in id_like_columns:
        duplicates = int(df[column].duplicated(keep=False).sum())
        key_checks[column] = {
            "duplicated_rows": duplicates,
            "duplicated_rate": round(duplicates / len(df), 4) if len(df) else 0,
            "distinct_count": int(df[column].nunique(dropna=True)),
        }

    profile["id_like_key_checks"] = key_checks
    return profile


def leakage_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []
    for column in df.columns:
        normalized = normalize_column(column)
        if normalized in TARGET_CANDIDATES:
            continue
        if any(pattern in normalized for pattern in LEAKAGE_PATTERNS):
            candidates.append(column)
    return candidates


def build_warnings(result: AuditResult) -> list[str]:
    warnings = []
    if result.duplicate_profile["exact_duplicate_rate"] > 0.01:
        warnings.append("Exact duplicate rate is above 1%; verify dataset grain before training.")
    if not result.target_profiles:
        warnings.append("No obvious target column was detected.")
    if result.leakage_candidates:
        warnings.append("Potential leakage columns detected; review before modeling.")

    for profile in result.columns:
        if profile["missing_rate"] > 0.2:
            warnings.append(f"Column `{profile['column']}` has more than 20% missing values.")
        if profile["sentinel_rate"] > 0.05:
            warnings.append(f"Column `{profile['column']}` has more than 5% sentinel text values.")

    for profile in result.numeric_ranges:
        if profile["outside_clinical_range"]:
            warnings.append(
                f"Column `{profile['column']}` has {profile['outside_clinical_range']} values outside expected clinical range."
            )
        normalized = normalize_column(profile["column"])
        if "hba1c" in normalized and profile["median"] > 20:
            warnings.append(
                f"Column `{profile['column']}` looks like HbA1c in IFCC mmol/mol, not percent; confirm units before modeling."
            )
        if ("mg_dl" in normalized or "mgdl" in normalized) and "glucose" in normalized and profile["median"] < 30:
            warnings.append(
                f"Column `{profile['column']}` is named mg/dL but values look like mmol/L; confirm units before mapping to `BS`."
            )

    return warnings


def audit(path: Path, name: str, header_row: int = 0) -> AuditResult:
    df = read_dataset(path, header_row)
    result = AuditResult(
        summary={
            "name": name,
            "source_path": str(path),
            "header_row": header_row,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        columns=column_profiles(df),
        numeric_ranges=numeric_profiles(df),
        target_profiles=target_profiles(df),
        duplicate_profile=duplicate_profile(df),
        leakage_candidates=leakage_candidates(df),
        warnings=[],
    )
    result.warnings = build_warnings(result)
    return result


def write_reports(result: AuditResult, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    base_name = slugify(result.summary["name"])
    json_path = report_dir / f"{base_name}_audit.json"
    md_path = report_dir / f"{base_name}_audit.md"

    payload = {
        "summary": result.summary,
        "columns": result.columns,
        "numeric_ranges": result.numeric_ranges,
        "target_profiles": result.target_profiles,
        "duplicate_profile": result.duplicate_profile,
        "leakage_candidates": result.leakage_candidates,
        "warnings": result.warnings,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def render_markdown(result: AuditResult) -> str:
    lines = [
        f"# Dataset audit: {result.summary['name']}",
        "",
        "## Summary",
        "",
        f"- Source path: `{result.summary['source_path']}`",
        f"- Header row: {result.summary['header_row']}",
        f"- Rows: {result.summary['rows']}",
        f"- Columns: {result.summary['columns']}",
        "",
        "## Warnings",
        "",
    ]

    if result.warnings:
        lines.extend([f"- {warning}" for warning in result.warnings])
    else:
        lines.append("- No immediate warnings detected.")

    lines.extend(["", "## Target candidates", ""])
    if result.target_profiles:
        for target in result.target_profiles:
            lines.append(f"- `{target['column']}`: {target['distinct_count']} distinct values")
            for value, count in target["top_values"].items():
                lines.append(f"  - `{value}`: {count}")
    else:
        lines.append("- No target candidates detected.")

    lines.extend(["", "## Duplicate profile", ""])
    lines.append(f"- Exact duplicate rows: {result.duplicate_profile['exact_duplicate_rows']}")
    lines.append(f"- Exact duplicate rate: {result.duplicate_profile['exact_duplicate_rate']:.2%}")
    if result.duplicate_profile["id_like_key_checks"]:
        for column, key_check in result.duplicate_profile["id_like_key_checks"].items():
            lines.append(
                f"- `{column}` duplicated rows: {key_check['duplicated_rows']} "
                f"({key_check['duplicated_rate']:.2%})"
            )

    lines.extend(["", "## Potential leakage columns", ""])
    if result.leakage_candidates:
        lines.extend([f"- `{column}`" for column in result.leakage_candidates])
    else:
        lines.append("- None detected by keyword scan.")

    lines.extend(["", "## Numeric ranges", ""])
    lines.append("| Column | Min | Median | Max | IQR outliers | Clinical range | Outside range |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for profile in result.numeric_ranges:
        clinical_range = profile["clinical_range"] or ""
        outside = "" if profile["outside_clinical_range"] is None else profile["outside_clinical_range"]
        lines.append(
            f"| `{profile['column']}` | {profile['min']} | {profile['median']} | {profile['max']} | "
            f"{profile['iqr_outliers']} | {clinical_range} | {outside} |"
        )

    lines.extend(["", "## Column profile", ""])
    lines.append("| Column | Type | Missing | Sentinel text | Distinct |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for profile in result.columns:
        lines.append(
            f"| `{profile['column']}` | `{profile['dtype']}` | {profile['missing_rate']:.2%} | "
            f"{profile['sentinel_rate']:.2%} | {profile['distinct_count']} |"
        )

    lines.extend(
        [
            "",
            "## Manual review before inclusion",
            "",
            "- Confirm license and permitted use.",
            "- Confirm one-row grain and whether repeat patient visits exist.",
            "- Confirm target definition and label source.",
            "- Confirm units, especially glucose and body temperature.",
            "- Remove post-outcome, derived-flag, and direct-label leakage columns.",
            "- Decide whether this dataset joins the main model or needs a separate model.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile an external maternal-health dataset before model inclusion.")
    parser.add_argument("--input", required=True, type=Path, help="Path to CSV, XLSX, XLS, or JSON dataset.")
    parser.add_argument("--name", default=None, help="Human-readable dataset name for report files.")
    parser.add_argument("--header-row", type=int, default=0, help="Zero-based row index to use as column names.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for audit reports.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    path = args.input.resolve()
    name = args.name or path.stem
    result = audit(path, name, args.header_row)
    json_path, md_path = write_reports(result, args.report_dir)
    print(f"Saved JSON audit: {json_path}")
    print(f"Saved Markdown audit: {md_path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
