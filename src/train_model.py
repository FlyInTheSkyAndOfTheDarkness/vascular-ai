from __future__ import annotations

import json
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = PROJECT_ROOT / "maternal+health+risk.zip"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_FILE = DATA_DIR / "Maternal Health Risk Data Set.csv"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
CLASS_NAMES = ["low risk", "mid risk", "high risk"]
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}


def ensure_dataset() -> Path:
    """Extract the provided UCI zip if the CSV is not present yet."""
    if DATA_FILE.exists():
        return DATA_FILE

    if not DATA_ZIP.exists():
        raise FileNotFoundError(
            f"Dataset was not found. Put the UCI zip at {DATA_ZIP} or the CSV at {DATA_FILE}."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP) as archive:
        archive.extractall(DATA_DIR)

    if not DATA_FILE.exists():
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files were extracted from {DATA_ZIP}.")
        return csv_files[0]

    return DATA_FILE


def load_dataset() -> pd.DataFrame:
    data_path = ensure_dataset()
    df = pd.read_csv(data_path)

    expected_columns = set(FEATURES + [TARGET])
    missing_columns = sorted(expected_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[FEATURES + [TARGET]].copy()
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET].isin(CLASS_TO_ID)]

    if df.empty:
        raise ValueError("Dataset is empty after validation.")

    return df


def train() -> dict:
    df = load_dataset()
    x = df[FEATURES]
    y = df[TARGET].map(CLASS_TO_ID)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_NAMES),
        n_estimators=350,
        learning_rate=0.045,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.2,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "class_order": CLASS_NAMES,
        "feature_names": FEATURES,
        "rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    background = x_train.sample(n=min(100, len(x_train)), random_state=42)
    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "class_names": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "metrics": metrics,
        "shap_background": background,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "maternal_risk_xgboost.joblib"
    metrics_path = MODELS_DIR / "metrics.json"
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"model_path": model_path, "metrics_path": metrics_path, "metrics": metrics}


if __name__ == "__main__":
    result = train()
    print(f"Saved model: {result['model_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Balanced accuracy: {result['metrics']['balanced_accuracy']:.4f}")
