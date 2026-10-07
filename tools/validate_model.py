"""Reproduce saved metrics and measure duplicate leakage without replacing the model.

Run from the repository root: python tools/validate_model.py
"""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_model import CLASS_NAMES, CLASS_TO_ID, FEATURES, TARGET, load_dataset


def scores(y, prediction):
    return {
        'accuracy': float(accuracy_score(y, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
        'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }


def main():
    data = load_dataset()
    bundle = joblib.load(ROOT / 'models/maternal_risk_xgboost.joblib')
    model = bundle['model']
    x, y = data[FEATURES], data[TARGET].map(CLASS_TO_ID)
    train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=y)
    groups = pd.util.hash_pandas_object(x, index=False)
    overlap = groups.iloc[test].isin(set(groups.iloc[train]))
    prediction = model.predict(x.iloc[test])
    report = {
        'dataset_rows': len(data), 'unique_feature_profiles': int(groups.nunique()),
        'duplicate_rows': int(data.duplicated().sum()),
        'conflicting_feature_profiles': int((data.groupby(FEATURES)[TARGET].nunique() > 1).sum()),
        'class_order': CLASS_NAMES,
        'saved_model_holdout': scores(y.iloc[test], prediction),
        'saved_metrics_match': bool(np.isclose(accuracy_score(y.iloc[test], prediction), bundle['metrics']['accuracy'])),
        'holdout_rows': len(test),
        'holdout_rows_with_training_feature_match': int(overlap.sum()),
        'holdout_seen_profiles': scores(y.iloc[test][overlap.to_numpy()], prediction[overlap.to_numpy()]),
        'holdout_unseen_profiles': scores(y.iloc[test][~overlap.to_numpy()], prediction[~overlap.to_numpy()]),
        'group_holdouts': [],
    }
    for seed in [42, 43, 44, 45, 46]:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(x, y, groups))
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
        candidate = clone(model)
        candidate.fit(x.iloc[train_idx], y.iloc[train_idx])
        result = scores(y.iloc[test_idx], candidate.predict(x.iloc[test_idx]))
        result.update(seed=seed, train_rows=len(train_idx), test_rows=len(test_idx), feature_overlap=0)
        report['group_holdouts'].append(result)
    report['group_holdout_mean_accuracy'] = float(np.mean([r['accuracy'] for r in report['group_holdouts']]))
    report['group_holdout_mean_balanced_accuracy'] = float(np.mean([r['balanced_accuracy'] for r in report['group_holdouts']]))
    report['limitations'] = [
        'Group holdouts keep identical six-feature profiles together; patient identifiers are unavailable.',
        'Five overlapping random holdouts are a sensitivity check, not independent external validation.',
        'The target contains low/mid/high maternal risk labels, not preeclampsia outcomes.',
        'The shipped model and its metrics were not replaced.',
    ]
    target = ROOT / 'reports/local_validation/model_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
