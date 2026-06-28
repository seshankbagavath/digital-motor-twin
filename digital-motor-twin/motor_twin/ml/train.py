"""Train bearing-prognostics models and persist them to models/.

Three models share one feature schema (motor_twin/ml/features.py):
  * RUL regressor      -- GradientBoosting, predicts remaining files-to-failure
  * Failure classifier -- RandomForest, P(bearing in failure stage)
  * Anomaly detector   -- IsolationForest, unsupervised novelty score

Data source is pluggable:
  --source synthetic     (default) physics-grounded run-to-failure generator
  --source ims --path P  real NASA IMS test-set folder P

Run-grouped splitting prevents leakage between train and test bearings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              IsolationForest)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from .features import FEATURE_NAMES
from . import synthetic, ims_loader

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
FAILURE_RUL_FRAC = 0.15   # last 15% of life is labelled "failure stage"


def _load_synthetic():
    X, rul, sev, run_id = synthetic.make_dataset(n_runs=6, n_files=120)
    return X, rul, sev, run_id


def _load_ims(path: str):
    # Use each available channel as an independent run.
    import numpy as np
    sample = np.loadtxt(ims_loader.list_files(path)[0])
    n_ch = sample.shape[1] if sample.ndim == 2 else 1
    Xs, ruls, ids = [], [], []
    for ch in range(n_ch):
        X, rul, _ = ims_loader.load_ims_run(path, channel=ch)
        Xs.append(X); ruls.append(rul); ids.append(np.full(len(rul), ch))
    X = np.vstack(Xs); rul = np.concatenate(ruls); run_id = np.concatenate(ids)
    # severity proxy from normalised RUL within each run
    sev = np.zeros_like(rul)
    for r in np.unique(run_id):
        m = run_id == r
        sev[m] = 1.0 - rul[m] / rul[m].max()
    return X, rul, sev, run_id


def train(source: str = "synthetic", path: str | None = None) -> dict:
    if source == "ims":
        if not path:
            raise SystemExit("--source ims requires --path to a test-set folder")
        X, rul, sev, run_id = _load_ims(path)
    else:
        X, rul, sev, run_id = _load_synthetic()

    # Group-aware split: hold out the last run entirely.
    runs = np.unique(run_id)
    test_runs = runs[-1:]
    test = np.isin(run_id, test_runs)
    train_m = ~test

    # Failure-stage label from RUL fraction within each run
    y_fail = np.zeros(len(rul), dtype=int)
    for r in runs:
        m = run_id == r
        thr = rul[m].max() * FAILURE_RUL_FRAC
        y_fail[m] = (rul[m] <= thr).astype(int)

    # --- RUL regressor -------------------------------------------------
    rul_model = Pipeline([
        ("scale", StandardScaler()),
        ("gb", GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                         learning_rate=0.05, subsample=0.9)),
    ])
    rul_model.fit(X[train_m], rul[train_m])
    rul_pred = rul_model.predict(X[test])
    rul_mae = mean_absolute_error(rul[test], rul_pred)
    rul_r2 = r2_score(rul[test], rul_pred)

    # --- Failure classifier -------------------------------------------
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                      class_weight="balanced", random_state=0)),
    ])
    clf.fit(X[train_m], y_fail[train_m])
    proba = clf.predict_proba(X[test])[:, 1]
    auc = roc_auc_score(y_fail[test], proba) if len(np.unique(y_fail[test])) > 1 else float("nan")

    # --- Anomaly detector (fit on healthy data only) -------------------
    healthy = train_m & (sev < 0.1)
    iso = Pipeline([
        ("scale", StandardScaler()),
        ("iso", IsolationForest(n_estimators=200, contamination=0.05, random_state=0)),
    ])
    iso.fit(X[healthy])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rul_model, MODEL_DIR / "rul_regressor.joblib")
    joblib.dump(clf, MODEL_DIR / "failure_classifier.joblib")
    joblib.dump(iso, MODEL_DIR / "anomaly_detector.joblib")

    metrics = {
        "source": source,
        "n_samples": int(len(rul)),
        "n_train": int(train_m.sum()),
        "n_test": int(test.sum()),
        "feature_names": FEATURE_NAMES,
        "rul_mae_files": round(float(rul_mae), 3),
        "rul_r2": round(float(rul_r2), 3),
        "failure_auc": round(float(auc), 3),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train bearing prognostics models")
    ap.add_argument("--source", choices=["synthetic", "ims"], default="synthetic")
    ap.add_argument("--path", help="IMS test-set folder (for --source ims)")
    args = ap.parse_args()
    m = train(args.source, args.path)
    print(json.dumps(m, indent=2))
