"""Inference wrapper: load trained models and score a vibration window or features.

Lazily loads the joblib artifacts in models/ once. Degrades gracefully if the
models haven't been trained yet (returns None fields) so the rest of the twin
keeps running.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib

from .features import extract, to_vector, FEATURE_NAMES

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@lru_cache(maxsize=1)
def _models():
    try:
        return (
            joblib.load(MODEL_DIR / "rul_regressor.joblib"),
            joblib.load(MODEL_DIR / "failure_classifier.joblib"),
            joblib.load(MODEL_DIR / "anomaly_detector.joblib"),
        )
    except FileNotFoundError:
        return None


def models_available() -> bool:
    return _models() is not None


def predict_from_features(feats: dict[str, float],
                          file_interval_min: float = 10.0) -> dict:
    """Score a feature dict. ``file_interval_min`` converts RUL-in-files to hours
    (IMS records a file roughly every 10 min)."""
    m = _models()
    if m is None:
        return {"available": False}
    rul_model, clf, iso = m
    x = to_vector(feats).reshape(1, -1)

    rul_files = float(rul_model.predict(x)[0])
    rul_files = max(rul_files, 0.0)
    fail_prob = float(clf.predict_proba(x)[0, 1])
    # IsolationForest: decision_function > 0 normal, < 0 anomalous.
    anomaly_score = float(-iso.named_steps["iso"].decision_function(
        iso.named_steps["scale"].transform(x))[0])

    return {
        "available": True,
        "rul_files": rul_files,
        "rul_hours": rul_files * file_interval_min / 60.0,
        "failure_prob": fail_prob,
        "anomaly_score": anomaly_score,
        "is_anomaly": bool(iso.predict(x)[0] == -1),
    }


def predict_from_signal(signal: np.ndarray, fs: float = 20000.0,
                        shaft_rpm: float = 2000.0, **kw) -> dict:
    """Extract features from a raw window, then score."""
    return predict_from_features(extract(signal, fs=fs, shaft_rpm=shaft_rpm), **kw)
