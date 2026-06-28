"""Machine-learning bearing prognostics for the motor digital twin.

Shared feature schema (features.py) feeds three models trained by train.py:
RUL regression, failure-stage classification, and anomaly detection. Data comes
either from the synthetic run-to-failure generator (synthetic.py) or the real
NASA IMS dataset (ims_loader.py).
"""
from .features import extract, to_vector, FEATURE_NAMES, BearingGeometry
from .predict import (predict_from_features, predict_from_signal,
                      models_available)

__all__ = [
    "extract", "to_vector", "FEATURE_NAMES", "BearingGeometry",
    "predict_from_features", "predict_from_signal", "models_available",
]
