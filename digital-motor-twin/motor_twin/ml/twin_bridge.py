"""Bridge the physics twin to the ML prognostics models.

The dynamic twin tracks scalar health state (bearing condition, accumulated
wear, speed). The ML models consume a *vibration window*. This module
synthesizes a representative window for the twin's current health using the
same generator the models were trained on, extracts features, and returns a
prediction -- closing the loop from physics state to learned prognostics.
"""
from __future__ import annotations

import numpy as np

from .synthetic import _window
from .features import extract
from .predict import predict_from_features, models_available


def predict_for_state(state: dict, op_bearing_condition: float,
                      op_lubrication: float = 1.0,
                      seed: int | None = None) -> dict:
    """Score the twin's current health with the ML models.

    Effective defect severity combines the user's bearing-condition setting,
    poor lubrication, and physics-accumulated wear.
    """
    if not models_available():
        return {"available": False}

    severity = (1.0 - op_bearing_condition)
    severity += 0.5 * (1.0 - op_lubrication)
    severity += state.get("bearing_wear", 0.0)
    severity = float(np.clip(severity, 0.0, 1.0))

    # The prognostic models are calibrated to the IMS test bearing, whose rig
    # runs at a fixed 2000 rpm. We synthesize and featurize at that reference
    # speed so the window matches the training distribution; the motor's own
    # electrical speed drives the physics twin, not the bearing-vibration model.
    ims_rpm = 2000.0
    rng = np.random.default_rng(seed)
    sig = _window(severity, fs=20000.0, dur=0.1, shaft_rpm=ims_rpm,
                  defect="bpfo", rng=rng)
    feats = extract(sig, fs=20000.0, shaft_rpm=ims_rpm)
    pred = predict_from_features(feats)
    pred["severity"] = severity
    return pred
