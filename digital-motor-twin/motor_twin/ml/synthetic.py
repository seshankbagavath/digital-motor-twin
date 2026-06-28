"""Physics-grounded synthetic run-to-failure data for bearing prognostics.

Generates raw vibration windows whose statistics evolve like a real degrading
bearing: a healthy bearing shows low-amplitude broadband noise; as a localized
defect develops, periodic impulses appear at a bearing defect frequency
(modulated by shaft rotation), driving up RMS, kurtosis and crest factor, and
finally broadband energy near failure.

The windows are passed through the SAME `features.extract` used for real IMS
data, so models trained here transfer to the NASA dataset unchanged. This lets
the full pipeline train and validate offline; swap in `ims_loader` to retrain
on real data.
"""
from __future__ import annotations

import numpy as np

from .features import extract, BearingGeometry, FEATURE_NAMES

_GEOM = BearingGeometry()


def _window(severity: float, fs: float, dur: float, shaft_rpm: float,
            defect: str, rng: np.random.Generator) -> np.ndarray:
    """One vibration window at a given degradation severity in [0, 1]."""
    n = int(fs * dur)
    t = np.arange(n) / fs

    # Baseline machine vibration: shaft harmonics + broadband noise
    fr = shaft_rpm / 60.0
    sig = 0.05 * np.sin(2 * np.pi * fr * t)
    sig += 0.02 * np.sin(2 * np.pi * 2 * fr * t)
    sig += rng.normal(0, 0.03, n)

    # Defect impulse train at the chosen bearing defect frequency.
    fdef = _GEOM.defect_frequencies(shaft_rpm)[defect]
    impulse_amp = 0.6 * severity ** 1.5
    if fdef > 0 and impulse_amp > 0:
        period = fs / fdef
        decay = np.exp(-np.linspace(0, 8, int(period)))  # ringdown shape
        ring_f = 4000 + 1500 * severity                   # excited resonance
        ring = decay * np.sin(2 * np.pi * ring_f * np.arange(len(decay)) / fs)
        train = np.zeros(n)
        for k in range(int(n / period)):
            i = int(k * period + rng.normal(0, period * 0.01))  # jitter/slip
            if 0 <= i < n - len(ring):
                # amplitude modulated by shaft rotation (inner-race signature)
                mod = 1.0 + 0.4 * np.sin(2 * np.pi * fr * t[i])
                train[i:i + len(ring)] += ring * mod
        sig += impulse_amp * train

    # Near end of life: rising broadband energy (spalling / multiple defects)
    sig += rng.normal(0, 0.08 * severity ** 2, n)
    return sig


def make_run_to_failure(
    n_files: int = 120,
    fs: float = 20000.0,
    dur: float = 0.1,           # 0.1 s window keeps generation fast
    shaft_rpm: float = 2000.0,
    defect: str = "bpfo",
    knee: float = 0.55,         # fraction of life before degradation accelerates
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one bearing test to failure.

    Returns
    -------
    X    : (n_files, n_features) feature matrix
    rul  : (n_files,) remaining useful life in files (cycles) until failure
    sev  : (n_files,) ground-truth severity in [0, 1]
    """
    rng = np.random.default_rng(seed)
    rows, sev = [], []
    for i in range(n_files):
        life = i / (n_files - 1)
        # Flat-then-accelerating degradation (typical bearing P-F curve)
        if life < knee:
            s = 0.05 * (life / knee)
        else:
            s = 0.05 + 0.95 * ((life - knee) / (1 - knee)) ** 2.2
        s = float(np.clip(s + rng.normal(0, 0.01), 0, 1))
        sig = _window(s, fs, dur, shaft_rpm, defect, rng)
        feats = extract(sig, fs=fs, shaft_rpm=shaft_rpm)
        rows.append([feats[k] for k in FEATURE_NAMES])
        sev.append(s)

    X = np.array(rows)
    rul = (n_files - 1 - np.arange(n_files)).astype(float)
    return X, rul, np.array(sev)


def make_dataset(n_runs: int = 6, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Several run-to-failure experiments stacked together.

    Returns X, rul, severity, run_id. Different seeds/defects emulate the
    independent bearing channels in the IMS test sets.
    """
    defects = ["bpfo", "bpfi", "bsf", "bpfo", "bpfi", "bsf"]
    Xs, ruls, sevs, ids = [], [], [], []
    for r in range(n_runs):
        X, rul, sev = make_run_to_failure(
            defect=defects[r % len(defects)], seed=r, **kw)
        Xs.append(X); ruls.append(rul); sevs.append(sev)
        ids.append(np.full(len(rul), r))
    return (np.vstack(Xs), np.concatenate(ruls),
            np.concatenate(sevs), np.concatenate(ids))
