"""Vibration feature extraction for bearing prognostics.

A single extractor turns a raw accelerometer window (e.g. 1 s @ 20 kHz, the
NASA IMS sampling scheme) into a fixed feature vector. The SAME function is
used for real IMS files and for the synthetic run-to-failure generator, so a
model trained on one transfers directly to the other.

Feature groups
--------------
Time domain : rms, peak, kurtosis, skewness, crest/shape/impulse factors
Frequency   : spectral centroid, band energies, and energy at the four bearing
              defect frequencies (BPFO/BPFI/BSF/FTF) computed from geometry.

Reference: Randall, "Rotating Machine Vibration"; the IMS test rig uses
Rexnord ZA-2115 double-row bearings (16 rollers, pitch dia 71.5 mm, roller
dia 8.4 mm, contact angle 15.17 deg) at a 2000 rpm shaft.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import kurtosis, skew


@dataclass(frozen=True)
class BearingGeometry:
    """Geometry used to derive characteristic defect frequencies."""
    n_rollers: int = 16
    pitch_dia_mm: float = 71.5
    roller_dia_mm: float = 8.4
    contact_angle_deg: float = 15.17

    def defect_frequencies(self, shaft_rpm: float) -> dict[str, float]:
        """Return BPFO/BPFI/BSF/FTF [Hz] for a given shaft speed."""
        fr = shaft_rpm / 60.0                      # shaft freq [Hz]
        n = self.n_rollers
        d, dp = self.roller_dia_mm, self.pitch_dia_mm
        ca = np.cos(np.radians(self.contact_angle_deg))
        ratio = d / dp * ca
        return {
            "ftf": fr / 2.0 * (1 - ratio),                 # cage
            "bpfo": n / 2.0 * fr * (1 - ratio),            # outer race
            "bpfi": n / 2.0 * fr * (1 + ratio),            # inner race
            "bsf": dp / (2 * d) * fr * (1 - ratio ** 2),   # roller
        }


# Stable, documented feature order (also the model input order)
FEATURE_NAMES = [
    "rms", "peak", "kurtosis", "skew", "crest_factor",
    "shape_factor", "impulse_factor", "spectral_centroid",
    "band_lo", "band_mid", "band_hi",
    "e_ftf", "e_bpfo", "e_bpfi", "e_bsf",
]

_GEOM = BearingGeometry()


def _band_energy(freqs: np.ndarray, mag: np.ndarray, f0: float, bw: float) -> float:
    """Spectral energy within +/- bw of f0 (and its 2nd harmonic)."""
    if f0 <= 0:
        return 0.0
    m = ((np.abs(freqs - f0) <= bw) | (np.abs(freqs - 2 * f0) <= bw))
    return float(np.sum(mag[m] ** 2))


def extract(signal: np.ndarray, fs: float = 20000.0,
            shaft_rpm: float = 2000.0,
            geom: BearingGeometry = _GEOM) -> dict[str, float]:
    """Extract the feature dict from one vibration window."""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = len(x)

    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-12
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x))) or 1e-12

    # FFT magnitude (single-sided)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total_e = float(np.sum(spec ** 2)) or 1e-12

    defects = geom.defect_frequencies(shaft_rpm)
    bw = 5.0  # Hz tolerance around each defect line

    feats = {
        "rms": rms,
        "peak": peak,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        "skew": float(skew(x, bias=False)),
        "crest_factor": peak / rms,
        "shape_factor": rms / mean_abs,
        "impulse_factor": peak / mean_abs,
        "spectral_centroid": float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12)),
        "band_lo": float(np.sum(spec[freqs < 1000] ** 2) / total_e),
        "band_mid": float(np.sum(spec[(freqs >= 1000) & (freqs < 5000)] ** 2) / total_e),
        "band_hi": float(np.sum(spec[freqs >= 5000] ** 2) / total_e),
        "e_ftf": _band_energy(freqs, spec, defects["ftf"], bw) / total_e,
        "e_bpfo": _band_energy(freqs, spec, defects["bpfo"], bw) / total_e,
        "e_bpfi": _band_energy(freqs, spec, defects["bpfi"], bw) / total_e,
        "e_bsf": _band_energy(freqs, spec, defects["bsf"], bw) / total_e,
    }
    return feats


def to_vector(feats: dict[str, float]) -> np.ndarray:
    """Feature dict -> ordered numpy vector (model input)."""
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)
