"""Loader for the NASA IMS (Intelligent Maintenance Systems) bearing dataset.

The IMS run-to-failure data (NASA Prognostics Center of Excellence repository)
consists of three test sets. Each *file* is a 1-second snapshot sampled at
20 kHz; the filename is a timestamp, and files are recorded every ~10 min until
a bearing fails. Each file is whitespace-separated with one column per
accelerometer channel (Set 1: 8 channels, Sets 2 & 3: 4 channels).

This module turns a directory of those files into the SAME feature matrix used
by the synthetic generator, with RUL labels = (time until the last file).

Download
--------
The raw archive (~6 GB unpacked) is *not* committed. Get it from the NASA PCoE
"Bearing Data Set" page, unpack a test set, and point ``load_ims_run`` at the
folder, e.g.::

    from motor_twin.ml.ims_loader import load_ims_run
    X, rul, t = load_ims_run("data/raw/2nd_test", channel=0)

Then retrain with ``train.py --source ims --path data/raw/2nd_test``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .features import extract, FEATURE_NAMES

IMS_INFO_URL = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"


def list_files(folder: str | Path) -> list[Path]:
    """IMS files are timestamp-named with no extension; sort chronologically."""
    p = Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]
    return sorted(files, key=lambda f: f.name)


def load_ims_run(folder: str | Path, channel: int = 0,
                 fs: float = 20000.0, shaft_rpm: float = 2000.0,
                 max_files: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Featurize one IMS test set for a single accelerometer channel.

    Returns X (n_files, n_features), rul (files-to-failure), and the ordered
    list of source filenames.
    """
    files = list_files(folder)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No IMS data files found in {folder!r}")

    rows, names = [], []
    for f in files:
        data = np.loadtxt(f)
        col = data[:, channel] if data.ndim == 2 else data
        feats = extract(col, fs=fs, shaft_rpm=shaft_rpm)
        rows.append([feats[k] for k in FEATURE_NAMES])
        names.append(f.name)

    X = np.array(rows)
    rul = (len(files) - 1 - np.arange(len(files))).astype(float)
    return X, rul, names


def download_hint() -> str:
    return (
        "The IMS bearing dataset is distributed by the NASA Prognostics Center "
        f"of Excellence. Download the 'Bearing Data Set' from:\n  {IMS_INFO_URL}\n"
        "Unpack a test set (e.g. '2nd_test') into data/raw/ and pass its path "
        "to load_ims_run() or train.py --source ims --path <folder>."
    )
