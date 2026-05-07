"""Respiration modality: cleaning + (future) breath-rate / breath-phase helpers.

Centralizes the 0.05–1 Hz bandpass + NaN-interp + z-score recipe that was
previously duplicated inline across the analysis and figure scripts.

Public API
----------
- :func:`clean_respiration` — cleans a raw respiration trace (pd.Series or
  array-like) and returns a numpy array on the same time grid.
- :func:`bandpass_respiration` — the bandpass primitive itself, exposed for
  callers that want to apply the same filter to an already-NaN-cleaned array.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


DEFAULT_FS = 256.0
#: Default cutoffs (Hz) for respiration bandpass (covers ~3–60 breaths/min).
DEFAULT_LO = 0.05
DEFAULT_HI = 1.0
DEFAULT_ORDER = 4


def bandpass_respiration(
    x: np.ndarray,
    fs: float = DEFAULT_FS,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass (SOS form) for respiration."""
    ny = 0.5 * fs
    sos = butter(order, [lo / ny, hi / ny], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def clean_respiration(
    series: Union[pd.Series, np.ndarray],
    fs: float = DEFAULT_FS,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Clean a raw respiration trace.

    Steps (matching the canonical pipeline used by the coupling analyses):

    1. Linear-interpolate NaNs (both directions).
    2. Center on the mean.
    3. Scale to [-1, 1] before filtering for numerical stability.
    4. Zero-phase Butterworth bandpass (default 0.05–1 Hz, order 4).
    5. Z-score the result.

    Parameters
    ----------
    series : pd.Series or array-like
        Raw respiration samples at ``fs`` Hz.
    fs : float
        Sampling rate (default 256 Hz, the merged-table rate).
    lo, hi : float
        Bandpass cutoffs in Hz (default 0.05–1 Hz).
    order : int
        Butterworth order (default 4).

    Returns
    -------
    np.ndarray
        Cleaned, z-scored respiration on the same time grid as the input.
    """
    if isinstance(series, pd.Series):
        x = series.astype(float).to_numpy()
    else:
        x = np.asarray(series, dtype=float)

    # 1) interpolate NaNs
    s = pd.Series(x).interpolate(method="linear", limit_direction="both").to_numpy()
    # 2) center
    s = s - np.nanmean(s)
    # 3) pre-filter scale
    mx = np.nanmax(np.abs(s)) or 1.0
    s = s / mx
    # 4) bandpass
    s = bandpass_respiration(s, fs=fs, lo=lo, hi=hi, order=order)
    # 5) z-score
    mu = float(np.mean(s))
    sd = float(np.std(s)) + 1e-12
    return (s - mu) / sd
