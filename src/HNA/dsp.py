"""
Shared DSP helpers for the HNA toolbox.

Consolidates filter design, envelope extraction, NaN interpolation, z-scoring,
and resampling that were previously duplicated across coupling.py, the audio
envelope script, and the audio-EEG correlation script.

Conventions:
- All filtering uses second-order sections (`output='sos'`) for numerical stability.
- All filters use zero-phase application via `sosfiltfilt`.
- `interpolate_nan` does linear interpolation in the interior with constant
  padding at the edges. Returns a float array even if the input was integer.
"""

from __future__ import annotations

from math import gcd
from typing import Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly


# -----------------------------
# NaN handling and z-score
# -----------------------------
def interpolate_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior NaNs; constant-pad NaN edges.

    Returns a float copy. Raises ValueError if the input is empty or all NaN.
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if n == 0:
        raise ValueError("interpolate_nan: input is empty.")
    mask = np.isnan(x)
    if not mask.any():
        return x
    if mask.all():
        raise ValueError("interpolate_nan: all values are NaN.")
    idx = np.arange(n)
    # constant-pad leading NaNs with first valid value
    if mask[0]:
        first_valid = int(np.flatnonzero(~mask)[0])
        x[:first_valid] = x[first_valid]
    # constant-pad trailing NaNs with last valid value
    if mask[-1]:
        last_valid = int(np.flatnonzero(~mask)[-1])
        x[last_valid + 1:] = x[last_valid]
    # linear-interpolate interior gaps
    mask = np.isnan(x)
    if mask.any():
        x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize to zero mean, unit variance. Robust to zero-variance inputs."""
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + eps)


# -----------------------------
# Filter design + application
# -----------------------------
def butter_sos(cut: float | Sequence[float], fs: float, btype: str, order: int = 4) -> np.ndarray:
    """Design a Butterworth filter and return its second-order sections.

    btype: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'.
    """
    nyq = fs * 0.5
    Wn = np.atleast_1d(cut) / nyq
    if np.any(Wn <= 0) or np.any(Wn >= 1):
        raise ValueError(f"butter_sos: cutoff(s) {cut} Hz invalid for fs={fs} Hz.")
    return butter(order, Wn, btype=btype, output="sos")


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass between `lo` and `hi` Hz."""
    sos = butter_sos([lo, hi], fs=fs, btype="bandpass", order=order)
    return sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, cut: float, order: int = 4) -> np.ndarray:
    """Zero-phase lowpass at `cut` Hz."""
    sos = butter_sos(cut, fs=fs, btype="lowpass", order=order)
    return sosfiltfilt(sos, x)


def highpass(x: np.ndarray, fs: float, cut: float, order: int = 2) -> np.ndarray:
    """Zero-phase highpass at `cut` Hz."""
    sos = butter_sos(cut, fs=fs, btype="highpass", order=order)
    return sosfiltfilt(sos, x)


# -----------------------------
# Envelope and resampling
# -----------------------------
def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope via the analytic signal."""
    return np.abs(hilbert(np.asarray(x, dtype=float)))


def resample_to(x: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Polyphase resample from `fs_from` Hz to `fs_to` Hz, GCD-simplified.

    Returns a float32 array. If `fs_from == fs_to`, returns x as float32.
    """
    fs_from = int(round(fs_from))
    fs_to = int(round(fs_to))
    if fs_from == fs_to:
        return np.asarray(x, dtype=np.float32, order="C")
    g = gcd(fs_from, fs_to)
    up = fs_to // g
    down = fs_from // g
    return resample_poly(x, up, down).astype(np.float32)


# -----------------------------
# Convenience: prep a pair of signals for windowed analysis
# -----------------------------
def detrend_zscore_pair(s1: np.ndarray, s2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly detrend and z-score two signals after NaN-interpolating both."""
    from scipy import signal as _sig
    s1 = interpolate_nan(s1); s2 = interpolate_nan(s2)
    s1 = _sig.detrend(s1, type="linear")
    s2 = _sig.detrend(s2, type="linear")
    return zscore(s1), zscore(s2)
