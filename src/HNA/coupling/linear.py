"""Linear (time-domain) coupling: lagged cross-correlation.

The simplest coupling family: do amplitudes co-vary linearly, possibly
with a time lag? The single estimator here is :func:`windowed_xcorr`,
which slides a window over the two signals, computes Pearson-normalized
cross-correlation up to ``±max_lag_sec``, and reports the per-window peak
correlation and its lag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from ..dsp import interpolate_nan as _nan_interp, zscore as _zscore


@dataclass
class XCorrResult:
    times_s: np.ndarray              # window centers (seconds)
    lags_s: np.ndarray               # lag axis (seconds)
    xcorr: Optional[np.ndarray]      # (n_windows, n_lags) if return_matrix=True else None
    peak_r: np.ndarray               # max correlation per window
    peak_lag_s: np.ndarray           # lag (s) at which correlation peaks per window


def windowed_xcorr(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    win_sec: float = 180.0,
    step_sec: float = 10.0,
    max_lag_sec: float = 30.0,
    detrend: bool = True,
    zscore: bool = True,
    return_matrix: bool = False,
) -> XCorrResult:
    """Sliding-window normalized cross-correlation with lag analysis.

    Parameters
    ----------
    s1, s2 : array-like
        1-D signals at the same sample rate.
    fs : float
    win_sec, step_sec : float
        Window length / step in seconds.
    max_lag_sec : float
        Maximum lag (both directions) over which the peak is searched.
    detrend, zscore : bool
        Whether to linearly detrend / z-score each signal before
        correlation (default True).
    return_matrix : bool
        If True, also returns the full ``(n_windows, n_lags)`` correlation
        matrix as ``XCorrResult.xcorr``. Default False (saves memory).

    Returns
    -------
    :class:`XCorrResult`
    """
    s1 = _nan_interp(s1)
    s2 = _nan_interp(s2)
    if detrend:
        s1 = signal.detrend(s1, type="linear")
        s2 = signal.detrend(s2, type="linear")
    if zscore:
        s1, s2 = _zscore(s1), _zscore(s2)

    N = len(s1)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    L = int(round(max_lag_sec * fs))
    if W <= 1 or W > N:
        raise ValueError("Window size must be >1 and <= length of signals.")

    starts = np.arange(0, N - W + 1, H)
    lags = np.arange(-L, L + 1)
    lags_s = lags / fs

    peak_r = []
    peak_lag_s = []
    xcorr_mat = [] if return_matrix else None
    times_s = []

    for st in starts:
        seg1 = s1[st:st + W].copy()
        seg2 = s2[st:st + W].copy()
        seg1 -= seg1.mean()
        seg2 -= seg2.mean()
        denom = (np.std(seg1) * np.std(seg2) * len(seg1) + 1e-12)
        full = signal.correlate(seg1, seg2, mode="full") / denom
        center = len(full) // 2
        corr = full[center - L:center + L + 1]
        if return_matrix:
            xcorr_mat.append(corr)
        i_max = int(np.argmax(corr))
        peak_r.append(float(corr[i_max]))
        peak_lag_s.append(float(lags_s[i_max]))
        times_s.append((st + W / 2) / fs)

    return XCorrResult(
        times_s=np.asarray(times_s),
        lags_s=lags_s,
        xcorr=np.asarray(xcorr_mat) if return_matrix else None,
        peak_r=np.asarray(peak_r),
        peak_lag_s=np.asarray(peak_lag_s),
    )
