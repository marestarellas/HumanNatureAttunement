"""
Oceanographic timestack analysis.

A single pixel column of the downscaled clip is sampled at every frame,
then collapsed to a 1-D row-mean signal whose Welch-PSD peak inside a
[0.05, 2] Hz band is the dominant wave frequency. `extract_timestack`
returns a clip-level summary; `extract_timestack_windowed` slides the
analysis to deliver a frame-rate peak-frequency series.

Lives in the "whole-image" tier of the framework (the underlying signal
is one number per frame) and the oscillatory feature family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import signal as _sps

from ._common import iter_frames


@dataclass
class TimestackResult:
    fps: float
    column_index: int
    dominant_freq_hz: float
    dominant_period_s: float
    psd_freqs: np.ndarray
    psd_power: np.ndarray
    timestack_image: np.ndarray            # (n_frames, height_small)


def extract_timestack(path: str, fps: float, target_long: int = 192,
                      column_frac: float = 0.5,
                      stride: int = 1, max_frames: Optional[int] = None,
                      f_min: float = 0.05, f_max: float = 2.0
                      ) -> TimestackResult:
    """
    Sample one vertical pixel column over time; FFT its row-mean to find
    the dominant wave frequency. `column_frac` selects which column (0..1).
    """
    rows = []
    h_small = None
    col_idx = None
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        if h_small is None:
            h_small = gray.shape[0]
            col_idx = int(np.clip(column_frac * gray.shape[1], 0,
                                  gray.shape[1] - 1))
        rows.append(gray[:, col_idx].astype(np.float32))
    if not rows:
        raise RuntimeError("No frames decoded.")
    ts = np.stack(rows, axis=0)            # (n_frames, height)
    sig = ts.mean(axis=1)
    sig = sig - sig.mean()
    nper = min(len(sig), int(round(fps * 8)))
    if nper < 16:
        nper = len(sig)
    f, P = _sps.welch(sig, fs=fps, nperseg=nper)
    band = (f >= f_min) & (f <= f_max)
    if band.any():
        i = int(np.argmax(P[band]))
        peak_f = float(f[band][i])
    else:
        peak_f = float("nan")
    return TimestackResult(
        fps=fps, column_index=int(col_idx) if col_idx is not None else 0,
        dominant_freq_hz=peak_f,
        dominant_period_s=(1.0 / peak_f)
        if peak_f and np.isfinite(peak_f) else float("nan"),
        psd_freqs=f, psd_power=P, timestack_image=ts)


def extract_timestack_windowed(
    path: str, fps: float,
    win_sec: float = 4.0, step_sec: float = 0.5,
    target_long: int = 192,
    column_frac: float = 0.5,
    stride: int = 1, max_frames: Optional[int] = None,
    f_min: float = 0.05, f_max: float = 2.0,
) -> Dict[str, np.ndarray]:
    """
    Sliding-window timestack: per-window Welch PSD on the column row-mean
    signal, take the band-restricted argmax. Returns frame-rate series

      timestack_w_peak_freq_hz(t)
      timestack_w_peak_power(t)

    Useful when the dominant wave period drifts over the clip.
    """
    rows = []
    h_small = None
    col_idx = None
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        if h_small is None:
            h_small = gray.shape[0]
            col_idx = int(np.clip(column_frac * gray.shape[1], 0,
                                  gray.shape[1] - 1))
        rows.append(gray[:, col_idx].astype(np.float32))
    if not rows:
        raise RuntimeError("No frames decoded.")
    ts = np.stack(rows, axis=0)
    sig = ts.mean(axis=1)
    T = sig.size
    fs = float(fps)
    W = max(16, int(round(win_sec * fs)))
    H = max(1, int(round(step_sec * fs)))
    if W > T:
        raise RuntimeError(f"Window {win_sec}s ({W} samples) > clip length "
                           f"({T} samples).")
    starts = np.arange(0, T - W + 1, H)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    centers = starts + W // 2

    n_w = starts.size
    pf = np.full(n_w, np.nan, dtype=np.float32)
    pp = np.full(n_w, np.nan, dtype=np.float32)
    nper = min(W, max(16, int(round(fs * 8))))

    for wi, st in enumerate(starts):
        seg = sig[st:st + W]
        seg = seg - seg.mean()
        if np.std(seg) < 1e-12:
            continue
        try:
            f_, P_ = _sps.welch(seg, fs=fs, nperseg=min(seg.size, nper))
        except Exception:
            continue
        band = (f_ >= f_min) & (f_ <= f_max)
        if not band.any():
            continue
        i = int(np.argmax(P_[band]))
        pf[wi] = float(f_[band][i])
        pp[wi] = float(np.log10(P_[band][i] + 1.0))

    t_centers = centers.astype(np.float64) / fs
    t_frames = np.arange(T) / fs

    def _to_framerate(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if not finite.any():
            return np.full(T, np.nan, dtype=np.float32)
        a = arr.astype(np.float64).copy()
        a[~finite] = np.interp(t_centers[~finite],
                               t_centers[finite], a[finite])
        return np.interp(t_frames, t_centers, a).astype(np.float32)

    return {
        "timestack_w_peak_freq_hz": _to_framerate(pf),
        "timestack_w_peak_power": _to_framerate(pp),
    }


__all__ = ["TimestackResult", "extract_timestack", "extract_timestack_windowed"]
