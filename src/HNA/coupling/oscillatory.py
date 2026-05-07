"""Oscillatory coupling family: spectral, phase, and cross-frequency.

This module groups every method that lives in the frequency / phase
domain:

- **Spectral** — :func:`band_coherence`, :func:`band_coherence_windowed`
  (Welch coherence, optionally windowed and band-averaged).
- **Phase** — :func:`plv_phase_sync`, :func:`windowed_plv`,
  :func:`wpli_phase_sync`, :func:`windowed_wpli` (PLV and the debiased
  weighted PLI).
- **Cross-frequency** — :func:`tort_modulation_index`, :func:`canolty_mvl`,
  :func:`windowed_pac`, :func:`comodulogram` (phase-amplitude coupling).

A few internal helpers are exposed (with leading underscore) so external
notebook code that imported them from the legacy flat ``coupling.py`` keeps
working: :func:`_butter_bandpass`, :func:`_dominant_freq`,
:func:`_wpli_from_phase`, :func:`_coh_choose_nperseg`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as _sig
from scipy.signal import hilbert as _hilbert

from ..dsp import (
    interpolate_nan as _nan_interp,
    butter_sos as _butter_sos,
    bandpass as _bandpass,
)


# =====================================================================
# Shared internal helpers
# =====================================================================
def _butter_bandpass(lo, hi, fs, order: int = 4):
    """Build a Butterworth SOS bandpass — kept under the legacy name so
    notebooks that imported ``_butter_bandpass`` from the old flat module
    still work."""
    return _butter_sos([lo, hi], fs=fs, btype="bandpass", order=order)


def _dominant_freq(x: np.ndarray, fs: float, fmin: float = 0.05, fmax: float = 0.5) -> float:
    """Return the Welch-PSD peak frequency in ``[fmin, fmax]``.

    Used by PLV/wPLI to pick a target frequency from the second signal
    when the caller has not specified ``f0``.
    """
    nper = min(len(x), int(round(fs * 300)))
    f, Pxx = _sig.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="constant")
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return (fmin + fmax) / 2.0
    return float(f[m][np.argmax(Pxx[m])])


def _wpli_from_phase(dphi: np.ndarray) -> float:
    """Weighted phase-lag index from a phase-difference time series."""
    s = np.sin(np.asarray(dphi, dtype=float))
    den = np.mean(np.abs(s)) + 1e-12
    num = np.abs(np.mean(s))
    return float(num / den) if np.isfinite(den) and den > 0 else float("nan")


def _coh_choose_nperseg(W: int, fs: float, target_sec: float = 60,
                        min_segments: int = 4) -> Tuple[int, int]:
    """Pick ``(nperseg, noverlap)`` so each window has enough Welch segments."""
    nper = int(fs * min(target_sec, max(8, W / 2)))
    nper = max(32, nper)
    while True:
        nover = nper // 2
        step = nper - nover
        nseg = 1 + max(0, (W - nper) // max(1, step))
        if nseg >= min_segments or nper <= 32:
            return nper, nover
        nper = int(nper * 0.8)


# =====================================================================
# Spectral: Welch coherence
# =====================================================================
@dataclass
class CoherenceResult:
    f: np.ndarray
    Cxy: np.ndarray
    peak_f: float
    peak_coh: float
    band_avg_coh: float
    times_s: Optional[np.ndarray] = None
    band_avg_coh_win: Optional[np.ndarray] = None


def band_coherence(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    fmin: float = 0.05,
    fmax: float = 0.5,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    detrend: str = "constant",
    windowed: bool = True,
    win_sec: float = 180.0,
    step_sec: float = 30.0,
) -> CoherenceResult:
    """Welch magnitude-squared coherence with optional windowed band-avg series."""
    s1 = _nan_interp(s1)
    s2 = _nan_interp(s2)
    if nperseg is None:
        nperseg = min(len(s1), int(round(fs * 300)))
    if noverlap is None:
        noverlap = nperseg // 2

    f, Cxy = _sig.coherence(s1, s2, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=detrend)
    band_mask = (f >= fmin) & (f <= fmax)
    if not np.any(band_mask):
        raise ValueError("No frequencies inside requested band. Adjust fmin/fmax or nperseg.")

    band_f = f[band_mask]
    band_C = Cxy[band_mask]
    i_peak = np.argmax(band_C)
    peak_f, peak_coh = float(band_f[i_peak]), float(band_C[i_peak])
    band_avg = float(np.mean(band_C))

    times_s = None
    band_avg_win = None
    if windowed:
        W = int(round(win_sec * fs))
        H = int(round(step_sec * fs))
        if W < nperseg:
            W = nperseg
        starts = np.arange(0, len(s1) - W + 1, H)
        ts, vals = [], []
        for st in starts:
            seg1 = s1[st:st + W]
            seg2 = s2[st:st + W]
            f_w, C_w = _sig.coherence(seg1, seg2, fs=fs, nperseg=nperseg,
                                      noverlap=noverlap, detrend=detrend)
            m = (f_w >= fmin) & (f_w <= fmax)
            vals.append(np.mean(C_w[m]) if np.any(m) else np.nan)
            ts.append((st + W / 2) / fs)
        times_s = np.asarray(ts)
        band_avg_win = np.asarray(vals)

    return CoherenceResult(
        f=f, Cxy=Cxy, peak_f=peak_f, peak_coh=peak_coh,
        band_avg_coh=band_avg, times_s=times_s, band_avg_coh_win=band_avg_win,
    )


def band_coherence_windowed(
    s1, s2, fs,
    fmin: float = 0.08, fmax: float = 0.35,
    win_sec: float = 180.0, step_sec: float = 30.0,
    detrend: str = "constant",
):
    """Windowed band-averaged coherence with auto-chosen Welch segments.

    Returns a dict (rather than a dataclass) for direct CSV/JSON export
    parity with the legacy implementation.
    """
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)
    N = len(s1)
    assert N == len(s2)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    starts = np.arange(0, max(N - W + 1, 0), H)

    nper_g, nover_g = _coh_choose_nperseg(N, fs, target_sec=60, min_segments=8)
    f, Cxy = _sig.coherence(s1, s2, fs=fs, nperseg=nper_g, noverlap=nover_g, detrend=detrend)
    m = (f >= fmin) & (f <= fmax)
    peak_f = float(f[m][np.argmax(Cxy[m])]) if np.any(m) else float("nan")
    peak_coh = float(np.max(Cxy[m])) if np.any(m) else float("nan")
    band_avg = float(np.mean(Cxy[m])) if np.any(m) else float("nan")

    times, band_avg_win = [], []
    for st in starts:
        seg1 = s1[st:st + W]
        seg2 = s2[st:st + W]
        nper, nover = _coh_choose_nperseg(W, fs, target_sec=60, min_segments=4)
        fw, Cw = _sig.coherence(seg1, seg2, fs=fs, nperseg=nper, noverlap=nover, detrend=detrend)
        mw = (fw >= fmin) & (fw <= fmax)
        if np.sum(mw) >= 4 and np.isfinite(Cw[mw]).any():
            band_avg_win.append(float(np.nanmean(Cw[mw])))
        else:
            band_avg_win.append(np.nan)
        times.append((st + W / 2) / fs)

    return {
        "f": f, "Cxy": Cxy,
        "peak_f": peak_f, "peak_coh": peak_coh, "band_avg_coh": band_avg,
        "times_s": np.asarray(times), "band_avg_coh_win": np.asarray(band_avg_win),
        "params": dict(fmin=fmin, fmax=fmax, win_sec=win_sec, step_sec=step_sec),
    }


# =====================================================================
# Phase: PLV
# =====================================================================
@dataclass
class PLVResult:
    f0: float
    band: Tuple[float, float]
    plv: float
    mean_phase_diff: float
    preferred_lag_s: float


def plv_phase_sync(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    f0: Optional[float] = None,
    bw_hz: float = 0.12,
    fmin_search: float = 0.05,
    fmax_search: float = 0.5,
    order: int = 4,
) -> PLVResult:
    """Phase-locking value at a target frequency (or auto-detected from ``s2``)."""
    s1 = _nan_interp(s1)
    s2 = _nan_interp(s2)
    if f0 is None:
        f0 = _dominant_freq(s2, fs, fmin=fmin_search, fmax=fmax_search)
    half = bw_hz / 2.0
    lo, hi = max(1e-3, f0 - half), max(f0 + half, f0 + 1e-3)

    sos = _butter_bandpass(lo, hi, fs, order=order)
    x1 = _sig.sosfiltfilt(sos, s1.astype(float))
    x2 = _sig.sosfiltfilt(sos, s2.astype(float))

    phi1 = np.angle(_hilbert(x1))
    phi2 = np.angle(_hilbert(x2))
    dphi = np.angle(np.exp(1j * (phi1 - phi2)))

    plv = np.abs(np.mean(np.exp(1j * dphi)))
    mean_phase = np.angle(np.mean(np.exp(1j * dphi)))
    preferred_lag_s = mean_phase / (2 * np.pi * f0)

    return PLVResult(
        f0=f0, band=(lo, hi), plv=float(plv),
        mean_phase_diff=float(mean_phase),
        preferred_lag_s=float(preferred_lag_s),
    )


def windowed_plv(
    s1, s2, fs,
    win_sec: float = 180.0, step_sec: float = 30.0,
    f0: Optional[float] = None, bw_hz: float = 0.12,
    fmin_search: float = 0.05, fmax_search: float = 0.5,
    order: int = 4,
):
    """Sliding-window PLV time series."""
    s1 = _nan_interp(s1)
    s2 = _nan_interp(s2)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    starts = np.arange(0, max(len(s1) - W + 1, 0), H)

    times, plv, mean_phi, lag_s = [], [], [], []
    for st in starts:
        seg1, seg2 = s1[st:st + W], s2[st:st + W]
        f0w = f0 if f0 is not None else _dominant_freq(seg2, fs, fmin_search, fmax_search)
        half = bw_hz / 2
        sos = _butter_bandpass(max(1e-3, f0w - half), f0w + half, fs, order=order)
        x1 = _sig.sosfiltfilt(sos, seg1)
        x2 = _sig.sosfiltfilt(sos, seg2)
        dphi = np.angle(np.exp(1j * (np.angle(_hilbert(x1)) - np.angle(_hilbert(x2)))))
        e = np.exp(1j * dphi)
        plv.append(np.abs(np.mean(e)))
        mphi = np.angle(np.mean(e))
        mean_phi.append(mphi)
        lag_s.append(mphi / (2 * np.pi * f0w) if f0w > 0 else np.nan)
        times.append((st + W / 2) / fs)

    return {
        "times_s": np.asarray(times),
        "plv": np.asarray(plv),
        "mean_phase_diff": np.asarray(mean_phi),
        "preferred_lag_s": np.asarray(lag_s),
    }


# =====================================================================
# Phase: wPLI
# =====================================================================
@dataclass
class WPLIResult:
    f0: float
    band: Tuple[float, float]
    wpli: float


def wpli_phase_sync(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    f0: Optional[float] = None,
    bw_hz: float = 0.12,
    fmin_search: float = 0.05,
    fmax_search: float = 0.5,
    order: int = 4,
) -> WPLIResult:
    """Weighted phase-lag index at a target frequency (or auto-detected)."""
    s1 = _nan_interp(np.asarray(s1, float))
    s2 = _nan_interp(np.asarray(s2, float))
    if f0 is None:
        f0 = _dominant_freq(s2, fs, fmin=fmin_search, fmax=fmax_search)

    half = bw_hz / 2.0
    lo, hi = max(1e-3, f0 - half), max(f0 + half, f0 + 1e-3)
    sos = _butter_bandpass(lo, hi, fs, order=order)
    x1 = _sig.sosfiltfilt(sos, s1)
    x2 = _sig.sosfiltfilt(sos, s2)

    phi1 = np.angle(_hilbert(x1))
    phi2 = np.angle(_hilbert(x2))
    dphi = np.angle(np.exp(1j * (phi1 - phi2)))

    return WPLIResult(f0=f0, band=(lo, hi), wpli=_wpli_from_phase(dphi))


def windowed_wpli(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    win_sec: float = 180.0,
    step_sec: float = 30.0,
    f0: Optional[float] = None,
    bw_hz: float = 0.12,
    fmin_search: float = 0.05,
    fmax_search: float = 0.5,
    order: int = 4,
):
    """Sliding-window wPLI time series."""
    s1 = _nan_interp(np.asarray(s1, float))
    s2 = _nan_interp(np.asarray(s2, float))

    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    if W <= 1 or W > len(s1):
        raise ValueError("Window size must be >1 and <= length of signals.")

    starts = np.arange(0, len(s1) - W + 1, H)
    wpli_vals, times_s = [], []
    lohi_cache: Optional[Tuple[float, float]] = None

    for st in starts:
        seg1 = s1[st:st + W]
        seg2 = s2[st:st + W]

        if f0 is None:
            f0_w = _dominant_freq(seg2, fs, fmin_search, fmax_search)
            half = bw_hz / 2.0
            lo, hi = max(1e-3, f0_w - half), max(f0_w + half, f0_w + 1e-3)
        else:
            if lohi_cache is None:
                half = bw_hz / 2.0
                lohi_cache = (max(1e-3, f0 - half), max(f0 + half, f0 + 1e-3))
            lo, hi = lohi_cache

        sos = _butter_bandpass(lo, hi, fs, order=order)
        x1 = _sig.sosfiltfilt(sos, seg1)
        x2 = _sig.sosfiltfilt(sos, seg2)

        phi1 = np.angle(_hilbert(x1))
        phi2 = np.angle(_hilbert(x2))
        dphi = np.angle(np.exp(1j * (phi1 - phi2)))

        wpli_vals.append(_wpli_from_phase(dphi))
        times_s.append((st + W / 2) / fs)

    return {
        "times_s": np.asarray(times_s),
        "wpli": np.asarray(wpli_vals),
        "band": (lo, hi) if lohi_cache is not None else None,
    }


# =====================================================================
# Cross-frequency: phase-amplitude coupling (PAC)
# =====================================================================
@dataclass
class TortMIResult:
    mi: float
    n_bins: int
    distribution: np.ndarray
    bin_edges: np.ndarray
    low_band: Tuple[float, float]
    high_band: Tuple[float, float]


@dataclass
class CanoltyMVLResult:
    mvl: float
    mvl_norm: float
    preferred_phase_rad: float
    low_band: Tuple[float, float]
    high_band: Tuple[float, float]


def _phase_and_amp(
    low_signal: np.ndarray,
    high_signal: np.ndarray,
    fs: float,
    low_band: Tuple[float, float],
    high_band: Tuple[float, float],
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bandpass + Hilbert → (low-band phase, high-band amplitude)."""
    lo = _bandpass(np.asarray(low_signal, dtype=float),
                   fs=fs, lo=low_band[0], hi=low_band[1], order=order)
    hi = _bandpass(np.asarray(high_signal, dtype=float),
                   fs=fs, lo=high_band[0], hi=high_band[1], order=order)
    phase = np.angle(_hilbert(lo))
    amp = np.abs(_hilbert(hi))
    return phase, amp


def tort_modulation_index(
    low_signal: np.ndarray,
    high_signal: np.ndarray,
    fs: float,
    low_band: Tuple[float, float],
    high_band: Tuple[float, float],
    n_bins: int = 18,
    order: int = 4,
) -> TortMIResult:
    """Tort et al. 2010 KL-divergence modulation index, ∈ ``[0, 1]``."""
    phase, amp = _phase_and_amp(low_signal, high_signal, fs=fs,
                                 low_band=low_band, high_band=high_band, order=order)

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean_amp = np.zeros(n_bins, dtype=float)
    for k in range(n_bins):
        mask = bin_idx == k
        mean_amp[k] = amp[mask].mean() if mask.any() else 0.0

    total = mean_amp.sum()
    if not np.isfinite(total) or total <= 0:
        return TortMIResult(float("nan"), n_bins, mean_amp, bin_edges,
                            low_band, high_band)

    p = mean_amp / total
    eps = 1e-12
    h = -np.sum(p * np.log(p + eps))
    mi = (np.log(n_bins) - h) / np.log(n_bins)
    return TortMIResult(float(mi), n_bins, mean_amp, bin_edges, low_band, high_band)


def canolty_mvl(
    low_signal: np.ndarray,
    high_signal: np.ndarray,
    fs: float,
    low_band: Tuple[float, float],
    high_band: Tuple[float, float],
    order: int = 4,
) -> CanoltyMVLResult:
    """Canolty et al. 2006 mean vector length (raw + amplitude-normalized)."""
    phase, amp = _phase_and_amp(low_signal, high_signal, fs=fs,
                                 low_band=low_band, high_band=high_band, order=order)
    z = amp * np.exp(1j * phase)
    z_mean = z.mean()
    mvl = float(np.abs(z_mean))
    a_mean = float(np.mean(amp))
    mvl_norm = mvl / a_mean if a_mean > 0 else float("nan")
    pref_phase = float(np.angle(z_mean))
    return CanoltyMVLResult(mvl, mvl_norm, pref_phase, low_band, high_band)


def windowed_pac(
    low_signal: np.ndarray,
    high_signal: np.ndarray,
    fs: float,
    low_band: Tuple[float, float],
    high_band: Tuple[float, float],
    win_sec: float = 30.0,
    step_sec: float = 5.0,
    method: str = "tort",
    n_bins: int = 18,
    order: int = 4,
) -> dict:
    """Sliding-window PAC time series.

    ``method`` is ``"tort"`` (Tort MI) or ``"canolty"`` (normalized MVL).
    """
    method = method.lower()
    if method not in ("tort", "canolty"):
        raise ValueError(f"Unknown PAC method {method!r}; choose 'tort' or 'canolty'.")

    low_signal = np.asarray(low_signal, dtype=float)
    high_signal = np.asarray(high_signal, dtype=float)
    n = min(len(low_signal), len(high_signal))
    win = int(win_sec * fs)
    step = max(1, int(step_sec * fs))

    starts = np.arange(0, n - win + 1, step)
    times = (starts + win / 2.0) / fs
    values = np.empty(starts.size, dtype=float)
    for i, s in enumerate(starts):
        e = s + win
        if method == "tort":
            res = tort_modulation_index(
                low_signal[s:e], high_signal[s:e], fs=fs,
                low_band=low_band, high_band=high_band,
                n_bins=n_bins, order=order,
            )
            values[i] = res.mi
        else:
            res = canolty_mvl(
                low_signal[s:e], high_signal[s:e], fs=fs,
                low_band=low_band, high_band=high_band, order=order,
            )
            values[i] = res.mvl_norm
    return {
        "times_s": times,
        "value": values,
        "method": method,
        "low_band": low_band,
        "high_band": high_band,
        "win_sec": win_sec,
        "step_sec": step_sec,
    }


def comodulogram(
    low_signal: np.ndarray,
    high_signal: np.ndarray,
    fs: float,
    low_freqs: np.ndarray,
    high_freqs: np.ndarray,
    low_bw: float = 1.0,
    high_bw: float = 4.0,
    method: str = "tort",
    n_bins: int = 18,
    order: int = 4,
) -> dict:
    """PAC comodulogram on a 2-D grid of (low-freq, high-freq) pairs.

    Returns a dict with ``low_freqs``, ``high_freqs``, and a ``matrix`` of
    shape ``(len(high_freqs), len(low_freqs))``.
    """
    method = method.lower()
    low_freqs = np.asarray(low_freqs, dtype=float)
    high_freqs = np.asarray(high_freqs, dtype=float)
    M = np.full((high_freqs.size, low_freqs.size), np.nan, dtype=float)

    for j, fL in enumerate(low_freqs):
        lb = (max(fL - low_bw / 2, 1e-3), fL + low_bw / 2)
        for i, fH in enumerate(high_freqs):
            hb = (max(fH - high_bw / 2, 1e-3), fH + high_bw / 2)
            if method == "tort":
                M[i, j] = tort_modulation_index(
                    low_signal, high_signal, fs=fs,
                    low_band=lb, high_band=hb, n_bins=n_bins, order=order,
                ).mi
            elif method == "canolty":
                M[i, j] = canolty_mvl(
                    low_signal, high_signal, fs=fs,
                    low_band=lb, high_band=hb, order=order,
                ).mvl_norm
            else:
                raise ValueError(f"Unknown method {method!r}")
    return {
        "low_freqs": low_freqs, "high_freqs": high_freqs, "matrix": M,
        "method": method, "low_bw": low_bw, "high_bw": high_bw,
    }
