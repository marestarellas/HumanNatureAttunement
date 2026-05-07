"""
Nonlinear-dynamics summaries on any 1-D signal.

`complexity_summary(x, fs)` returns a dict of ~20 measures (entropies,
fractal dimensions, Hjorth parameters, DFA / Hurst / LZ, spectral
descriptors). Missing optional libraries (`antropy`, `neurokit2`) leave
the corresponding keys at NaN.

`windowed_complexity(x, fs, ...)` slides a window over the signal and
computes a chosen subset of those measures at each step, *promoting
complexity itself to a 1-D signal* that can be cross-correlated with
neural complexity series.

This file lives at the bottom of the framework: every signal produced
by the spatial-tier extractors (whole-image, per-patch, per-pixel)
flows through `complexity_summary` to fill the third feature-family
column.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as _sps

from ._common import _HAVE_ANTROPY, _HAVE_NK, _ant, _nk


def _hjorth(x: np.ndarray) -> Tuple[float, float, float]:
    """Hjorth (1970) activity, mobility, complexity."""
    x = np.asarray(x, float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    activity = float(var_x)
    mobility = float(np.sqrt(var_dx / var_x)) if var_x > 0 else np.nan
    if var_dx > 0 and mobility > 0:
        complexity = float(np.sqrt(var_ddx / var_dx) / mobility)
    else:
        complexity = float("nan")
    return activity, mobility, complexity


def complexity_summary(x: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Nonlinear-dynamics summary of a 1-D signal. Returned keys:

      perm_entropy, sample_entropy, approx_entropy, spectral_entropy,
      svd_entropy, higuchi_fd, katz_fd, petrosian_fd,
      hjorth_mobility, hjorth_complexity,
      dfa_alpha, hurst, lz_complexity,
      mse_area, psd_slope,
      peak_freq, peak_power, spectral_centroid, spectral_bandwidth,
      band_power_low, band_power_mid, band_power_high

    Missing optional libraries (antropy / neurokit2) -> NaN for those keys.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    keys = ("perm_entropy sample_entropy approx_entropy spectral_entropy "
            "svd_entropy higuchi_fd katz_fd petrosian_fd "
            "hjorth_mobility hjorth_complexity "
            "dfa_alpha hurst lz_complexity mse_area psd_slope "
            "peak_freq peak_power spectral_centroid spectral_bandwidth "
            "band_power_low band_power_mid band_power_high").split()
    out: Dict[str, float] = {k: float("nan") for k in keys}
    if x.size < 32 or np.std(x) < 1e-12:
        return out

    if _HAVE_ANTROPY:
        try: out["perm_entropy"] = float(_ant.perm_entropy(x, normalize=True))
        except Exception: pass
        try: out["sample_entropy"] = float(_ant.sample_entropy(x))
        except Exception: pass
        try: out["approx_entropy"] = float(_ant.app_entropy(x))
        except Exception: pass
        try: out["spectral_entropy"] = float(
                _ant.spectral_entropy(x, sf=fs, normalize=True))
        except Exception: pass
        try: out["svd_entropy"] = float(_ant.svd_entropy(x, normalize=True))
        except Exception: pass
        try: out["higuchi_fd"] = float(_ant.higuchi_fd(x))
        except Exception: pass
        try: out["katz_fd"] = float(_ant.katz_fd(x))
        except Exception: pass
        try: out["petrosian_fd"] = float(_ant.petrosian_fd(x))
        except Exception: pass

    _, mob, cmpl = _hjorth(x)
    out["hjorth_mobility"] = mob
    out["hjorth_complexity"] = cmpl

    if _HAVE_NK:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                v, _ = _nk.fractal_dfa(x); out["dfa_alpha"] = float(v)
            except Exception: pass
            try:
                v, _ = _nk.fractal_hurst(x); out["hurst"] = float(v)
            except Exception: pass
            try:
                v, _ = _nk.complexity_lempelziv(x); out["lz_complexity"] = float(v)
            except Exception: pass
            try:
                v, _ = _nk.entropy_multiscale(x, scale="default", method="MSEn")
                out["mse_area"] = float(v)
            except Exception: pass

    f, P = _sps.welch(x, fs=fs, nperseg=min(len(x), 256))
    m = (f > 0) & np.isfinite(P)
    if m.sum() > 4:
        f_pos = f[m]; P_pos = P[m]
        slope, _ = np.polyfit(np.log10(f_pos[P_pos > 0]),
                              np.log10(P_pos[P_pos > 0]), 1)
        out["psd_slope"] = float(slope)
        i_pk = int(np.argmax(P_pos))
        out["peak_freq"] = float(f_pos[i_pk])
        out["peak_power"] = float(P_pos[i_pk])
        Pn = P_pos / (P_pos.sum() + 1e-12)
        centroid = float((f_pos * Pn).sum())
        out["spectral_centroid"] = centroid
        out["spectral_bandwidth"] = float(
            np.sqrt(((f_pos - centroid) ** 2 * Pn).sum()))
        for name, lo, hi in (("low", 0.05, 0.25),
                             ("mid", 0.25, 0.50),
                             ("high", 0.50, 2.00)):
            sel = (f_pos >= lo) & (f_pos < hi)
            out[f"band_power_{name}"] = (
                float(P_pos[sel].sum()) if sel.any() else float("nan"))
    else:
        for k in ("peak_freq", "peak_power", "spectral_centroid",
                  "spectral_bandwidth",
                  "band_power_low", "band_power_mid", "band_power_high"):
            out[k] = float("nan")
    return out


def windowed_complexity(x: np.ndarray, fs: float,
                        win_sec: float = 4.0, step_sec: float = 0.25,
                        measures: Tuple[str, ...] = (
                            "perm_entropy", "hjorth_complexity",
                            "higuchi_fd", "spectral_entropy"),
                        ) -> Dict[str, np.ndarray]:
    """
    Time-resolved complexity: slide a window of `win_sec` seconds with hop
    `step_sec` and compute the requested measures at each step. Returns a
    dict with key 'times_s' plus one 1-D array per measure -- so
    complexity *itself* becomes a coupling-ready signal.

    `measures` is a subset of the keys returned by `complexity_summary`.
    """
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nan)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    if W < 16:
        raise ValueError("win_sec * fs must be >= 16 samples")
    starts = np.arange(0, max(len(x) - W + 1, 0), H)
    out: Dict[str, List[float]] = {m: [] for m in measures}
    times: List[float] = []
    for st in starts:
        seg = x[st:st + W]
        seg = seg[np.isfinite(seg)]
        if seg.size < 16 or np.std(seg) < 1e-12:
            for m in measures:
                out[m].append(np.nan)
        else:
            s = complexity_summary(seg, fs=fs)
            for m in measures:
                out[m].append(s.get(m, np.nan))
        times.append((st + W / 2) / fs)
    res: Dict[str, np.ndarray] = {"times_s": np.asarray(times)}
    for m in measures:
        res[m] = np.asarray(out[m])
    return res


__all__ = ["complexity_summary", "windowed_complexity", "_hjorth"]
