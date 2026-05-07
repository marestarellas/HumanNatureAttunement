"""Fractal / scaling exponents for any 1-D signal.

Per-signal scalar measures of self-similarity and long-range structure:

- :func:`higuchi_fd`     — Higuchi fractal dimension (antropy).
- :func:`katz_fd`        — Katz fractal dimension (antropy).
- :func:`petrosian_fd`   — Petrosian fractal dimension (antropy).
- :func:`dfa_alpha`      — DFA scaling exponent α (antropy).
- :func:`hurst_rs`       — Hurst exponent via classical R/S (numpy).

For convenience, :func:`all_fractals` runs every measure and returns a
single dict with the legacy column names so it can be dropped straight
into :func:`HNA.features.windowed.windowed_channel_features`.

Multi-scale curves (DFA fluctuation function, MSE) live in
:mod:`HNA.coupling.complexity` because their primary use is in
*complexity matching* between two signals; they are not single-signal
features.
"""
from __future__ import annotations

import numpy as np
import antropy as ant


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension. Returns ``np.nan`` on failure."""
    try:
        return float(ant.higuchi_fd(np.asarray(x, dtype=float), kmax=kmax))
    except Exception:  # noqa: BLE001
        return float("nan")


def katz_fd(x: np.ndarray) -> float:
    """Katz fractal dimension. Returns ``np.nan`` on failure."""
    try:
        return float(ant.katz_fd(np.asarray(x, dtype=float)))
    except Exception:  # noqa: BLE001
        return float("nan")


def petrosian_fd(x: np.ndarray) -> float:
    """Petrosian fractal dimension. Returns ``np.nan`` on failure."""
    try:
        return float(ant.petrosian_fd(np.asarray(x, dtype=float)))
    except Exception:  # noqa: BLE001
        return float("nan")


def dfa_alpha(x: np.ndarray) -> float:
    """Detrended-fluctuation-analysis (DFA) scaling exponent α.

    α≈0.5 is white noise, α≈1 is pink/1-f noise, α≈1.5 is Brownian motion.
    """
    try:
        return float(ant.detrended_fluctuation(np.asarray(x, dtype=float)))
    except Exception:  # noqa: BLE001
        return float("nan")


def hurst_rs(
    x: np.ndarray,
    min_window: int = 10,
    n_scales: int = 20,
) -> float:
    """Hurst exponent via classical rescaled-range (R/S) analysis.

    Returns ``np.nan`` if the signal is too short (< 4 * ``min_window``)
    or numerically degenerate.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4 * min_window:
        return float("nan")
    max_window = n // 2
    if max_window <= min_window:
        return float("nan")

    sizes = np.unique(np.logspace(
        np.log10(min_window), np.log10(max_window), n_scales,
    ).astype(int))
    sizes = sizes[sizes >= 4]

    rs_means = []
    valid_sizes = []
    for w in sizes:
        n_windows = n // w
        if n_windows < 1:
            continue
        chunks = x[: n_windows * w].reshape(n_windows, w)
        means = chunks.mean(axis=1, keepdims=True)
        Y = np.cumsum(chunks - means, axis=1)
        R = Y.max(axis=1) - Y.min(axis=1)
        S = chunks.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(S > 0, R / S, np.nan)
        rs = rs[np.isfinite(rs)]
        if rs.size == 0:
            continue
        rs_means.append(rs.mean())
        valid_sizes.append(w)

    if len(valid_sizes) < 4:
        return float("nan")

    log_w = np.log(valid_sizes)
    log_rs = np.log(rs_means)
    if not np.all(np.isfinite(log_rs)):
        return float("nan")
    slope, _ = np.polyfit(log_w, log_rs, 1)
    return float(slope)


def all_fractals(x: np.ndarray, kmax: int = 10) -> dict:
    """Compute every fractal measure and return one dict.

    Keys (matching the legacy column order from the deprecated
    ``compute_fractal_features``):
    ``higuchi_fd, katz_fd, petrosian_fd, dfa_alpha, hurst``.
    """
    return {
        "higuchi_fd":   higuchi_fd(x, kmax=kmax),
        "katz_fd":      katz_fd(x),
        "petrosian_fd": petrosian_fd(x),
        "dfa_alpha":    dfa_alpha(x),
        "hurst":        hurst_rs(x),
    }
