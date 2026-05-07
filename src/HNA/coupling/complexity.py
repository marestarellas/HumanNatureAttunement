"""Linear coupling on complexity features.

Despite the subpackage name, **all methods here are linear comparisons
applied to a complexity feature**. The complexity character lives in the
*feature* (DFA α, F(s), MSE, FOOOF aperiodic exponent) — the comparison
step itself is a plain Pearson r (or a |Δ| similarity score on two
scalars). This subpackage exists as an organisational unit because all
of its methods share the same conceptual question — *do two signals
share scaling / regularity structure?* — even though the operation that
implements that question is linear.

A genuine "complexity coupling family" — methods whose *coupling step*
is itself non-linear and scale-aware — would include detrended
cross-correlation analysis (DCCA), multiscale cross-entropy, and
multiscale cross-mutual-information. None of these are in HNA today.
They are noted as future work in the methods report.

The original Marmelat & Delignières (2012) observation: interpersonal
coordination (and human–environment attunement) is sometimes better
captured by *matched scaling* than by sample-by-sample correlation —
two participants walking together don't synchronise step-for-step, but
their gait-variability scaling exponents become very similar.

The module exposes four estimators, each operating on a different
**observable** of complexity:

**Scalar (one number per signal)**
- :func:`exponent_matching`     — ``1 - |α_x - α_y| / α_max`` ∈ [0, 1]
- :func:`exponent_correlation`  — Pearson r of α scalars across many cells

**Curve over scales (one vector per signal, indexed by scale)**
- :func:`fluctuation_curve`     — DFA fluctuation function ``F(s)`` for one signal
- :func:`fluctuation_matching`  — log-log Pearson r of two F(s) curves
- :func:`mse_curve`             — multiscale (sample) entropy curve
- :func:`mse_matching`          — Pearson r of two MSE curves

**Trace over time (one vector per signal, indexed by window)**
- :func:`windowed_exponent`     — slide a window, return scaling exponent per window
- :func:`complexity_coupling`   — windowed_exponent on each signal, then
  Pearson / MI of the two traces (the only one of the four that can take
  any standard coupling metric on the resulting traces)
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ..features.fractal import dfa_alpha, hurst_rs


# ----------------------------------------------------------------------
# Static estimators
# ----------------------------------------------------------------------
def exponent_matching(
    x: np.ndarray,
    y: np.ndarray,
    exponent_fn: Callable[[np.ndarray], float] = dfa_alpha,
    alpha_max: float = 2.0,
) -> dict:
    """Static complexity matching score in [0, 1].

    Computes a single scaling exponent for each signal and returns
    ``1 - |α_x - α_y| / α_max`` so that 1.0 = exactly matched and 0.0
    means the two exponents differ by ``α_max`` or more.

    Parameters
    ----------
    x, y : array-like
        1-D signals (need not be the same length).
    exponent_fn : callable(x) -> float
        Per-signal exponent estimator. Default :func:`dfa_alpha`. Other
        sensible choices: :func:`hurst_rs`, :func:`higuchi_fd` (with
        ``alpha_max=2``), or a closure around
        :func:`HNA.features.aperiodic.aperiodic_features`.
    alpha_max : float
        Normalization constant — the maximum plausible distance between
        the two exponents. Default 2.0 (covers DFA range 0.5–2.0 and
        FOOOF aperiodic exponents typical of EEG / 1/f signals).

    Returns
    -------
    dict
        ``alpha_x``, ``alpha_y``, ``delta`` (signed ``α_x - α_y``),
        ``matching`` (the [0,1] score).
    """
    a_x = float(exponent_fn(np.asarray(x, dtype=float)))
    a_y = float(exponent_fn(np.asarray(y, dtype=float)))
    if not (np.isfinite(a_x) and np.isfinite(a_y)):
        return {"alpha_x": a_x, "alpha_y": a_y,
                "delta": float("nan"), "matching": float("nan")}
    delta = a_x - a_y
    matching = max(0.0, 1.0 - abs(delta) / alpha_max)
    return {"alpha_x": a_x, "alpha_y": a_y,
            "delta": float(delta), "matching": float(matching)}


def exponent_correlation(
    alphas_x: np.ndarray,
    alphas_y: np.ndarray,
) -> dict:
    """Across-cell Pearson r of paired exponent values.

    Use when you have many (subject × condition) cells and want to ask
    "does HRV scaling track audio scaling across the whole dataset?" —
    the classic Marmelat–Delignières framing.

    Parameters
    ----------
    alphas_x, alphas_y : array-like
        Paired vectors of exponent values; one entry per (subject, condition).

    Returns
    -------
    dict
        ``r``, ``p``, ``n``.
    """
    from scipy import stats as sps
    a = np.asarray(alphas_x, dtype=float)
    b = np.asarray(alphas_y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return {"r": float("nan"), "p": float("nan"), "n": int(valid.sum())}
    r, p = sps.pearsonr(a[valid], b[valid])
    return {"r": float(r), "p": float(p), "n": int(valid.sum())}


# ----------------------------------------------------------------------
# Multi-scale: DFA fluctuation curves
# ----------------------------------------------------------------------
def _dfa_fluctuations(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Compute F(s) at each scale for the cumulative-sum-detrended signal."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    Y = np.cumsum(x)
    n = Y.size
    F = np.full(scales.size, np.nan)
    for i, s in enumerate(scales):
        s = int(s)
        if s < 4 or s > n // 2:
            continue
        n_segs = n // s
        segs = Y[: n_segs * s].reshape(n_segs, s)
        # Detrend each segment with a linear fit (float64 to avoid int overflow at large s).
        t = np.arange(s, dtype=np.float64)
        sx = t.sum()
        sxx = (t * t).sum()
        sy = segs.sum(axis=1)
        sxy = (segs * t).sum(axis=1)
        denom = float(s) * sxx - sx * sx
        slope = (float(s) * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / float(s)
        # Residuals
        residual_var = ((segs - (intercept[:, None] + slope[:, None] * t)) ** 2).mean(axis=1)
        F[i] = float(np.sqrt(residual_var.mean()))
    return F


def fluctuation_curve(
    x: np.ndarray,
    scales: Optional[Sequence[int]] = None,
) -> dict:
    """DFA fluctuation function F(s) at each scale.

    The log-log slope of ``F(s)`` vs ``s`` is the DFA exponent α; this
    function exposes the curve itself so you can compare the *shape*
    between signals (multi-scale matching), not just the slope.

    Parameters
    ----------
    x : array-like
        1-D signal.
    scales : sequence of int, optional
        Window sizes (samples). Defaults to ~20 log-spaced values
        between 8 and ``len(x) // 4``.

    Returns
    -------
    dict
        ``scales`` (np.ndarray), ``F`` (np.ndarray), ``alpha`` (float
        slope of log F vs log s; same as :func:`dfa_alpha` up to numerical
        differences).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if scales is None:
        s_max = max(16, n // 4)
        scales = np.unique(np.logspace(np.log10(8), np.log10(s_max), 20).astype(int))
    scales = np.asarray(scales, dtype=int)
    F = _dfa_fluctuations(x, scales)
    valid = np.isfinite(F) & (F > 0)
    if valid.sum() < 3:
        return {"scales": scales, "F": F, "alpha": float("nan")}
    slope, _ = np.polyfit(np.log(scales[valid]), np.log(F[valid]), 1)
    return {"scales": scales, "F": F, "alpha": float(slope)}


def fluctuation_matching(
    x: np.ndarray,
    y: np.ndarray,
    scales: Optional[Sequence[int]] = None,
) -> dict:
    """Compare two log F(s) curves via Pearson correlation.

    Returns
    -------
    dict
        ``scales``, ``F_x``, ``F_y``, ``r``, ``p``, ``alpha_x``,
        ``alpha_y``, ``delta_alpha``.
    """
    from scipy import stats as sps

    if scales is None:
        n = min(len(x), len(y))
        s_max = max(16, n // 4)
        scales = np.unique(np.logspace(np.log10(8), np.log10(s_max), 20).astype(int))
    scales = np.asarray(scales, dtype=int)

    cx = fluctuation_curve(x, scales=scales)
    cy = fluctuation_curve(y, scales=scales)
    Fx, Fy = cx["F"], cy["F"]
    valid = np.isfinite(Fx) & np.isfinite(Fy) & (Fx > 0) & (Fy > 0)
    if valid.sum() < 3:
        return {"scales": scales, "F_x": Fx, "F_y": Fy,
                "r": float("nan"), "p": float("nan"),
                "alpha_x": cx["alpha"], "alpha_y": cy["alpha"],
                "delta_alpha": float("nan")}
    r, p = sps.pearsonr(np.log(Fx[valid]), np.log(Fy[valid]))
    return {
        "scales": scales, "F_x": Fx, "F_y": Fy,
        "r": float(r), "p": float(p),
        "alpha_x": cx["alpha"], "alpha_y": cy["alpha"],
        "delta_alpha": float(cx["alpha"] - cy["alpha"]),
    }


# ----------------------------------------------------------------------
# Multi-scale: multiscale entropy curves
# ----------------------------------------------------------------------
def _coarse_grain(x: np.ndarray, tau: int) -> np.ndarray:
    """Non-overlapping mean coarse-graining used by Costa-style MSE."""
    n = (x.size // tau) * tau
    if n == 0:
        return np.empty(0)
    return x[:n].reshape(-1, tau).mean(axis=1)


def mse_curve(
    x: np.ndarray,
    scales: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20),
    m: int = 2,
    r_factor: float = 0.15,
) -> dict:
    """Multiscale (sample) entropy curve at each coarse-graining factor τ.

    Parameters
    ----------
    x : array-like
        1-D signal.
    scales : sequence of int
        Coarse-graining factors τ.
    m : int
        Embedding dimension for sample entropy (default 2).
    r_factor : float
        Tolerance as a fraction of the original-signal standard deviation
        (default 0.15 — common in the MSE literature).

    Returns
    -------
    dict
        ``scales`` (np.ndarray), ``mse`` (np.ndarray of sample entropy values
        per scale; NaN where the coarse-grained series is too short).
    """
    import antropy as ant
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    r_abs = r_factor * sd  # tolerance in original units (kept across scales)
    scales = np.asarray(scales, dtype=int)
    mse = np.full(scales.size, np.nan)
    for i, tau in enumerate(scales):
        cg = _coarse_grain(x, int(tau))
        if cg.size < (m + 2) * 5:
            continue
        try:
            mse[i] = float(ant.sample_entropy(cg))
        except Exception:  # noqa: BLE001
            continue
    return {"scales": scales, "mse": mse}


def mse_matching(
    x: np.ndarray,
    y: np.ndarray,
    scales: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20),
    m: int = 2,
    r_factor: float = 0.15,
) -> dict:
    """Pearson correlation between two MSE curves."""
    from scipy import stats as sps
    cx = mse_curve(x, scales=scales, m=m, r_factor=r_factor)
    cy = mse_curve(y, scales=scales, m=m, r_factor=r_factor)
    a, b = cx["mse"], cy["mse"]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return {"scales": cx["scales"], "mse_x": a, "mse_y": b,
                "r": float("nan"), "p": float("nan")}
    r, p = sps.pearsonr(a[valid], b[valid])
    return {"scales": cx["scales"], "mse_x": a, "mse_y": b,
            "r": float(r), "p": float(p)}


# ----------------------------------------------------------------------
# Time-resolved: windowed exponent + α(t) coupling
# ----------------------------------------------------------------------
def windowed_exponent(
    x: np.ndarray,
    fs: float,
    win_sec: float = 30.0,
    step_sec: float = 5.0,
    exponent_fn: Callable[[np.ndarray], float] = dfa_alpha,
) -> dict:
    """Sliding-window scaling exponent α(t).

    Returns
    -------
    dict
        ``times_s`` (window centers, seconds), ``exponent`` (one α per window).
    """
    x = np.asarray(x, dtype=float)
    win = int(win_sec * fs)
    step = max(1, int(step_sec * fs))
    starts = np.arange(0, x.size - win + 1, step)
    times = (starts + win / 2.0) / fs
    alphas = np.array([float(exponent_fn(x[s:s + win])) for s in starts])
    return {"times_s": times, "exponent": alphas,
            "win_sec": win_sec, "step_sec": step_sec}


def complexity_coupling(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    win_sec: float = 30.0,
    step_sec: float = 5.0,
    exponent_fn: Callable[[np.ndarray], float] = dfa_alpha,
    method: str = "pearson",
) -> dict:
    """Coupling between the two scaling-exponent traces α_x(t) and α_y(t).

    This composes with the rest of the toolbox: ``windowed_exponent`` is
    applied to each signal, then *any* standard coupling metric can be
    computed on the resulting α(t) traces. Two simple methods are built
    in here; for PLV / coherence / wPLI / windowed_xcorr, drop the
    α(t) traces into the corresponding helper from
    :mod:`HNA.coupling` directly.

    Parameters
    ----------
    x, y : array-like
        1-D signals (same fs).
    fs : float
    win_sec, step_sec : float
        Window settings for the exponent trace.
    exponent_fn : callable(x) -> float
        Per-window scaling-exponent estimator.
    method : str
        ``"pearson"`` (default) or ``"mi"`` (sklearn k-NN MI).

    Returns
    -------
    dict
        Always includes ``alpha_x`` (np.ndarray α_x(t)), ``alpha_y``
        (np.ndarray α_y(t)), ``times_s``, ``n_windows`` and ``coupling``
        (the scalar association). Pearson also returns ``p``.
    """
    rx = windowed_exponent(x, fs=fs, win_sec=win_sec, step_sec=step_sec,
                            exponent_fn=exponent_fn)
    ry = windowed_exponent(y, fs=fs, win_sec=win_sec, step_sec=step_sec,
                            exponent_fn=exponent_fn)
    a, b = rx["exponent"], ry["exponent"]
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    times = rx["times_s"][:n]
    valid = np.isfinite(a) & np.isfinite(b)
    out = {"alpha_x": a, "alpha_y": b, "times_s": times,
           "n_windows": int(valid.sum()), "method": method}

    if valid.sum() < 3:
        out["coupling"] = float("nan")
        if method == "pearson":
            out["p"] = float("nan")
        return out

    method = method.lower()
    if method == "pearson":
        from scipy import stats as sps
        r, p = sps.pearsonr(a[valid], b[valid])
        out["coupling"] = float(r)
        out["p"] = float(p)
    elif method == "mi":
        from sklearn.feature_selection import mutual_info_regression
        mi = float(mutual_info_regression(
            a[valid].reshape(-1, 1), b[valid],
            n_neighbors=3, random_state=42,
        )[0])
        out["coupling"] = mi
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'pearson' or 'mi'.")
    return out
