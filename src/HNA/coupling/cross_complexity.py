"""Cross-complexity coupling: genuinely scale- / regularity-aware bivariate measures.

Until this submodule existed, every method named "complexity" in the HNA
toolbox was actually a *linear* comparison applied to a univariate complexity
feature (cf. :mod:`HNA.coupling.complexity`). The estimators in this file
fill the missing column of the framework matrix: they are bivariate by
construction, scale-aware, and non-linear.

Public API
----------
- :func:`dcca` --- Detrended Cross-Correlation Analysis. Returns the
  cross-fluctuation function :math:`F_{xy}(s)` over scales and the joint
  scaling exponent :math:`\\lambda_{xy}` (Podobnik & Stanley 2008).
- :func:`dcca_rho` --- the DCCA correlation coefficient
  :math:`\\rho_{xy}(s) = F_{xy}^2(s) / [F_x(s)\\, F_y(s)]` in :math:`[-1, 1]`,
  the bivariate analogue of Pearson r at each scale (Zebende 2011).
- :func:`cross_sample_entropy` --- bivariate sample entropy, the regularity
  of the joint dynamics at the native sampling rate (Pincus & Singer 1996).
- :func:`multiscale_cross_entropy` --- coarse-grain at each :math:`\\tau`
  then cross-sample-entropy, giving a regularity-vs-scale curve
  (Yan et al. 2008).

All four methods are pure numpy / scipy and ship without external
dependencies. They are organised here (rather than in
:mod:`HNA.coupling.complexity`) because their *coupling step itself* is
non-linear and scale-aware, in contrast to the four "linear coupling on
a complexity feature" methods that share the older filename.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------
# DCCA (Detrended Cross-Correlation Analysis)
# ---------------------------------------------------------------------
def _detrend_polyfit(segs: np.ndarray, order: int = 1) -> np.ndarray:
    """Vectorised polynomial detrending of an array of segments.

    ``segs`` shape ``(n_seg, s)``. Returns residuals after subtracting the
    least-squares polynomial of given ``order`` from each segment.
    """
    s = segs.shape[1]
    t = np.arange(s, dtype=float)
    # ``np.polyfit`` accepts 2-D y and returns coefs shape (order+1, n_seg).
    coefs = np.polyfit(t, segs.T, deg=order)
    trend = np.polyval(coefs, t.reshape(-1, 1)).T   # (n_seg, s)
    return segs - trend


def dcca(
    x: np.ndarray,
    y: np.ndarray,
    scales: Optional[Sequence[int]] = None,
    order: int = 1,
) -> dict:
    """Detrended Cross-Correlation Analysis.

    Parameters
    ----------
    x, y : array-like
        Two 1-D signals of equal length (truncated to the shorter).
    scales : sequence of int, optional
        Window sizes (samples) at which to evaluate the fluctuation
        function. Defaults to ~25 log-spaced values from 8 to ``len(x)//4``.
    order : int
        Polynomial detrending order (1 = linear, the standard choice).

    Returns
    -------
    dict
        ``scales`` (np.ndarray of int), ``F_xy`` (signed cross-fluctuation
        per scale), ``F_x`` and ``F_y`` (univariate fluctuation functions
        for each signal --- handy for normalisation / debugging),
        ``lambda_xy`` (the joint scaling exponent obtained by least-
        squares fitting :math:`\\log |F_{xy}|` vs :math:`\\log s`).

    Notes
    -----
    The cross-fluctuation function is signed: at any scale where the
    detrended residuals of x and y anti-correlate within windows,
    :math:`F_{xy}` will be negative. The scaling exponent
    :math:`\\lambda_{xy}` is fitted on :math:`|F_{xy}|`, so its
    interpretation is the same as univariate DFA :math:`\\alpha`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # Integrated profiles (cumulative-sum of mean-centred signal).
    X = np.cumsum(x - x.mean())
    Y = np.cumsum(y - y.mean())

    if scales is None:
        scales = np.unique(
            np.logspace(np.log10(8), np.log10(max(16, n // 4)), 25).astype(int)
        )
    scales = np.asarray(scales, dtype=int)

    F_xy = np.full(scales.size, np.nan, dtype=float)
    F_x = np.full(scales.size, np.nan, dtype=float)
    F_y = np.full(scales.size, np.nan, dtype=float)

    for i, s in enumerate(scales):
        if s < 4 or s > n // 2:
            continue
        n_seg = n // s
        if n_seg < 1:
            continue
        Xseg = X[: n_seg * s].reshape(n_seg, s)
        Yseg = Y[: n_seg * s].reshape(n_seg, s)

        rX = _detrend_polyfit(Xseg, order=order)
        rY = _detrend_polyfit(Yseg, order=order)

        # Fluctuation functions (mean over segments of mean-of-products).
        F_xy[i] = float((rX * rY).mean())
        F_x[i] = float((rX * rX).mean())
        F_y[i] = float((rY * rY).mean())

    # Joint scaling exponent: slope of log|F_xy| vs log s.
    valid = np.isfinite(F_xy) & (np.abs(F_xy) > 0) & np.isfinite(F_x) & np.isfinite(F_y)
    if valid.sum() >= 4:
        # |F_xy(s)| ~ s^{2*lambda_xy} (because F is mean-of-products,
        # which scales like the square of the linear fluctuation).
        slope, _ = np.polyfit(
            np.log(scales[valid]),
            np.log(np.abs(F_xy[valid])),
            1,
        )
        lambda_xy = float(slope / 2.0)
    else:
        lambda_xy = float("nan")

    return {
        "scales": scales,
        "F_xy": F_xy,
        "F_x": F_x,
        "F_y": F_y,
        "lambda_xy": lambda_xy,
    }


def dcca_rho(
    x: np.ndarray,
    y: np.ndarray,
    scales: Optional[Sequence[int]] = None,
    order: int = 1,
) -> dict:
    """DCCA correlation coefficient :math:`\\rho_{xy}(s) \\in [-1, 1]`.

    Defined as :math:`F_{xy}(s) / \\sqrt{F_x(s) \\, F_y(s)}` --- the
    bivariate analogue of Pearson r at each scale.

    Returns a dict with ``scales`` and ``rho``, plus the underlying
    fluctuation functions ``F_xy``, ``F_x``, ``F_y``.
    """
    res = dcca(x, y, scales=scales, order=order)
    Fxy, Fx, Fy = res["F_xy"], res["F_x"], res["F_y"]
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(Fx * Fy)
        rho = np.where(denom > 0, Fxy / denom, np.nan)
    # Numerical clip: rounding can push slightly past +/- 1.
    rho = np.clip(rho, -1.0, 1.0)
    return {
        "scales": res["scales"],
        "rho": rho,
        "F_xy": Fxy,
        "F_x": Fx,
        "F_y": Fy,
        "lambda_xy": res["lambda_xy"],
    }


# ---------------------------------------------------------------------
# Cross-sample-entropy (Pincus & Singer 1996)
# ---------------------------------------------------------------------
def cross_sample_entropy(
    x: np.ndarray,
    y: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    r_factor: float = 0.15,
) -> float:
    """Cross-sample-entropy of two equal-length 1-D signals.

    Counts the proportion of length-:math:`m` template vector pairs
    :math:`(X^m_i, Y^m_j)` that match within tolerance :math:`r`, then
    repeats at :math:`m+1`, and returns
    :math:`-\\log(B_{m+1} / B_m)` --- the bivariate analogue of sample
    entropy. Returns ``np.nan`` if either count is zero.

    Parameters
    ----------
    x, y : array-like
        1-D signals of (preferably equal) length.
    m : int
        Embedding dimension (default 2).
    r : float, optional
        Absolute distance tolerance. Pass either ``r`` or ``r_factor``.
    r_factor : float
        Tolerance as a fraction of the std of the concatenated
        ``[x, y]`` signal (default 0.15, the standard MSE choice).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if r is None:
        sd = float(np.std(np.concatenate([x, y])))
        r = r_factor * sd
    if r <= 0 or n <= m + 1:
        return float("nan")

    def _count_matches(A: np.ndarray, B: np.ndarray, r_: float) -> float:
        """Fraction of pairs (i, j) with ||A_i - B_j||_inf < r_."""
        n_a, n_b = A.shape[0], B.shape[0]
        if n_a == 0 or n_b == 0:
            return 0.0
        # Process row-by-row to keep memory bounded.
        count = 0
        for i in range(n_a):
            d = np.max(np.abs(B - A[i]), axis=1)
            count += int(np.sum(d < r_))
        return count / (n_a * n_b)

    # Embed: create sliding-window matrices of length m and m+1.
    Xm = np.lib.stride_tricks.sliding_window_view(x, m)
    Ym = np.lib.stride_tricks.sliding_window_view(y, m)
    Xm1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    Ym1 = np.lib.stride_tricks.sliding_window_view(y, m + 1)
    # Drop the last template of length m so Xm and Xm1 are aligned.
    Xm = Xm[: len(Xm1)]
    Ym = Ym[: len(Ym1)]

    Bm = _count_matches(Xm, Ym, r)
    Bm1 = _count_matches(Xm1, Ym1, r)
    if Bm <= 0 or Bm1 <= 0:
        return float("nan")
    return float(-np.log(Bm1 / Bm))


# ---------------------------------------------------------------------
# Multiscale cross-sample-entropy (coarse-graining + cross-SampEn)
# ---------------------------------------------------------------------
def _coarse_grain(z: np.ndarray, tau: int) -> np.ndarray:
    """Costa-style non-overlapping mean coarse-graining."""
    n = (z.size // tau) * tau
    if n == 0:
        return np.empty(0, dtype=float)
    return z[:n].reshape(-1, tau).mean(axis=1)


def multiscale_cross_entropy(
    x: np.ndarray,
    y: np.ndarray,
    scales: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20),
    m: int = 2,
    r_factor: float = 0.15,
) -> dict:
    """Multiscale cross-sample-entropy curve.

    For each coarse-graining factor :math:`\\tau`, the two signals are
    averaged in non-overlapping windows of length :math:`\\tau` and
    cross-sample-entropy is computed on the coarse-grained pair. The
    tolerance :math:`r` is fixed at the *original-signal* combined std
    (standard MSE convention) so that values across scales are
    comparable.

    Returns ``{"scales": np.ndarray, "cross_sampen": np.ndarray}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sd = float(np.std(np.concatenate([x, y])))
    r_abs = r_factor * sd
    scales = np.asarray(scales, dtype=int)
    out = np.full(scales.size, np.nan, dtype=float)
    for i, tau in enumerate(scales):
        if tau <= 0:
            continue
        xc = _coarse_grain(x, int(tau))
        yc = _coarse_grain(y, int(tau))
        if xc.size < (m + 2) * 5 or yc.size < (m + 2) * 5:
            continue
        out[i] = cross_sample_entropy(xc, yc, m=m, r=r_abs)
    return {"scales": scales, "cross_sampen": out}
