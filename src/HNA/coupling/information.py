"""Information-theoretic coupling: MI, effective MI, Granger, TE.

The information family asks: "is there *any* statistical dependence
between the two signals — linear or non-linear — and does that
dependence have a temporal direction?"

Methods
-------
- :func:`windowed_mi` — sliding-window mutual information using sklearn's
  k-NN estimator (Kraskov-style). Upward-biased on autocorrelated
  signals; use for *relative* comparison across conditions.
- :func:`effective_mi` / :func:`windowed_effective_mi` — bias-corrected
  MI: subtract the mean MI of phase-shuffle surrogates that share each
  signal's spectrum. Bound the absolute value back to "dependence above
  the spectral floor".
- :func:`granger_bivariate` / :func:`granger_score` /
  :func:`windowed_granger` — linear-Gaussian directional coupling via
  bivariate VAR (statsmodels). Granger is the linear special case of
  transfer entropy and so lives in the same family.
- :func:`transfer_entropy` — kNN-based transfer entropy stub. Imports
  ``copent`` lazily; raises with a helpful message if not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..dsp import interpolate_nan as _nan_interp
from ..surrogates import surrogate_test


# =====================================================================
# Mutual information (raw, sliding-window)
# =====================================================================
def windowed_mi(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    win_sec: float = 120.0,
    step_sec: float = 10.0,
    n_neighbors: int = 3,
    rng_seed: int = 42,
) -> dict:
    """Sliding-window mutual information between two 1-D signals.

    Uses ``sklearn.feature_selection.mutual_info_regression`` (kNN /
    Kraskov-like) on each window. Returns a dict with ``times_s`` (window
    centers) and ``mi`` (per-window MI). The estimator is upward-biased
    on autocorrelated signals — for absolute values, use
    :func:`effective_mi` instead.
    """
    from sklearn.feature_selection import mutual_info_regression  # lazy

    s1 = _nan_interp(np.asarray(s1, float))
    s2 = _nan_interp(np.asarray(s2, float))
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]

    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    if W <= 1 or W > n:
        raise ValueError("Window size must be >1 and <= length of signals.")

    starts = np.arange(0, n - W + 1, H)
    times_s = np.empty(len(starts), float)
    mi_vals = np.empty(len(starts), float)

    for i, st in enumerate(starts):
        x = s1[st:st + W].reshape(-1, 1)
        y = s2[st:st + W]
        try:
            mi_vals[i] = float(mutual_info_regression(
                x, y, n_neighbors=n_neighbors, random_state=rng_seed)[0])
        except Exception:  # noqa: BLE001
            mi_vals[i] = float("nan")
        times_s[i] = (st + W / 2.0) / fs

    return {
        "times_s": times_s, "mi": mi_vals,
        "n_neighbors": n_neighbors, "win_sec": win_sec, "step_sec": step_sec,
    }


# =====================================================================
# Effective (bias-corrected) MI
# =====================================================================
def effective_mi(
    x: np.ndarray,
    y: np.ndarray,
    fs: Optional[float] = None,
    n_surrogates: int = 200,
    n_neighbors: int = 3,
    method: Literal["phase_shuffle", "time_shift"] = "phase_shuffle",
    rng_seed: Optional[int] = 0,
) -> dict:
    """Bias-corrected mutual information.

    Subtracts the mean MI obtained on phase-shuffled surrogates (which
    share each signal's spectrum but break the inter-signal dependence)
    from the raw MI. The result is a one-number estimate of dependence
    above the spectral autocorrelation floor.

    Returns dict with ``mi_observed``, ``mi_null_mean``, ``mi_null_std``,
    ``mi_effective`` (= observed - null_mean), ``z``, ``p``, ``null``.
    """
    from sklearn.feature_selection import mutual_info_regression  # lazy

    def _mi_metric(a: np.ndarray, b: np.ndarray) -> float:
        return float(mutual_info_regression(
            a.reshape(-1, 1), b, n_neighbors=n_neighbors, random_state=42,
        )[0])

    obs, null, p, z = surrogate_test(
        _mi_metric, x, y,
        n=n_surrogates, method=method, surrogate_target="y",
        higher_is_better=True, rng_seed=rng_seed,
    )
    null_mean = float(np.nanmean(null))
    null_std = float(np.nanstd(null))
    return {
        "mi_observed": obs,
        "mi_null_mean": null_mean,
        "mi_null_std": null_std,
        "mi_effective": float(obs - null_mean),
        "z": z,
        "p": p,
        "null": null,
    }


def windowed_effective_mi(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    win_sec: float = 60.0,
    step_sec: float = 10.0,
    n_surrogates: int = 100,
    n_neighbors: int = 3,
    method: Literal["phase_shuffle", "time_shift"] = "phase_shuffle",
    rng_seed: Optional[int] = 0,
) -> dict:
    """Sliding-window :func:`effective_mi`.

    Returns a dict with arrays ``times_s, mi_observed, mi_null_mean,
    mi_effective, p, z`` plus the window settings.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    win = int(win_sec * fs)
    step = max(1, int(step_sec * fs))
    starts = np.arange(0, n - win + 1, step)
    if starts.size == 0:
        return {
            "times_s": np.empty(0), "mi_observed": np.empty(0),
            "mi_null_mean": np.empty(0), "mi_effective": np.empty(0),
            "p": np.empty(0), "z": np.empty(0),
            "win_sec": win_sec, "step_sec": step_sec,
        }

    times = (starts + win / 2.0) / fs
    obs = np.empty(starts.size)
    nm = np.empty(starts.size)
    eff = np.empty(starts.size)
    p_arr = np.empty(starts.size)
    z_arr = np.empty(starts.size)
    for i, s in enumerate(starts):
        e = s + win
        r = effective_mi(
            x[s:e], y[s:e], fs=fs,
            n_surrogates=n_surrogates,
            n_neighbors=n_neighbors,
            method=method,
            rng_seed=None if rng_seed is None else rng_seed + i,
        )
        obs[i] = r["mi_observed"]
        nm[i] = r["mi_null_mean"]
        eff[i] = r["mi_effective"]
        p_arr[i] = r["p"]
        z_arr[i] = r["z"]
    return {
        "times_s": times,
        "mi_observed": obs,
        "mi_null_mean": nm,
        "mi_effective": eff,
        "p": p_arr,
        "z": z_arr,
        "win_sec": win_sec,
        "step_sec": step_sec,
    }


# =====================================================================
# Granger causality (linear-Gaussian directional information)
# =====================================================================
@dataclass
class GrangerResult:
    lag: int
    x_to_y_F: float
    x_to_y_p: float
    y_to_x_F: float
    y_to_x_p: float
    ic: str


def granger_bivariate(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 10,
    ic: Literal["aic", "bic", "hqic", "fpe"] = "aic",
    detrend: bool = True,
) -> GrangerResult:
    """Bivariate Granger causality both directions via statsmodels VAR."""
    from statsmodels.tsa.api import VAR  # lazy
    from scipy import signal as _sps

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if detrend:
        x = _sps.detrend(x, type="linear")
        y = _sps.detrend(y, type="linear")

    df = pd.DataFrame({"x": x - x.mean(), "y": y - y.mean()})
    model = VAR(df)
    sel = model.select_order(maxlags=max_lag)
    lag = int(getattr(sel, ic))
    if lag < 1:
        lag = 1
    fit = model.fit(lag, trend="c")

    res_x_to_y = fit.test_causality("y", causing="x", kind="f")
    res_y_to_x = fit.test_causality("x", causing="y", kind="f")

    return GrangerResult(
        lag=lag,
        x_to_y_F=float(res_x_to_y.test_statistic),
        x_to_y_p=float(res_x_to_y.pvalue),
        y_to_x_F=float(res_y_to_x.test_statistic),
        y_to_x_p=float(res_y_to_x.pvalue),
        ic=ic,
    )


def granger_score(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """Single signed directionality score: positive => x drives y.

    Computed as ``log10(p_y_to_x) - log10(p_x_to_y)``; NaN-safe (returns
    0.0 if the fit fails).
    """
    try:
        r = granger_bivariate(x, y, **kwargs)
    except Exception:  # noqa: BLE001
        return 0.0
    eps = 1e-300
    return float(np.log10(max(r.y_to_x_p, eps)) - np.log10(max(r.x_to_y_p, eps)))


def windowed_granger(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    win_sec: float = 60.0,
    step_sec: float = 10.0,
    max_lag: int = 10,
    ic: Literal["aic", "bic", "hqic", "fpe"] = "aic",
    detrend: bool = True,
) -> dict:
    """Sliding-window bivariate Granger.

    Returns a dict with arrays ``times_s, x_to_y_F, x_to_y_p, y_to_x_F,
    y_to_x_p, lag`` plus the window settings.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    win = int(win_sec * fs)
    step = max(1, int(step_sec * fs))
    starts = np.arange(0, n - win + 1, step)
    if starts.size == 0:
        return {
            "times_s": np.empty(0), "x_to_y_F": np.empty(0),
            "x_to_y_p": np.empty(0), "y_to_x_F": np.empty(0),
            "y_to_x_p": np.empty(0), "lag": np.empty(0, dtype=int),
            "win_sec": win_sec, "step_sec": step_sec,
        }

    times = (starts + win / 2.0) / fs
    F_xy = np.full(starts.size, np.nan)
    P_xy = np.full(starts.size, np.nan)
    F_yx = np.full(starts.size, np.nan)
    P_yx = np.full(starts.size, np.nan)
    lags = np.zeros(starts.size, dtype=int)
    for i, s in enumerate(starts):
        e = s + win
        try:
            r = granger_bivariate(x[s:e], y[s:e], max_lag=max_lag, ic=ic,
                                  detrend=detrend)
            F_xy[i] = r.x_to_y_F
            P_xy[i] = r.x_to_y_p
            F_yx[i] = r.y_to_x_F
            P_yx[i] = r.y_to_x_p
            lags[i] = r.lag
        except Exception:  # noqa: BLE001
            continue
    return {
        "times_s": times,
        "x_to_y_F": F_xy, "x_to_y_p": P_xy,
        "y_to_x_F": F_yx, "y_to_x_p": P_yx,
        "lag": lags,
        "win_sec": win_sec, "step_sec": step_sec,
    }


# =====================================================================
# Transfer entropy (optional dependency: copent)
# =====================================================================
def transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    k: int = 4,
) -> dict:
    """kNN-based transfer entropy in both directions (lazy ``copent`` import).

    Raises ``ImportError`` with an install hint if ``copent`` is not
    available.
    """
    try:
        import copent
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "transfer_entropy() requires the 'copent' package. "
            "Install with: pip install copent"
        ) from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    te_xy = float(copent.transent(x, y, lag=lag, k=k))
    te_yx = float(copent.transent(y, x, lag=lag, k=k))
    return {"te_x_to_y": te_xy, "te_y_to_x": te_yx, "lag": lag, "k": k}


# =====================================================================
# Equal-frequency binning helper (used by transfer_entropy_binned + pid_2source)
# =====================================================================
def _equal_freq_bin(arr, n_bins):
    """Equal-frequency binning to ``n_bins`` integer levels.

    If the input has at most ``n_bins`` unique values (e.g. already
    discrete), they are returned mapped 1-to-1 via ``np.searchsorted``,
    avoiding the ``np.digitize`` edge-value collapse.
    """
    arr = np.asarray(arr, dtype=float)
    uniq = np.unique(arr)
    if len(uniq) <= n_bins:
        return np.searchsorted(uniq, arr).astype(int)
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(arr, q)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    idx = np.digitize(arr, edges) - 1
    return np.clip(idx, 0, n_bins - 1).astype(int)


# =====================================================================
# Schreiber-style binned TE via pyinform
# =====================================================================
def transfer_entropy_binned(
    x: np.ndarray,
    y: np.ndarray,
    history: int = 1,
    n_bins: int = 8,
) -> dict:
    """Schreiber 2000 transfer entropy via the ``pyinform`` package.

    Continuous data is equal-frequency-binned to ``n_bins`` levels per
    signal, then ``pyinform.transfer_entropy`` computes the binned TE in
    bits in both directions.

    Lazy import: raises ``ImportError`` with an install hint if
    ``pyinform`` is not available.
    """
    try:
        from pyinform import transfer_entropy as _pyte
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "transfer_entropy_binned() requires the 'pyinform' package. "
            "Install with: pip install pyinform. "
            "For a no-deps continuous estimator, use transfer_entropy_knn()."
        ) from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    xi = _equal_freq_bin(x[:n], n_bins)
    yi = _equal_freq_bin(y[:n], n_bins)
    te_xy = float(_pyte(xi, yi, k=history))
    te_yx = float(_pyte(yi, xi, k=history))
    return {"te_x_to_y": te_xy, "te_y_to_x": te_yx,
            "history": history, "n_bins": n_bins}


# =====================================================================
# Partial Information Decomposition (PID) via dit
# =====================================================================
def pid_2source(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    n_bins: int = 4,
    measure: str = "broja",
) -> dict:
    """Williams-Beer 2-source PID via the ``dit`` package.

    Decomposes :math:`I(\\{X_1, X_2\\}; Y)` into four non-negative atoms
    that sum to the joint MI:

    .. math::
       I(\\{X_1, X_2\\}; Y) = I_\\text{red} + I_\\text{unq}^{X_1}
                                + I_\\text{unq}^{X_2} + I_\\text{syn}.

    Continuous data is equal-frequency-binned to ``n_bins`` levels per
    variable. ``measure="broja"`` uses the Bertschinger et al. 2014
    non-negative decomposition (recommended); ``"wb"`` falls back to the
    original Williams-Beer 2010 :math:`I_\\min` measure.

    Returns ``{redundant, unique_x1, unique_x2, synergistic, mi_joint,
    measure, n_bins}`` (atoms in bits).

    Lazy import: raises ``ImportError`` if ``dit`` is not available.
    Note: ``dit`` PID is exponentially expensive in alphabet size; raise
    ``n_bins`` carefully (8 is roughly the upper limit for fast runs).
    """
    try:
        import dit
        from dit.pid import PID_BROJA, PID_WB
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "pid_2source() requires the 'dit' package. "
            "Install with: pip install dit"
        ) from e

    x1 = _equal_freq_bin(np.asarray(x1, dtype=float), n_bins)
    x2 = _equal_freq_bin(np.asarray(x2, dtype=float), n_bins)
    y = _equal_freq_bin(np.asarray(y, dtype=float), n_bins)
    n = min(len(x1), len(x2), len(y))
    x1, x2, y = x1[:n], x2[:n], y[:n]

    counts = {}
    for a, b, c in zip(x1, x2, y):
        key = (int(a), int(b), int(c))
        counts[key] = counts.get(key, 0) + 1
    outcomes = list(counts.keys())
    pmf = np.array([counts[o] for o in outcomes], dtype=float)
    pmf /= pmf.sum()
    d = dit.Distribution(outcomes, pmf)

    cls = PID_BROJA if measure.lower() == "broja" else PID_WB
    pid = cls(d)

    def _atom(key):
        try:
            return float(pid.get_pi(key))
        except Exception:
            return float("nan")

    # dit antichain key syntax (verified against pid._pis dict):
    #   ((0,),)        -> unique to source 0
    #   ((1,),)        -> unique to source 1
    #   ((0,), (1,))   -> redundancy (intersection of {0} and {1})
    #   ((0, 1),)      -> synergy (joint atom)
    return {
        "redundant":   _atom(((0,), (1,))),
        "unique_x1":   _atom(((0,),)),
        "unique_x2":   _atom(((1,),)),
        "synergistic": _atom(((0, 1),)),
        "mi_joint":    float(pid._total),
        "measure":     measure,
        "n_bins":      n_bins,
    }


# =====================================================================
# Integrated Information Decomposition (Phi-ID) via phyid
# =====================================================================
def phi_id(
    x: np.ndarray,
    y: np.ndarray,
    tau: int = 1,
    kind: str = "gaussian",
    redundancy: str = "MMI",
) -> dict:
    """Full 16-atom Integrated Information Decomposition (Mediano et al. 2021).

    Lazy wrapper around the ``phyid`` package. Decomposes the time-shifted
    information flow :math:`(X_t, Y_t) \\to (X_{t+\\tau}, Y_{t+\\tau})`
    into 16 atoms organised as a 4 x 4 lattice (4 source atom-types
    {redundant, unique-X, unique-Y, synergistic} crossed with the same
    4 target atom-types). Returns the **time-averaged** value of every
    atom plus several aggregate coupling summaries.

    Atom keys (``source-to-target``):
        ``rtr, rtx, rty, rts, xtr, xtx, xty, xts,
        ytr, ytx, yty, yts, str, stx, sty, sts``

    Aggregates returned:
        * ``transfer_x_to_y``  = ``rty + xty + sty`` --- canonical TE
          analogue: information landing on Y from sources containing X.
        * ``transfer_y_to_x``  = ``rtx + ytx + stx``
        * ``storage``          = ``rtr + xtx + yty + sts``
        * ``synergy``          = ``sts``  --- info present in joint
          dynamics that no marginal carries (the canonical "emergence"
          atom).
        * ``total``            = sum of all 16 atoms.

    Parameters
    ----------
    x, y : array-like, shape (N,)
    tau : int
        Time-lag for the source -> target transition (default 1).
    kind : str
        ``"gaussian"`` (continuous, analytic) or ``"discrete"``.
    redundancy : str
        ``"MMI"`` (Minimum Mutual Information; default) or ``"CCS"``
        (common change in surprisal). MMI is the most widely-used.

    Lazy import: raises ``ImportError`` with an install hint if
    ``phyid`` is not available.
    """
    try:
        from phyid.calculate import calc_PhiID
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "phi_id() requires the 'phyid' package "
            "(Mediano et al. 2021). Install with: "
            "pip install git+https://github.com/Imperial-MIND-lab/integrated-info-decomp"
        ) from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    atoms, _ = calc_PhiID(x, y, tau=tau, kind=kind, redundancy=redundancy)

    means = {k: float(np.nanmean(v)) for k, v in atoms.items()}

    transfer_x_to_y = means.get("rty", 0) + means.get("xty", 0) + means.get("sty", 0)
    transfer_y_to_x = means.get("rtx", 0) + means.get("ytx", 0) + means.get("stx", 0)
    storage = (means.get("rtr", 0) + means.get("xtx", 0)
               + means.get("yty", 0) + means.get("sts", 0))
    synergy = means.get("sts", 0)
    total = sum(means.values())

    return {
        "atoms":           means,
        "transfer_x_to_y": transfer_x_to_y,
        "transfer_y_to_x": transfer_y_to_x,
        "storage":         storage,
        "synergy":         synergy,
        "total":           total,
        "tau":             tau,
        "kind":            kind,
        "redundancy":      redundancy,
    }
