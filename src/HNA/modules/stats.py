"""
Statistical helpers for the HNA toolbox.

Grouped by purpose:
  - Linear: ``fisher_z`` / ``inv_fisher_z``                  (Pearson r transformation)
  - Multiple comparisons: ``fdr_bh``                          (Benjamini-Hochberg FDR)
  - Group tests: ``friedman_with_posthoc``                    (omnibus + paired Wilcoxon)
  - Circular: ``rayleigh_test``, ``circular_mean``, ``circular_R`` (preferred-phase / phase-locking)
  - Slope tests: ``per_subject_slopes`` and ``slopes_one_sample_wilcoxon`` (used by Analysis C)

These are generic enough to be reused across modalities (EEG, ECG/HRV,
respiration, audio) and across coupling methods.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Optional

import numpy as np


# -----------------------------
# Linear stats
# -----------------------------
def fisher_z(r: np.ndarray | float) -> np.ndarray | float:
    """Fisher's z transformation for Pearson correlations."""
    r_clipped = np.clip(np.asarray(r, float), -0.9999, 0.9999)
    return np.arctanh(r_clipped)


def inv_fisher_z(z: np.ndarray | float) -> np.ndarray | float:
    """Inverse Fisher transformation."""
    return np.tanh(np.asarray(z, float))


# -----------------------------
# Multiple comparisons
# -----------------------------
def fdr_bh(pvals: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg FDR. Returns (p_adjusted, reject) arrays."""
    from statsmodels.stats.multitest import multipletests
    pvals = np.asarray(pvals, float)
    valid = np.isfinite(pvals)
    p_adj = np.full_like(pvals, np.nan, dtype=float)
    reject = np.zeros_like(pvals, dtype=bool)
    if valid.any():
        rej_v, p_v, *_ = multipletests(pvals[valid], alpha=alpha, method="fdr_bh")
        p_adj[valid] = p_v
        reject[valid] = rej_v
    return p_adj, reject


# -----------------------------
# Repeated-measures group tests
# -----------------------------
def friedman_with_posthoc(data_matrix: np.ndarray,
                          condition_labels: Sequence[str]) -> dict:
    """Run Friedman omnibus + pairwise Wilcoxon post-hoc tests.

    Parameters
    ----------
    data_matrix : np.ndarray of shape (n_subjects, n_conditions)
        Cell (i, j) is subject i's score under condition j. NaNs are dropped row-wise.
    condition_labels : sequence of str
        Names for the n_conditions columns. Used in the post-hoc dict keys.

    Returns
    -------
    dict with keys:
        n           : number of complete-case subjects
        friedman_chi2, friedman_p
        posthoc     : list of {pair, stat, p}
    """
    from itertools import combinations
    from scipy import stats as sps

    data_matrix = np.asarray(data_matrix, float)
    valid = np.all(np.isfinite(data_matrix), axis=1)
    M = data_matrix[valid]
    out = {"n": int(M.shape[0]), "friedman_chi2": float("nan"),
           "friedman_p": float("nan"), "posthoc": []}

    if M.shape[0] >= 3 and M.shape[1] >= 2:
        try:
            chi2, p = sps.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
            out["friedman_chi2"] = float(chi2)
            out["friedman_p"] = float(p)
        except Exception:
            pass

    # Pairwise Wilcoxon
    for i, j in combinations(range(M.shape[1]), 2):
        if M.shape[0] < 3:
            out["posthoc"].append({"pair": f"{condition_labels[i]}_vs_{condition_labels[j]}",
                                   "stat": float("nan"), "p": float("nan")})
            continue
        try:
            res = sps.wilcoxon(M[:, i], M[:, j], alternative="two-sided",
                               zero_method="wilcox", nan_policy="omit")
            out["posthoc"].append({"pair": f"{condition_labels[i]}_vs_{condition_labels[j]}",
                                   "stat": float(res.statistic), "p": float(res.pvalue)})
        except Exception:
            out["posthoc"].append({"pair": f"{condition_labels[i]}_vs_{condition_labels[j]}",
                                   "stat": float("nan"), "p": float("nan")})
    return out


# -----------------------------
# Circular statistics
# -----------------------------
def circular_mean(angles: np.ndarray) -> float:
    """Circular mean of a 1-D array of angles (radians)."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.angle(np.exp(1j * a).mean()))


def circular_R(angles: np.ndarray) -> float:
    """Mean resultant length R in [0, 1] for a 1-D array of angles (radians)."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.abs(np.exp(1j * a).mean()))


def rayleigh_test(angles: np.ndarray) -> Tuple[float, float]:
    """Rayleigh test for circular non-uniformity.

    Returns (R, p) using the asymptotic Mardia-Jupp approximation,
    valid for n >= 5; lower n gives an upward-biased (conservative) p.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return float("nan"), float("nan")
    R = circular_R(a)
    z = n * R**2
    # Mardia & Jupp (2000) approximation
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - z * n)) - (1 + 2 * n)))
    return R, p


# -----------------------------
# Slope tests (used by time-resolved coupling; see scripts/figures/analysis_C_*)
# -----------------------------
def per_subject_slopes(times_per_subj: Iterable[np.ndarray],
                       values_per_subj: Iterable[np.ndarray],
                       normalize_time: bool = True) -> np.ndarray:
    """Compute one linear-trend slope per subject.

    Each subject's series is fitted with simple linear regression
    (``scipy.stats.linregress``). Returns the array of slopes (NaN for any
    subject with fewer than 4 finite samples).
    """
    from scipy.stats import linregress
    out = []
    for t, v in zip(times_per_subj, values_per_subj):
        t = np.asarray(t, float); v = np.asarray(v, float)
        m = np.isfinite(t) & np.isfinite(v)
        if m.sum() < 4:
            out.append(float("nan"))
            continue
        tt = t[m]; vv = v[m]
        if normalize_time:
            span = tt.max() - tt.min()
            if span > 0:
                tt = (tt - tt.min()) / span
        out.append(float(linregress(tt, vv).slope))
    return np.asarray(out)


def slopes_one_sample_wilcoxon(slopes: np.ndarray) -> Tuple[float, float, int]:
    """One-sample Wilcoxon vs zero on per-subject slopes.

    Returns (mean_slope, p, n_used).
    """
    from scipy.stats import wilcoxon
    s = np.asarray(slopes, float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return (float(np.mean(s)) if len(s) else float("nan")), float("nan"), int(len(s))
    try:
        p = float(wilcoxon(s, alternative="two-sided",
                           zero_method="wilcox", nan_policy="omit").pvalue)
    except Exception:
        p = float("nan")
    return float(np.mean(s)), p, int(len(s))
