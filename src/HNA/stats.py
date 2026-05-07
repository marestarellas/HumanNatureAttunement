"""
Statistical helpers for the HNA toolbox.

Grouped by purpose:
  - Linear: ``fisher_z`` / ``inv_fisher_z``                  (Pearson r transformation)
  - Multiple comparisons: ``fdr_bh``                          (Benjamini-Hochberg FDR)
  - Group tests: ``friedman_with_posthoc``                    (omnibus + paired Wilcoxon)
  - Circular: ``rayleigh_test``, ``circular_mean``, ``circular_R`` (preferred-phase / phase-locking)
  - Slope tests: ``per_subject_slopes`` and ``slopes_one_sample_wilcoxon`` (used by Analysis C)
  - Cluster permutation: ``cluster_permutation_paired_1d`` /
    ``cluster_permutation_two_sample_1d`` (Maris/Oostenveld correction
    for time- or frequency-resolved repeated-measures contrasts).

These are generic enough to be reused across modalities (EEG, ECG/HRV,
respiration, audio) and across coupling methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple, Optional

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


# =====================================================================
# Cluster-based permutation testing (Maris & Oostenveld 2007)
# =====================================================================

# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass
class Cluster:
    start: int                  # inclusive
    stop: int                   # exclusive
    mass: float                 # signed sum of stat values in the cluster
    sign: int                   # +1 or -1
    p_value: float


@dataclass
class ClusterResult:
    stat_observed: np.ndarray   # per-time-point statistic, shape (T,)
    threshold: float            # absolute statistic threshold used
    clusters: List[Cluster] = field(default_factory=list)
    null_max_mass: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_permutations: int = 0

    @property
    def significant_mask(self) -> np.ndarray:
        """Boolean mask over time of points belonging to a p<0.05 cluster."""
        m = np.zeros_like(self.stat_observed, dtype=bool)
        for c in self.clusters:
            if c.p_value < 0.05:
                m[c.start:c.stop] = True
        return m


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _paired_t(diff: np.ndarray) -> np.ndarray:
    """Per-column one-sample t against zero. ``diff`` shape (n_subjects, T)."""
    n = diff.shape[0]
    mean = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1)
    se = sd / np.sqrt(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(se > 0, mean / se, 0.0)
    return t


def _two_sample_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Welch t per column. ``a`` shape (n_a, T), ``b`` shape (n_b, T)."""
    ma, va = a.mean(axis=0), a.var(axis=0, ddof=1)
    mb, vb = b.mean(axis=0), b.var(axis=0, ddof=1)
    se = np.sqrt(va / a.shape[0] + vb / b.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(se > 0, (ma - mb) / se, 0.0)
    return t


def _find_clusters(
    stat: np.ndarray, threshold: float
) -> List[tuple[int, int, int]]:
    """Find contiguous runs where ``|stat| > threshold``.

    Returns a list of ``(start, stop, sign)`` where ``stop`` is exclusive.
    """
    above = np.abs(stat) > threshold
    out = []
    i = 0
    n = stat.size
    while i < n:
        if above[i]:
            sign = 1 if stat[i] > 0 else -1
            j = i
            while j < n and above[j] and (np.sign(stat[j]) == sign):
                j += 1
            out.append((i, j, sign))
            i = j
        else:
            i += 1
    return out


def _cluster_masses(stat: np.ndarray, threshold: float) -> List[tuple[int, int, int, float]]:
    """Return ``[(start, stop, sign, mass), ...]`` for every cluster."""
    clusters = _find_clusters(stat, threshold)
    return [(s, e, sg, float(stat[s:e].sum())) for (s, e, sg) in clusters]


def _max_abs_mass(stat: np.ndarray, threshold: float) -> float:
    out = _cluster_masses(stat, threshold)
    if not out:
        return 0.0
    return float(max(abs(c[3]) for c in out))


# ---------------------------------------------------------------------
# Paired / one-sample
# ---------------------------------------------------------------------
def cluster_permutation_paired_1d(
    diff: np.ndarray,
    n_permutations: int = 1000,
    threshold: Optional[float] = None,
    threshold_alpha: float = 0.05,
    rng_seed: int = 0,
) -> ClusterResult:
    """One-sample / paired cluster permutation test on per-subject differences.

    Parameters
    ----------
    diff : np.ndarray, shape (n_subjects, n_timepoints)
        Per-subject difference series (NATURE - REST, post - pre, ...).
    n_permutations : int
        Number of sign-flip permutations (default 1000).
    threshold : float, optional
        Absolute cluster-forming threshold on the paired-t statistic. If
        ``None``, uses the two-tailed t critical value at
        ``threshold_alpha`` with ``df = n_subjects - 1``.
    threshold_alpha : float
        Used only when ``threshold`` is ``None`` (default 0.05).
    rng_seed : int
        Seed for the sign-flip permutations.

    Returns
    -------
    :class:`ClusterResult`
    """
    diff = np.asarray(diff, dtype=float)
    if diff.ndim != 2:
        raise ValueError("diff must be 2-D: (n_subjects, n_timepoints)")
    n, T = diff.shape

    if threshold is None:
        from scipy import stats as sps
        df = max(n - 1, 1)
        threshold = float(sps.t.ppf(1 - threshold_alpha / 2, df))

    stat_obs = _paired_t(diff)
    obs = _cluster_masses(stat_obs, threshold)

    rng = np.random.default_rng(rng_seed)
    null_max = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=(n, 1))
        stat_perm = _paired_t(signs * diff)
        null_max[i] = _max_abs_mass(stat_perm, threshold)

    clusters: List[Cluster] = []
    for (s, e, sg, mass) in obs:
        # Two-sided p: fraction of permutation max-cluster-masses ≥ |observed|
        p_val = float((null_max >= abs(mass)).sum() + 1) / (n_permutations + 1)
        clusters.append(Cluster(start=s, stop=e, mass=mass, sign=sg, p_value=p_val))

    # Sort: most extreme (smallest p) first.
    clusters.sort(key=lambda c: c.p_value)

    return ClusterResult(
        stat_observed=stat_obs,
        threshold=threshold,
        clusters=clusters,
        null_max_mass=null_max,
        n_permutations=n_permutations,
    )


# ---------------------------------------------------------------------
# Two-sample
# ---------------------------------------------------------------------
def cluster_permutation_two_sample_1d(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 1000,
    threshold: Optional[float] = None,
    threshold_alpha: float = 0.05,
    rng_seed: int = 0,
) -> ClusterResult:
    """Independent-samples cluster permutation test.

    Parameters
    ----------
    a, b : np.ndarray
        Each of shape ``(n, n_timepoints)``. Rows are observations.
    n_permutations : int
        Number of label-shuffle permutations.
    threshold : float, optional
        Absolute cluster-forming threshold on the Welch-t statistic. If
        ``None``, uses the two-tailed t critical value at ``threshold_alpha``
        with ``df = n_a + n_b - 2`` (a conservative pooled-df approximation).
    threshold_alpha : float
        Used only when ``threshold`` is ``None`` (default 0.05).
    rng_seed : int
        RNG seed for the permutations.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("a and b must be 2-D with the same n_timepoints")
    n_a, T = a.shape
    n_b = b.shape[0]

    if threshold is None:
        from scipy import stats as sps
        df = max(n_a + n_b - 2, 1)
        threshold = float(sps.t.ppf(1 - threshold_alpha / 2, df))

    stat_obs = _two_sample_t(a, b)
    obs = _cluster_masses(stat_obs, threshold)

    pooled = np.vstack([a, b])
    n_total = n_a + n_b
    rng = np.random.default_rng(rng_seed)

    null_max = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        idx = rng.permutation(n_total)
        ap = pooled[idx[:n_a]]
        bp = pooled[idx[n_a:]]
        stat_perm = _two_sample_t(ap, bp)
        null_max[i] = _max_abs_mass(stat_perm, threshold)

    clusters: List[Cluster] = []
    for (s, e, sg, mass) in obs:
        p_val = float((null_max >= abs(mass)).sum() + 1) / (n_permutations + 1)
        clusters.append(Cluster(start=s, stop=e, mass=mass, sign=sg, p_value=p_val))
    clusters.sort(key=lambda c: c.p_value)

    return ClusterResult(
        stat_observed=stat_obs,
        threshold=threshold,
        clusters=clusters,
        null_max_mass=null_max,
        n_permutations=n_permutations,
    )
