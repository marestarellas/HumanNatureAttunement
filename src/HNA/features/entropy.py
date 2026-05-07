"""Entropy / complexity scalars for any 1-D signal.

Each function operates on a single 1-D array and returns a float
(``np.nan`` on numerical failure). They are pure thin wrappers around
``antropy`` so the toolbox does not duplicate library code.

The convenience aggregator :func:`all_entropies` returns a dict keyed by
the legacy column names used elsewhere in the project so the resulting
DataFrames slot into existing analyses without renames.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import antropy as ant


def lzc(x: np.ndarray, normalize: bool = True) -> float:
    """Lempel–Ziv complexity of a binarized version of ``x`` (median-split).

    Returns the normalized LZC by default (range ``[0, 1]``).
    """
    try:
        x = np.asarray(x, dtype=float)
        median = np.median(x)
        binary_seq = "".join("1" if v > median else "0" for v in x)
        return float(ant.lziv_complexity(binary_seq, normalize=normalize))
    except Exception:  # noqa: BLE001
        return float("nan")


def perm_entropy(x: np.ndarray, order: int = 3, normalize: bool = True) -> float:
    """Permutation entropy."""
    try:
        return float(ant.perm_entropy(np.asarray(x, dtype=float),
                                      order=order, normalize=normalize))
    except Exception:  # noqa: BLE001
        return float("nan")


def spectral_entropy(x: np.ndarray, fs: float, normalize: bool = True) -> float:
    """Spectral entropy (Welch-based)."""
    try:
        return float(ant.spectral_entropy(np.asarray(x, dtype=float),
                                          sf=fs, method="welch",
                                          normalize=normalize))
    except Exception:  # noqa: BLE001
        return float("nan")


def svd_entropy(x: np.ndarray, normalize: bool = True) -> float:
    """SVD entropy."""
    try:
        return float(ant.svd_entropy(np.asarray(x, dtype=float),
                                     normalize=normalize))
    except Exception:  # noqa: BLE001
        return float("nan")


def sample_entropy(x: np.ndarray) -> float:
    """Sample entropy."""
    try:
        return float(ant.sample_entropy(np.asarray(x, dtype=float)))
    except Exception:  # noqa: BLE001
        return float("nan")


def all_entropies(x: np.ndarray, fs: Optional[float] = None) -> dict:
    """Compute every entropy metric and return a single dict.

    Parameters
    ----------
    x : array-like
        1-D signal.
    fs : float, optional
        Sampling rate. Required for :func:`spectral_entropy`. If ``None``,
        spectral entropy is set to NaN.

    Returns
    -------
    dict
        Keys: ``lzc, perm_entropy, spectral_entropy, svd_entropy,
        sample_entropy`` — matches the legacy ``compute_entropy_features``
        column order.
    """
    return {
        "lzc": lzc(x),
        "perm_entropy": perm_entropy(x),
        "spectral_entropy": (
            spectral_entropy(x, fs=fs) if fs is not None else float("nan")
        ),
        "svd_entropy": svd_entropy(x),
        "sample_entropy": sample_entropy(x),
    }
