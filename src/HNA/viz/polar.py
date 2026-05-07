"""Polar-axis helpers for circular / phase data."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def polar_phase_plot(
    ax: plt.PolarAxes,
    phases: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    n_bins: int = 24,
    color: str = "#3B7DD8",
    show_mean_vector: bool = True,
    cardinal_only: bool = True,
    title: Optional[str] = None,
) -> plt.PolarAxes:
    """Render a circular histogram + (optional) mean-vector arrow into ``ax``.

    Parameters
    ----------
    ax : matplotlib polar Axes
        Use ``plt.subplots(subplot_kw={'projection': 'polar'})`` to make one.
    phases : array-like, shape (n,)
        Phase angles in radians, range ``[-π, π]`` or ``[0, 2π]``.
    weights : array-like, shape (n,), optional
        Per-sample weight (e.g. PLV or coupling strength). If ``None``,
        each sample gets weight 1.
    n_bins : int
        Number of phase bins (default 24 → 15° each).
    color : str
        Bin / arrow color.
    show_mean_vector : bool
        Whether to overlay an arrow at the (weighted) mean phase, with
        length equal to the resultant length R (∈ [0, 1]).
    cardinal_only : bool
        If True (default), only show the 0°, 90°, 180°, 270° tick labels —
        useful when the figure is small or the title would overlap 90°/270°.
    title : str, optional

    Returns
    -------
    The same ``ax``.
    """
    phases = np.asarray(phases, dtype=float)
    if weights is None:
        weights = np.ones_like(phases)
    else:
        weights = np.asarray(weights, dtype=float)

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 2 * np.pi / n_bins

    counts = np.zeros(n_bins, dtype=float)
    idx = np.digitize(phases, bin_edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    for k, w in zip(idx, weights):
        counts[k] += w
    if counts.sum() > 0:
        counts = counts / counts.sum()

    ax.bar(bin_centers, counts, width=width, color=color,
           edgecolor="white", linewidth=0.6, alpha=0.85)

    if show_mean_vector and phases.size > 0:
        z = np.sum(weights * np.exp(1j * phases))
        wsum = weights.sum()
        if wsum > 0:
            R = np.abs(z) / wsum
            mu = np.angle(z)
            ax.annotate(
                "",
                xy=(mu, R * counts.max() if counts.max() > 0 else R),
                xytext=(mu, 0),
                arrowprops=dict(arrowstyle="->", color="#222", lw=1.4),
            )

    if cardinal_only:
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0°", "", "180°", ""])

    ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    return ax
