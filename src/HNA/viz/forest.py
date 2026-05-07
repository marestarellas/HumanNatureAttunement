"""Per-subject "forest" plots of paired contrasts.

A forest plot stacks per-subject point estimates (with optional CIs) along
one ordered axis and a vertical zero-line, so you can see at a glance
whether the effect is consistent in direction across subjects.

The helper here is intentionally lightweight: it takes a 1-D vector of
per-subject deltas (and optional CIs) and writes a publication-ready panel
into a Matplotlib axis. For the multi-axis nature-vs-rest grid, call this
once per subplot.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def forest_plot(
    ax: plt.Axes,
    deltas: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    ci_low: Optional[Sequence[float]] = None,
    ci_high: Optional[Sequence[float]] = None,
    color: str = "#3B7DD8",
    color_negative: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Δ (effect size)",
    zero_line: bool = True,
) -> plt.Axes:
    """Render a forest plot of per-subject deltas into ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Existing axis to draw into. Use ``plt.subplots()`` first.
    deltas : array-like of float, shape (n_subjects,)
        Per-subject effect sizes (e.g. ``mean(NATURE) - mean(REST)``).
    labels : sequence of str, optional
        Y-axis labels (one per subject). Defaults to ``S1, S2, …``.
    ci_low, ci_high : array-like of float, optional
        Lower and upper bounds of a per-subject confidence interval.
    color : str
        Marker / errorbar color for positive deltas (default blue).
    color_negative : str, optional
        Color for negative deltas. If ``None`` (default), uses ``color``.
    title : str, optional
    xlabel : str
        X-axis label (default ``"Δ (effect size)"``).
    zero_line : bool
        Whether to draw a vertical reference line at zero (default True).

    Returns
    -------
    The same ``ax``, after drawing.
    """
    deltas = np.asarray(deltas, dtype=float)
    n = deltas.size
    y = np.arange(n)[::-1]   # first subject at top, like classic forest plots
    if labels is None:
        labels = [f"S{i+1}" for i in range(n)]

    cn = color_negative or color
    point_colors = [color if d >= 0 else cn for d in deltas]

    # Error bars
    if ci_low is not None and ci_high is not None:
        cil = np.asarray(ci_low, dtype=float)
        cih = np.asarray(ci_high, dtype=float)
        for yi, d, lo, hi, c in zip(y, deltas, cil, cih, point_colors):
            ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, alpha=0.7)

    ax.scatter(deltas, y, c=point_colors, s=42, zorder=3,
               edgecolor="white", linewidth=0.8)

    if zero_line:
        ax.axvline(0, color="#888", lw=0.9, ls="--", zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
    return ax
