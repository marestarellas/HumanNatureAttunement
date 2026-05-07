"""
Paper-ready plotting helpers for the HNA toolbox.

Provides:
- A consistent matplotlib style for figures destined for reports/papers.
- A canonical condition palette (used across resp/HRV/EEG comparisons).
- Significance label helpers (stars + p-value formatting).
- Save helpers that emit both PNG (for previewing) and PDF (for typesetting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Canonical colors and labels
# -----------------------------
CONDITION_COLORS = {
    "RS1":   "#5DA399",   # teal - distinct from VIZ blue
    "RS2":   "#7B5BA6",   # muted purple - distinct from AUD rose
    "VIZ":   "#3B7DD8",   # blue
    "AUD":   "#C9325F",   # rose
    "MULTI": "#E08E1A",   # amber
}

MODALITY_COLORS = {
    "audio":       "#C9325F",
    "respiration": "#3B7DD8",
    "hrv":         "#E08E1A",
    "eeg":         "#2EAA70",
}

CONDITION_ORDER = ("RS1", "VIZ", "AUD", "MULTI", "RS2")


# -----------------------------
# Style
# -----------------------------
PAPER_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "regular",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#222222",
    "axes.grid": True,
    "grid.color": "#D4D4D4",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "grid.alpha": 0.7,

    "lines.linewidth": 1.6,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 0.6,

    "errorbar.capsize": 0,
    "boxplot.medianprops.color": "#222222",
    "boxplot.medianprops.linewidth": 1.4,
}


def use_paper_style():
    """Set matplotlib rcParams for paper-quality figures (call at top of figure scripts)."""
    mpl.rcParams.update(PAPER_RC)


# -----------------------------
# Significance helpers
# -----------------------------
def sig_stars(p, trend_threshold: float = 0.10):
    """Return APA-style significance stars for a p-value.

    Adds a tilde (~) for trend-level (default p < 0.10) results, between
    `*` and `ns`. Set ``trend_threshold=None`` to disable.
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if trend_threshold is not None and p < trend_threshold:
        return "~"
    return "ns"


def fmt_p(p):
    """Format a p-value for figure annotations."""
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"


# -----------------------------
# Layout helpers
# -----------------------------
def _figsize(kind: str = "single") -> tuple[float, float]:
    """Standard figure widths for publication.

    kind in {"single","1.5col","double","square","wide"}.
    Returns (width_in, height_in) following common journal templates.
    """
    return {
        "single":  (3.5, 2.6),
        "1.5col":  (5.0, 3.4),
        "double":  (7.2, 3.4),
        "square":  (4.0, 4.0),
        "wide":    (8.5, 3.0),
    }.get(kind, (5.0, 3.4))


def add_significance_bar(ax, x1: float, x2: float, y: float, text: str,
                         height_frac: float = 0.02, color: str = "#222222",
                         fontsize: float = 9.5):
    """Draw a bracket between x1 and x2 at height y with label `text` above it."""
    y_top = y * (1 + height_frac) if y >= 0 else y * (1 - height_frac)
    ax.plot([x1, x1, x2, x2], [y, y_top, y_top, y], color=color, lw=0.9)
    ax.text((x1 + x2) / 2.0, y_top, text, ha="center", va="bottom",
            fontsize=fontsize, color=color)


# -----------------------------
# Save helper
# -----------------------------
def save_figure(fig, out_path, formats: Sequence[str] = ("png", "pdf")):
    """Save a figure to one or more formats. `out_path` is the path *without* extension."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p)
        paths.append(p)
    return paths
