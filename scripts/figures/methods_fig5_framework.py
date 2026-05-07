#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 5 - The features x coupling design space.

A modern grid that lays out the conceptual framework central to the
toolbox: every coupling analysis combines (a) a feature derived from
each signal with (b) a coupling method comparing them. The three
feature rows are RAW SIGNAL, OSCILLATORY FEATURES (Hilbert envelope /
phase / PSD bands), and COMPLEXITY FEATURES (DFA alpha, fractal
dimension, FOOOF aperiodic, multiscale entropy). The four coupling
columns are LINEAR, OSCILLATORY, INFORMATION, COMPLEXITY.

Each cell either names the canonical method that lives there in the
HNA toolbox (bold) or describes the pattern in plain prose. Two
illuminating identities the figure makes visible:

* "complexity_coupling" = LINEAR coupling applied to a COMPLEXITY
  feature trace (alpha(t)).
* "PAC" = OSCILLATORY coupling between two OSCILLATORY features
  (slow-band phase X fast-band amplitude).

Output
------
figures/report/Methods5_features_x_coupling.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.viz import use_paper_style, save_figure


# Family palette (matches the rest of the methods figures).
FAMILY_COLORS = {
    "linear":      "#3B7DD8",
    "oscillatory": "#5DA399",
    "information": "#7B5BA6",
    "complexity":  "#E08E1A",
}

# Each cell:
#   (headline, body, kind)
#   kind in {"named", "pattern", "rare"}:
#     "named"   -> the cell IS a named HNA toolbox method (highlight)
#     "pattern" -> the cell is a general approach (plain)
#     "rare"    -> uncommon / not standard (faded)
CELLS = [
    # Row 0: raw signal
    [
        ("windowed_xcorr",
         "lag-search Pearson\non raw samples",
         "named"),
        ("(feature step needed)",
         "PLV / wPLI / coherence\nimply Hilbert / Welch first",
         "rare"),
        ("MI, effective MI,\nGranger, TE",
         "global statistical\ndependence",
         "named"),
    ],
    # Row 1: oscillatory features (envelope / phase / band-power)
    [
        ("xcorr / corr",
         "of envelopes\nor band-power",
         "pattern"),
        ("PLV / wPLI / coherence;\nPAC (Tort, Canolty)",
         "same-band phase or\nslow phase x fast amp",
         "named"),
        ("MI of envelopes /\nband-power traces",
         "non-linear envelope\ncoupling",
         "pattern"),
    ],
    # Row 2: complexity features (fractality + entropy)
    # All four toolbox-named methods land in the LINEAR cell here, because
    # they are all linear comparisons applied to a complexity feature
    # (scalar alpha, F(s) / MSE curve over scales, alpha(t) trace over time).
    [
        ("exponent_matching\nfluctuation_matching\nmse_matching\ncomplexity_coupling",
         "scalar / scale curves / time trace;\nall = Pearson on the feature",
         "named"),
        ("(rare)",
         "phase-locking on a\ncomplexity trace is\nnot standard",
         "rare"),
        ("MI between\ncomplexity traces",
         "non-linear coupling\nof scaling / entropy",
         "pattern"),
    ],
]

ROW_LABELS = [
    ("Raw signal",
     "x[t] itself"),
    ("Oscillatory features",
     "envelope, phase,\nband-power"),
    ("Complexity features",
     "fractality + entropy:\nDFA, FD, FOOOF, MSE"),
]

COL_LABELS = [
    ("Linear",       "linear",      "lag-zero / lag-search\nPearson"),
    ("Oscillatory",  "oscillatory", "phase / spectrum /\nphase-amplitude"),
    ("Information",  "information", "MI, effective MI,\nGranger, TE"),
]


# --------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------- #
def _draw_cell(ax, x, y, w, h, headline, body, kind, accent_color):
    """Draw one cell at (x, y) with size (w, h)."""
    if kind == "named":
        face = "#FFFFFF"
        edge = accent_color
        edge_w = 1.6
        head_color = accent_color
        head_weight = "bold"
        head_size = 10.5
        body_color = "#333333"
        body_alpha = 1.0
    elif kind == "pattern":
        face = "#FAFBFC"
        edge = "#D6DAE0"
        edge_w = 0.9
        head_color = "#1F2A37"
        head_weight = "bold"
        head_size = 10.0
        body_color = "#5A6470"
        body_alpha = 1.0
    else:  # rare
        face = "#F5F6F8"
        edge = "#E1E5EB"
        edge_w = 0.8
        head_color = "#7A828F"
        head_weight = "normal"
        head_size = 9.5
        body_color = "#9CA3AE"
        body_alpha = 1.0

    pad = 0.012  # visual gap between cells
    box = FancyBboxPatch(
        (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=edge_w,
    )
    ax.add_patch(box)

    # Headline (slightly above center)
    ax.text(x + w / 2, y + h * 0.62, headline,
            ha="center", va="center", fontsize=head_size,
            fontweight=head_weight, color=head_color,
            transform=ax.transData, alpha=body_alpha)
    # Body (below)
    ax.text(x + w / 2, y + h * 0.30, body,
            ha="center", va="center", fontsize=8.5,
            color=body_color, alpha=body_alpha)

    # Subtle filled corner accent for "named" cells (visual priority without
    # overlapping the centred text).
    if kind == "named":
        corner = Rectangle(
            (x + pad, y + h - pad - 0.015),
            0.025, 0.015,
            facecolor=accent_color, edgecolor="none",
        )
        ax.add_patch(corner)


def _draw_col_header(ax, x, y, w, h, label, family, sub):
    color = FAMILY_COLORS[family]
    box = FancyBboxPatch(
        (x + 0.012, y + 0.012), w - 0.024, h - 0.024,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=color, edgecolor="none",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.65, label,
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="white")
    ax.text(x + w / 2, y + h * 0.30, sub,
            ha="center", va="center", fontsize=8.7,
            color="white", alpha=0.92)


def _draw_row_header(ax, x, y, w, h, label, sub):
    box = FancyBboxPatch(
        (x + 0.012, y + 0.012), w - 0.024, h - 0.024,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor="#2D3748", edgecolor="none",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.65, label,
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color="white", rotation=0)
    ax.text(x + w / 2, y + h * 0.30, sub,
            ha="center", va="center", fontsize=8.2,
            color="#CBD5E0", alpha=0.95)


# --------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------- #
def make_figure(out_path: Path):
    use_paper_style()

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # Layout (figure-fraction units 0..1).
    margin_top    = 0.07
    margin_bottom = 0.04
    margin_left   = 0.02
    margin_right  = 0.02

    head_h = 0.13   # column header height
    rowh_w = 0.22   # row header width

    grid_x0 = margin_left + rowh_w
    grid_x1 = 1 - margin_right
    grid_y0 = margin_bottom
    grid_y1 = 1 - margin_top - head_h

    n_rows = len(CELLS)
    n_cols = len(CELLS[0])

    cell_w = (grid_x1 - grid_x0) / n_cols
    cell_h = (grid_y1 - grid_y0) / n_rows

    # Column headers (top strip).
    for j, (lab, fam, sub) in enumerate(COL_LABELS):
        x = grid_x0 + j * cell_w
        y = grid_y1
        _draw_col_header(ax, x, y, cell_w, head_h, lab, fam, sub)

    # Row headers (left strip).
    for i, (lab, sub) in enumerate(ROW_LABELS):
        # Rows go top-to-bottom in our content; matplotlib y is bottom-to-top
        y = grid_y1 - (i + 1) * cell_h
        _draw_row_header(ax, margin_left, y, rowh_w, cell_h, lab, sub)

    # Cells.
    for i, row in enumerate(CELLS):
        y = grid_y1 - (i + 1) * cell_h
        for j, (headline, body, kind) in enumerate(row):
            x = grid_x0 + j * cell_w
            family = COL_LABELS[j][1]
            accent = FAMILY_COLORS[family]
            _draw_cell(ax, x, y, cell_w, cell_h, headline, body, kind, accent)

    # Title (above the column headers).
    fig.text(0.02, 0.97,
             "The features x coupling design space",
             fontsize=15.5, fontweight="bold", ha="left", va="top",
             color="#1F2A37")

    save_figure(fig, out_path)
    plt.close()
    print(f"  Saved: {out_path.name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures" / "report" / "Methods5_features_x_coupling")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.out)


if __name__ == "__main__":
    main()
