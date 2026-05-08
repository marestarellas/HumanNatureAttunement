#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-envelope cardiac-audio coupling fingerprints (two figures).

Reads ``results/multi_envelope_hrv/hrv_audio_multi_envelope.csv``
produced by ``scripts/analysis/compute_hrv_audio_multi_envelope.py``
and emits TWO complementary heatmap families:

A) **Slow-trend** (xcorr peak |r|, mutual information): one figure
   per HRV feature (HRV_HF, HRV_MeanNN, HRV_RMSSD, ...). The HRV-
   feature label here is meaningful because both metrics operate on
   the windowed HRV-feature trace at 4 Hz.

B) **Oscillatory** (PLV, wPLI, band-avg coherence): a single figure
   without a per-feature suffix. These three metrics are computed
   on the instantaneous-HR signal at 4 Hz and do not depend on which
   HRV feature is under test, so a per-feature filename would be
   misleading.

Outputs (under ``reports/preliminary_results/figures/``):
    F_HRV_slow_trend_<feat>.{png,pdf}      one per HRV feature
    F_HR_oscillatory.{png,pdf}             one figure
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import use_paper_style, save_figure


METRIC_LABELS = {
    "xcorr_peak_r": "xcorr peak |r|",
    "mi":           "mutual information",
    "coh_band_avg": "coherence (avg)",
    "plv":          "PLV",
    "wpli":         "wPLI",
}

ENV_PRETTY = {
    "env_broad":      "broad (60 Hz LP)",
    "env_swell_0p2":  "swell 0.2 Hz LP",
    "env_swell_0p1":  "swell 0.1 Hz LP",
    "env_hrv_lf":     "HRV LF (0.04-0.15)",
    "env_hrv_hf":     "HRV HF (0.15-0.40)",
    "env_splash_1_5": "splash 1-5 Hz",
    "env_delta":      "delta 0.5-4",
    "env_theta":      "theta 4-8",
    "env_alpha":      "alpha 8-13",
    "env_beta_low":   "low-beta 13-20",
    "env_beta_high":  "high-beta 20-30",
    "env_gamma1":     "gamma1 30-50",
}

HRV_LABELS = {
    "HRV_HF":     "HRV-HF (0.15-0.40 Hz)",
    "HRV_MeanNN": "HRV-MeanNN (mean RR interval)",
    "HRV_RMSSD":  "HRV-RMSSD (vagal/parasympathetic)",
    "HRV_SDNN":   "HRV-SDNN (overall variability)",
    "HRV_LnHF":   "HRV-LnHF (log HF)",
    "HRV_LF":     "HRV-LF (caution: 30 s window may be too short)",
}

# Slow-trend metrics: operate on the windowed HRV-feature trace.
SLOW_METRICS = ["xcorr_peak_r", "mi"]
# Oscillatory metrics: operate on instantaneous HR.
OSC_METRICS = ["coh_band_avg", "plv", "wpli"]


def _draw_panel(ax, cells, conditions, env_order, metric, show_y):
    vmin = 0.0
    vmax = float(np.nanmax(cells)) if np.isfinite(np.nanmax(cells)) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.imshow(cells, aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, origin="upper")
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            v = cells[r, c]
            if np.isnan(v):
                continue
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.55 * vmax else "black",
                    fontsize=8.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=10, fontweight="bold")
    ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
    ax.tick_params(top=False, bottom=False, length=0)
    ax.spines[:].set_visible(False)
    if show_y:
        ax.set_yticks(range(len(env_order)))
        ax.set_yticklabels([ENV_PRETTY.get(e, e) for e in env_order],
                            fontsize=9)
    else:
        ax.tick_params(left=False, labelleft=False)
    return im


def _aggregate(sub_df: pd.DataFrame, metric: str, env_order, conditions):
    cells = np.full((len(env_order), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        sub = sub_df[sub_df["condition"] == cond]
        if sub.empty:
            continue
        piv = sub.pivot(index="envelope", columns="subject_id",
                         values=metric)
        piv = piv.reindex(env_order)
        cells[:, j] = piv.mean(axis=1).to_numpy()
    return cells


def make_slow_trend(df, hrv_feature, conditions, env_order, output_dir):
    """One figure per HRV feature; xcorr + MI panels (windowed HRV trace)."""
    use_paper_style()
    sub_all = df[df["hrv_feature"] == hrv_feature]
    if sub_all.empty:
        print(f"  No rows for {hrv_feature}; skipping slow-trend")
        return None

    metrics = SLOW_METRICS
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(2.8 * len(metrics) + 1.8,
                  0.40 * len(env_order) + 1.6),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        cells = _aggregate(sub_all, metric, env_order, conditions)
        im = _draw_panel(ax, cells, conditions, env_order, metric,
                          show_y=(i == 0))
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                             fraction=0.06, pad=0.08, shrink=0.85)
        cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    saved = save_figure(
        fig, output_dir / f"F_HRV_slow_trend_{hrv_feature}"
    )
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")
    return saved


def make_oscillatory(df, conditions, env_order, output_dir):
    """Single figure: PLV / wPLI / coherence on instantaneous HR.

    The 3 metrics do not depend on the HRV feature label; the values
    are duplicated across hrv_feature rows by the upstream computation
    script. We collapse them by taking the first non-NaN value per
    (subject, condition, envelope) tuple.
    """
    use_paper_style()
    # Collapse the redundant hrv_feature axis: same (subject, condition,
    # envelope, oscillatory metric) rows are duplicated across features.
    df_osc = (df.drop_duplicates(
        subset=["subject_id", "condition", "envelope"],
    ))
    metrics = OSC_METRICS
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(2.8 * len(metrics) + 1.8,
                  0.40 * len(env_order) + 1.6),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        cells = _aggregate(df_osc, metric, env_order, conditions)
        im = _draw_panel(ax, cells, conditions, env_order, metric,
                          show_y=(i == 0))
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                             fraction=0.06, pad=0.08, shrink=0.85)
        cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    saved = save_figure(fig, output_dir / "F_HR_oscillatory")
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")
    return saved


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=Path,
                   default=ROOT / "results" / "multi_envelope_hrv"
                                / "hrv_audio_multi_envelope.csv")
    p.add_argument("--conditions", nargs="+",
                   default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--env-order", nargs="+",
                   default=list(ENV_PRETTY.keys()))
    p.add_argument("--hrv-features", nargs="+", default=None,
                   help="Default: every HRV feature found in the CSV.")
    p.add_argument("--figures-dir", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Missing CSV: {args.input_csv}. "
            f"Run scripts/analysis/compute_hrv_audio_multi_envelope.py first."
        )
    df = pd.read_csv(args.input_csv)
    feats = args.hrv_features or sorted(df["hrv_feature"].unique())

    # A) slow-trend: one figure per HRV feature
    for feat in feats:
        make_slow_trend(df, feat, args.conditions, args.env_order,
                         args.figures_dir)

    # B) oscillatory: single figure across the dataset
    make_oscillatory(df, args.conditions, args.env_order,
                      args.figures_dir)


if __name__ == "__main__":
    main()
