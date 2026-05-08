#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis D - Cross-modal coupling co-variance.

Asks: when a subject shows strong respiration<->audio coupling under a given
condition, does that same subject also show strong HRV<->audio coupling and
strong EEG<->audio coupling? In other words, is "attunement" a single
multi-modal state, or are the channels independent?

Pulls three coupling metrics per (subject, condition):
  - Resp <-> audio swell PLV  (from coupling_<COND>.json)
  - HRV (RMSSD) <-> audio swell PLV  (from hrv_audio_coupling_<COND>_HRV_RMSSD.json)
  - EEG alpha <-> audio band-correlation, channel-averaged
    (from results/audio_eeg_correlation/audio_eeg_correlation_results.csv)

Outputs:
  - reports/preliminary_results/figures/cross_modal_coupling.{png,pdf}
      A 4-panel figure: 3 pairwise scatters + a 3x3 Spearman rho heatmap.
  - results/cross_modal/cross_modal_long.csv : long table of (subject, cond, modality, value).

Usage:
    python scripts/figures/analysis_cross_modal_coupling.py \\
        --subjects 2 3 4 5 6
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import (
    use_paper_style, CONDITION_COLORS, save_figure, fmt_p,
)


def _plv_scalar(d: dict) -> float:
    """Coupling JSON stores plv as either a dict or a scalar; return the scalar."""
    plv = d.get("plv", np.nan)
    if isinstance(plv, dict):
        return float(plv.get("plv", np.nan))
    return float(plv)


def _load_resp_audio_plv(data_dir: Path, subj: int, cond: str):
    p = data_dir / "processed" / f"sub-{subj:02d}" / "tables" / f"coupling_{cond}.json"
    if not p.exists():
        return np.nan
    with open(p) as f:
        return _plv_scalar(json.load(f))


def _load_hrv_audio_plv(data_dir: Path, subj: int, cond: str, hrv_feature="HRV_RMSSD"):
    p = (data_dir / "processed" / f"sub-{subj:02d}" / "tables"
         / f"hrv_audio_coupling_{cond}_{hrv_feature}.json")
    if not p.exists():
        return np.nan
    with open(p) as f:
        return _plv_scalar(json.load(f))


def _load_eeg_alpha_corr(results_dir: Path, subj: int, cond: str, band: str = "alpha"):
    """Mean across-channel correlation_direct between EEG band and audio envelope."""
    csv = results_dir / "audio_eeg_correlation" / "audio_eeg_correlation_results.csv"
    if not csv.exists():
        return np.nan
    df = pd.read_csv(csv)
    sub = df[(df["subject_id"] == subj) & (df["condition"] == cond) & (df["band"] == band)]
    if sub.empty:
        return np.nan
    return float(np.nanmean(sub["correlation_direct"].abs()))


def collect(args):
    rows = []
    for subj in args.subjects:
        for cond in args.conditions:
            row = {
                "subject_id": subj,
                "condition": cond,
                "resp_audio_plv": _load_resp_audio_plv(args.data_dir, subj, cond),
                "hrv_audio_plv": _load_hrv_audio_plv(args.data_dir, subj, cond, args.hrv_feature),
                "eeg_alpha_corr": _load_eeg_alpha_corr(args.results_dir, subj, cond, args.eeg_band),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def plot_cross_modal(df: pd.DataFrame, conditions, output_dir: Path,
                     hrv_feature: str, eeg_band: str):
    use_paper_style()
    metrics = ["resp_audio_plv", "hrv_audio_plv", "eeg_alpha_corr"]
    labels = ["Resp <-> audio (PLV)",
              f"HRV ({hrv_feature.replace('HRV_', '')}) <-> audio (PLV)",
              f"EEG {eeg_band} <-> audio (|r|)"]

    df_clean = df.dropna(subset=metrics)

    fig = plt.figure(figsize=(8.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

    # Pairwise scatters (top row)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, k])
        x = df_clean[metrics[i]].values
        y = df_clean[metrics[j]].values
        for cond in conditions:
            mask = (df_clean["condition"] == cond).values
            ax.scatter(x[mask], y[mask], s=42,
                       color=CONDITION_COLORS.get(cond, "#666"),
                       alpha=0.85, edgecolor="white", linewidth=0.7,
                       label=cond if k == 0 else None)
        if len(x) >= 3:
            rho, p_val = spearmanr(x, y)
            ax.set_title(f"rho={rho:.2f}, {fmt_p(p_val)}", fontsize=11)
        ax.set_xlabel(labels[i], fontsize=10)
        ax.set_ylabel(labels[j], fontsize=10)
        ax.tick_params(labelsize=9)

    # Spearman heatmap (bottom, spans full width)
    ax_h = fig.add_subplot(gs[1, :])
    rho_mat = np.zeros((3, 3))
    p_mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                rho_mat[i, j] = 1.0
                p_mat[i, j] = 0.0
            else:
                xi = df_clean[metrics[i]].values
                xj = df_clean[metrics[j]].values
                if len(xi) >= 3:
                    r, p = spearmanr(xi, xj)
                    rho_mat[i, j] = r
                    p_mat[i, j] = p
    im = ax_h.imshow(rho_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    short = ["Resp", "HRV", "EEG"]
    ax_h.set_xticks(range(3)); ax_h.set_xticklabels(short)
    ax_h.set_yticks(range(3)); ax_h.set_yticklabels(short)
    for i in range(3):
        for j in range(3):
            ax_h.text(j, i, f"rho={rho_mat[i, j]:.2f}\n{fmt_p(p_mat[i, j])}",
                      ha="center", va="center", fontsize=10,
                      color="white" if abs(rho_mat[i, j]) > 0.55 else "#222")
    ax_h.set_title(f"Cross-modal Spearman correlation matrix (n={len(df_clean)} (subj, cond) pairs)",
                   fontsize=11)
    ax_h.spines[:].set_visible(False)
    cbar = plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman ρ")

    # Single legend at top-left
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
               frameon=False, ncol=len(conditions), fontsize=10)
    fig.suptitle("Cross-modal audio coupling: is attunement a unitary state?",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    saved = save_figure(fig, output_dir / "cross_modal_coupling")
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--hrv-feature", default="HRV_RMSSD")
    p.add_argument("--eeg-band", default="alpha")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "preliminary_results" / "figures")
    p.add_argument("--cross-modal-dir", type=Path, default=ROOT / "results" / "cross_modal")
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.cross_modal_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args)
    out_csv = args.cross_modal_dir / "cross_modal_long.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved CSV: {out_csv}  (rows={len(df)})")
    plot_cross_modal(df, args.conditions, args.figures_dir,
                     args.hrv_feature, args.eeg_band)


if __name__ == "__main__":
    main()
