#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-audio band-matched correlation across the three multisensory
conditions: 6 bands x 3 violins per band.

Replaces the earlier pair-figures
(Fig4_violins_VIZ_vs_AUD / VIZ_vs_MULTI / AUD_vs_MULTI) with a single
consolidated figure. Stats inset come from the LMM results in
``results/eeg_audio_phase_coupling/eeg_audio_phase_coupling_stats.csv``.
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
from HNA.viz import use_paper_style, save_figure, sig_stars


CONDITIONS = ["VIZ", "AUD", "MULTI"]
COND_COLORS = {
    "VIZ":   "#3B7DD8",
    "AUD":   "#E08E1A",
    "MULTI": "#5DA399",
}
BANDS = ("delta", "theta", "alpha", "low_beta", "high_beta", "gamma1")
BAND_LABELS = {
    "delta": "delta (0.5-4 Hz)",
    "theta": "theta (4-8 Hz)",
    "alpha": "alpha (8-13 Hz)",
    "low_beta": "low-beta (13-20 Hz)",
    "high_beta": "high-beta (20-30 Hz)",
    "gamma1": "gamma1 (30-50 Hz)",
}


def plot_grid(per_subject_means: pd.DataFrame, stats_df: pd.DataFrame,
                value_col: str, value_label: str, output_path: Path):
    """Per-subject-mean violins per band x condition, with LMM-derived
    pairwise stars and an omnibus p-value inset."""
    use_paper_style()
    n_panels = len(BANDS)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.0 * n_cols + 0.4,
                                        3.2 * n_rows + 0.4),
                              sharey=False)
    axes = np.atleast_2d(axes).flatten()

    for k, band in enumerate(BANDS):
        ax = axes[k]
        sub = per_subject_means[per_subject_means["band"] == band]
        data = [sub[sub["condition"] == c][value_col].values
                for c in CONDITIONS]

        if all(len(d) > 0 for d in data):
            parts = ax.violinplot(data, positions=range(len(CONDITIONS)),
                                    widths=0.7, showmeans=False,
                                    showextrema=False)
            for pc, c in zip(parts["bodies"], CONDITIONS):
                pc.set_facecolor(COND_COLORS[c]); pc.set_alpha(0.55)
                pc.set_edgecolor(COND_COLORS[c]); pc.set_linewidth(0.9)
            # Paired connectors + subject points
            pivot = (sub.pivot(index="subject_id", columns="condition",
                                 values=value_col)
                       .reindex(columns=CONDITIONS))
            for _, row in pivot.iterrows():
                if row.notna().all():
                    ax.plot(range(len(CONDITIONS)), row.values,
                             color="#9aa0a6", alpha=0.55, lw=0.9)
            rng = np.random.default_rng(0)
            for j, c in enumerate(CONDITIONS):
                vals = sub[sub["condition"] == c][value_col].values
                xj = rng.normal(j, 0.045, len(vals))
                ax.scatter(xj, vals, s=22, color="black", alpha=0.7,
                            zorder=6)
            means = [sub[sub["condition"] == c][value_col].mean()
                       for c in CONDITIONS]
            ax.plot(range(len(CONDITIONS)), means, "-D",
                     color="#f5b400", lw=1.7, ms=8, mec="black",
                     mew=0.6, zorder=10)

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, fontsize=10, fontweight="bold")
        ax.set_title(BAND_LABELS[band], fontsize=11.5, fontweight="bold")
        if k % n_cols == 0:
            ax.set_ylabel(value_label, fontsize=11)
        ax.grid(True, axis="y", alpha=0.30)

        # LMM stats from CSV
        srow = stats_df[stats_df["band"] == band]
        if not srow.empty:
            r = srow.iloc[0]
            omnibus_p = r["omnibus_p"]
            ax.text(0.97, 0.04,
                     f"LMM omnibus p={omnibus_p:.3g} {sig_stars(omnibus_p)}",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=8.5, color="#444",
                     bbox=dict(facecolor="white", edgecolor="none",
                                alpha=0.7, pad=1.2))
            # Pairwise brackets
            pairs = [
                ("VIZ", "AUD",   r["p_AUD_vs_VIZ"]),
                ("VIZ", "MULTI", r["p_MULTI_vs_VIZ"]),
                ("AUD", "MULTI", r["p_AUD_vs_MULTI"]),
            ]
            ymax = float(np.nanmax([np.nanmax(d) if len(d) else np.nan
                                       for d in data]))
            ymin = float(np.nanmin([np.nanmin(d) if len(d) else np.nan
                                       for d in data]))
            yspan = max(1e-6, ymax - ymin)
            level = ymax + 0.06 * yspan
            for a, b, pv in pairs:
                if not np.isfinite(pv):
                    continue
                ia = CONDITIONS.index(a); ib = CONDITIONS.index(b)
                stars = sig_stars(pv)
                ax.plot([ia, ib], [level, level], color="#444", lw=0.8)
                ax.text((ia + ib) / 2.0, level, stars,
                         ha="center", va="bottom",
                         fontsize=11 if stars not in ("ns", "n/a") else 9,
                         fontweight="bold" if stars not in ("ns", "n/a") else "normal",
                         color="#222" if stars not in ("ns", "n/a") else "#888")
                level += 0.10 * yspan

    for k in range(len(BANDS), len(axes)):
        axes[k].set_visible(False)

    fig.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close()
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corr-csv", type=Path,
                   default=ROOT / "results" / "audio_eeg_correlation"
                                / "audio_eeg_correlation_results.csv")
    p.add_argument("--stats-csv", type=Path,
                   default=ROOT / "results" / "eeg_audio_phase_coupling"
                                / "eeg_audio_phase_coupling_stats.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures"
                                / "eeg_audio_correlation_3cond")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.corr_csv)
    if "correlation_direct" not in df.columns:
        raise SystemExit("Expected `correlation_direct` column not found")
    df = df[df["condition"].isin(CONDITIONS)]
    df = df.rename(columns={"correlation_direct": "value"})
    # Per-subject mean across channels
    per_sub = (df.groupby(["subject_id", "condition", "band"], as_index=False)
                  ["value"].mean())
    per_sub = per_sub.rename(columns={"value": "correlation_direct"})

    stats = pd.DataFrame()
    if args.stats_csv.exists():
        sd = pd.read_csv(args.stats_csv)
        stats = sd[sd["metric"] == "correlation_direct"]

    plot_grid(per_sub, stats,
                value_col="correlation_direct",
                value_label="band-matched Pearson r (mean across 32 channels)",
                output_path=args.out)


if __name__ == "__main__":
    main()
