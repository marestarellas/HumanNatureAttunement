#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-ready feature comparison across conditions.

Combines representative features from three modalities into one figure:
  - HRV         : HRV_HF, HRV_MeanNN, HRV_RMSSD
  - EEG power   : alpha_rel, theta_rel, high_beta_rel  (channel-averaged)
  - EEG complex.: lzc, sample_entropy, perm_entropy    (channel-averaged)

Layout: 3 rows (modalities) x 3 cols (features). Each panel shows a
violin per condition with paired connectors, mean line, Friedman p
inside the panel, and pairwise-Wilcoxon brackets between every pair.

Inputs:
  data/processed/sub-XX/tables/features_<COND>.csv          (EEG features)
  data/processed/sub-XX/tables/hrv_features_<COND>.csv      (HRV features)

Output:
  figures/report/Fig_features_grid.{png,pdf}

Usage:
    python scripts/figures/analysis_features_grid.py \\
        --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI
"""
from __future__ import annotations
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from HNA.modules.viz import use_paper_style, CONDITION_COLORS, save_figure, sig_stars


# Modality / feature map. Each row is a modality, each tuple is (column, label).
MODALITIES = {
    "HRV": [
        ("HRV_HF",     "HF"),
        ("HRV_MeanNN", "Mean NN"),
        ("HRV_RMSSD",  "RMSSD"),
    ],
    "EEG power (rel.)": [
        ("alpha_rel",     "alpha"),
        ("theta_rel",     "theta"),
        ("high_beta_rel", "high beta"),
    ],
    "EEG complexity": [
        ("lzc",            "LZC"),
        ("sample_entropy", "Sample entropy"),
        ("perm_entropy",   "Perm. entropy"),
    ],
}


def _load_eeg_subject_means(subjects, conditions, data_dir):
    """Per-subject, per-condition mean of each EEG feature, averaged across channels and windows."""
    rows = []
    for s in subjects:
        sub = f"sub-{s:02d}"
        for c in conditions:
            p = data_dir / "processed" / sub / "tables" / f"features_{c}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            # Columns we want
            for feat, _ in MODALITIES["EEG power (rel.)"] + MODALITIES["EEG complexity"]:
                if feat not in df.columns:
                    continue
                rows.append({
                    "subject_id": s, "condition": c, "feature": feat,
                    "value": float(df[feat].mean(skipna=True)),
                })
    return pd.DataFrame(rows)


def _load_hrv_subject_means(subjects, conditions, data_dir):
    rows = []
    for s in subjects:
        sub = f"sub-{s:02d}"
        for c in conditions:
            p = data_dir / "processed" / sub / "tables" / f"hrv_features_{c}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            for feat, _ in MODALITIES["HRV"]:
                if feat not in df.columns:
                    continue
                rows.append({
                    "subject_id": s, "condition": c, "feature": feat,
                    "value": float(df[feat].mean(skipna=True)),
                })
    return pd.DataFrame(rows)


def _friedman_with_posthoc(sub, conditions):
    """Run Friedman + pairwise Wilcoxon on a per-subject series."""
    pivot = sub.pivot(index="subject_id", columns="condition", values="value")
    pivot = pivot.reindex(columns=conditions).dropna()
    out = {"friedman_p": np.nan, "posthoc": []}
    if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
        try:
            chi2, p = sps.friedmanchisquare(*[pivot[c].values for c in conditions])
            out["friedman_p"] = float(p)
        except Exception:
            pass
        for a, b in combinations(conditions, 2):
            try:
                res = sps.wilcoxon(pivot[a].values, pivot[b].values,
                                   alternative="two-sided", zero_method="wilcox",
                                   nan_policy="omit")
                out["posthoc"].append({"a": a, "b": b, "p": float(res.pvalue)})
            except Exception:
                out["posthoc"].append({"a": a, "b": b, "p": float("nan")})
    return out


def plot_panel(ax, sub, conditions, label):
    data = [sub[sub["condition"] == c]["value"].values for c in conditions]

    parts = ax.violinplot(data, positions=range(len(conditions)),
                          widths=0.7, showmeans=False, showextrema=False)
    for pc, c in zip(parts["bodies"], conditions):
        pc.set_facecolor(CONDITION_COLORS.get(c, "#666"))
        pc.set_alpha(0.55)
        pc.set_edgecolor(CONDITION_COLORS.get(c, "#666"))
        pc.set_linewidth(0.9)

    pivot = (sub.pivot(index="subject_id", columns="condition", values="value")
               .reindex(columns=conditions))
    for _, row in pivot.iterrows():
        if row.notna().all():
            ax.plot(range(len(conditions)), row.values,
                    color="#9aa0a6", alpha=0.55, lw=0.9)
    rng = np.random.default_rng(0)
    for k, c in enumerate(conditions):
        vals = sub[sub["condition"] == c]["value"].values
        xj = rng.normal(k, 0.045, len(vals))
        ax.scatter(xj, vals, s=22, color="black", alpha=0.7, zorder=6)

    means = [sub[sub["condition"] == c]["value"].mean() for c in conditions]
    ax.plot(range(len(conditions)), means, "-D", color="#f5b400",
            lw=1.7, ms=8, mec="black", mew=0.6, zorder=10)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=10)
    ax.grid(True, axis="y", alpha=0.30)
    ax.set_title(label, fontsize=11, fontweight="bold")

    # Stats
    stats = _friedman_with_posthoc(sub, conditions)
    fp = stats["friedman_p"]
    if np.isfinite(fp):
        ax.text(0.97, 0.04, f"Friedman p={fp:.3g} {sig_stars(fp)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color="#444",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1.2))
    # All pairwise brackets stacked
    ymax = float(np.nanmax([np.nanmax(d) if len(d) else np.nan for d in data]))
    ymin = float(np.nanmin([np.nanmin(d) if len(d) else np.nan for d in data]))
    yspan = max(1e-6, ymax - ymin)
    level = ymax + 0.06 * yspan
    for entry in stats["posthoc"]:
        p_val = entry["p"]
        if not np.isfinite(p_val):
            continue
        a, b = entry["a"], entry["b"]
        if a not in conditions or b not in conditions:
            continue
        ia = conditions.index(a); ib = conditions.index(b)
        stars = sig_stars(p_val)
        ax.plot([ia, ib], [level, level], color="#444", lw=0.8)
        ax.text((ia + ib) / 2.0, level, stars,
                ha="center", va="bottom",
                fontsize=11 if stars not in ("ns", "n/a") else 9,
                fontweight="bold" if stars not in ("ns", "n/a") else "normal",
                color="#222" if stars not in ("ns", "n/a") else "#888")
        level += 0.10 * yspan


def plot_grid(eeg_df, hrv_df, conditions, output_path):
    use_paper_style()
    n_rows = len(MODALITIES)
    n_cols = max(len(v) for v in MODALITIES.values())
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.7 * n_rows),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for r, (mod_name, feats) in enumerate(MODALITIES.items()):
        # Pick the right source df
        source = hrv_df if mod_name == "HRV" else eeg_df
        for c, (col, label) in enumerate(feats):
            ax = axes[r, c]
            sub = source[source["feature"] == col]
            if sub.empty:
                ax.axis("off")
                continue
            plot_panel(ax, sub, conditions, label)
            if c == 0:
                ax.set_ylabel(mod_name, fontsize=11, fontweight="bold")

    # Title removed (in caption).
    fig.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close()
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures" / "report" / "Fig_features_grid")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    eeg_df = _load_eeg_subject_means(args.subjects, args.conditions, args.data_dir)
    hrv_df = _load_hrv_subject_means(args.subjects, args.conditions, args.data_dir)
    print(f"  EEG rows: {len(eeg_df)}   HRV rows: {len(hrv_df)}")
    plot_grid(eeg_df, hrv_df, args.conditions, args.out)


if __name__ == "__main__":
    main()
