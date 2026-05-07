#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis C - Time-resolved coupling within each condition.

For each condition, aggregate the windowed PLV / wPLI / xcorr peak time
series stored in the per-condition coupling JSONs. Re-express each subject's
windowed series on a normalized time axis (0..1 within the condition) and
compute the across-subject mean and SEM at each time bin.

The question is whether coupling **builds up over the listening period** —
which would be the signature of an entrainment process — or stays flat
(suggesting any coupling that exists is steady, not engaged by stimulus).

Outputs:
  - reports/preliminary_results/figures/C_time_resolved_resp_audio.{png,pdf}

Usage:
    python scripts/figures/analysis_C_time_resolved_coupling.py \\
        --subjects 2 3 4 5 6 --metric wpli
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps
from itertools import combinations

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import (
    use_paper_style, CONDITION_COLORS, save_figure, fmt_p, sig_stars,
)


METRIC_KEYS = {
    "wpli":  ("wpli", "win_wpli", "win_times_s"),
    "plv":   ("plv", "win_plv", "win_times_s"),
    "xcorr": ("xcorr", "peak_r", "times_s"),
    "coh":   ("coherence", "band_avg_coh_win", "times_s"),
    "mi":    ("mi", "win_mi", "win_times_s"),
}

METRIC_LABELS = {
    "wpli":  "wPLI (resp <-> audio)",
    "plv":   "PLV (resp <-> audio)",
    "xcorr": "Cross-corr peak r",
    "coh":   "Band-avg coherence",
    "mi":    "Mutual information",
}


def _load_subj_metric(data_dir: Path, subj: int, cond: str, metric: str):
    block, value_key, time_key = METRIC_KEYS[metric]
    p = data_dir / "processed" / f"sub-{subj:02d}" / "tables" / f"coupling_{cond}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    if block not in d:
        return None
    section = d[block]
    if value_key not in section or time_key not in section:
        return None
    times = np.asarray(section[time_key], float)
    vals = np.asarray(section[value_key], float)
    if len(times) == 0:
        return None
    return times, vals


def _resample_to_norm(times: np.ndarray, vals: np.ndarray, n_grid: int = 30):
    """Map (times in seconds) to normalized 0..1 axis with `n_grid` bins.

    Uses linear interpolation; fills NaN where extrapolation would be needed.
    """
    if times.max() <= times.min():
        return np.full(n_grid, np.nan)
    norm_t = (times - times.min()) / (times.max() - times.min())
    grid = np.linspace(0, 1, n_grid)
    return np.interp(grid, norm_t, vals, left=np.nan, right=np.nan)


def aggregate(args, conditions, metric):
    """Returns dict[condition] -> (grid, mean, sem, arr, slopes_per_subj).

    ``slopes_per_subj`` is a list of per-subject linear-trend slopes (units of
    ``metric`` per unit normalized time). One slope per subject who contributed
    a non-trivial trace to that condition.
    """
    n_grid = args.n_grid
    grid = np.linspace(0, 1, n_grid)
    by_cond = {}
    for cond in conditions:
        rows = []
        slopes = []
        for subj in args.subjects:
            r = _load_subj_metric(args.data_dir, subj, cond, metric)
            if r is None:
                continue
            times, vals = r
            row = _resample_to_norm(times, vals, n_grid=n_grid)
            rows.append(row)
            # Per-subject slope on the original (unresampled) time axis
            mask = np.isfinite(vals) & np.isfinite(times)
            if mask.sum() >= 4:
                t_norm = (times[mask] - times[mask].min()) / max(1e-9, times[mask].max() - times[mask].min())
                lr = sps.linregress(t_norm, vals[mask])
                slopes.append(float(lr.slope))
        if not rows:
            continue
        arr = np.vstack(rows)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / max(1.0, np.sqrt(arr.shape[0]))
        by_cond[cond] = (grid, mean, sem, arr, np.array(slopes))
    return by_cond


def slope_stats(by_cond, conditions):
    """Returns:
        per_cond : dict[cond] -> {"mean_slope", "p_vs_zero"}  (one-sample Wilcoxon vs 0)
        pairwise : list of dicts [{"a","b","stat","p"}]       (paired Wilcoxon)
    """
    per_cond = {}
    for cond in conditions:
        if cond not in by_cond:
            continue
        slopes = by_cond[cond][4]
        if len(slopes) < 3:
            per_cond[cond] = {"mean_slope": float(np.nanmean(slopes)) if len(slopes) else float("nan"),
                              "p_vs_zero": float("nan"), "n": int(len(slopes))}
            continue
        try:
            res = sps.wilcoxon(slopes, alternative="two-sided",
                               zero_method="wilcox", nan_policy="omit")
            p = float(res.pvalue)
        except Exception:
            p = float("nan")
        per_cond[cond] = {"mean_slope": float(np.nanmean(slopes)),
                          "p_vs_zero": p, "n": int(len(slopes))}

    pairwise = []
    if len(conditions) >= 2:
        # Build per-subject paired slopes between conditions where the same subject contributed.
        # Easier: assume subjects are aligned by order across conditions (small n=5).
        for a, b in combinations(conditions, 2):
            if a not in by_cond or b not in by_cond:
                continue
            sa = by_cond[a][4]; sb = by_cond[b][4]
            n = min(len(sa), len(sb))
            if n < 3:
                pairwise.append({"a": a, "b": b, "n": n, "stat": float("nan"), "p": float("nan")})
                continue
            try:
                res = sps.wilcoxon(sa[:n], sb[:n], alternative="two-sided",
                                   zero_method="wilcox", nan_policy="omit")
                pairwise.append({"a": a, "b": b, "n": n, "stat": float(res.statistic), "p": float(res.pvalue)})
            except Exception as e:
                pairwise.append({"a": a, "b": b, "n": n, "stat": float("nan"), "p": float("nan")})
    return per_cond, pairwise


def plot(by_cond, conditions, metric, output_dir: Path,
         per_cond_stats=None, pairwise_stats=None):
    use_paper_style()
    fig = plt.figure(figsize=(9.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.50)
    ax = fig.add_subplot(gs[0, 0])
    ax_slopes = fig.add_subplot(gs[0, 1])

    # Left panel: per-condition mean +/- SEM trajectories.
    for cond in conditions:
        if cond not in by_cond:
            continue
        grid, mean, sem, arr, slopes = by_cond[cond]
        color = CONDITION_COLORS.get(cond, "#666666")
        n_subj = arr.shape[0]
        ax.plot(grid, mean, lw=2.0, color=color, label=f"{cond} (n={n_subj})")
        ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)

    # Build a compact stats legend (replaces inline labels that overlapped the right panel).
    if per_cond_stats:
        lines = []
        for cond in conditions:
            if cond not in per_cond_stats:
                continue
            s = per_cond_stats[cond]
            lines.append(f"{cond}: slope={s['mean_slope']:+.3f}  "
                         f"{fmt_p(s['p_vs_zero'])} {sig_stars(s['p_vs_zero'])}")
        if lines:
            ax.text(0.02, 0.97, "Per-subject buildup slopes\n" + "\n".join(lines),
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.5, color="#222",
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              alpha=0.9, pad=3.0))

    ax.set_xlabel("Normalized time within condition (0 = start, 1 = end)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlim(0, 1)
    # Title removed (in caption).
    ax.legend(loc="lower left", frameon=False)

    # Right panel: per-subject slopes per condition (strip + boxplot) with pairwise p-values.
    box_data, box_labels, box_colors = [], [], []
    for cond in conditions:
        if cond not in by_cond:
            continue
        slopes = by_cond[cond][4]
        if len(slopes) == 0:
            continue
        box_data.append(slopes)
        box_labels.append(cond)
        box_colors.append(CONDITION_COLORS.get(cond, "#666666"))

    if box_data:
        positions = np.arange(len(box_data))
        bp = ax_slopes.boxplot(box_data, positions=positions, widths=0.55,
                               patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.30); patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color("#222222"); med.set_linewidth(1.2)
        for k, slopes in enumerate(box_data):
            ax_slopes.scatter(np.full_like(slopes, positions[k]) + np.random.uniform(-0.07, 0.07, len(slopes)),
                              slopes, color=box_colors[k], s=28,
                              edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
        ax_slopes.axhline(0, color="#888888", lw=0.9, ls="--")
        ax_slopes.set_xticks(positions); ax_slopes.set_xticklabels(box_labels)
        ax_slopes.set_ylabel("Per-subject slope")
        ax_slopes.set_title("Buildup slope (per subject)\nWilcoxon vs 0", fontsize=10.5)

        # pairwise brackets (only show if any p < 0.1 to avoid clutter)
        if pairwise_stats:
            ymax = max(np.nanmax(s) for s in box_data) if box_data else 0.0
            ymin = min(np.nanmin(s) for s in box_data) if box_data else 0.0
            yspan = max(1e-6, ymax - ymin)
            level = ymax + 0.06 * yspan
            for entry in pairwise_stats:
                if entry["a"] not in box_labels or entry["b"] not in box_labels:
                    continue
                ia = box_labels.index(entry["a"]); ib = box_labels.index(entry["b"])
                p = entry["p"]
                if p is None or np.isnan(p):
                    continue
                ax_slopes.plot([ia, ib], [level, level], color="#444444", lw=0.9)
                ax_slopes.text((ia + ib)/2.0, level + 0.01 * yspan, sig_stars(p),
                               ha="center", va="bottom", fontsize=10, color="#222222")
                level += 0.10 * yspan

    fig.tight_layout()
    saved = save_figure(fig, output_dir / f"C_time_resolved_resp_audio_{metric}")
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--metric", choices=list(METRIC_KEYS.keys()), default="wpli")
    p.add_argument("--n-grid", type=int, default=30,
                   help="Number of bins on the normalized time axis (default: 30)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "preliminary_results" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    by_cond = aggregate(args, args.conditions, args.metric)
    if not by_cond:
        print("No data loaded.")
        return
    per_cond_stats, pairwise_stats = slope_stats(by_cond, args.conditions)
    print("\nSlope statistics (per-subject linear trend across normalized time):")
    for cond, s in per_cond_stats.items():
        print(f"  {cond}: mean slope={s['mean_slope']:.3f}, n={s['n']}, "
              f"Wilcoxon p={s['p_vs_zero']:.4g}")
    if pairwise_stats:
        print("\nPairwise (paired Wilcoxon on slopes):")
        for r in pairwise_stats:
            print(f"  {r['a']} vs {r['b']}: stat={r['stat']:.3g}, p={r['p']:.4g}")
    plot(by_cond, args.conditions, args.metric, args.figures_dir,
         per_cond_stats=per_cond_stats, pairwise_stats=pairwise_stats)


if __name__ == "__main__":
    main()
