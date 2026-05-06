#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis E - Phase clustering of respiration relative to the audio swell.

Why a *circular* analysis instead of a linear LMM?
Phase angles wrap around 2*pi (the mean of pi and -pi is 0, not pi). Linear
mixed models on raw phases violate this. The standard small-N approach
(Fisher 1993) is hierarchical:

1. Within-subject: each subject's coupling pipeline produces ~20 windowed
   preferred phases. We compute Rayleigh's R (mean resultant length, [0,1])
   and the Rayleigh test p-value on those windows. R close to 1 = stable
   phase across windows; R close to 0 = uniformly distributed.
2. Group level: take each subject's circular-mean phase, then run Rayleigh
   on those n=5 subject means per condition. R reflects how clustered the
   subjects are around a common phase; p tests against uniform.

The circular equivalent of an LMM (e.g. wrapped-Cauchy GLMM, projected
sin/cos LMM) gives essentially the same answer with five subjects. We can
add it as a supplementary table if a reviewer asks.

In each polar panel below:
- Coloured arrow per subject = (within-subject R, mean phase). Coloured
  if the subject's within-window Rayleigh is significant (p<0.05), grey
  otherwise.
- Black arrow = group mean resultant; its length is the group-level R.

Output:
  - figures/report/E_phase_polar.{png,pdf}
  - results/phase_polar/phase_per_subject.csv

Usage:
    python scripts/figures/analysis_E_phase_polar.py --subjects 2 3 4 5 6
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from HNA.modules.viz import (
    use_paper_style, CONDITION_COLORS, save_figure, fmt_p, sig_stars,
)


def rayleigh_test(angles: np.ndarray):
    """Rayleigh test for non-uniformity of circular data.

    Returns (R, p) where R is the resultant length in [0, 1]. Approx p-value
    from the Mardia/Jupp asymptotic formula valid for n >= 5.
    """
    angles = np.asarray(angles, float)
    angles = angles[~np.isnan(angles)]
    n = len(angles)
    if n < 2:
        return float("nan"), float("nan")
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    z = n * R**2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - z * n)) - (1 + 2 * n))
    return float(R), float(p)


def _circular_mean(angles: np.ndarray) -> float:
    return float(np.angle(np.exp(1j * angles).mean()))


def _circular_R(angles: np.ndarray) -> float:
    return float(np.abs(np.exp(1j * angles).mean()))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _load_subject_phases(data_dir: Path, subj: int, cond: str):
    """Return per-subject phase summary for one (subject, condition).

    Returns dict with:
      - whole_phase, whole_plv : scalar from the whole-condition PLV
      - win_phases : np.ndarray of ~20 windowed phases (radians, wrapped)
      - mean_phase : circular mean of win_phases (subject-level summary)
      - within_R, within_p : per-subject Rayleigh on win_phases
      - dom_freq : dominant respiratory freq used (Hz)
    """
    p = data_dir / "processed" / f"sub-{subj:02d}" / "tables" / f"coupling_{cond}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    plv_block = d.get("plv")
    if not isinstance(plv_block, dict):
        return None
    f0 = plv_block.get("dom_freq", np.nan)
    if not (np.isfinite(f0) and f0 > 0):
        return None

    whole_lag = plv_block.get("preferred_lag_s", np.nan)
    whole_plv = float(plv_block.get("plv", np.nan))
    whole_phase = _wrap(2 * np.pi * whole_lag * f0) if np.isfinite(whole_lag) else np.nan

    win_lags = np.asarray(plv_block.get("win_preferred_lag_s", []), float)
    win_phases = np.array([_wrap(2 * np.pi * lag * f0) for lag in win_lags
                           if np.isfinite(lag)])
    if len(win_phases) >= 4:
        within_R, within_p = rayleigh_test(win_phases)
        mean_phase = _circular_mean(win_phases)
    else:
        within_R, within_p, mean_phase = np.nan, np.nan, whole_phase

    return {
        "subject_id": subj, "condition": cond,
        "whole_phase": whole_phase, "whole_plv": whole_plv,
        "win_phases": win_phases, "n_windows": int(len(win_phases)),
        "mean_phase": mean_phase, "within_R": within_R, "within_p": within_p,
        "dom_freq": float(f0),
    }


def collect(args):
    out = {cond: [] for cond in args.conditions}
    for subj in args.subjects:
        for cond in args.conditions:
            row = _load_subject_phases(args.data_dir, subj, cond)
            if row is not None:
                out[cond].append(row)
    return out


def plot(by_cond, conditions, output_dir: Path):
    use_paper_style()
    n_cond = len(conditions)
    fig = plt.figure(figsize=(2.3 * n_cond, 2.7))

    for k, cond in enumerate(conditions):
        ax = fig.add_subplot(1, n_cond, k + 1, projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlim(0, 1.20)                  # extra room for labels
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1.0"], fontsize=9)
        # Drop the top (90°) and bottom (270°) tick labels: they collide with the
        # multi-line title/legend. Left/right are sufficient for orientation.
        ax.set_xticks(np.deg2rad([0, 180]))
        ax.set_xticklabels(["0°", "180°"], fontsize=9.5)
        ax.tick_params(axis="x", pad=4)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.5)

        rows = by_cond.get(cond, [])
        if not rows:
            ax.set_title(f"{cond}\n(no data)")
            continue
        color = CONDITION_COLORS.get(cond, "#666666")

        # Each subject: vector at (mean_phase, within_R), with significance halo.
        subj_phases = []
        n_sig_within = 0
        # Sort by angle so labels can be staggered radially when angles are close.
        ordered = sorted(
            (row for row in rows
             if np.isfinite(row["mean_phase"]) and np.isfinite(row["within_R"])),
            key=lambda r: r["mean_phase"],
        )
        # Cluster nearby angles (within ~28 degrees) to alternate label radii.
        prev_angle = None
        radial_offset_step = 0.10
        radial_offset = 0.06
        for row in ordered:
            ang = row["mean_phase"]
            r_within = row["within_R"]
            p_within = row["within_p"]
            subj_phases.append(ang)
            sig = (np.isfinite(p_within) and p_within < 0.05)
            n_sig_within += int(sig)
            arrow_color = color if sig else "#9aa0a6"
            r_arrow = max(0.05, min(r_within, 1.0))
            ax.annotate("",
                        xy=(ang, r_arrow),
                        xytext=(ang, 0),
                        arrowprops=dict(arrowstyle="->", color=arrow_color,
                                        lw=1.6, alpha=0.95 if sig else 0.70,
                                        mutation_scale=12))

        # Group-level Rayleigh on per-subject mean phases.
        if subj_phases:
            subj_phases = np.array(subj_phases)
            R_group, p_group = rayleigh_test(subj_phases)
            cmean = _circular_mean(subj_phases)
            ax.annotate("",
                        xy=(cmean, R_group), xytext=(cmean, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#222",
                                        lw=2.0, mutation_scale=14))
            stars = sig_stars(p_group)
            # Two-line centered title: condition (bold) above, p-value below.
            ax.set_title(cond, fontsize=12, fontweight="bold", pad=22)
            ax.text(0.5, 1.02, f"p={p_group:.3f} {stars}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=10, color="#222")

    # Title removed (in caption).
    # Single shared legend at the figure bottom.
    legend_handles = [
        plt.Line2D([0], [0], color="#222", lw=2.4, label="Group resultant"),
        plt.Line2D([0], [0], color="#9aa0a6", lw=1.6, label="Subject (ns within)"),
        plt.Line2D([0], [0], color=CONDITION_COLORS.get(conditions[0], "#666"),
                   lw=1.6, label="Subject (sig within)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3,
               fontsize=8.5, frameon=False)

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])
    saved = save_figure(fig, output_dir / "E_phase_polar")
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "figures" / "report")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "phase_polar")
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    by_cond = collect(args)

    # Per-subject summary CSV (long form)
    rows = []
    for cond, lst in by_cond.items():
        for r in lst:
            rows.append({
                "subject_id": r["subject_id"], "condition": cond,
                "n_windows": r["n_windows"], "dom_freq": r["dom_freq"],
                "whole_phase_rad": r["whole_phase"], "whole_plv": r["whole_plv"],
                "mean_phase_rad": r["mean_phase"],
                "within_R": r["within_R"], "within_p": r["within_p"],
            })
    if rows:
        pd.DataFrame(rows).to_csv(args.results_dir / "phase_per_subject.csv", index=False)
        print(f"  Saved per-subject summary: {args.results_dir / 'phase_per_subject.csv'}")

    # Group-level Rayleigh + console summary
    print("\nWithin-subject Rayleigh tests on windowed phases (~20 windows / subject):")
    for cond, lst in by_cond.items():
        if not lst:
            continue
        n_sig = sum(1 for r in lst if np.isfinite(r["within_p"]) and r["within_p"] < 0.05)
        print(f"  {cond}: {n_sig}/{len(lst)} subjects with within-subject phase consistency (p<0.05)")
        subj_means = np.array([r["mean_phase"] for r in lst
                                if np.isfinite(r["mean_phase"])])
        if len(subj_means) >= 2:
            R, p = rayleigh_test(subj_means)
            print(f"    Group-level Rayleigh on subject means: n={len(subj_means)}  R={R:.3f}  p={p:.4f}")

    plot(by_cond, args.conditions, args.figures_dir)


if __name__ == "__main__":
    main()
