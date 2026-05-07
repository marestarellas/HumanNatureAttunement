#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis B - Per-subject surrogate-based PLV significance.

For each (subject, condition), generate N surrogates of the audio swell
envelope by phase-randomization, compute the PLV between respiration and
each surrogate, and compare the observed PLV to that null distribution.

Outputs:
  - figures/report/B_surrogate_resp_audio.{png,pdf}
      A panel-per-condition figure showing each subject's observed PLV
      relative to its own surrogate distribution (z-score + p-value).
  - results/surrogate_tests/resp_audio_plv_surrogates.csv
      Per-subject z, p, and n_significant counts.

Design rationale: with n=5 subjects, group-level p-values are under-powered,
but per-subject surrogate tests are well-powered (each test uses ~hundreds
of samples). Reporting "K of N subjects significant" is a more compelling
small-sample readout than a Friedman p-value.

Usage:
    python scripts/figures/analysis_B_surrogate_significance.py \\
        --subjects 2 3 4 5 6 --n-surrogates 500
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.utils import extract_condition_data
from HNA.dsp import interpolate_nan
from HNA.coupling import plv_phase_sync
from HNA.modalities.respiration import clean_respiration
from HNA.surrogates import surrogate_test
from HNA.viz import (
    use_paper_style, CONDITION_COLORS, save_figure, sig_stars,
)

FS = 256.0
ENV_COL = "env_swell_0p2"
PLV_BW_HZ = 0.1


def _plv_metric(resp: np.ndarray, env: np.ndarray) -> float:
    """Wrap plv_phase_sync to return a single PLV value."""
    res = plv_phase_sync(resp, env, fs=FS, bw_hz=PLV_BW_HZ)
    return float(res.plv)


def _process_subject(subj: int, conditions: list[str], data_dir: Path,
                     n_surrogates: int, rng_seed_base: int = 42):
    sub_folder = f"sub-{subj:02d}"
    p = data_dir / "processed" / sub_folder / "tables" / "merged_annotated_with_audio.csv"
    if not p.exists():
        print(f"  SKIP {sub_folder}: missing merged CSV")
        return None
    df = pd.read_csv(p, low_memory=False)
    df["respiration_clean"] = clean_respiration(df["respiration"], fs=FS)

    rows = []
    for cond in conditions:
        seg = extract_condition_data(df, cond)
        if seg is None:
            print(f"    no {cond}")
            continue
        if ENV_COL not in seg.columns:
            print(f"    {cond}: missing {ENV_COL}")
            continue
        resp = seg["respiration_clean"].to_numpy(float)
        env = interpolate_nan(seg[ENV_COL].to_numpy(float))
        try:
            obs, null, p_val, z = surrogate_test(
                _plv_metric, resp, env,
                n=n_surrogates, method="phase_shuffle",
                surrogate_target="y", higher_is_better=True,
                rng_seed=rng_seed_base + subj * 100,
            )
        except Exception as e:
            print(f"    {cond}: surrogate failed: {e}")
            continue
        rows.append({
            "subject_id": subj, "condition": cond,
            "observed_plv": obs, "null_mean": float(np.nanmean(null)),
            "null_p95": float(np.nanpercentile(null, 95)),
            "z": z, "p": p_val, "significant_p05": bool(p_val < 0.05),
        })
        print(f"    {cond}: PLV={obs:.3f}, null95%={np.nanpercentile(null, 95):.3f}, "
              f"z={z:.2f}, p={p_val:.4g}")
    return rows


def _plot(df: pd.DataFrame, conditions: list[str], output_dir: Path):
    use_paper_style()
    n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_cond, figsize=(2.4 * n_cond, 3.6),
                             sharey=True)
    if n_cond == 1:
        axes = [axes]

    subjects = sorted(df["subject_id"].unique())
    x = np.arange(len(subjects))

    for ax, cond in zip(axes, conditions):
        sub_df = df[df["condition"] == cond].set_index("subject_id").reindex(subjects)
        obs = sub_df["observed_plv"].to_numpy()
        null95 = sub_df["null_p95"].to_numpy()
        sig = sub_df["significant_p05"].fillna(False).to_numpy()
        color = CONDITION_COLORS.get(cond, "#666666")

        # Bars: observed PLV
        bars = ax.bar(x, obs, color=color, alpha=0.85, width=0.65,
                      edgecolor="white", linewidth=0.8)
        # Null 95th percentile as horizontal lines per subject
        for xi, n95 in zip(x, null95):
            if np.isfinite(n95):
                ax.hlines(n95, xi - 0.3, xi + 0.3, colors="#333333",
                          linewidth=1.2, linestyles="--")
        # Significance stars above bars
        for xi, ob, s, p in zip(x, obs, sig, sub_df["p"]):
            if not np.isfinite(ob):
                continue
            label = sig_stars(p)
            ax.text(xi, ob + 0.012, label, ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    color="#222222" if s else "#888888")

        n_sig = int(sig.sum())
        ax.set_title(f"{cond}\n{n_sig}/{len(subjects)} sig.", fontsize=11.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s:02d}" for s in subjects])
        ax.set_ylim(0, max(0.6, float(np.nanmax(df["observed_plv"]) * 1.15)))
        ax.set_xlim(-0.6, len(subjects) - 0.4)

    axes[0].set_ylabel("Respiration <-> audio PLV")
    # Single legend for null line.
    axes[-1].plot([], [], color="#333333", linestyle="--", lw=1.2,
                  label="surrogate 95th pct.")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Per-subject phase-locking vs. phase-shuffled audio surrogates",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    saved = save_figure(fig, output_dir / "B_surrogate_resp_audio")
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["RS1", "VIZ", "AUD", "MULTI", "RS2"])
    p.add_argument("--n-surrogates", type=int, default=500)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "figures" / "report")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "surrogate_tests")
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running surrogate tests with n={args.n_surrogates} per (subject, condition)...")
    rows = []
    for s in args.subjects:
        print(f"\n  sub-{s:02d}")
        r = _process_subject(s, args.conditions, args.data_dir, args.n_surrogates)
        if r:
            rows.extend(r)

    if not rows:
        print("No results generated.")
        return

    df = pd.DataFrame(rows)
    out_csv = args.results_dir / "resp_audio_plv_surrogates.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved CSV: {out_csv}")

    print("\nFigure...")
    _plot(df, args.conditions, args.figures_dir)


if __name__ == "__main__":
    main()
