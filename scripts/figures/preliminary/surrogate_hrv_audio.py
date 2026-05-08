#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surrogate test for HR-audio oscillatory coupling (coherence, PLV, wPLI).

Per (subject, condition):
  - Build instantaneous-HR trace at 4 Hz (1/RR cubic-interp from R-peaks).
  - Match audio swell envelope (env_swell_0p2) to the same 4 Hz grid.
  - Compute observed coherence / PLV / wPLI.
  - Generate n=200 phase-shuffle surrogates of the audio envelope (preserves
    spectrum), recompute each metric per surrogate -> null distribution.
  - Report z = (observed - null mean) / null std and one-sided p.

Output:
    reports/preliminary_results/figures/Fig_surrogate_hrv_audio_oscillatory.{png,pdf}
    results/surrogate_hrv_audio/surrogate_hrv_audio.csv

The figure is two stacked rows:
  Row 1: observed values per (condition, subject), with the per-subject null
         95% interval shown as a thin error bar
  Row 2: z-scores per (condition, subject); horizontal line at z=1.96 = 5%
         one-sided

This script is diagnostic. It is NOT in the report unless you decide to
add it after inspection.

Usage:
    PYTHONPATH=src python scripts/figures/preliminary/surrogate_hrv_audio.py \\
        --subjects 2 3 4 5 6 --data-dir <DATA>
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
from HNA.utils import get_condition_segments
from HNA.modalities.ecg import instantaneous_hr_signal
from HNA.surrogates import surrogate_test
from HNA.coupling import (
    plv_phase_sync, wpli_phase_sync, band_coherence_windowed,
)


FS_AUDIO = 256.0
FS_HR = 4.0
COH_FMIN, COH_FMAX = 0.01, 0.5
PLV_BW = 0.10

CONDITIONS = ["RS1", "VIZ", "AUD", "MULTI", "RS2"]
COND_COLORS = {
    "RS1":   "#C9325F",
    "RS2":   "#C9325F",
    "VIZ":   "#3B7DD8",
    "AUD":   "#E08E1A",
    "MULTI": "#5DA399",
}

N_SURROGATES = 200


# --------------------------------------------------------------------- #
# Per-subject signal preparation
# --------------------------------------------------------------------- #
def _build_signals(df: pd.DataFrame, cond: str, ecg_dir: Path,
                    env_col: str = "env_swell_0p2"):
    """Return (hr_inst, env_at_4hz) for the given condition or None."""
    indices = get_condition_segments(df, df["condition_names"].unique())
    s = indices.get(f"{cond}_start"); e = indices.get(f"{cond}_stop")
    if s is None or e is None:
        return None, None
    s, e = int(s), int(e)
    r = df.iloc[s:e].copy()
    if env_col not in r.columns:
        return None, None
    audio_time = r["time_s"].to_numpy(float) - r["time_s"].iloc[0]
    env = r[env_col].to_numpy(float)
    if not np.any(np.isfinite(env)) or len(env) < int(FS_AUDIO * 30):
        return None, None
    rpeaks_file = ecg_dir / f"rpeaks_{cond}.npy"
    if not rpeaks_file.exists():
        return None, None
    rpeaks_seg = np.load(rpeaks_file)
    if len(rpeaks_seg) < 4:
        return None, None
    seg_duration = (e - s) / FS_AUDIO
    n_target = int(round(seg_duration * FS_HR))
    hr_inst = instantaneous_hr_signal(
        rpeaks_seg, fs_in=FS_AUDIO, fs_target=FS_HR, n_samples=n_target,
    )
    t_target = np.arange(n_target) / FS_HR
    env_at_4hz = np.interp(
        t_target,
        audio_time[np.isfinite(env)],
        env[np.isfinite(env)],
    )
    m = np.isfinite(hr_inst) & np.isfinite(env_at_4hz)
    if (m.sum() < int(FS_HR * 60)
            or float(np.std(hr_inst[m])) < 1e-9
            or float(np.std(env_at_4hz[m])) < 1e-9):
        return None, None
    return hr_inst[m], env_at_4hz[m]


# --------------------------------------------------------------------- #
# Coupling-metric closures (used by surrogate_test)
# --------------------------------------------------------------------- #
def _coh_metric(x, y):
    res = band_coherence_windowed(x, y, fs=FS_HR, fmin=COH_FMIN, fmax=COH_FMAX,
                                   win_sec=120.0, step_sec=30.0)
    return float(res["band_avg_coh"])


def _plv_metric(x, y):
    return float(plv_phase_sync(x, y, fs=FS_HR, bw_hz=PLV_BW,
                                  fmin_search=0.02, fmax_search=0.5).plv)


def _wpli_metric(x, y):
    return float(wpli_phase_sync(x, y, fs=FS_HR, bw_hz=PLV_BW,
                                   fmin_search=0.02, fmax_search=0.5).wpli)


METRICS = [
    ("coh_band_avg", "Band-avg coherence", _coh_metric),
    ("plv",          "PLV",                _plv_metric),
    ("wpli",         "wPLI",               _wpli_metric),
]


# --------------------------------------------------------------------- #
# Per-subject surrogate run
# --------------------------------------------------------------------- #
def _run_subject(subj: int, data_dir: Path, n_surrogates: int):
    rows = []
    sub = f"sub-{subj:02d}"
    sdir = data_dir / "processed" / sub
    merged = sdir / "tables" / "merged_annotated_with_audio.csv"
    if not merged.exists():
        print(f"  SKIP {sub}: no merged CSV")
        return rows
    df = pd.read_csv(merged, low_memory=False)
    if "time_s" not in df.columns:
        df["time_s"] = np.arange(len(df)) / FS_AUDIO
    ecg_dir = sdir / "ecg_processed"

    for cond in CONDITIONS:
        hr, env = _build_signals(df, cond, ecg_dir)
        if hr is None or env is None:
            print(f"    {cond}: skipped (missing signals)")
            continue
        for key, _, fn in METRICS:
            try:
                obs, null, p, z = surrogate_test(
                    fn, hr, env, n=n_surrogates,
                    method="phase_shuffle",
                    surrogate_target="y",
                    higher_is_better=True,
                    rng_seed=int(subj * 1000 + hash(key + cond) % 1000),
                )
            except Exception as e:  # noqa: BLE001
                print(f"    {sub}/{cond}/{key}: surrogate failed ({e})")
                continue
            rows.append({
                "subject_id": subj, "condition": cond, "metric": key,
                "observed": float(obs),
                "null_mean": float(np.nanmean(null)),
                "null_std": float(np.nanstd(null)),
                "null_p2_5": float(np.nanpercentile(null, 2.5)),
                "null_p97_5": float(np.nanpercentile(null, 97.5)),
                "p_one_sided": float(p),
                "z": float(z),
            })
        print(f"    {cond}: ok ({n_surrogates} surrogates / 3 metrics)")
    return rows


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #
def _plot(df: pd.DataFrame, output_path: Path):
    use_paper_style()
    metrics = [m[0] for m in METRICS]
    metric_labels = {m[0]: m[1] for m in METRICS}

    fig, axes = plt.subplots(2, len(metrics),
                              figsize=(4.4 * len(metrics) + 0.6, 6.4),
                              sharex=True)
    cond_x = {c: i for i, c in enumerate(CONDITIONS)}

    for j, metric in enumerate(metrics):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue

        # --- Top row: observed value + per-subject null 95% interval ---
        ax = axes[0, j]
        for _, row in sub.iterrows():
            x = cond_x[row["condition"]] + (
                np.random.default_rng(int(row["subject_id"] * 7 + j)).uniform(-0.18, 0.18)
            )
            ax.plot([x, x],
                     [row["null_p2_5"], row["null_p97_5"]],
                     color="#9aa0a6", lw=1.2, alpha=0.75, zorder=1)
            ax.scatter(x, row["observed"],
                        s=46, color=COND_COLORS.get(row["condition"], "#000"),
                        edgecolors="black", linewidths=0.6, zorder=4)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel("observed\n(grey bar = null 95% CI)",
                           fontsize=10.5)

        # --- Bottom row: z-score per (subject, condition) ---
        ax = axes[1, j]
        rng = np.random.default_rng(0)
        for _, row in sub.iterrows():
            x = cond_x[row["condition"]] + rng.uniform(-0.18, 0.18)
            ax.scatter(x, row["z"], s=46,
                        color=COND_COLORS.get(row["condition"], "#000"),
                        edgecolors="black", linewidths=0.6, zorder=4)
        ax.axhline(1.96, color="#444", ls="--", lw=0.9, zorder=1)
        ax.axhline(0.0, color="#444", ls=":", lw=0.6, alpha=0.6, zorder=1)
        ax.text(len(CONDITIONS) - 0.1, 1.96, "  z=1.96 (one-sided p<.05)",
                 va="bottom", ha="right", fontsize=8.5, color="#444")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel("z = (obs − null mean) / null std",
                           fontsize=10.5)

    fig.suptitle(
        "HR–audio oscillatory coupling vs. phase-shuffle null "
        f"(n_surr={N_SURROGATES})",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close(fig)
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+",
                   default=[2, 3, 4, 5, 6])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--n-surrogates", type=int, default=N_SURROGATES)
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures"
                                / "Fig_surrogate_hrv_audio_oscillatory")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "surrogate_hrv_audio")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.subjects:
        print(f"\n[sub-{s:02d}]")
        rows.extend(_run_subject(s, args.data_dir, args.n_surrogates))
    if not rows:
        print("No rows -- nothing to plot.")
        return

    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "surrogate_hrv_audio.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")
    _plot(df, args.out)


if __name__ == "__main__":
    main()
