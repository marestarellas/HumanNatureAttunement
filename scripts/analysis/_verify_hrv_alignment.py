"""Visually verify HRV-to-audio alignment for one (subject, condition).

Plots the interpolated HRV signal on the HRV time axis alongside the
audio swell envelope on the same axis. If alignment is correct, the
shapes should be temporally coherent (same total duration, same origin).

Usage:
    python scripts/analysis/_verify_hrv_alignment.py
    python scripts/analysis/_verify_hrv_alignment.py --subject 3 --condition AUD \\
        --hrv-feature HRV_HF --env-col env_swell_0p2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from HNA.utils import get_condition_segments
from HNA.viz import use_paper_style, save_figure


FS_AUDIO = 256.0
FS_HRV = 4.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=3,
                   help="Subject ID (default: 3).")
    p.add_argument("--condition", default="AUD",
                   help="Condition label (default: AUD).")
    p.add_argument("--hrv-feature", default="HRV_HF",
                   help="HRV feature column to plot (default: HRV_HF).")
    p.add_argument("--env-col", default="env_swell_0p2",
                   help="Audio envelope column from the merged CSV (default: env_swell_0p2).")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data",
                   help=f"Data root (default: {ROOT / 'data'}).")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures" / "ZZ_hrv_alignment_check",
                   help="Figure output path (without extension).")
    return p.parse_args()


def main():
    args = parse_args()
    use_paper_style()
    sub = f"sub-{args.subject:02d}"
    sdir = args.data_dir / "processed" / sub

    # 1. Pull condition slice from merged CSV
    df = pd.read_csv(sdir / "tables" / "merged_annotated_with_audio.csv", low_memory=False)
    indices = get_condition_segments(df, df["condition_names"].unique())
    start = int(indices[f"{args.condition}_start"])
    stop = int(indices[f"{args.condition}_stop"])
    r = df.iloc[start:stop].copy()
    audio_time_abs = r["time_s"].to_numpy(float)
    audio_time = audio_time_abs - audio_time_abs[0]
    audio_env = r[args.env_col].to_numpy(float)

    # 2. Pull HRV time series for this condition
    hrv_df = pd.read_csv(sdir / "tables" / f"hrv_features_{args.condition}.csv")
    hrv_centers = (hrv_df["time_start"].values + hrv_df["time_end"].values) / 2.0
    hrv_vals = hrv_df[args.hrv_feature].values

    # 3. Interpolate HRV to 4 Hz grid
    valid = np.isfinite(hrv_vals)
    f_hrv = interp1d(hrv_centers[valid], hrv_vals[valid], kind="linear",
                     bounds_error=False, fill_value="extrapolate")
    hrv_grid = np.arange(hrv_centers[valid].min(), hrv_centers[valid].max(), 1 / FS_HRV)
    hrv_signal = f_hrv(hrv_grid)

    # 4. Interpolate audio envelope onto HRV grid
    f_aud = interp1d(audio_time, audio_env, kind="linear",
                     bounds_error=False, fill_value=np.nan)
    aud_on_hrv = f_aud(hrv_grid)

    # 5. Diagnostics
    print(f"  Condition '{args.condition}' for {sub}:")
    print(f"    audio_time:    [{audio_time.min():.2f}, {audio_time.max():.2f}] s   "
          f"({len(audio_time)} samples @ {FS_AUDIO:.0f} Hz)")
    print(f"    hrv_centers:   [{hrv_centers.min():.2f}, {hrv_centers.max():.2f}] s   "
          f"({len(hrv_centers)} windows)")
    print(f"    hrv_grid:      [{hrv_grid.min():.2f}, {hrv_grid.max():.2f}] s   "
          f"({len(hrv_grid)} samples @ {FS_HRV:.0f} Hz)")
    print(f"    audio on HRV grid: {np.sum(np.isfinite(aud_on_hrv))}/{len(aud_on_hrv)} finite")

    # 6. Plot HRV + audio overlay
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(hrv_grid, (hrv_signal - np.nanmean(hrv_signal)) / np.nanstd(hrv_signal),
            color="#E08E1A", lw=1.5, label=f"{args.hrv_feature} (z)")
    ax.plot(hrv_grid, (aud_on_hrv - np.nanmean(aud_on_hrv)) / np.nanstd(aud_on_hrv),
            color="#3B7DD8", lw=1.5, alpha=0.85, label=f"{args.env_col} on HRV grid (z)")
    ax.set_xlabel("Time within condition (s)")
    ax.set_ylabel("Normalized amplitude (z)")
    ax.set_title(f"HRV <-> audio alignment check  {sub} {args.condition}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, args.out)
    plt.close()
    print(f"\n  Saved: {args.out.name}.png (+ pdf)")


if __name__ == "__main__":
    main()
