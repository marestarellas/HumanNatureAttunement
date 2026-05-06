"""Visually verify HRV-to-audio alignment for one (subject, condition).

Plots the interpolated HRV signal on the HRV time axis alongside the
audio swell envelope on the same axis. If alignment is correct, the
shapes should be temporally coherent (same total duration, same origin).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from HNA.modules.utils import get_condition_segments
from HNA.modules.viz import use_paper_style, save_figure

DATA = Path(r"C:/Users/skite/Documents/Github/HumanNatureAttunement/data")
SUB = 3
COND = "AUD"
HRV_FEATURE = "HRV_HF"
ENV_COL = "env_swell_0p2"
FS_AUDIO = 256.0
FS_HRV = 4.0


def main():
    use_paper_style()
    sub = f"sub-{SUB:02d}"
    sdir = DATA / "processed" / sub

    # 1. Pull condition slice from merged CSV
    df = pd.read_csv(sdir / "tables" / "merged_annotated_with_audio.csv", low_memory=False)
    indices = get_condition_segments(df, df["condition_names"].unique())
    start = int(indices[f"{COND}_start"]); stop = int(indices[f"{COND}_stop"])
    r = df.iloc[start:stop].copy()
    audio_time_abs = r["time_s"].to_numpy(float)
    audio_time = audio_time_abs - audio_time_abs[0]
    audio_env = r[ENV_COL].to_numpy(float)

    # 2. Pull HRV time series for this condition
    hrv_df = pd.read_csv(sdir / "tables" / f"hrv_features_{COND}.csv")
    hrv_centers = (hrv_df["time_start"].values + hrv_df["time_end"].values) / 2.0
    hrv_vals = hrv_df[HRV_FEATURE].values

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
    print(f"  Condition '{COND}' for {sub}:")
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
            color="#E08E1A", lw=1.5, label=f"{HRV_FEATURE} (z)")
    ax.plot(hrv_grid, (aud_on_hrv - np.nanmean(aud_on_hrv)) / np.nanstd(aud_on_hrv),
            color="#3B7DD8", lw=1.5, alpha=0.85, label=f"{ENV_COL} on HRV grid (z)")
    ax.set_xlabel("Time within condition (s)")
    ax.set_ylabel("Normalized amplitude (z)")
    ax.set_title(f"HRV <-> audio alignment check  {sub} {COND}",
                 fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figures" / "report" / "ZZ_hrv_alignment_check"
    save_figure(fig, out)
    plt.close()
    print(f"\n  Saved: {out.name}.png (+ pdf)")


if __name__ == "__main__":
    main()
