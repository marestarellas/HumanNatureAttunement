#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep all 5 coupling methods against all 12 audio envelopes for each HRV feature,
per (subject, condition).

For each (subject, condition, HRV feature, audio envelope), we compute:
- xcorr   : peak Pearson r within +-30 s lag (windowed)
- coh     : Welch coherence band-averaged in [0.01, 0.5] Hz
- plv     : phase-locking value at the dominant HRV-band frequency
- wpli    : weighted phase-lag index
- mi      : mutual information (kNN/Kraskov-like)

Output:
    results/multi_envelope_hrv/hrv_audio_multi_envelope.csv

Usage:
    python scripts/analysis/run_hrv_audio_multi_envelope.py \\
        --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI \\
        --hrv-features HRV_LF HRV_HF HRV_MeanNN
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.utils import get_condition_segments
from HNA.modules.coupling import (
    windowed_xcorr, band_coherence_windowed,
    plv_phase_sync, wpli_phase_sync,
)


FS_AUDIO = 256.0
FS_HRV = 4.0   # interpolation rate for HRV time series

DEFAULT_ENV_COLS = [
    "env_broad",
    "env_swell_0p2", "env_swell_0p1",
    "env_hrv_lf", "env_hrv_hf",
    "env_splash_1_5",
    "env_delta", "env_theta", "env_alpha",
    "env_beta_low", "env_beta_high", "env_gamma1",
]

DEFAULT_HRV_FEATURES = ["HRV_LF", "HRV_HF", "HRV_MeanNN"]


def _interp_to_grid(times: np.ndarray, vals: np.ndarray, fs: float):
    valid = np.isfinite(vals)
    if valid.sum() < 4:
        return None, None
    t = times[valid]; v = vals[valid]
    grid = np.arange(t.min(), t.max(), 1.0 / fs)
    return grid, interp1d(t, v, kind="linear", bounds_error=False,
                          fill_value="extrapolate")(grid)


def _audio_to_hrv_grid(env: np.ndarray, audio_time: np.ndarray,
                       hrv_time: np.ndarray):
    f = interp1d(audio_time, env, kind="linear",
                 bounds_error=False, fill_value=np.nan)
    return f(hrv_time)


def compute_metrics(hrv: np.ndarray, env: np.ndarray, fs: float):
    """5 coupling metrics; HRV-tuned default windows / band."""
    xc = windowed_xcorr(hrv, env, fs=fs, win_sec=120.0, step_sec=10.0,
                        max_lag_sec=30.0)
    xcorr_peak = float(np.nanmean(np.abs(xc.peak_r)))

    coh = band_coherence_windowed(hrv, env, fs=fs, fmin=0.01, fmax=0.5,
                                  win_sec=120.0, step_sec=30.0)
    coh_band_avg = float(coh["band_avg_coh"])

    plv = plv_phase_sync(hrv, env, fs=fs, bw_hz=0.10,
                         fmin_search=0.02, fmax_search=0.5)
    plv_val = float(plv.plv)

    wpli = wpli_phase_sync(hrv, env, fs=fs, bw_hz=0.10,
                           fmin_search=0.02, fmax_search=0.5)
    wpli_val = float(wpli.wpli)

    try:
        mi_val = float(mutual_info_regression(
            hrv.reshape(-1, 1), env, n_neighbors=3, random_state=42)[0])
    except Exception:
        mi_val = float("nan")

    return {
        "xcorr_peak_r": xcorr_peak,
        "coh_band_avg": coh_band_avg,
        "plv": plv_val,
        "wpli": wpli_val,
        "mi": mi_val,
    }


def process_subject(subj: int, conditions, env_cols, hrv_features, data_dir: Path):
    sub = f"sub-{subj:02d}"
    sdir = data_dir / "processed" / sub
    merged = sdir / "tables" / "merged_annotated_with_audio.csv"
    if not merged.exists():
        print(f"  SKIP {sub}: no merged CSV")
        return []
    df = pd.read_csv(merged, low_memory=False)
    if "time_s" not in df.columns:
        df["time_s"] = np.arange(len(df)) / FS_AUDIO

    indices = get_condition_segments(df, df["condition_names"].unique())
    segs = {}
    for k, v in indices.items():
        if isinstance(k, str) and k.endswith("_start") and v is not None:
            base = k[:-6]
            stop = indices.get(base + "_stop", None)
            if stop is not None:
                segs[base] = (int(v), int(stop))

    rows = []
    for cond in conditions:
        if cond not in segs:
            continue
        start, stop = segs[cond]
        r = df.iloc[start:stop].copy()
        # HRV table's time_start/time_end are condition-relative (0..duration),
        # so zero the audio time axis to match.
        audio_time = r["time_s"].to_numpy(float)
        audio_time = audio_time - audio_time[0]

        hrv_file = sdir / "tables" / f"hrv_features_{cond}.csv"
        if not hrv_file.exists():
            print(f"    {cond}: no HRV file, skipping")
            continue
        hrv_df = pd.read_csv(hrv_file)
        if "time_start" not in hrv_df.columns or "time_end" not in hrv_df.columns:
            continue
        hrv_centers = (hrv_df["time_start"].values + hrv_df["time_end"].values) / 2.0

        for feat in hrv_features:
            if feat not in hrv_df.columns:
                continue
            hrv_grid, hrv_signal = _interp_to_grid(hrv_centers, hrv_df[feat].values, FS_HRV)
            if hrv_signal is None or len(hrv_signal) < int(FS_HRV * 60):
                continue
            for env_col in env_cols:
                if env_col not in r.columns:
                    continue
                env_full = r[env_col].to_numpy(float)
                env_on_hrv = _audio_to_hrv_grid(env_full, audio_time, hrv_grid)
                m = np.isfinite(hrv_signal) & np.isfinite(env_on_hrv)
                if m.sum() < int(FS_HRV * 60):
                    continue
                hrv_v = hrv_signal[m]; env_v = env_on_hrv[m]
                try:
                    metrics = compute_metrics(hrv_v, env_v, fs=FS_HRV)
                except Exception as e:
                    print(f"    {cond}/{feat}/{env_col}: failed ({e})")
                    continue
                row = {"subject_id": subj, "condition": cond,
                       "hrv_feature": feat, "envelope": env_col}
                row.update(metrics)
                rows.append(row)
            print(f"    {cond:<6} {feat:<14}  ({len([1 for x in rows if x['hrv_feature']==feat and x['condition']==cond])} envelopes)")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--envelopes", nargs="+", default=DEFAULT_ENV_COLS)
    p.add_argument("--hrv-features", nargs="+", default=DEFAULT_HRV_FEATURES)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "multi_envelope_hrv")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.subjects:
        print(f"\nsub-{s:02d}")
        rows.extend(process_subject(s, args.conditions, args.envelopes,
                                    args.hrv_features, args.data_dir))
    if not rows:
        print("No results.")
        return
    df = pd.DataFrame(rows)
    out = args.results_dir / "hrv_audio_multi_envelope.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}  rows={len(df)}")


if __name__ == "__main__":
    main()
