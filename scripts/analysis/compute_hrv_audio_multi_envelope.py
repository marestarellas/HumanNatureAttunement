#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep all 5 coupling methods against all 12 audio envelopes per
(subject, condition, HRV feature).

Two cardiac signals are used, each appropriate for a different family:

* **Slow-trend metrics** (xcorr, MI) run on the **windowed HRV-feature
  trace** (HRV_MeanNN / HRV_RMSSD / ... computed in 30 s windows with
  90% overlap, linearly resampled to 4 Hz). Effective bandwidth
  ~0--0.017 Hz.
* **Oscillatory metrics** (PLV, wPLI, coherence) run on the
  **instantaneous-HR trace at 4 Hz** (built per-condition from R-peaks
  via 1/RR cubic interpolation). Effective bandwidth ~0--2 Hz, which
  is what the bandpass-Hilbert phase analyses actually require to
  resolve the swell band.

The CSV output schema is unchanged so downstream figures (Figure F /
HRV multi-envelope heatmap) work without modification: the value of
PLV / wPLI / coherence is the SAME across all rows that share
(subject, condition, envelope) regardless of `hrv_feature`, because
those metrics no longer depend on the HRV feature -- they depend only
on the instantaneous-HR trace + audio envelope.

For each (subject, condition, HRV feature, audio envelope), we compute:
- xcorr   : peak Pearson r within +-30 s lag (windowed); HRV-feature trace
- coh     : Welch coherence band-averaged in [0.01, 0.5] Hz; instantaneous HR
- plv     : phase-locking value at the dominant audio-band frequency; instantaneous HR
- wpli    : weighted phase-lag index; instantaneous HR
- mi      : mutual information (kNN/Kraskov-like); HRV-feature trace

Output:
    results/multi_envelope_hrv/hrv_audio_multi_envelope.csv

Usage:
    python scripts/analysis/compute_hrv_audio_multi_envelope.py \\
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

from HNA.utils import get_condition_segments
from HNA.modalities.ecg import instantaneous_hr_signal
from HNA.coupling import (
    windowed_xcorr, band_coherence_windowed,
    plv_phase_sync, wpli_phase_sync,
)


FS_AUDIO = 256.0
FS_HRV = 4.0   # 4 Hz grid; common to both windowed-trace and instantaneous-HR

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


def compute_slow_trend(hrv_trace: np.ndarray, env: np.ndarray, fs: float):
    """Slow-trend metrics on the windowed HRV-feature trace.

    Cross-correlation and MI both work on signals with bandwidth limited
    to the HRV trace's natural ~0.017 Hz Nyquist; no narrowband-Hilbert
    step is involved.
    """
    xc = windowed_xcorr(hrv_trace, env, fs=fs, win_sec=120.0, step_sec=10.0,
                        max_lag_sec=30.0)
    xcorr_peak = float(np.nanmean(np.abs(xc.peak_r)))

    try:
        mi_val = float(mutual_info_regression(
            hrv_trace.reshape(-1, 1), env, n_neighbors=3, random_state=42)[0])
    except Exception:  # noqa: BLE001
        mi_val = float("nan")

    return {"xcorr_peak_r": xcorr_peak, "mi": mi_val}


def compute_oscillatory(hr_inst: np.ndarray, env: np.ndarray, fs: float):
    """Oscillatory metrics on the instantaneous-HR trace.

    PLV / wPLI / coherence require a signal with real spectral content
    in the swell band; the windowed HRV-feature trace does not have it,
    but a 4 Hz instantaneous-HR trace does.
    """
    coh = band_coherence_windowed(hr_inst, env, fs=fs, fmin=0.01, fmax=0.5,
                                  win_sec=120.0, step_sec=30.0)
    coh_band_avg = float(coh["band_avg_coh"])

    plv = plv_phase_sync(hr_inst, env, fs=fs, bw_hz=0.10,
                         fmin_search=0.02, fmax_search=0.5)
    plv_val = float(plv.plv)

    wpli = wpli_phase_sync(hr_inst, env, fs=fs, bw_hz=0.10,
                           fmin_search=0.02, fmax_search=0.5)
    wpli_val = float(wpli.wpli)

    return {
        "coh_band_avg": coh_band_avg,
        "plv": plv_val,
        "wpli": wpli_val,
    }


def process_subject(subj: int, conditions, env_cols, hrv_features, data_dir: Path):
    sub = f"sub-{subj:02d}"
    sdir = data_dir / "processed" / sub
    ecg_dir = sdir / "ecg_processed"
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

        # Build instantaneous-HR trace once per condition (used by all
        # oscillatory metrics, regardless of HRV feature or audio envelope).
        # R-peaks in rpeaks_<COND>.npy are saved as condition-relative
        # indices by 05_preprocess_ecg.py (no rebasing needed).
        rpeaks_file = ecg_dir / f"rpeaks_{cond}.npy"
        hr_inst = None
        hr_grid = None
        if rpeaks_file.exists():
            rpeaks_seg = np.load(rpeaks_file)
            if len(rpeaks_seg) >= 4:
                seg_duration = (stop - start) / FS_AUDIO
                n_target = int(round(seg_duration * FS_HRV))
                try:
                    hr_inst = instantaneous_hr_signal(
                        rpeaks_seg, fs_in=FS_AUDIO,
                        fs_target=FS_HRV, n_samples=n_target,
                    )
                    hr_grid = np.arange(n_target) / FS_HRV
                except Exception as e:  # noqa: BLE001
                    print(f"    {cond}: instantaneous HR failed ({e})")
                    hr_inst = None
        if hr_inst is None:
            print(f"    {cond}: no R-peaks/instantaneous HR -> "
                  f"oscillatory metrics will be NaN for this condition")

        # Cache oscillatory metrics per env (independent of hrv_feature)
        osc_cache: dict[str, dict] = {}
        if hr_inst is not None:
            for env_col in env_cols:
                if env_col not in r.columns:
                    continue
                env_full = r[env_col].to_numpy(float)
                env_on_hr = _audio_to_hrv_grid(env_full, audio_time, hr_grid)
                m = np.isfinite(hr_inst) & np.isfinite(env_on_hr)
                if m.sum() < int(FS_HRV * 60):
                    continue
                try:
                    osc_cache[env_col] = compute_oscillatory(
                        hr_inst[m], env_on_hr[m], fs=FS_HRV,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"    {cond}/HR_inst/{env_col}: oscillatory failed ({e})")

        # Slow-trend metrics: per HRV feature
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
                    slow = compute_slow_trend(hrv_v, env_v, fs=FS_HRV)
                except Exception as e:  # noqa: BLE001
                    print(f"    {cond}/{feat}/{env_col}: slow-trend failed ({e})")
                    continue
                # Pull oscillatory values from per-condition cache (same
                # for every hrv_feature row that shares (cond, envelope))
                osc = osc_cache.get(env_col, {
                    "coh_band_avg": float("nan"),
                    "plv": float("nan"),
                    "wpli": float("nan"),
                })
                row = {"subject_id": subj, "condition": cond,
                       "hrv_feature": feat, "envelope": env_col,
                       "slow_signal": f"{feat}_30s_window_at_4Hz",
                       "osc_signal": ("hr_instantaneous_4Hz"
                                       if hr_inst is not None else "unavailable")}
                row.update(slow)
                row.update(osc)
                rows.append(row)
            print(f"    {cond:<6} {feat:<14}  ("
                  f"{len([1 for x in rows if x['hrv_feature']==feat and x['condition']==cond])} envelopes)")
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
