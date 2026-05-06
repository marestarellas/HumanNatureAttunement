#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio-EEG correlation using the pre-computed band-organized envelope columns.

For each (subject, condition, band), this script reads the corresponding
``env_<band>`` column from ``merged_annotated_with_audio.csv`` (already
band-filtered, log-normalized, at 256 Hz) and Pearson-correlates it with
each band-filtered EEG channel.

Why a separate script?
- The original ``compute_audio_eeg_correlation.py`` re-extracts the audio
  envelope from each per-condition WAV at audio rate (~44 kHz) and
  bandpass-filters at extremely low normalized cutoffs (e.g.
  ``0.5/22050 ≈ 2e-5`` for the delta band). ``sosfiltfilt`` handles this
  but the filter is borderline numerically.
- The merged-CSV envelopes are produced in
  ``02_compute_audio_envelopes.py`` using two stable processing rates
  (50 Hz / 200 Hz), then resampled to 256 Hz. Same intent, cleaner numerics.

Both should agree to a good approximation; this script lets us check that
explicitly and produce a parallel topomap set.

Output CSV: same schema as the original, written to
``results/audio_eeg_correlation_band_matched/audio_eeg_correlation_results.csv``.
That makes ``run_correlation_stats.py --results-dir <new>`` and
``plot_correlation_changes.py --results-dir <new>`` work unchanged.

Usage:
    python scripts/analysis/compute_audio_eeg_correlation_band_matched.py \\
        --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, correlate
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.utils import extract_condition_data


# Mapping EEG band -> the audio envelope column that was prefiltered to the same band.
BAND_TO_ENV_COL = {
    "delta":     ("env_delta",     (0.5, 4)),
    "theta":     ("env_theta",     (4.0, 8)),
    "alpha":     ("env_alpha",     (8.0, 13)),
    "low_beta":  ("env_beta_low",  (13.0, 20)),
    "high_beta": ("env_beta_high", (20.0, 30)),
    "gamma1":    ("env_gamma1",    (30.0, 50)),
}

EEG_SR = 256


def _bandpass_eeg(x: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4):
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    padlen = min(len(x) // 4, 500)
    return sosfiltfilt(sos, x, padlen=padlen)


def _process_subject_condition(subject_id: int, condition: str,
                               data_dir: Path, max_lag_ms: float = 500.0):
    sub = f"sub-{subject_id:02d}"
    merged_csv = data_dir / "processed" / sub / "tables" / "merged_annotated_with_audio.csv"
    if not merged_csv.exists():
        print(f"  SKIP {sub}: missing merged CSV")
        return []
    df = pd.read_csv(merged_csv, low_memory=False)
    seg = extract_condition_data(df, condition)
    if seg is None or len(seg) < int(EEG_SR * 30):
        print(f"  {sub} {condition}: no/short segment")
        return []

    # Identify EEG channels: anything that doesn't look like a condition / time / physio /
    # envelope column. Names typically start with "EEG-" or are 32-channel labels.
    exclude = {
        "time_s", "condition_names", "physio_triggers", "condition_triggers",
        "ecg", "respiration", "eeg_triggers", "respiration_clean",
        "env_broad", "env_swell_0p2", "env_swell_0p1",
        "env_hrv_lf", "env_hrv_hf", "env_splash_1_5",
        "env_delta", "env_theta", "env_alpha",
        "env_beta_low", "env_beta_high", "env_gamma1",
    }
    eeg_channels = [c for c in seg.columns
                    if c not in exclude and c.startswith("EEG-")]
    if not eeg_channels:
        print(f"  {sub} {condition}: no EEG-* channels")
        return []

    rows = []
    max_lag_samples = int(max_lag_ms * EEG_SR / 1000)

    for band_name, (env_col, (lowcut, highcut)) in BAND_TO_ENV_COL.items():
        if env_col not in seg.columns:
            continue
        env = seg[env_col].to_numpy(float)
        if not np.all(np.isfinite(env)) or env.std() < 1e-10:
            continue
        for ch in eeg_channels:
            try:
                eeg_raw = seg[ch].to_numpy(float)
                if not np.all(np.isfinite(eeg_raw)):
                    continue
                eeg = _bandpass_eeg(eeg_raw, EEG_SR, lowcut, highcut)
                if eeg.std() < 1e-10:
                    continue
                n = min(len(env), len(eeg))
                e_, eg = env[:n], eeg[:n]
                # Direct correlation
                r_direct, p_direct = pearsonr(e_, eg)
                # Lagged xcorr
                ez = (e_ - e_.mean()) / (e_.std() + 1e-12)
                gz = (eg - eg.mean()) / (eg.std() + 1e-12)
                xc = correlate(gz, ez, mode="same", method="auto") / n
                center = len(xc) // 2
                xc_w = xc[center - max_lag_samples:center + max_lag_samples + 1]
                if len(xc_w) == 0:
                    continue
                lags = np.arange(-max_lag_samples, max_lag_samples + 1)
                idx_max = int(np.argmax(np.abs(xc_w)))
                rows.append({
                    "subject_id": subject_id, "condition": condition,
                    "channel": ch, "band": band_name,
                    "lowcut": lowcut, "highcut": highcut,
                    "correlation_direct": float(r_direct),
                    "pvalue_direct": float(p_direct),
                    "correlation_max_lagged": float(xc_w[idx_max]),
                    "optimal_lag_ms": float(lags[idx_max] * 1000.0 / EEG_SR),
                    "env_source": env_col,
                })
            except Exception as e:
                print(f"    {ch}/{band_name}: {e}")
                continue
        print(f"    {band_name}: done ({sum(1 for r in rows if r['band'] == band_name)} ch)")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", required=True)
    p.add_argument("--conditions", type=str, nargs="+", required=True)
    p.add_argument("--max-lag-ms", type=float, default=500.0)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "audio_eeg_correlation_band_matched")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for s in args.subjects:
        for c in args.conditions:
            try:
                rows = _process_subject_condition(s, c, args.data_dir,
                                                  max_lag_ms=args.max_lag_ms)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  Subject {s} {c}: failed ({e})")
    if not all_rows:
        print("No results generated.")
        return
    df = pd.DataFrame(all_rows)
    out = args.output_dir / "audio_eeg_correlation_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  rows={len(df)} subjects={sorted(df['subject_id'].unique())}")


if __name__ == "__main__":
    main()
