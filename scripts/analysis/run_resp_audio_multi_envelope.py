#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep all 5 coupling methods against all 12 audio envelopes per (subject, condition).

Methods
-------
- xcorr        : peak Pearson r within +-30 s lag (windowed)
- coh          : Welch coherence band-averaged in [0.04, 0.5] Hz
- plv          : phase-locking value at the dominant respiratory frequency
- wpli         : weighted phase-lag index (debiased phase consistency)
- mi           : mutual information (kNN estimator from sklearn)

Outputs
-------
- results/multi_envelope/resp_audio_multi_envelope.csv   long-form per-cell metric values

Usage
-----
    python scripts/analysis/run_resp_audio_multi_envelope.py \\
        --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.utils import extract_condition_data
from HNA.modules.dsp import interpolate_nan
from HNA.modules.coupling import (
    windowed_xcorr, band_coherence_windowed,
    plv_phase_sync, wpli_phase_sync,
)


FS = 256.0

# Default envelope columns to sweep over (matches the new band-organized output of
# scripts/preprocessing/02_compute_audio_envelopes.py).
DEFAULT_ENV_COLS = [
    "env_broad",
    "env_swell_0p2", "env_swell_0p1",
    "env_hrv_lf", "env_hrv_hf",
    "env_splash_1_5",
    "env_delta", "env_theta", "env_alpha",
    "env_beta_low", "env_beta_high", "env_gamma1",
]


# --------------------------- helpers ---------------------------
def _clean_respiration(series: pd.Series, fs: float = FS) -> np.ndarray:
    """Same cleaning as run_resp_audio_coupling.py: 0.05-1 Hz BP + zscore."""
    sos = butter(4, [0.05 / (fs / 2), 1.0 / (fs / 2)], btype="band", output="sos")
    x = interpolate_nan(series.to_numpy(float))
    x -= np.nanmean(x)
    mx = np.nanmax(np.abs(x)) or 1.0
    x = x / mx
    x = sosfiltfilt(sos, x)
    return (x - x.mean()) / (x.std() + 1e-12)


def _mi_score(x: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> float:
    """Continuous mutual information via sklearn's kNN-based estimator (Kraskov-like)."""
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float)
    return float(mutual_info_regression(x, y, n_neighbors=n_neighbors,
                                        random_state=42)[0])


# --------------------------- per-(subject, condition, env) ---------------------------
def compute_metrics(resp: np.ndarray, env: np.ndarray, fs: float = FS):
    """Compute all 5 metrics. Returns a dict keyed by short metric name."""
    # xcorr (windowed; report mean peak |r|)
    xc = windowed_xcorr(resp, env, fs=fs, win_sec=120.0, step_sec=10.0,
                        max_lag_sec=30.0)
    xcorr_peak = float(np.nanmean(np.abs(xc.peak_r)))

    # coherence
    coh = band_coherence_windowed(resp, env, fs=fs, fmin=0.04, fmax=0.5,
                                  win_sec=120.0, step_sec=30.0)
    coh_band_avg = float(coh["band_avg_coh"])

    # PLV (centered on dominant respiratory frequency)
    plv = plv_phase_sync(resp, env, fs=fs, bw_hz=0.12)
    plv_val = float(plv.plv)

    # wPLI (centered on dominant respiratory frequency)
    wpli = wpli_phase_sync(resp, env, fs=fs, bw_hz=0.12)
    wpli_val = float(wpli.wpli)

    # MI: continuous kNN (Kraskov-like)
    mi_val = _mi_score(resp, env)

    return {
        "xcorr_peak_r": xcorr_peak,
        "coh_band_avg": coh_band_avg,
        "plv": plv_val,
        "wpli": wpli_val,
        "mi": mi_val,
    }


# --------------------------- per-subject ---------------------------
def process_subject(subj: int, conditions, env_cols, data_dir: Path):
    """Returns a list of dict rows for the long-form CSV."""
    sub = f"sub-{subj:02d}"
    p = data_dir / "processed" / sub / "tables" / "merged_annotated_with_audio.csv"
    if not p.exists():
        print(f"  SKIP {sub}: no merged CSV")
        return []
    df = pd.read_csv(p, low_memory=False)
    df["respiration_clean"] = _clean_respiration(df["respiration"], fs=FS)

    rows = []
    for cond in conditions:
        seg = extract_condition_data(df, cond)
        if seg is None:
            print(f"    {cond}: no segment")
            continue
        resp = seg["respiration_clean"].to_numpy(float)
        for env_col in env_cols:
            if env_col not in seg.columns:
                continue
            env = interpolate_nan(seg[env_col].to_numpy(float))
            try:
                m = compute_metrics(resp, env, fs=FS)
            except Exception as e:
                print(f"    {cond}/{env_col}: failed ({e})")
                continue
            row = {"subject_id": subj, "condition": cond, "envelope": env_col}
            row.update(m)
            rows.append(row)
            print(f"    {cond:<6} {env_col:<14}  "
                  f"xcorr={m['xcorr_peak_r']:.3f}  coh={m['coh_band_avg']:.3f}  "
                  f"plv={m['plv']:.3f}  wpli={m['wpli']:.3f}  mi={m['mi']:.3f}")
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--envelopes", nargs="+", default=DEFAULT_ENV_COLS,
                   help="Envelope column names to sweep over.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "multi_envelope")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.subjects:
        print(f"\nsub-{s:02d}")
        rows.extend(process_subject(s, args.conditions, args.envelopes, args.data_dir))

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    out = args.results_dir / "resp_audio_multi_envelope.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved long-form CSV: {out}  (rows={len(df)})")


if __name__ == "__main__":
    main()
