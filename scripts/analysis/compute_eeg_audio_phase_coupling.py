#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-audio band-matched PHASE coupling: PLV + wPLI per (subject, condition,
channel, band).

This is the phase-only complement of the existing band-matched correlation
analysis (``compute_eeg_audio_correlation_band_matched.py``):

  - audio side  : the band-organised envelope column ``env_<band>`` from
                  the merged CSV (audio amplitude envelope bandpassed to
                  the same EEG band: env_delta @ 0.5-4 Hz, env_theta @ 4-8,
                  env_alpha @ 8-13, env_beta_low @ 13-20, env_beta_high
                  @ 20-30, env_gamma1 @ 30-50). All at 256 Hz.
  - EEG side    : raw EEG bandpass-filtered to the same band (zero-phase
                  4th-order Butter; matches the band-matched correlation
                  pipeline).
  - PLV / wPLI  : on the (eeg_bandpassed, env_<band>) pair, with f0 set to
                  the band centre and bw_hz set to the band width so the
                  internal bandpass-Hilbert in plv_phase_sync /
                  wpli_phase_sync re-isolates the same band before phase
                  extraction. The result tests whether the EEG rhythm in
                  band X phase-locks with audio amplitude modulations in
                  band X (cortical-tracking style).

Output:
    results/eeg_audio_phase_coupling/eeg_audio_phase_coupling.csv
        rows: (subject_id, condition, channel, band, lowcut, highcut,
               plv, wpli, plv_pref_lag_s, plv_dom_freq, n_samples)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.utils import extract_condition_data
from HNA.coupling import plv_phase_sync, wpli_phase_sync


EEG_SR = 256


# Band -> (audio envelope column, lowcut, highcut, f0_centre, bw_hz)
BAND_PARAMS = {
    "delta":     ("env_delta",     0.5,  4.0,   2.25,  3.5),
    "theta":     ("env_theta",     4.0,  8.0,   6.0,   4.0),
    "alpha":     ("env_alpha",     8.0,  13.0,  10.5,  5.0),
    "low_beta":  ("env_beta_low",  13.0, 20.0,  16.5,  7.0),
    "high_beta": ("env_beta_high", 20.0, 30.0,  25.0,  10.0),
    "gamma1":    ("env_gamma1",    30.0, 50.0,  40.0,  20.0),
}


def _bandpass_eeg(x: np.ndarray, fs: float,
                   lowcut: float, highcut: float, order: int = 4):
    """Zero-phase 4th-order Butter bandpass; matches the existing
    band-matched correlation pipeline."""
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq],
                  btype="bandpass", output="sos")
    return sosfiltfilt(sos, x.astype(float))


def process_subject_condition(subject_id: int, condition: str,
                                data_dir: Path):
    sub = f"sub-{subject_id:02d}"
    merged_csv = data_dir / "processed" / sub / "tables" / "merged_annotated_with_audio.csv"
    if not merged_csv.exists():
        print(f"  SKIP {sub}/{condition}: missing merged CSV")
        return []
    df = pd.read_csv(merged_csv, low_memory=False)
    seg = extract_condition_data(df, condition)
    if seg is None or len(seg) < int(EEG_SR * 30):
        print(f"  SKIP {sub}/{condition}: not enough samples")
        return []

    exclude = {"time_s", "respiration", "ecg",
                "condition", "condition_names", "condition_label",
                "audio", "AUDIO_SYNC", "AUDIO", "AUDIO_TIME",
                "env_broad", "env_swell_0p2", "env_swell_0p1",
                "env_hrv_lf", "env_hrv_hf", "env_splash_1_5",
                "env_delta", "env_theta", "env_alpha",
                "env_beta_low", "env_beta_high", "env_gamma1"}
    eeg_channels = [c for c in seg.columns
                     if c not in exclude and c.startswith("EEG-")]
    if not eeg_channels:
        print(f"  {sub} {condition}: no EEG-* channels")
        return []

    rows = []
    for band_name, (env_col, lowcut, highcut, f0, bw) in BAND_PARAMS.items():
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

                plv = plv_phase_sync(eg, e_, fs=EEG_SR,
                                      f0=f0, bw_hz=bw)
                wpli = wpli_phase_sync(eg, e_, fs=EEG_SR,
                                        f0=f0, bw_hz=bw)
            except Exception as exc:  # noqa: BLE001
                print(f"  {sub}/{condition}/{band_name}/{ch}: {exc}")
                continue
            rows.append({
                "subject_id": subject_id, "condition": condition,
                "channel": ch, "band": band_name,
                "lowcut": lowcut, "highcut": highcut,
                "plv": float(plv.plv),
                "wpli": float(wpli.wpli),
                "plv_pref_lag_s": float(plv.preferred_lag_s),
                "plv_dom_freq": float(plv.f0),
                "n_samples": int(n),
            })
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+",
                   default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+",
                   default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "eeg_audio_phase_coupling")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.subjects:
        for c in args.conditions:
            print(f"[sub-{s:02d}] {c}")
            rows.extend(process_subject_condition(s, c, args.data_dir))
    if not rows:
        print("No rows.")
        return
    out = args.results_dir / "eeg_audio_phase_coupling.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Saved: {out}  rows={len(rows)}")


if __name__ == "__main__":
    main()
