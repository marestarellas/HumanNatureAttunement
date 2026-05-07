#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute coupling between HRV features and audio envelope.

Similar to respiration-audio coupling but uses HRV time series (RMSSD, MeanNN, etc.)
as the cardiac signal coupled with the audio envelope.

Usage:
    python compute_hrv_audio_coupling.py --subjects 2 3 4 5 6 --aggregate
    python compute_hrv_audio_coupling.py --subjects 2 --hrv-feature HRV_RMSSD
"""

from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace

# HNA utils
from HNA.utils import get_condition_segments
from HNA.coupling import (
    windowed_xcorr, band_coherence_windowed,
    plv_phase_sync, windowed_plv,
    wpli_phase_sync, windowed_wpli,
    plot_coupling_over_time, plot_coherence_results,
)
from HNA.modalities.ecg import (
    interpolate_hrv_to_regular_grid,
    match_audio_to_hrv,  # noqa: F401  (kept for downstream callers)
)

# ---------- repo paths ----------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_FIGURES_DIR = ROOT / "figures"

# ---------- defaults ----------
FS_AUDIO = 256.0        # Audio sampling rate
FS_HRV_TARGET = 4.0     # Interpolate HRV to this rate (4 Hz gives good temporal resolution)
XC_WIN = 120.0
XC_STEP = 10.0
XC_LAG = 30.0           # Longer lag for cardiac-audio coupling
COH_FMIN, COH_FMAX = 0.01, 0.3  # Lower frequencies for HRV (cardiac rhythms are slower)
PLV_BW = 0.1
WPLI_WIN, WPLI_STEP = 120.0, 10.0

# Default HRV features to use (can specify via CLI)
DEFAULT_HRV_FEATURES = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN']


# HRV<->time-grid and HRV<->audio resampling helpers now live in
# ``HNA.modalities.ecg`` (imported at the top of the file).


def save_coupling_plots(plots_dir: Path, cond: str, hrv_feature: str,
                       xc, coh_dict, plv_win_dict, env_col: str = "",
                       hrv_signal=None, env_matched=None, fs=None):
    """Save coupling plots into ``plots_dir`` (typically figures/per_subject/sub-XX)."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    coh_for_plot = SimpleNamespace(**coh_dict) if isinstance(coh_dict, dict) else coh_dict

    # 1) time-series summary (xcorr + PLV + coherence time series)
    fig1 = plot_coupling_over_time(xc, coh_for_plot, plv_win_dict)
    env_label = f" ({env_col})" if env_col else ""
    fig1.suptitle(f"{cond} - {hrv_feature} <-> Audio{env_label}", fontsize=14, fontweight='bold')
    fig1.savefig(plots_dir / f"{cond}_hrv_audio_coupling_{hrv_feature}.png", dpi=160)
    plt.close(fig1)

    # 2) spectrum + windowed band-avg
    fig2 = plot_coherence_results(coh_dict, band=(COH_FMIN, COH_FMAX),
                                  title=f"Coherence - {cond} ({hrv_feature}){env_label}")
    fig2.savefig(plots_dir / f"{cond}_hrv_audio_coherence_{hrv_feature}.png", dpi=160)
    plt.close(fig2)

    # 3) signal alignment validation (3-panel: overlay, lag, sliding correlation)
    if hrv_signal is not None and env_matched is not None and fs is not None:
        from HNA.coupling import plot_signal_alignment_validation
        fig3 = plot_signal_alignment_validation(
            hrv_signal, env_matched, fs=fs,
            cond_label=f"{cond} ({hrv_feature})",
            env_label=env_col or "audio envelope",
            signal1_label=hrv_feature,
        )
        fig3.savefig(plots_dir / f"{cond}_hrv_audio_alignment_{hrv_feature}.png", dpi=160)
        plt.close(fig3)


def process_subject(subj: str, data_dir: Path = DEFAULT_DATA_DIR, figures_dir: Path = DEFAULT_FIGURES_DIR,
                   hrv_features=None, overwrite=False,
                   envelope_pref=("env_swell_0p2","env_swell_0p3","env_swell_0p3hz","env_broad")):
    """
    Compute HRV-audio coupling for one subject.
    
    Parameters
    ----------
    subj : str
        Subject ID (e.g., '02')
    hrv_features : list of str, optional
        HRV features to analyze. If None, uses DEFAULT_HRV_FEATURES
    overwrite : bool
        Overwrite existing results
    envelope_pref : tuple
        Preferred audio envelope column names
    """
    if hrv_features is None:
        hrv_features = DEFAULT_HRV_FEATURES

    processed = Path(data_dir) / "processed"
    sdir = processed / f"sub-{int(subj):02d}"
    tables = sdir / "tables"
    plots_dir = Path(figures_dir) / "per_subject" / f"sub-{int(subj):02d}"
    
    # Load merged data for audio
    merged = tables / "merged_annotated_with_audio.csv"
    if not merged.exists():
        raise FileNotFoundError(f"[{subj}] missing {merged}")

    df = pd.read_csv(merged, low_memory=False)
    if "time_s" not in df.columns:
        df["time_s"] = np.arange(len(df)) / FS_AUDIO

    # Choose envelope column
    env_col = next((c for c in envelope_pref if c in df.columns), None)
    if env_col is None:
        raise RuntimeError(f"[{subj}] no envelope column found. Tried: {envelope_pref}")

    # Get condition segments
    indices = get_condition_segments(df, df["condition_names"].unique())
    
    # Define pairing function
    def pair_segments(indices_dict):
        segs = {}
        for k, v in indices_dict.items():
            if isinstance(k, str) and k.endswith("_start") and v is not None:
                base = k[:-6]
                stop = indices_dict.get(base + "_stop", None)
                if stop is not None:
                    segs[base] = (int(v), int(stop))
        return segs
    
    segs = pair_segments(indices)

    # Process each condition
    for cond, (start, stop) in segs.items():
        if cond.upper() == "AUDIO_SYNC":
            continue
        
        print(f"\n[{subj}] Processing {cond}")
        
        # Load HRV features for this condition
        hrv_file = tables / f"hrv_features_{cond}.csv"
        if not hrv_file.exists():
            print(f"  WARNING: {hrv_file} not found, skipping {cond}")
            continue
        
        hrv_df = pd.read_csv(hrv_file)
        
        # Get audio segment
        r = df.iloc[start:stop].copy()
        env = r[env_col].to_numpy(dtype=float)
        audio_time = r["time_s"].to_numpy(dtype=float)
        
        # Process each HRV feature
        for hrv_feat in hrv_features:
            if hrv_feat not in hrv_df.columns:
                print(f"  WARNING: {hrv_feat} not in HRV features")
                continue
            
            print(f"    {hrv_feat}:")
            
            # Check if already processed
            out_json = tables / f"hrv_audio_coupling_{cond}_{hrv_feat}.json"
            if out_json.exists() and not overwrite:
                print(f"      EXISTS (use --overwrite)")
                continue
            
            try:
                # Interpolate HRV to regular grid
                hrv_time, hrv_signal = interpolate_hrv_to_regular_grid(
                    hrv_df, hrv_feat, FS_HRV_TARGET
                )
                
                # Validate interpolated signal
                if not np.all(np.isfinite(hrv_signal)):
                    nan_count = np.sum(~np.isfinite(hrv_signal))
                    raise ValueError(f"Interpolation produced {nan_count} NaN/Inf values")
                
                # Match audio to HRV time grid
                audio_time_cond = audio_time - audio_time[0]  # Start from 0
                env_matched = np.interp(hrv_time, audio_time_cond, env)
                
                # Ensure equal length
                min_len = min(len(hrv_signal), len(env_matched))
                hrv_signal = hrv_signal[:min_len]
                env_matched = env_matched[:min_len]
                
                # Final validation before coupling
                if not np.all(np.isfinite(hrv_signal)):
                    raise ValueError(f"HRV signal contains NaN/Inf after matching")
                if not np.all(np.isfinite(env_matched)):
                    raise ValueError(f"Audio envelope contains NaN/Inf after matching")
                
                print(f"      Signals: {len(hrv_signal)} samples @ {FS_HRV_TARGET} Hz "
                      f"({len(hrv_signal)/FS_HRV_TARGET:.1f}s)")
                
                # Compute coupling metrics
                xc = windowed_xcorr(hrv_signal, env_matched, fs=FS_HRV_TARGET, 
                                   win_sec=XC_WIN, step_sec=XC_STEP, max_lag_sec=XC_LAG)
                
                coh = band_coherence_windowed(hrv_signal, env_matched, fs=FS_HRV_TARGET,
                                             fmin=COH_FMIN, fmax=COH_FMAX,
                                             win_sec=XC_WIN, step_sec=XC_STEP)
                
                plv = plv_phase_sync(hrv_signal, env_matched, fs=FS_HRV_TARGET, 
                                    bw_hz=PLV_BW)
                plv_win = windowed_plv(hrv_signal, env_matched, fs=FS_HRV_TARGET,
                                      win_sec=XC_WIN, step_sec=XC_STEP)
                
                wpli_g = wpli_phase_sync(hrv_signal, env_matched, fs=FS_HRV_TARGET,
                                        bw_hz=PLV_BW)
                wpli_w = windowed_wpli(hrv_signal, env_matched, fs=FS_HRV_TARGET,
                                      win_sec=WPLI_WIN, step_sec=WPLI_STEP)
                
                # Save results
                payload = {
                    "subject": subj,
                    "condition": cond,
                    "hrv_feature": hrv_feat,
                    "fs": FS_HRV_TARGET,
                    "env_col": env_col,
                    "xcorr": {
                        "mean_peak_r": float(np.nanmean(xc.peak_r)),
                        "mean_peak_lag_s": float(np.nanmean(xc.peak_lag_s)),
                        "times_s": xc.times_s.tolist(),
                        "peak_r": xc.peak_r.tolist(),
                        "peak_lag_s": xc.peak_lag_s.tolist(),
                    },
                    "coherence": {
                        "peak_f": coh["peak_f"],
                        "peak_coh": coh["peak_coh"],
                        "band_avg_coh": coh["band_avg_coh"],
                        "times_s": coh["times_s"].tolist(),
                        "band_avg_coh_win": coh["band_avg_coh_win"].tolist(),
                        "band": [COH_FMIN, COH_FMAX],
                    },
                    "plv": {
                        "plv": plv.plv,
                        "preferred_lag_s": plv.preferred_lag_s,
                        "dom_freq": plv.f0,
                        "win_times_s": plv_win["times_s"].tolist(),
                        "win_plv": plv_win["plv"].tolist(),
                        "win_preferred_lag_s": plv_win["preferred_lag_s"].tolist(),
                    },
                    "wpli": {
                        "wpli": wpli_g.wpli,
                        "band": list(wpli_g.band),
                        "win_times_s": wpli_w["times_s"].tolist(),
                        "win_wpli": wpli_w["wpli"].tolist(),
                    },
                }
                
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                
                print(f"      Saved: {out_json.name}")
                
                # Save plots (write to figures/per_subject/sub-XX, not data/.../plots)
                save_coupling_plots(plots_dir, cond, hrv_feat, xc, coh, plv_win, env_col,
                                    hrv_signal=hrv_signal, env_matched=env_matched, fs=FS_HRV_TARGET)
                
            except Exception as e:
                print(f"      ERROR: {e}")
                continue

    print(f"[{subj}] Complete")


def aggregate_group(hrv_features=None, data_dir: Path = DEFAULT_DATA_DIR, save_to: Path = None):
    """Aggregate HRV-audio coupling across subjects."""
    if hrv_features is None:
        hrv_features = DEFAULT_HRV_FEATURES

    processed = Path(data_dir) / "processed"
    if save_to is None:
        save_to = processed / "group_hrv_audio_coupling.csv"

    print("\n" + "="*80)
    print("AGGREGATING HRV-AUDIO COUPLING")
    print("="*80)

    rows = []
    for sub_dir in sorted(processed.glob("sub-*")):
        tables_dir = sub_dir / "tables"
        
        for json_file in tables_dir.glob("hrv_audio_coupling_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                row = {
                    "subject": data["subject"],
                    "condition": data["condition"],
                    "hrv_feature": data["hrv_feature"],
                    "env_col": data["env_col"],
                    "xcorr_mean_peak_r": data["xcorr"]["mean_peak_r"],
                    "xcorr_mean_peak_lag_s": data["xcorr"]["mean_peak_lag_s"],
                    "coh_band_avg": data["coherence"]["band_avg_coh"],
                    "coh_peak": data["coherence"]["peak_coh"],
                    "coh_peak_f": data["coherence"]["peak_f"],
                    "plv": data["plv"]["plv"],
                    "plv_pref_lag_s": data["plv"]["preferred_lag_s"],
                    "plv_dom_f": data["plv"]["dom_freq"],
                    "wpli": data["wpli"]["wpli"],
                }
                rows.append(row)
            except Exception as e:
                print(f"  WARNING: Error reading {json_file.name}: {e}")
                continue
    
    if not rows:
        print("No results found.")
        return
    
    grp = pd.DataFrame(rows).sort_values(["subject", "condition", "hrv_feature"])
    grp.to_csv(save_to, index=False)
    print(f"\nSaved: {save_to}")
    print(f"Total: {len(grp)} rows ({len(grp)/len(hrv_features):.0f} per feature)")


def parse_args():
    p = argparse.ArgumentParser(description="HRV <-> Audio coupling analysis")
    p.add_argument("-s", "--subjects", nargs="*", default=None,
                   help="Subjects like 02 03 05. Default: auto-detect")
    p.add_argument("--hrv-feature", nargs="+", default=None,
                   help=f"HRV features to analyze (default: {DEFAULT_HRV_FEATURES})")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing results")
    p.add_argument("--aggregate", action="store_true",
                   help="After processing, write group summary CSV")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Data root (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
                   help=f"Figures root (default: {DEFAULT_FIGURES_DIR})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processed = Path(args.data_dir) / "processed"

    # Get subjects
    if args.subjects:
        subs = [f"{int(s):02d}" for s in args.subjects]
    else:
        subs = [p.name.split("-")[-1] for p in sorted(processed.glob("sub-*"))]

    # Get HRV features
    hrv_feats = args.hrv_feature if args.hrv_feature else DEFAULT_HRV_FEATURES

    print(f"\nProcessing {len(subs)} subjects")
    print(f"HRV features: {hrv_feats}")

    # Process subjects
    for s in subs:
        try:
            process_subject(s, data_dir=args.data_dir, figures_dir=args.figures_dir,
                            hrv_features=hrv_feats, overwrite=args.overwrite)
        except Exception as e:
            print(f"\nERROR processing subject {s}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate
    if args.aggregate:
        aggregate_group(hrv_features=hrv_feats, data_dir=args.data_dir)
    
    print("\nAll done.")
