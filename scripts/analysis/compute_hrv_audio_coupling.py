#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute coupling between cardiac signal and audio envelope.

This script uses TWO complementary cardiac derivations, each appropriate
for a different family of coupling metrics:

* **Instantaneous heart-rate (BPM) at 4 Hz**, built per-condition from
  R-peak indices via 1/RR interpolation. Bandwidth ~0--2 Hz Nyquist;
  carries beat-to-beat variability including respiratory sinus
  arrhythmia and audio-swell-band oscillations. Used for the
  **oscillatory metrics** PLV / wPLI / coherence at the swell band.
* **Windowed HRV-feature traces** (HRV_MeanNN, HRV_RMSSD, HRV_SDNN)
  computed in 30 s windows with 90% overlap, then linearly resampled
  to 4 Hz. Effective bandwidth ~0--0.017 Hz (Nyquist of the windowed
  series). Used for the **slow-trend metrics** cross-correlation
  (and MI on the slow envelope).

Mixing the two prevents narrowband-Hilbert phase analyses from being
applied to traces whose effective bandwidth sits well below the swell
band --- which would have made PLV / wPLI / coherence meaningless on a
windowed RMSSD/MeanNN/SDNN trace.

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
    instantaneous_hr_signal,
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
    ecg_dir = sdir / "ecg_processed"
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

        # Load R-peaks for this condition (for instantaneous HR)
        rpeaks_file = ecg_dir / f"rpeaks_{cond}.npy"
        rpeaks_cond = np.load(rpeaks_file) if rpeaks_file.exists() else None
        if rpeaks_cond is None:
            print(f"  NOTE: {rpeaks_file.name} missing -> oscillatory metrics "
                  f"(PLV / wPLI / coherence) will be skipped")

        # Get audio segment
        r = df.iloc[start:stop].copy()
        env = r[env_col].to_numpy(dtype=float)
        audio_time = r["time_s"].to_numpy(dtype=float)

        # ----- ONE-TIME (per condition) -----
        # Build instantaneous HR (4 Hz) for the oscillatory metrics.
        # Time grid follows the audio segment so xcorr/coherence can be
        # compared against the same matched envelope.
        hr_inst = None
        env_at_4hz = None
        if rpeaks_cond is not None and len(rpeaks_cond) >= 4:
            seg_duration_s = (stop - start) / FS_AUDIO
            n_target = int(round(seg_duration_s * FS_HRV_TARGET))
            # R-peaks in rpeaks_<COND>.npy are saved by 05_preprocess_ecg.py
            # from `preprocess_ecg_segment(ecg_for_this_condition, fs=FS)`,
            # so the indices are already condition-relative (starting at 0
            # within the condition slice). No rebasing needed.
            rpeaks_seg = rpeaks_cond
            try:
                hr_inst = instantaneous_hr_signal(
                    rpeaks_seg, fs_in=FS_AUDIO,
                    fs_target=FS_HRV_TARGET, n_samples=n_target,
                )
                # Match audio to the same 4 Hz grid the HR trace lives on.
                # Pre-clean NaN in the audio envelope so np.interp doesn't
                # propagate them onto the target grid.
                audio_time_cond = audio_time - audio_time[0]
                env_finite_mask = np.isfinite(env)
                if env_finite_mask.sum() < 2:
                    raise ValueError("audio envelope all-NaN in this segment")
                env_clean = np.interp(
                    audio_time_cond,
                    audio_time_cond[env_finite_mask],
                    env[env_finite_mask],
                )
                t_target = np.arange(n_target) / FS_HRV_TARGET
                env_at_4hz = np.interp(t_target, audio_time_cond, env_clean)
                # Ensure equal length defensively
                m = min(len(hr_inst), len(env_at_4hz))
                hr_inst, env_at_4hz = hr_inst[:m], env_at_4hz[:m]
                if not np.all(np.isfinite(hr_inst)):
                    # Fall back: replace residual NaN with mean (helper
                    # already does this; defensive double-check)
                    mu = float(np.nanmean(hr_inst))
                    hr_inst = np.where(np.isfinite(hr_inst), hr_inst, mu)
                if not np.all(np.isfinite(env_at_4hz)):
                    mu = float(np.nanmean(env_at_4hz))
                    env_at_4hz = np.where(
                        np.isfinite(env_at_4hz), env_at_4hz, mu
                    )
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: instantaneous HR build failed ({e}); "
                      f"oscillatory metrics will be skipped")
                hr_inst = None
                env_at_4hz = None

        # Compute PLV / wPLI / coherence ONCE on (hr_inst, env_at_4hz)
        plv_payload = None
        wpli_payload = None
        coh_payload = None
        coh_full = None       # full dict (with 'f' and 'Cxy') for plotting

        # Sanity-check: signals must be finite, non-empty, and have
        # variance (PLV / wPLI / coherence are undefined on a constant
        # signal --- _nan_interp would raise "all values are NaN" on
        # one that became NaN after filtering).
        oscillatory_ok = (
            hr_inst is not None and env_at_4hz is not None
            and len(hr_inst) >= int(FS_HRV_TARGET * 60)  # >= 60 s
            and np.all(np.isfinite(hr_inst))
            and np.all(np.isfinite(env_at_4hz))
            and float(np.std(hr_inst)) > 1e-9
            and float(np.std(env_at_4hz)) > 1e-9
        )
        if oscillatory_ok:
            try:
                print(f"    [oscillatory] hr_inst ({len(hr_inst)} samp @ "
                      f"{FS_HRV_TARGET} Hz) <-> {env_col}")
                plv_obj = plv_phase_sync(hr_inst, env_at_4hz,
                                          fs=FS_HRV_TARGET, bw_hz=PLV_BW)
                plv_win_dict = windowed_plv(hr_inst, env_at_4hz,
                                             fs=FS_HRV_TARGET,
                                             win_sec=XC_WIN, step_sec=XC_STEP)
                wpli_obj = wpli_phase_sync(hr_inst, env_at_4hz,
                                            fs=FS_HRV_TARGET, bw_hz=PLV_BW)
                wpli_win_dict = windowed_wpli(hr_inst, env_at_4hz,
                                               fs=FS_HRV_TARGET,
                                               win_sec=WPLI_WIN,
                                               step_sec=WPLI_STEP)
                coh_dict = band_coherence_windowed(
                    hr_inst, env_at_4hz, fs=FS_HRV_TARGET,
                    fmin=COH_FMIN, fmax=COH_FMAX,
                    win_sec=XC_WIN, step_sec=XC_STEP,
                )
                coh_full = coh_dict  # keep raw dict for plotting
            except Exception as e:  # noqa: BLE001
                print(f"    WARNING: oscillatory metrics failed ({e}); "
                      f"skipping PLV / wPLI / coherence for this condition")
                plv_obj = wpli_obj = coh_dict = None
                plv_win_dict = wpli_win_dict = None
                coh_full = None
                oscillatory_ok = False
        if oscillatory_ok and plv_obj is not None:
            plv_payload = {
                "plv": plv_obj.plv,
                "preferred_lag_s": plv_obj.preferred_lag_s,
                "dom_freq": plv_obj.f0,
                "win_times_s": plv_win_dict["times_s"].tolist(),
                "win_plv": plv_win_dict["plv"].tolist(),
                "win_preferred_lag_s": plv_win_dict["preferred_lag_s"].tolist(),
                "signal": "hr_instantaneous_4Hz",
            }
            wpli_payload = {
                "wpli": wpli_obj.wpli,
                "band": list(wpli_obj.band),
                "win_times_s": wpli_win_dict["times_s"].tolist(),
                "win_wpli": wpli_win_dict["wpli"].tolist(),
                "signal": "hr_instantaneous_4Hz",
            }
            coh_payload = {
                "peak_f": coh_dict["peak_f"],
                "peak_coh": coh_dict["peak_coh"],
                "band_avg_coh": coh_dict["band_avg_coh"],
                "times_s": coh_dict["times_s"].tolist(),
                "band_avg_coh_win": coh_dict["band_avg_coh_win"].tolist(),
                "band": [COH_FMIN, COH_FMAX],
                "signal": "hr_instantaneous_4Hz",
            }

            # Save the oscillatory payload once per condition (independent
            # of HRV-feature loop below).
            osc_json = tables / f"hrv_audio_coupling_{cond}_hr_instantaneous.json"
            if overwrite or not osc_json.exists():
                with open(osc_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "subject": subj, "condition": cond,
                        "signal": "hr_instantaneous_4Hz",
                        "fs": FS_HRV_TARGET, "env_col": env_col,
                        "plv": plv_payload, "wpli": wpli_payload,
                        "coherence": coh_payload,
                    }, f, ensure_ascii=False)
                print(f"    [oscillatory] saved {osc_json.name}")

        # ----- PER-HRV-FEATURE LOOP (slow-trend xcorr) -----
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
                
                print(f"      [slow-trend] {hrv_feat}: "
                      f"{len(hrv_signal)} samp @ {FS_HRV_TARGET} Hz "
                      f"({len(hrv_signal)/FS_HRV_TARGET:.1f} s)")

                # Slow-trend metric: cross-correlation only.
                # PLV / wPLI / coherence are NOT computed on the windowed
                # HRV-feature trace (its bandwidth ~0-0.017 Hz lies below
                # the swell band; narrowband-Hilbert phase analyses on it
                # are meaningless). They are computed once per condition
                # on the instantaneous-HR signal, above.
                xc = windowed_xcorr(hrv_signal, env_matched, fs=FS_HRV_TARGET,
                                    win_sec=XC_WIN, step_sec=XC_STEP,
                                    max_lag_sec=XC_LAG)

                # Save results
                payload = {
                    "subject": subj,
                    "condition": cond,
                    "hrv_feature": hrv_feat,
                    "fs": FS_HRV_TARGET,
                    "env_col": env_col,
                    "signal": f"{hrv_feat}_30s_window_at_4Hz",
                    "xcorr": {
                        "mean_peak_r": float(np.nanmean(xc.peak_r)),
                        "mean_peak_lag_s": float(np.nanmean(xc.peak_lag_s)),
                        "times_s": xc.times_s.tolist(),
                        "peak_r": xc.peak_r.tolist(),
                        "peak_lag_s": xc.peak_lag_s.tolist(),
                    },
                    # PLV / wPLI / coherence sit in the per-condition
                    # ``hrv_audio_coupling_{cond}_hr_instantaneous.json``
                    # file (they don't depend on which HRV feature is
                    # under test).
                }

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)

                print(f"      Saved: {out_json.name}")

                # Save plots: the time-resolved coupling plot mixes xcorr
                # (slow-trend) with PLV / coherence (oscillatory) so that
                # each panel shows its appropriate cardiac derivation.
                if plv_payload is not None:
                    plv_win_for_plot = {
                        "times_s": np.asarray(plv_payload["win_times_s"]),
                        "plv": np.asarray(plv_payload["win_plv"]),
                        "preferred_lag_s": np.asarray(
                            plv_payload["win_preferred_lag_s"]
                        ),
                    }
                else:
                    plv_win_for_plot = {
                        "times_s": np.array([]), "plv": np.array([]),
                        "preferred_lag_s": np.array([]),
                    }
                save_coupling_plots(
                    plots_dir, cond, hrv_feat, xc,
                    coh_full if coh_full is not None else {
                        "f": np.array([]), "Cxy": np.array([]),
                        "peak_f": np.nan, "peak_coh": np.nan,
                        "band_avg_coh": np.nan, "times_s": np.array([]),
                        "band_avg_coh_win": np.array([]),
                    },
                    plv_win_for_plot, env_col,
                    hrv_signal=hrv_signal, env_matched=env_matched,
                    fs=FS_HRV_TARGET,
                )

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

        # Cache the per-condition oscillatory payload (PLV / wPLI / coh
        # computed once on hr_instantaneous_4Hz). Keyed by condition.
        osc_cache: dict[str, dict] = {}
        for osc_json in tables_dir.glob(
                "hrv_audio_coupling_*_hr_instantaneous.json"):
            try:
                with open(osc_json, "r") as f:
                    payload = json.load(f)
                osc_cache[payload["condition"]] = payload
            except Exception:  # noqa: BLE001
                pass

        for json_file in sorted(tables_dir.glob("hrv_audio_coupling_*.json")):
            # Skip the per-condition oscillatory payload (one row per
            # condition there, not per HRV feature).
            if json_file.name.endswith("_hr_instantaneous.json"):
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                cond = data["condition"]
                osc = osc_cache.get(cond, {})
                osc_plv = osc.get("plv") or {}
                osc_wpli = osc.get("wpli") or {}
                osc_coh = osc.get("coherence") or {}

                row = {
                    "subject": data["subject"],
                    "condition": cond,
                    "hrv_feature": data["hrv_feature"],
                    "env_col": data["env_col"],
                    # Slow-trend metric (per HRV feature)
                    "xcorr_mean_peak_r": data["xcorr"]["mean_peak_r"],
                    "xcorr_mean_peak_lag_s": data["xcorr"]["mean_peak_lag_s"],
                    # Oscillatory metrics (per condition; same value
                    # repeated across the three HRV-feature rows)
                    "coh_band_avg": osc_coh.get("band_avg_coh", float("nan")),
                    "coh_peak": osc_coh.get("peak_coh", float("nan")),
                    "coh_peak_f": osc_coh.get("peak_f", float("nan")),
                    "plv": osc_plv.get("plv", float("nan")),
                    "plv_pref_lag_s": osc_plv.get("preferred_lag_s", float("nan")),
                    "plv_dom_f": osc_plv.get("dom_freq", float("nan")),
                    "wpli": osc_wpli.get("wpli", float("nan")),
                    "osc_signal": osc_plv.get("signal", "unavailable"),
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
