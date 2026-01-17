#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from types import SimpleNamespace

# HNA utils
from HNA.modules.utils import get_condition_segments
from HNA.modules.coupling import (
    windowed_xcorr, band_coherence_windowed, band_coherence,
    plv_phase_sync, windowed_plv,
    wpli_phase_sync, windowed_wpli,
    plot_coupling_over_time, plot_coherence_results,
)

# ---------- repo paths ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# ---------- defaults ----------
FS = 256.0
XC_WIN = 120.0
XC_STEP = 10.0
XC_LAG = 10.0
COH_FMIN, COH_FMAX = 0.05, 0.5
PLV_BW = 0.12
WPLI_WIN, WPLI_STEP = 120.0, 10.0

# ---------- respiration cleaner (as notebook) ----------
def _bandpass_sos(x, fs, lo=0.05, hi=1.0, order=4):
    ny = 0.5*fs
    sos = butter(order, [lo/ny, hi/ny], btype="band", output="sos")
    return sosfiltfilt(sos, x)

def clean_respiration(series: pd.Series, fs=FS) -> np.ndarray:
    x = series.astype(float).to_numpy()
    # 1) interpolate NaNs
    s = pd.Series(x).interpolate(method="linear", limit_direction="both").to_numpy()
    # 2) center
    s = s - np.nanmean(s)
    # 3) scale before filtering
    mx = np.nanmax(np.abs(s)) or 1.0
    s = s / mx
    # 4) stable bandpass
    s = _bandpass_sos(s, fs=fs, lo=0.05, hi=1.0, order=4)
    # 5) z-score
    mu, sd = np.mean(s), np.std(s) + 1e-12
    return (s - mu) / sd

# ---------- segments ----------
def pair_segments(indices_dict: dict) -> dict[str, tuple[int,int]]:
    segs = {}
    for k, v in indices_dict.items():
        if isinstance(k, str) and k.endswith("_start") and v is not None:
            base = k[:-6]
            stop = indices_dict.get(base + "_stop", None)
            if stop is not None:
                segs[base] = (int(v), int(stop))
    return segs

# ---------- plotting wrappers ----------
def save_coupling_plots(subject_dir: Path, cond: str, xc, coh_dict_or_obj, plv_win_dict, env_col: str = ""):
    plots_dir = subject_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # NEW: make sure 'coh' has attributes for plot_coupling_over_time
    coh_for_plot = SimpleNamespace(**coh_dict_or_obj) if isinstance(coh_dict_or_obj, dict) else coh_dict_or_obj

    # 1) time-series summary (xcorr + PLV + coherence time series)
    fig1 = plot_coupling_over_time(xc, coh_for_plot, plv_win_dict)
    env_label = f" ({env_col})" if env_col else ""
    fig1.suptitle(f"{cond} — Respiration ↔ Audio{env_label}", fontsize=14, fontweight='bold')
    fig1.savefig(plots_dir / f"{cond}_coupling_timeseries.png", dpi=160); plt.close(fig1)

    # 2) spectrum + windowed band-avg (this helper already accepts dict or object)
    fig2 = plot_coherence_results(coh_dict_or_obj, band=(COH_FMIN, COH_FMAX),
                                  title=f"Coherence — {cond}{env_label}")
    fig2.savefig(plots_dir / f"{cond}_coherence.png", dpi=160); plt.close(fig2)

# ---------- per-subject processor ----------
def process_subject(subj: str, overwrite=False, envelope_pref=("env_swell_0p3","env_swell_0p3hz","env_broad")):
    sdir = PROCESSED / f"sub-{int(subj):02d}"
    tables = sdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    merged = tables / "merged_annotated_with_audio.csv"
    if not merged.exists():
        raise FileNotFoundError(f"[{subj}] missing {merged}")

    df = pd.read_csv(merged, low_memory=False)
    if "time_s" not in df.columns:
        df["time_s"] = np.arange(len(df)) / FS

    # choose envelope column robustly
    env_col = next((c for c in envelope_pref if c in df.columns), None)
    if env_col is None:
        raise RuntimeError(f"[{subj}] no envelope column found. Tried: {envelope_pref}")

    # clean respiration
    if "respiration_clean" not in df.columns or overwrite:
        df["respiration_clean"] = clean_respiration(df["respiration"], fs=FS)

    # segments per condition
    indices = get_condition_segments(df, df["condition_names"].unique())
    segs = pair_segments(indices)

    # outputs
    summary_rows = []
    results_dir = sdir / "tables"
    results_dir.mkdir(exist_ok=True)

    for cond, (start, stop) in segs.items():
        if cond.upper() == "AUDIO_SYNC":
            continue
        r = df.iloc[start:stop].copy()
        resp = r["respiration_clean"].to_numpy(dtype=float)
        env  = r[env_col].to_numpy(dtype=float)

        # ---- metrics ----
        xc = windowed_xcorr(resp, env, fs=FS, win_sec=XC_WIN, step_sec=XC_STEP, max_lag_sec=XC_LAG)
        coh = band_coherence_windowed(resp, env, fs=FS, fmin=COH_FMIN, fmax=COH_FMAX,
                                      win_sec=XC_WIN, step_sec=XC_STEP)
        # (coh above returns a dict compatible with plot_coherence_results)
        plv = plv_phase_sync(resp, env, fs=FS, bw_hz=PLV_BW)
        plv_win = windowed_plv(resp, env, fs=FS, win_sec=XC_WIN, step_sec=XC_STEP)
        wpli_g  = wpli_phase_sync(resp, env, fs=FS, bw_hz=PLV_BW)
        wpli_w  = windowed_wpli(resp, env, fs=FS, win_sec=WPLI_WIN, step_sec=WPLI_STEP)

        # ---- persist per-condition JSON (compact) ----
        out_json = results_dir / f"coupling_{cond}.json"
        if overwrite or (not out_json.exists()):
            payload = {
                "subject": subj, "condition": cond, "fs": FS,
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
        # ---- add to summary row ----
        summary_rows.append({
            "subject": subj, "condition": cond, "env_col": env_col,
            "xcorr_mean_peak_r": float(np.nanmean(xc.peak_r)),
            "xcorr_mean_peak_lag_s": float(np.nanmean(xc.peak_lag_s)),
            "coh_band_avg": coh["band_avg_coh"],
            "coh_peak": coh["peak_coh"],
            "coh_peak_f": coh["peak_f"],
            "plv": plv.plv,
            "plv_pref_lag_s": plv.preferred_lag_s,
            "plv_dom_f": plv.f0,
            "wpli": wpli_g.wpli,
        })

        # ---- plots ----
        save_coupling_plots(sdir, cond, xc, coh, plv_win, env_col)

    # write per-subject summary CSV
    out_csv = results_dir / "coupling_summary.csv"
    pd.DataFrame(summary_rows).sort_values(["condition"]).to_csv(out_csv, index=False)
    print(f"[{subj}] wrote {out_csv} and per-condition JSON/plots")

# ---------- aggregate across subjects ----------
def aggregate_group(save_to: Path = PROCESSED / "group_coupling_summary.csv"):
    rows = []
    for sub_dir in sorted(PROCESSED.glob("sub-*")):
        csv = sub_dir / "tables" / "coupling_summary.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            rows.append(df)
    if not rows:
        print("No per-subject summaries found.")
        return
    grp = pd.concat(rows, ignore_index=True)
    grp.to_csv(save_to, index=False)
    print(f"[group] wrote {save_to}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Respiration ↔ Audio coupling for subjects.")
    p.add_argument("-s","--subjects", nargs="*", default=None,
                   help="Subjects like 02 03 05. Default: auto-detect sub-* under data/processed/")
    p.add_argument("--overwrite", action="store_true", help="Overwrite per-condition JSON if present")
    p.add_argument("--aggregate", action="store_true", help="After processing, write group summary CSV")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.subjects:
        subs = [f"{int(s):02d}" for s in args.subjects]
    else:
        subs = [p.name.split("-")[-1] for p in sorted(PROCESSED.glob("sub-*"))]
    for s in subs:
        process_subject(s, overwrite=args.overwrite)
    if args.aggregate:
        aggregate_group()
    print("All done.")
