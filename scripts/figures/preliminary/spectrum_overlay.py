#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis A - Spectrum overlay: audio swell vs. respiration vs. instantaneous HR.

For each condition, computes the Welch PSD of:
  - the audio swell envelope (env_swell_0p2; 0.2 Hz lowpass)
  - cleaned respiration
  - the instantaneous heart-rate signal (interpolated to a regular grid)

PSDs are averaged across subjects and overlaid on a single log-frequency axis.
This shows what slow rhythms exist in the audio and where physiology has
natural peaks - a prerequisite for any entrainment claim.

Usage:
    python scripts/figures/analysis_spectrum_overlay.py \\
        --subjects 2 3 4 5 6 \\
        --data-dir /path/to/data
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sps
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from HNA.dsp import bandpass, lowpass, interpolate_nan
from HNA.utils import extract_condition_data
from HNA.viz import (
    use_paper_style, MODALITY_COLORS, save_figure, _figsize,
)

FS_EEG = 256.0           # Hz, also EEG/physio merged-table rate
FS_HR_TARGET = 4.0       # Hz, for instantaneous HR resampling
PSD_FMIN = 0.02          # Hz; below this Welch is unreliable for our window lengths
PSD_FMAX = 1.5           # Hz; covers swell, respiration, and slow HR modulation
WELCH_SEG_SEC = 60       # Welch segment length in seconds


# -------------------- signal builders --------------------
def _instantaneous_hr_signal(rpeak_indices: np.ndarray, fs_in: float, target_fs: float, n_samples: int) -> np.ndarray:
    """Build a regularly-sampled instantaneous-HR signal (in BPM) from R-peaks.

    Parameters
    ----------
    rpeak_indices : indices into the original signal at fs_in
    fs_in : sample rate of the index space
    target_fs : output sample rate
    n_samples : output length (in target_fs samples)
    """
    if len(rpeak_indices) < 4:
        return np.full(n_samples, np.nan)
    t_peaks = np.asarray(rpeak_indices, float) / fs_in
    rr = np.diff(t_peaks)
    bpm = 60.0 / rr
    # value at midpoint of each RR interval
    t_mid = t_peaks[:-1] + rr / 2.0
    t_target = np.arange(n_samples) / target_fs
    if not (t_target.min() >= t_mid.min() and t_target.max() <= t_mid.max()):
        # Pad: use first/last bpm beyond available range
        f = interp1d(t_mid, bpm, kind="cubic", fill_value=(bpm[0], bpm[-1]),
                     bounds_error=False)
    else:
        f = interp1d(t_mid, bpm, kind="cubic")
    return f(t_target)


def _condition_signals(df: pd.DataFrame, condition: str, rpeak_indices: np.ndarray):
    """Slice df to a condition; return (audio_env, respiration, hr_signal_at_4Hz).

    Returns None if any required field is missing.
    """
    seg = extract_condition_data(df, condition)
    if seg is None or len(seg) < int(FS_EEG * 30):
        return None
    audio = seg["env_swell_0p2"].to_numpy(dtype=float)
    resp = seg["respiration"].to_numpy(dtype=float)
    if np.all(np.isnan(audio)) or np.all(np.isnan(resp)):
        return None

    # Map R-peaks (indexed against the *whole* merged df at FS_EEG) to this slice.
    seg_start = int(seg.index[0])
    seg_stop = int(seg.index[-1] + 1)
    rpeaks_in = rpeak_indices[(rpeak_indices >= seg_start) & (rpeak_indices < seg_stop)] - seg_start

    # Build HR signal at 4 Hz spanning the slice.
    n_target = int(round(len(seg) / FS_EEG * FS_HR_TARGET))
    hr = _instantaneous_hr_signal(rpeaks_in, FS_EEG, FS_HR_TARGET, n_target)

    audio = interpolate_nan(audio)
    resp = interpolate_nan(resp)
    if np.any(~np.isfinite(hr)):
        # Fall back: replace any residual NaN with mean
        m = np.nanmean(hr)
        hr = np.where(np.isfinite(hr), hr, m)
    return audio, resp, hr


# -------------------- PSD --------------------
def _welch_psd(x: np.ndarray, fs: float):
    nper = int(min(len(x), WELCH_SEG_SEC * fs))
    nover = nper // 2
    f, Pxx = sps.welch(x, fs=fs, nperseg=nper, noverlap=nover, detrend="constant")
    return f, Pxx


def _normalize(p):
    """Normalize a PSD to its area-under-curve in the displayed range."""
    return p / np.trapz(p) if np.trapz(p) > 0 else p


# -------------------- per-subject extraction --------------------
def _per_subject_psd(subj: int, conditions: list[str], data_dir: Path):
    """Compute PSD curves per condition for one subject.

    Returns dict: {condition: {"audio": (f,P), "resp": (f,P), "hr": (f,P)}}.
    """
    sub = f"sub-{subj:02d}"
    merged = data_dir / "processed" / sub / "tables" / "merged_annotated_with_audio.csv"
    if not merged.exists():
        print(f"  SKIP {sub}: missing {merged.name}")
        return None
    df = pd.read_csv(merged, low_memory=False)

    # Combine R-peaks across conditions (mapped back into whole-df index space).
    # Per-condition rpeaks_<COND>.npy is indexed *within* the condition slice;
    # easier to re-detect rpeaks here, but we already saved them, so reconstruct:
    rpeak_dir = data_dir / "processed" / sub / "ecg_processed"
    rpeaks_all = []
    for c in ["RS1", "VIZ", "AUD", "MULTI", "RS2"]:
        rp_path = rpeak_dir / f"rpeaks_{c}.npy"
        if not rp_path.exists():
            continue
        rp_local = np.load(rp_path)
        # find condition start in df
        starts = df.index[df["condition_names"] == f"{c}_start"].tolist()
        if not starts:
            continue
        rpeaks_all.append(rp_local + int(starts[0]))
    if not rpeaks_all:
        print(f"  SKIP {sub}: no rpeaks")
        return None
    rpeaks = np.concatenate(rpeaks_all).astype(int)

    out = {}
    for cond in conditions:
        sigs = _condition_signals(df, cond, rpeaks)
        if sigs is None:
            continue
        audio, resp, hr = sigs
        f_a, p_a = _welch_psd(audio, FS_EEG)
        f_r, p_r = _welch_psd(resp, FS_EEG)
        f_h, p_h = _welch_psd(hr, FS_HR_TARGET)
        out[cond] = {"audio": (f_a, p_a), "resp": (f_r, p_r), "hr": (f_h, p_h)}
    return out


# -------------------- aggregation --------------------
def _interp_to_grid(curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> np.ndarray:
    """Interpolate (f, P) pairs onto a common log-frequency grid in linear P space."""
    out = np.full((len(curves), len(grid)), np.nan)
    for i, (f, p) in enumerate(curves):
        m = (f >= grid[0]) & (f <= grid[-1])
        if not m.any():
            continue
        out[i] = np.interp(grid, f[m], p[m], left=np.nan, right=np.nan)
    return out


def _aggregate(per_subj_psd: dict, conditions: list[str], grid: np.ndarray):
    """Aggregate PSDs across subjects per condition, per modality.

    Returns dict[condition][modality] -> (mean, sem) on `grid`.
    """
    out = {}
    for cond in conditions:
        modalities = {"audio": [], "resp": [], "hr": []}
        for subj, by_cond in per_subj_psd.items():
            if by_cond is None or cond not in by_cond:
                continue
            for k in modalities:
                f, p = by_cond[cond][k]
                modalities[k].append((f, _normalize(p)))
        out[cond] = {}
        for k, curves in modalities.items():
            if not curves:
                continue
            arr = _interp_to_grid(curves, grid)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / max(1, np.sqrt(np.sum(~np.isnan(arr[:, 0]))))
            out[cond][k] = (mean, sem)
    return out


# -------------------- plotting --------------------
def plot_spectrum_overlay(per_subj_psd, conditions: list[str], grid: np.ndarray,
                           output_dir: Path):
    """Single-panel spectrum overlay pooling all conditions and subjects.

    For each modality, collect every (subject, condition) PSD, interpolate to
    the common grid, then take the across-trace mean ± SEM.
    """
    use_paper_style()
    RESP_BAND = (0.15, 0.40)  # canonical breathing band reference

    pooled = {"audio": [], "resp": [], "hr": []}
    for subj, by_cond in per_subj_psd.items():
        if by_cond is None:
            continue
        for cond in conditions:
            if cond not in by_cond:
                continue
            for k in pooled:
                f, p = by_cond[cond][k]
                pooled[k].append((f, _normalize(p)))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axvspan(*RESP_BAND, color="#E6E2F0", alpha=0.55, lw=0, zorder=0,
               label="Resp. rate band\n(0.15–0.40 Hz)")
    n_traces = 0
    for label, key in [("Audio (swell 0.2 Hz)", "audio"),
                       ("Respiration", "resp"),
                       ("HR (instantaneous)", "hr")]:
        curves = pooled.get(key, [])
        if not curves:
            continue
        arr = _interp_to_grid(curves, grid)
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0) / max(1.0, np.sqrt(arr.shape[0]))
        color = MODALITY_COLORS[{"audio": "audio",
                                 "resp": "respiration",
                                 "hr": "hrv"}[key]]
        ax.plot(grid, m, lw=2.0, color=color, label=label)
        ax.fill_between(grid, m - s, m + s, color=color, alpha=0.18, lw=0)
        n_traces = max(n_traces, arr.shape[0])
    ax.set_xscale("log")
    ax.set_xlim(PSD_FMIN, PSD_FMAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    cond_label = "+".join(conditions)
    # Title removed (in caption).
    fig.tight_layout()
    saved = save_figure(fig, output_dir / "spectrum_overlay")
    plt.close(fig)
    print(f"  Saved: {saved[0].name} (+ pdf)")


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "preliminary_results" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    print("Computing per-subject PSDs...")
    per_subj = {}
    for s in args.subjects:
        print(f"  sub-{s:02d}")
        per_subj[s] = _per_subject_psd(s, args.conditions, args.data_dir)
    grid = np.geomspace(PSD_FMIN, PSD_FMAX, 200)
    plot_spectrum_overlay(per_subj, args.conditions, grid, args.figures_dir)
    print(f"Done. Output: {args.figures_dir}")


if __name__ == "__main__":
    main()
