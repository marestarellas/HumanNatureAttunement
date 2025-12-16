#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: build sea_envelopes_curves.csv for each subject.

Scans: data/processed/sub-XX/audio/*_cut.wav
Writes: data/processed/sub-XX/audio/sea_envelopes_curves.csv
        data/processed/sub-XX/audio/sea_envelopes_curves.png

Implements the updated notebook logic:
- Process envelope & sub-Hz filters at a low, stable rate (ENV_DS_FS, default 50 Hz)
- Then resample finished curves to OUTPUT_FS (default 256 Hz) for CSV/merge
- Safe log-normalization to avoid tiny-negative issues
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from math import gcd
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly

# ---------- Repo-relative paths ----------
ROOT = Path(__file__).resolve().parents[1]        # .../HumanNatureAttunement
PROCESSED = ROOT / "data" / "processed"

# ---------- Defaults (updated to match the new notebook) ----------
TARGET_SR_AUDIO = 22050   # resample raw audio before envelope
ENV_DS_FS       = 50      # do sub-Hz filtering at this low rate (stable)
OUTPUT_FS       = 256     # resample finished curves to this rate for CSV/merge
HP_CUT          = 20.0    # high-pass on raw audio (Hz); set None to disable
LPF_BROAD       = 10.0    # first low-pass on envelope before DS (Hz)
SWELL_1         = 0.2     # Hz (updated from 0.3 in the old version)
SWELL_2         = 0.1     # Hz
MID_BAND        = (1.0, 5.0)  # Hz on envelope ("splash" band)

# ---------- DSP helpers ----------
def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x

def _resample_1d(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    if int(fs_from) == int(fs_to):
        return x.astype(np.float32, copy=False)
    g = gcd(int(fs_from), int(fs_to))
    up, down = int(fs_to)//g, int(fs_from)//g
    return resample_poly(x, up, down).astype(np.float32)

def _butter_sos(cut, fs, btype, order=4):
    nyq = fs * 0.5
    Wn = np.atleast_1d(cut) / nyq
    return butter(order, Wn, btype=btype, output="sos")

def _highpass(x, fs, cut=20.0, order=2):
    sos = _butter_sos(cut, fs, 'highpass', order=order)
    return sosfiltfilt(sos, x)

def _lowpass(x, fs, cut=5.0, order=4):
    sos = _butter_sos(cut, fs, 'lowpass', order=order)
    return sosfiltfilt(sos, x)

def _bandpass(x, fs, lo, hi, order=4):
    sos = _butter_sos([lo, hi], fs, 'bandpass', order=order)
    return sosfiltfilt(sos, x)

def _envelope_hilbert(x):
    return np.abs(hilbert(x))

def _safe_log_norm(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    y = np.log10(1e-8 + x)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return y.astype(np.float32)

# ---------- Core ----------
def compute_envelopes(wav_path: Path,
                      target_sr_audio=TARGET_SR_AUDIO,
                      env_ds_fs=ENV_DS_FS,
                      output_fs=OUTPUT_FS,
                      hp_cut=HP_CUT,
                      lpf_broad=LPF_BROAD,
                      swell_cut_1=SWELL_1,
                      swell_cut_2=SWELL_2,
                      mid_band=MID_BAND):
    # Load mono as float64 for very low-cutoff stability
    x, fs = sf.read(str(wav_path), always_2d=False)
    x = _to_mono(np.asarray(x, dtype=np.float64))

    # Clean-up on raw audio
    if hp_cut is not None and hp_cut > 0:
        x = _highpass(x, fs=fs, cut=hp_cut, order=2)

    # Resample raw audio to friendly rate
    if int(fs) != int(target_sr_audio):
        x = _resample_1d(x, fs_from=fs, fs_to=target_sr_audio)
        fs = target_sr_audio

    # Envelope at audio rate, then a broadband LPF
    env = _envelope_hilbert(x)
    env_broad = _lowpass(env, fs=fs, cut=lpf_broad, order=4)

    # Downsample envelope to low rate for robust sub-Hz filtering
    env_ds = _resample_1d(env_broad, fs_from=fs, fs_to=env_ds_fs)
    fs_ds = env_ds_fs

    # Swell + mid bands at low rate (stable)
    env_swell1 = _lowpass(env_ds, fs=fs_ds, cut=swell_cut_1, order=4)
    env_swell2 = _lowpass(env_ds, fs=fs_ds, cut=swell_cut_2, order=4)
    env_splash = _bandpass(env_ds, fs=fs_ds, lo=mid_band[0], hi=mid_band[1], order=4)

    # Normalize for CSV/plot
    env_broad_n = _safe_log_norm(env_ds)
    swell1_n    = _safe_log_norm(env_swell1)
    swell2_n    = _safe_log_norm(env_swell2)
    splash_n    = _safe_log_norm(np.abs(env_splash))

    # Resample finished curves to OUTPUT_FS for saving/merge
    env_fs = fs_ds
    if int(output_fs) != int(fs_ds):
        env_broad_n = _resample_1d(env_broad_n, fs_from=fs_ds, fs_to=output_fs)
        swell1_n    = _resample_1d(swell1_n,    fs_from=fs_ds, fs_to=output_fs)
        swell2_n    = _resample_1d(swell2_n,    fs_from=fs_ds, fs_to=output_fs)
        splash_n    = _resample_1d(splash_n,    fs_from=fs_ds, fs_to=output_fs)
        env_fs = int(output_fs)

    # Timebase at save rate
    t = np.arange(len(env_broad_n), dtype=np.float64) / float(env_fs)

    # Build DataFrame (column names kept stable for downstream merge)
    df = pd.DataFrame({
        "time_s": t.astype(np.float32, copy=False),
        "env_broad":        env_broad_n,
        "env_swell_0p3":    swell1_n,
        "env_swell_0p1":    swell2_n,
        "env_splash_1_5":   splash_n,
    })
    return df, env_fs

def save_plot(df: pd.DataFrame, png_path: Path):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(df["time_s"], df["env_broad"],      label="Broad env (~10 Hz LP)")
    ax.plot(df["time_s"], df["env_swell_0p3"],  label="Swell @ 0.2–0.3 Hz")
    ax.plot(df["time_s"], df["env_swell_0p1"],  label="Swell @ 0.1 Hz")
    ax.plot(df["time_s"], df["env_splash_1_5"], label="Splash 1–5 Hz", alpha=0.6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized amplitude (log)")
    ax.set_title("Wave Sound Envelope Curves"); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

def newest_cut_wav(audio_dir: Path) -> Path | None:
    candidates = list(audio_dir.glob("*_cut.wav"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def process_subject_dir(sub_dir: Path, overwrite: bool = False):
    audio_dir = sub_dir / "audio"
    if not audio_dir.exists():
        print(f"[{sub_dir.name}] SKIP: no audio/ dir")
        return

    wav_path = newest_cut_wav(audio_dir)
    if wav_path is None:
        print(f"[{sub_dir.name}] SKIP: no *_cut.wav found")
        return

    out_csv = audio_dir / "sea_envelopes_curves.csv"
    out_png = audio_dir / "sea_envelopes_curves.png"
    if out_csv.exists() and not overwrite:
        print(f"[{sub_dir.name}] EXISTS: {out_csv.name} (use --overwrite to regenerate)")
        return

    df, fs_env = compute_envelopes(wav_path)
    df.to_csv(out_csv, index=False)
    save_plot(df, out_png)
    print(f"[{sub_dir.name}] OK → {out_csv.name} (rows={len(df)}, fs_env={fs_env} Hz)")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Build envelope CSVs for all subjects (updated).")
    p.add_argument("-s","--subjects", nargs="*", default=None,
                   help="Subjects like 02 03 05. Default: auto-detect sub-*/audio/*_cut.wav")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs")
    # optional expert knobs (default values match the notebook update)
    p.add_argument("--env-ds-fs", type=int, default=ENV_DS_FS, help="Processing rate for sub-Hz filters")
    p.add_argument("--output-fs", type=int, default=OUTPUT_FS, help="CSV/merge save rate")
    p.add_argument("--swell1", type=float, default=SWELL_1, help="First swell LP cutoff (Hz)")
    p.add_argument("--swell2", type=float, default=SWELL_2, help="Second swell LP cutoff (Hz)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # allow CLI overrides without touching the function signature everywhere
    def compute_envelopes_with_args(wav_path: Path):
        return compute_envelopes(
            wav_path,
            env_ds_fs=args.env_ds_fs,
            output_fs=args.output_fs,
            swell_cut_1=args.swell1,
            swell_cut_2=args.swell2
        )

    # find subject dirs
    if args.subjects:
        subs = [f"sub-{int(s):02d}" for s in args.subjects]
        sub_dirs = [PROCESSED / s for s in subs]
    else:
        sub_dirs = sorted([p for p in PROCESSED.glob("sub-*") if (p / "audio").exists()])

    # run
    for sd in sub_dirs:
        # tiny wrapper to pass CLI parameterization
        def process_with_args(sub_dir: Path, overwrite: bool=False):
            audio_dir = sub_dir / "audio"
            if not audio_dir.exists():
                print(f"[{sub_dir.name}] SKIP: no audio/ dir")
                return
            wav_path = newest_cut_wav(audio_dir)
            if wav_path is None:
                print(f"[{sub_dir.name}] SKIP: no *_cut.wav found")
                return
            out_csv = audio_dir / "sea_envelopes_curves.csv"
            out_png = audio_dir / "sea_envelopes_curves.png"
            if out_csv.exists() and not overwrite:
                print(f"[{sub_dir.name}] EXISTS: {out_csv.name} (use --overwrite to regenerate)")
                return
            df, fs_env = compute_envelopes_with_args(wav_path)
            df.to_csv(out_csv, index=False)
            save_plot(df, out_png)
            print(f"[{sub_dir.name}] OK → {out_csv.name} (rows={len(df)}, fs_env={fs_env} Hz)")

        process_with_args(sd, overwrite=args.overwrite)

    print("All done.")
