#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: build sea_envelopes_curves.csv for each subject.

Scans: data/processed/sub-XX/audio/*_cut.wav
Writes: data/processed/sub-XX/audio/sea_envelopes_curves.csv
        data/processed/sub-XX/audio/sea_envelopes_curves.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from math import gcd
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly
import matplotlib.pyplot as plt

# ---------- Repo-relative paths ----------
ROOT = Path(__file__).resolve().parents[1]        # .../HumanNatureAttunement
PROCESSED = ROOT / "data" / "processed"

# ---------- Defaults (chosen to match annotated tables @ 256 Hz) ----------
TARGET_SR_AUDIO = 22050   # resample raw audio before envelope
ENV_DS_FS       = 256     # envelope sample rate (matches annotated)
HP_CUT          = 20.0    # high-pass on raw audio (Hz); set None to disable
LPF_BROAD       = 10.0    # first low-pass on envelope before DS (Hz)
SWELL_1         = 0.3     # Hz
SWELL_2         = 0.1     # Hz
MID_BAND        = (1.0, 5.0)  # Hz on envelope ("splash" band)

# ---------- DSP helpers ----------
def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x

def _resample_poly(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out)//g, int(fs_in)//g
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

def _compress_log(x, eps=1e-8):
    y = np.log10(eps + np.asarray(x))
    return (y - y.min()) / (y.max() - y.min() + 1e-12)

# ---------- Core ----------
def compute_envelopes(wav_path: Path,
                      target_sr_audio=TARGET_SR_AUDIO,
                      env_ds_fs=ENV_DS_FS,
                      hp_cut=HP_CUT,
                      lpf_broad=LPF_BROAD,
                      swell_cut_1=SWELL_1,
                      swell_cut_2=SWELL_2,
                      mid_band=MID_BAND):
    x, fs = sf.read(str(wav_path), always_2d=False)
    x = _to_mono(np.asarray(x, dtype=float))

    if hp_cut is not None and hp_cut > 0:
        x = _highpass(x, fs=fs, cut=hp_cut, order=2)

    # resample raw audio to a friendly rate
    if int(fs) != int(target_sr_audio):
        x = _resample_poly(x, fs_in=fs, fs_out=target_sr_audio)
        fs = target_sr_audio

    # envelope on audio
    env = _envelope_hilbert(x)
    env_broad = _lowpass(env, fs=fs, cut=lpf_broad, order=4)

    # downsample envelope to env_ds_fs (matches annotated @ 256 Hz)
    if int(fs) != int(env_ds_fs):
        env_ds = _resample_poly(env_broad, fs_in=fs, fs_out=env_ds_fs)
        fs_ds = env_ds_fs
    else:
        env_ds, fs_ds = env_broad, fs

    # derive bands on the (downsampled) envelope
    env_swell1 = _lowpass(env_ds, fs=fs_ds, cut=swell_cut_1, order=4)
    env_swell2 = _lowpass(env_ds, fs=fs_ds, cut=swell_cut_2, order=4)
    env_splash = _bandpass(env_ds, fs=fs_ds, lo=mid_band[0], hi=mid_band[1], order=4)

    # compress/normalize for CSV & plotting
    env_broad_n = _compress_log(env_ds)
    swell1_n    = _compress_log(env_swell1)
    swell2_n    = _compress_log(env_swell2)
    splash_n    = _compress_log(np.abs(env_splash) + 1e-8)

    t = np.arange(len(env_broad_n), dtype=float) / float(fs_ds)

    df = pd.DataFrame({
        "time_s": t,
        "env_broad":        env_broad_n,
        "env_swell_0p3":    swell1_n,
        "env_swell_0p1":    swell2_n,
        "env_splash_1_5":   splash_n,
    })
    return df, fs_ds

def save_plot(df: pd.DataFrame, png_path: Path):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(df["time_s"], df["env_broad"],      label="Broad env (~10 Hz LP)")
    ax.plot(df["time_s"], df["env_swell_0p3"],  label="Swell @ 0.3 Hz")
    ax.plot(df["time_s"], df["env_swell_0p1"],  label="Swell @ 0.1 Hz")
    ax.plot(df["time_s"], df["env_splash_1_5"], label="Splash 1–5 Hz", alpha=0.6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized amplitude (log-compressed)")
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
    p = argparse.ArgumentParser(description="Build envelope CSVs for all subjects.")
    p.add_argument("-s","--subjects", nargs="*", default=None,
                   help="Subjects like 02 03 05. Default: auto-detect sub-*/audio/*_cut.wav")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.subjects:
        subs = [f"sub-{int(s):02d}" for s in args.subjects]
        sub_dirs = [PROCESSED / s for s in subs]
    else:
        sub_dirs = sorted([p for p in PROCESSED.glob("sub-*") if (p / "audio").exists()])

    for sd in sub_dirs:
        process_subject_dir(sd, overwrite=args.overwrite)

    print("All done.")
