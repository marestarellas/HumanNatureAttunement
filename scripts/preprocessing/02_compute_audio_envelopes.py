#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: build sea_envelopes_curves.csv for each subject.

Scans: data/processed/sub-XX/audio/*_cut.wav
Writes: data/processed/sub-XX/audio/sea_envelopes_curves.csv
        data/processed/sub-XX/audio/sea_envelopes_curves.png

Decomposes the audio amplitude envelope into a *band-organized* set of
columns so that downstream coupling can test each biological scale
without re-filtering on the fly:

  - env_broad        : Hilbert envelope, low-pass 60 Hz (broadband reference)
  - env_swell_0p1    : 0.1 Hz LP                      (very-slow swell)
  - env_swell_0p2    : 0.2 Hz LP                      (slow swell)
  - env_hrv_lf       : 0.04 - 0.15 Hz BP              (HRV LF range)
  - env_hrv_hf       : 0.15 - 0.40 Hz BP              (HRV HF / breathing range)
  - env_splash_1_5   : 1 - 5 Hz BP                    (legacy splash)
  - env_delta        : 0.5 - 4 Hz BP                  (EEG delta)
  - env_theta        : 4 - 8 Hz BP                    (EEG theta)
  - env_alpha        : 8 - 13 Hz BP                   (EEG alpha)
  - env_beta_low     : 13 - 20 Hz BP                  (EEG low-beta)
  - env_beta_high    : 20 - 30 Hz BP                  (EEG high-beta)
  - env_gamma1       : 30 - 50 Hz BP                  (EEG gamma1)

Implementation notes:
- Two processing rates are used for numerical stability of the filters:
    ENV_FS_SLOW (50 Hz)  for sub-Hz / mid-band filters (HRV, swell, delta..alpha)
    ENV_FS_FAST (200 Hz) for fast bands (beta_low, beta_high, gamma1)
- All output columns are resampled to OUTPUT_FS (256 Hz) so they share a
  common time axis with the EEG/physio merged table.
- Safe log-normalization is applied to each column for visualization &
  scale invariance; phase-based metrics are unaffected by this monotonic transform.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

# ---------- Repo-relative paths ----------
ROOT = Path(__file__).resolve().parents[2]        # .../HumanNatureAttunement
sys.path.insert(0, str(ROOT / "src"))
PROCESSED = ROOT / "data" / "processed"

from HNA.dsp import (
    bandpass as _bandpass,
    lowpass as _lowpass,
    highpass as _highpass,
    hilbert_envelope as _envelope_hilbert,
    resample_to as _resample_1d,
)

# ---------- Defaults ----------
TARGET_SR_AUDIO = 22050   # resample raw audio before envelope
ENV_FS_SLOW     = 50      # processing rate for sub-Hz / mid-band filters (stable)
ENV_FS_FAST     = 200     # processing rate for beta / gamma envelope filters
OUTPUT_FS       = 256     # resample finished curves to this rate for CSV/merge
HP_CUT          = 20.0    # high-pass on raw audio (Hz); set None to disable
LPF_BROAD       = 60.0    # broadband LP on envelope before DS (Hz). Sets the upper
                          # frequency limit for any band-pass filter applied below.

# Band specs: (column_name, kind, params, processing_fs)
#   kind = "lowpass" -> params = {"cut": Hz}
#   kind = "bandpass" -> params = {"lo": Hz, "hi": Hz}
ENV_BANDS = [
    ("env_broad",      "lowpass",  {"cut": LPF_BROAD},        ENV_FS_FAST),
    ("env_swell_0p2",  "lowpass",  {"cut": 0.2},              ENV_FS_SLOW),
    ("env_swell_0p1",  "lowpass",  {"cut": 0.1},              ENV_FS_SLOW),
    ("env_hrv_lf",     "bandpass", {"lo": 0.04, "hi": 0.15},  ENV_FS_SLOW),
    ("env_hrv_hf",     "bandpass", {"lo": 0.15, "hi": 0.40},  ENV_FS_SLOW),
    ("env_splash_1_5", "bandpass", {"lo": 1.0,  "hi": 5.0},   ENV_FS_SLOW),
    ("env_delta",      "bandpass", {"lo": 0.5,  "hi": 4.0},   ENV_FS_SLOW),
    ("env_theta",      "bandpass", {"lo": 4.0,  "hi": 8.0},   ENV_FS_SLOW),
    ("env_alpha",      "bandpass", {"lo": 8.0,  "hi": 13.0},  ENV_FS_SLOW),
    ("env_beta_low",   "bandpass", {"lo": 13.0, "hi": 20.0},  ENV_FS_FAST),
    ("env_beta_high",  "bandpass", {"lo": 20.0, "hi": 30.0},  ENV_FS_FAST),
    ("env_gamma1",     "bandpass", {"lo": 30.0, "hi": 50.0},  ENV_FS_FAST),
]

# ---------- DSP helpers ----------
def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _safe_log_norm(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    y = np.log10(1e-8 + x)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return y.astype(np.float32)

# ---------- Core ----------
def compute_envelopes(wav_path: Path,
                      target_sr_audio=TARGET_SR_AUDIO,
                      output_fs=OUTPUT_FS,
                      hp_cut=HP_CUT,
                      lpf_broad=LPF_BROAD,
                      env_bands=ENV_BANDS):
    """Compute a band-organized envelope decomposition of an audio file.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``time_s`` plus one column per entry in ``env_bands``
        (each log-normalized for scale invariance). All columns share the
        same time axis at ``output_fs`` Hz.
    env_fs : int
        Output sample rate (== ``output_fs``).
    """
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

    # Envelope at audio rate, then broadband LPF (anti-aliasing for downstream DS).
    env = _envelope_hilbert(x)
    env_broad_audio = _lowpass(env, fs=fs, cut=lpf_broad, order=4)

    # Pre-compute downsampled-envelope buffers at each processing rate.
    proc_fs_set = sorted(set(spec[3] for spec in env_bands))
    env_at_fs = {pfs: _resample_1d(env_broad_audio, fs_from=fs, fs_to=pfs)
                 for pfs in proc_fs_set}

    # Apply each band filter on the appropriate buffer.
    out_cols = {}
    for name, kind, params, proc_fs in env_bands:
        src = env_at_fs[proc_fs]
        if kind == "lowpass":
            sig = _lowpass(src, fs=proc_fs, cut=params["cut"], order=4)
        elif kind == "bandpass":
            sig = _bandpass(src, fs=proc_fs, lo=params["lo"], hi=params["hi"], order=4)
        else:
            raise ValueError(f"Unknown band kind {kind!r} for {name}")
        # Bandpass output oscillates around 0 -> take abs before log-norm.
        sig_n = _safe_log_norm(np.abs(sig))
        if int(proc_fs) != int(output_fs):
            sig_n = _resample_1d(sig_n, fs_from=proc_fs, fs_to=output_fs)
        out_cols[name] = sig_n

    # Align lengths (resampling can yield off-by-one differences across rates).
    n = min(len(v) for v in out_cols.values())
    out_cols = {k: v[:n].astype(np.float32, copy=False) for k, v in out_cols.items()}
    t = (np.arange(n, dtype=np.float64) / float(output_fs)).astype(np.float32, copy=False)

    df = pd.DataFrame({"time_s": t, **out_cols})
    return df, int(output_fs)

def save_plot(df: pd.DataFrame, png_path: Path):
    """QC plot: 3 stacked panels grouping the band-organized envelope set."""
    panels = [
        ("Broad and swell",
         [("env_broad",     "Broad (60 Hz LP)",  None),
          ("env_swell_0p2", "Swell (0.2 Hz LP)", None),
          ("env_swell_0p1", "Swell (0.1 Hz LP)", None)]),
        ("HRV and respiration bands",
         [("env_hrv_lf",   "HRV LF (0.04-0.15 Hz)", None),
          ("env_hrv_hf",   "HRV HF (0.15-0.40 Hz)", None),
          ("env_splash_1_5", "Splash (1-5 Hz)", 0.6)]),
        ("EEG bands",
         [("env_delta",     "Delta (0.5-4 Hz)",   None),
          ("env_theta",     "Theta (4-8 Hz)",     None),
          ("env_alpha",     "Alpha (8-13 Hz)",    None),
          ("env_beta_low",  "Low beta (13-20 Hz)", None),
          ("env_beta_high", "High beta (20-30 Hz)", None),
          ("env_gamma1",    "Gamma1 (30-50 Hz)",  None)]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 9), sharex=True)
    for ax, (title, traces) in zip(axes, panels):
        for col, label, alpha in traces:
            if col not in df.columns:
                continue
            kw = {} if alpha is None else {"alpha": alpha}
            ax.plot(df["time_s"], df[col], lw=1.0, label=label, **kw)
        ax.set_ylabel("Normalized\namplitude (log)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Wave Sound Envelope Curves (band decomposition)",
                 fontsize=12, fontweight="bold", y=1.0)
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
    print(f"[{sub_dir.name}] OK -> {out_csv.name} (rows={len(df)}, fs_env={fs_env} Hz)")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Build band-organized envelope CSVs for all subjects.")
    p.add_argument("-s","--subjects", nargs="*", default=None,
                   help="Subjects like 02 03 05. Default: auto-detect sub-*/audio/*_cut.wav")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs")
    p.add_argument("--output-fs", type=int, default=OUTPUT_FS, help="CSV/merge save rate (Hz)")
    p.add_argument("--lpf-broad", type=float, default=LPF_BROAD,
                   help=f"Broadband LP cutoff (Hz) on the audio-rate envelope. Default: {LPF_BROAD}")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED,
                   help="Processed data root containing sub-*/audio/*_cut.wav (default: <repo>/data/processed)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    def compute_envelopes_with_args(wav_path: Path):
        return compute_envelopes(
            wav_path,
            output_fs=args.output_fs,
            lpf_broad=args.lpf_broad,
        )

    # find subject dirs
    processed_root = Path(args.processed_dir)
    if args.subjects:
        subs = [f"sub-{int(s):02d}" for s in args.subjects]
        sub_dirs = [processed_root / s for s in subs]
    else:
        sub_dirs = sorted([p for p in processed_root.glob("sub-*") if (p / "audio").exists()])

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
            print(f"[{sub_dir.name}] OK -> {out_csv.name} (rows={len(df)}, fs_env={fs_env} Hz)")

        process_with_args(sd, overwrite=args.overwrite)

    print("All done.")
