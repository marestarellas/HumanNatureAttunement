#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# Defaults / Config
# -----------------------
SAVEPATH = "./data/processed"
FS_ANNOT_DEFAULT = 256.0               # Hz
ANNOT_FILE = "merged_annotated_cut.csv"
AUDIO_CANONICAL = "sea_envelopes_curves.csv"
OUT_NAME = "merged_annotated_with_audio.csv"
STRICT = True

# -----------------------
# Helpers
# -----------------------
def get_annot_csv(sub_dir: str):
    p = os.path.join(sub_dir, "tables", ANNOT_FILE)
    return p if os.path.exists(p) else None

def get_audio_csv(sub_dir: str):
    audio_dir = os.path.join(sub_dir, "audio")
    p0 = os.path.join(audio_dir, AUDIO_CANONICAL)
    if os.path.exists(p0):
        return p0
    pats = ["*envelope*.csv", "*envelop*.csv", "*curve*.csv", "*curves*.csv"]
    cands = []
    for pat in pats:
        cands.extend(glob.glob(os.path.join(audio_dir, pat)))
    return sorted(cands)[0] if cands else None

def _load_annot(annot_path, fs=FS_ANNOT_DEFAULT):
    # reduce dtype warnings; treat condition_names as string
    try:
        df = pd.read_csv(annot_path, dtype={"condition_names":"string"}, low_memory=False)
    except Exception:
        df = pd.read_csv(annot_path, low_memory=False)
    if "time_s" not in df.columns:
        n = len(df)
        df["time_s"] = np.arange(n, dtype=float) / float(fs)
    return df.sort_values("time_s").drop_duplicates(subset="time_s")

def _load_audio(audio_path):
    df = pd.read_csv(audio_path)
    if "time_s" not in df.columns:
        raise ValueError(f"[ERROR] '{audio_path}' has no 'time_s' column.")
    return df.sort_values("time_s").drop_duplicates(subset="time_s")

def append_audio_to_annotated(annot_path, audio_path, out_path, fs=FS_ANNOT_DEFAULT, tolerance=None):
    df_annot = _load_annot(annot_path, fs=fs)
    df_audio = _load_audio(audio_path)

    # Trim audio to length of annotated
    t_max = df_annot["time_s"].iloc[-1]
    df_audio = df_audio[df_audio["time_s"] <= t_max].copy()

    # Merge by nearest time
    if tolerance is None:
        tolerance = 0.5 / float(fs)           # half a sample @ FS
    df_out = pd.merge_asof(
        df_annot,
        df_audio,
        on="time_s",
        direction="nearest",
        tolerance=tolerance
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return df_out.shape

def process_subject(subj: str, savepath: str, overwrite: bool=False,
                    fs_annot: float=FS_ANNOT_DEFAULT, tol_sec: float|None=None,
                    quiet: bool=False):
    sub_dir = os.path.join(savepath, f"sub-{int(subj):02d}")
    annot_csv = get_annot_csv(sub_dir)
    audio_csv = get_audio_csv(sub_dir)

    if not annot_csv:
        msg = f"[{subj}] MISSING: {ANNOT_FILE} not found. Run the alignment script first."
        if STRICT: raise FileNotFoundError(msg)
        if not quiet: print("SKIP:", msg)
        return

    if not audio_csv:
        msg = f"[{subj}] MISSING: audio envelope CSV not found in {os.path.join(sub_dir,'audio')}."
        if STRICT: raise FileNotFoundError(msg)
        if not quiet: print("SKIP:", msg)
        return

    out_csv = os.path.join(sub_dir, "tables", OUT_NAME)
    if os.path.exists(out_csv) and not overwrite:
        if not quiet:
            print(f"[{subj}] EXISTS → {out_csv} (use --overwrite to regenerate)")
        return

    rows, cols = append_audio_to_annotated(
        annot_csv, audio_csv, out_csv, fs=fs_annot, tolerance=tol_sec
    )
    if not quiet:
        print(f"[{subj}] OK → {out_csv}  (rows, cols)=({rows}, {cols})")

# -----------------------
# CLI / Run
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Merge audio envelopes into annotated tables for subjects.")
    p.add_argument("-s","--subjects", nargs="*", default=None,
                   help="Subjects like 02 03 05. Default: process all sub-* found under data/processed.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing merged_annotated_with_audio.csv")
    p.add_argument("--quiet", action="store_true", help="Only print when files are created or on errors")
    p.add_argument("--fs-annot", type=float, default=FS_ANNOT_DEFAULT, help="Sampling rate (Hz) for annotated table if time_s is missing")
    p.add_argument("--tolerance-sec", type=float, default=None, help="Merge tolerance in seconds (default: 0.5/fs-annot)")
    p.add_argument("--savepath", type=str, default=SAVEPATH, help="Processed root (default: ./data/processed)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        # auto-discover subject folders
        root = Path(args.savepath)
        subjects = [p.name.split("-")[-1] for p in sorted(root.glob("sub-*"))]

    for s in subjects:
        process_subject(s, args.savepath, overwrite=args.overwrite,
                        fs_annot=args.fs_annot, tol_sec=args.tolerance_sec,
                        quiet=args.quiet)
    if not args.quiet:
        print("All done.")
