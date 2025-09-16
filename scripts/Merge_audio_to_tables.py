#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd

# -----------------------
# Config
# -----------------------
SAVEPATH = "./data/processed"
SUBJECTS = ["02", "03", "05", "06", "07"]  # extend as needed
FS_ANNOT = 256.0                            # Hz (used if annotated CSV has no time_s)
ANNOT_FILE = "merged_annotated_cut.csv"     # <- require the cut version
AUDIO_CANONICAL = "sea_envelopes_curves.csv"
OUT_NAME = "merged_annotated_with_audio.csv"
STRICT = True                               # raise if something is missing

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
    # try common patterns
    pats = ["*envelope*.csv", "*envelop*.csv", "*curve*.csv", "*curves*.csv"]
    cands = []
    for pat in pats:
        cands.extend(glob.glob(os.path.join(audio_dir, pat)))
    return sorted(cands)[0] if cands else None

def append_audio_to_annotated(annot_path, audio_path, out_path, fs=FS_ANNOT, tolerance=None):
    # --- Load annotated ---
    df_annot = pd.read_csv(annot_path)
    if "time_s" not in df_annot.columns:
        n = len(df_annot)
        df_annot["time_s"] = np.arange(n, dtype=float) / float(fs)
    df_annot = df_annot.sort_values("time_s").drop_duplicates(subset="time_s")

    # --- Load audio envelope ---
    df_audio = pd.read_csv(audio_path)
    if "time_s" not in df_audio.columns:
        raise ValueError(f"[ERROR] '{audio_path}' has no 'time_s' column.")
    df_audio = df_audio.sort_values("time_s").drop_duplicates(subset="time_s")

    # --- Trim audio to length of annotated ---
    t_max = df_annot["time_s"].iloc[-1]
    df_audio = df_audio[df_audio["time_s"] <= t_max].copy()

    # --- Merge by nearest time ---
    if tolerance is None:
        tolerance = 0.5 / float(fs)  # half an EEG sample
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

def process_subject(subj: str):
    sub_dir = os.path.join(SAVEPATH, f"sub-{int(subj):02d}")
    annot_csv = get_annot_csv(sub_dir)
    audio_csv = get_audio_csv(sub_dir)

    if not annot_csv:
        msg = f"[{subj}] MISSING: {ANNOT_FILE} not found. Run the alignment script first."
        if STRICT: raise FileNotFoundError(msg)
        print("SKIP:", msg); return

    if not audio_csv:
        msg = f"[{subj}] MISSING: audio envelope CSV not found in {os.path.join(sub_dir,'audio')}."
        if STRICT: raise FileNotFoundError(msg)
        print("SKIP:", msg); return

    out_csv = os.path.join(sub_dir, "tables", OUT_NAME)
    rows, cols = append_audio_to_annotated(annot_csv, audio_csv, out_csv, fs=FS_ANNOT)
    print(f"[{subj}] OK → {out_csv}  (rows, cols)=({rows}, {cols})")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    for s in SUBJECTS:
        process_subject(s)
    print("All done.")
