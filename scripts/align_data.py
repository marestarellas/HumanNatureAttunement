import os
import json
import numpy as np
import pandas as pd
import soundfile as sf

from HNA.modules.utils import (
    load_data,
    align_by_first_triggers,
    find_last_high_indices,
    annotate_conditions,
    get_sync_seconds_for_subject,
    load_audio_for_subject,
)

# -----------------------
# Config
# -----------------------
SAVEPATH = "../data/processed"
DATA_DIR = "../data"
# DATA_DIR = r'G:\My Drive\COSMIC FUTURE GROUNDED PRESENT\THE SEA PROJECT\PILOT\Sea_Project_data'
JSON_PATH = "../data/audio_sync.json"

conditions = {
    "02": ["MULTI", "VIZ", "AUD"],
    "03": ["AUD", "VIZ", "MULTI"],
    '04': ["VIZ", "AUD", "MULTI"],
    '05': ["VIZ", "AUD", "MULTI"],
    '06': ["AUD", "VIZ", "MULTI"],
    # '07': ["VIZ", "AUD", "MULTI"],
    # '08': ["MULTI", "AUD", "VIZ"],
}

SUBJECTS = ["02", "03", '04', "05", "06"]  # expand later when ready
# SUBJECTS = ["08"]  # expand later when ready
# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fmt_seconds(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:d}:{s:05.2f}"

def process_subject(subj: str):
    print(f"\n=== Processing subject {subj} ===")

    # ------- I/O folders (processed) -------
    sub_out_dir = os.path.join(SAVEPATH, f"sub-{int(subj):02d}")
    audio_out_dir = os.path.join(sub_out_dir, "audio")
    tables_out_dir = os.path.join(sub_out_dir, "tables")
    ensure_dir(audio_out_dir)
    ensure_dir(tables_out_dir)

    # ------- Load raw data -------
    eeg_data = load_data(subj, "eeg", data_dir=DATA_DIR)
    physio_data = load_data(subj, "physio", data_dir=DATA_DIR)
    if eeg_data is None or physio_data is None:
        print(f"  Missing EEG/physio data for subj {subj}, skipping.")
        return

    # ------- Align EEG/physio on first triggers -------
    physio_aligned, eeg_aligned = align_by_first_triggers(physio_data, eeg_data)

    # ------- Merge and annotate conditions -------
    merged_data = pd.concat([physio_aligned, eeg_aligned], axis=1)
    if subj == '08':
        condition_indices = find_last_high_indices(merged_data, threshold=1950)
    else:
        condition_indices = find_last_high_indices(merged_data, threshold=2000)
    print(f"  Found condition trigger indices: {condition_indices}")
    cond_list = conditions.get(subj)
    if cond_list is None:
        print(f"  No condition ordering for subj {subj}, skipping.")
        return

    # Subject-specific corrections
    if subj == '05':
        # Remove fifth trigger from the end (index -5)
        print(f"  Before removal: {len(condition_indices)} triggers")
        print(f"  Removing trigger at index {condition_indices[-5]}")
        condition_indices = condition_indices[:-5] + condition_indices[-4:]
        print(f"  After removal: {len(condition_indices)} triggers")
        
    if subj == '04':
        # add trigger 3 minutes after the last trigger
        print(f"  Before addition: {len(condition_indices)} triggers")
        last_trigger = condition_indices[-1]
        additional_trigger = last_trigger + 3 * 60 * 256  # 3 minutes later
        condition_indices.append(additional_trigger)
        print(f"  After addition: {len(condition_indices)} triggers")
        
    try:
        merged_annotated = annotate_conditions(merged_data, condition_indices, cond_list)
        u = merged_annotated["condition_names"].unique()
        print("  Unique condition labels:", u)
        audio_sync_count = int((merged_annotated["condition_names"] == "AUDIO_SYNC").sum())
        print("  Number of AUDIO_SYNC events:", audio_sync_count)
    except AssertionError as e:
        print(f"  Annotation failed for subj {subj}: {e}. Skipping.")
        return

    # ------- Save merged_annotated (full) -------
    out_csv_full = os.path.join(tables_out_dir, "merged_annotated.csv")
    merged_annotated.to_csv(out_csv_full, index=False)
    print(f"  Saved: {out_csv_full}")

    # ------- Trim dataframe at last AUDIO_SYNC -------
    mask = (merged_annotated["condition_names"] == "AUDIO_SYNC")
    if mask.any():
        last_sync_idx = np.flatnonzero(mask.to_numpy())[-1]
        merged_annotated_cut = merged_annotated.iloc[last_sync_idx:].reset_index(drop=True)
        out_csv_cut = os.path.join(tables_out_dir, "merged_annotated_cut.csv")
        merged_annotated_cut.to_csv(out_csv_cut, index=False)
        print(f"  Trimmed rows: {len(merged_annotated)} -> {len(merged_annotated_cut)} (cut at index {last_sync_idx})")
        print(f"  Saved: {out_csv_cut}")
    else:
        merged_annotated_cut = merged_annotated
        print("  WARNING: no AUDIO_SYNC found; not trimming dataframe.")

    # ------- Audio: cut from JSON sync time and save under processed -------
    try:
        sync_seconds = get_sync_seconds_for_subject(JSON_PATH, subj)
    except Exception as e:
        print(f"  WARNING: Could not read sync time for subj {subj}: {e}")
        sync_seconds = None

    try:
        audio, sr, in_wav_path = load_audio_for_subject(subj, DATA_DIR)
    except Exception as e:
        print(f"  WARNING: Could not load audio for subj {subj}: {e}")
        audio = None
        sr = None
        in_wav_path = None

    if (sync_seconds is not None) and (audio is not None) and (sr is not None):
        start_sample = int(round(sync_seconds * sr))
        if start_sample >= len(audio):
            print(
                f"  WARNING: Sync time ({sync_seconds:.2f}s) exceeds audio length ({len(audio)/sr:.2f}s). Skipping audio cut."
            )
        else:
            print(f"  Cutting audio from sample {start_sample} onward")
            print("   JSON sync time:", fmt_seconds(sync_seconds), f"({sync_seconds:.2f} s)")
            print("   From-sample time:", fmt_seconds(start_sample / sr), f"({start_sample/sr:.2f} s)")

            audio_cut = audio[start_sample:]
            # name based on input or fallback
            base = os.path.splitext(os.path.basename(in_wav_path or f"sub-{int(subj):02d}.wav"))[0]
            out_wav_path = os.path.join(audio_out_dir, f"{base}_cut.wav")
            sf.write(out_wav_path, audio_cut, sr)
            print(f"  Audio written: {out_wav_path}")
    else:
        print("  Skipping audio cut for this subject due to missing sync time or audio.")

    print(f"=== Done subject {subj} ===")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    for s in SUBJECTS:
        process_subject(s)
    print("\nAll done.")
