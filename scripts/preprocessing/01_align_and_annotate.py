#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 of the Human Nature Attunement preprocessing pipeline.

For each subject:
- Load raw EEG (EEG.csv + D in.csv) and physio (ExG/Analog AUX/D out) via HNA.utils.load_data
- Align EEG <-> physio on first triggers
- Detect the AUDIO_SYNC + condition-boundary pulses on the condition_triggers channel
- Apply per-subject trigger patches from config/subjects.json
- Annotate condition labels (RS1/<C1>/<C2>/<C3>/RS2 starts and stops + AUDIO_SYNC)
- Trim merged dataframe at the LAST AUDIO_SYNC pulse
- Cut the raw audio WAV at the JSON sync time

Outputs (per subject):
  data/processed/sub-XX/tables/merged_annotated.csv
  data/processed/sub-XX/tables/merged_annotated_cut.csv
  data/processed/sub-XX/audio/<base>_cut.wav
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from HNA.utils import (
    align_by_first_triggers,
    annotate_conditions,
    find_last_high_indices,
    get_sync_seconds_for_subject,  # noqa: F401  (kept for reference)
    load_audio_for_subject,
    load_data,
    parse_mmss,
)


# ---------- Repo-relative paths ----------
ROOT = Path(__file__).resolve().parents[2]              # .../HumanNatureAttunement
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"
DEFAULT_CONFIG = ROOT / "config" / "subjects.json"
DEFAULT_SAMPLING_RATE = 256


def _fmt_seconds(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:d}:{s:05.2f}"


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["subjects"]


def _normalize_subj_key(subj: str) -> str:
    """Turn '03', 'sub-03', '3', 'sub3' -> 'sub-03'."""
    s = str(subj).strip().lower().replace("sub-", "").replace("sub", "")
    return f"sub-{int(s):02d}"


def _apply_trigger_patches(condition_indices: list, patches: dict, sampling_rate: int) -> list:
    """Apply manual trigger fixes from config. Returns a new list."""
    out = list(condition_indices)
    if not patches:
        return out

    if "add_after_last_s" in patches:
        delay_s = float(patches["add_after_last_s"])
        if not out:
            print(f"  WARNING: add_after_last_s requested but no triggers detected; skipping.")
        else:
            extra = out[-1] + int(round(delay_s * sampling_rate))
            print(f"  Patch: appending trigger {delay_s:.1f}s after last -> idx {extra}")
            out.append(extra)

    if "remove_indices" in patches:
        for i in patches["remove_indices"]:
            try:
                removed = out.pop(int(i))
                print(f"  Patch: removed trigger at position {i} (was idx {removed})")
            except IndexError:
                print(f"  WARNING: remove_indices position {i} out of range ({len(out)} triggers); skipping.")

    return out


def process_subject(
    subj: str,
    cfg_entry: dict,
    data_dir: Path,
    out_dir: Path,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    overwrite: bool = False,
) -> bool:
    """Process one subject end-to-end. Returns True on success."""
    subj_key = _normalize_subj_key(subj)
    short = subj_key.split("-")[-1]
    print(f"\n=== Processing {subj_key} ===")

    # ---- I/O folders ----
    sub_out_dir = out_dir / subj_key
    audio_out_dir = sub_out_dir / "audio"
    tables_out_dir = sub_out_dir / "tables"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    tables_out_dir.mkdir(parents=True, exist_ok=True)

    out_full = tables_out_dir / "merged_annotated.csv"
    out_cut = tables_out_dir / "merged_annotated_cut.csv"
    if out_full.exists() and out_cut.exists() and not overwrite:
        print(f"  EXISTS: {out_full.name} and {out_cut.name} (use --overwrite to regenerate)")
        return True

    # ---- Load raw ----
    eeg_data = load_data(short, "eeg", data_dir=str(data_dir))
    physio_data = load_data(short, "physio", data_dir=str(data_dir))
    if eeg_data is None or physio_data is None:
        print(f"  Missing EEG/physio data for {subj_key}; skipping.")
        return False

    # ---- Align on first triggers ----
    physio_aligned, eeg_aligned = align_by_first_triggers(physio_data, eeg_data)
    merged_data = pd.concat([physio_aligned, eeg_aligned], axis=1)

    # ---- Detect condition triggers ----
    threshold = int(cfg_entry.get("trigger_threshold", 2000))
    condition_indices = find_last_high_indices(merged_data, threshold=threshold)
    print(f"  Threshold={threshold} -> {len(condition_indices)} candidate triggers")

    # ---- Apply per-subject patches ----
    patches = cfg_entry.get("trigger_patches") or {}
    condition_indices = _apply_trigger_patches(condition_indices, patches, sampling_rate)

    # ---- Annotate ----
    cond_list = cfg_entry.get("condition_order")
    if not cond_list or len(cond_list) != 3:
        print(f"  No valid 3-condition order in config for {subj_key}; skipping.")
        return False
    try:
        merged_annotated = annotate_conditions(
            merged_data, condition_indices, cond_list, sampling_rate=sampling_rate
        )
    except AssertionError as e:
        print(f"  Annotation failed for {subj_key}: {e}. Skipping.")
        return False

    print("  Unique condition labels:", merged_annotated["condition_names"].unique())
    audio_sync_count = int((merged_annotated["condition_names"] == "AUDIO_SYNC").sum())
    print(f"  Number of AUDIO_SYNC events: {audio_sync_count}")

    # ---- Save merged_annotated ----
    merged_annotated.to_csv(out_full, index=False)
    print(f"  Saved: {out_full}")

    # ---- Trim at last AUDIO_SYNC ----
    mask = merged_annotated["condition_names"] == "AUDIO_SYNC"
    if mask.any():
        last_sync_idx = int(np.flatnonzero(mask.to_numpy())[-1])
        merged_annotated_cut = merged_annotated.iloc[last_sync_idx:].reset_index(drop=True)
        merged_annotated_cut.to_csv(out_cut, index=False)
        print(f"  Trimmed rows: {len(merged_annotated)} -> {len(merged_annotated_cut)} (cut at {last_sync_idx})")
        print(f"  Saved: {out_cut}")
    else:
        print("  WARNING: no AUDIO_SYNC found; not writing _cut.csv.")

    # ---- Cut raw audio at JSON sync time ----
    audio_sync_value = cfg_entry.get("audio_sync")
    if audio_sync_value is None:
        print("  No audio_sync in config; skipping audio cut.")
        return True

    try:
        sync_seconds = parse_mmss(audio_sync_value)
    except Exception as e:
        print(f"  WARNING: could not parse audio_sync={audio_sync_value!r}: {e}")
        return True

    try:
        audio, sr, in_wav_path = load_audio_for_subject(short, str(data_dir))
    except Exception as e:
        print(f"  WARNING: could not load audio for {subj_key}: {e}")
        return True

    start_sample = int(round(sync_seconds * sr))
    if start_sample >= len(audio):
        print(
            f"  WARNING: sync time ({sync_seconds:.2f}s) exceeds audio length "
            f"({len(audio)/sr:.2f}s). Skipping audio cut."
        )
        return True

    print(f"  Cutting audio from sample {start_sample}")
    print(f"   JSON sync time: {_fmt_seconds(sync_seconds)} ({sync_seconds:.2f} s)")
    print(f"   From-sample time: {_fmt_seconds(start_sample / sr)} ({start_sample / sr:.2f} s)")

    audio_cut = audio[start_sample:]
    base = os.path.splitext(os.path.basename(in_wav_path or f"{subj_key}.wav"))[0]
    out_wav_path = audio_out_dir / f"{base}_cut.wav"
    sf.write(str(out_wav_path), audio_cut, sr)
    print(f"  Audio written: {out_wav_path}")

    print(f"=== Done {subj_key} ===")
    return True


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Step 1: align EEG/physio, annotate conditions, cut audio.")
    p.add_argument(
        "-s", "--subjects",
        nargs="*",
        default=None,
        help="Subjects (e.g. 02 03). Default: all subjects with status='ready' in config.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to subjects.json")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Raw data root")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Processed data root")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--sampling-rate",
        type=int,
        default=DEFAULT_SAMPLING_RATE,
        help="Sample rate (Hz). Default 256.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = _load_config(args.config)

    if args.subjects:
        keys = [_normalize_subj_key(s) for s in args.subjects]
    else:
        keys = [k for k, v in cfg.items() if v.get("status") == "ready"]

    print(f"Subjects to process: {keys}")
    n_ok = 0
    for k in keys:
        if k not in cfg:
            print(f"\n{k}: not in config; skipping.")
            continue
        ok = process_subject(
            subj=k,
            cfg_entry=cfg[k],
            data_dir=args.data_dir,
            out_dir=args.output_dir,
            sampling_rate=args.sampling_rate,
            overwrite=args.overwrite,
        )
        n_ok += int(ok)
    print(f"\nAll done. {n_ok}/{len(keys)} subjects processed successfully.")
