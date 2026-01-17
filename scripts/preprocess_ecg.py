#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess ECG data using NeuroKit2 and save cleaned signals per condition.

This script:
1. Loads merged_annotated_with_audio.csv for each subject
2. Cleans ECG signal using neurokit2
3. Detects R-peaks
4. Saves cleaned ECG and R-peaks for each condition (RS1, VIZ, AUD, MULTI, RS2)

Usage:
    python preprocess_ecg.py --subjects 2 3 4 5 6
    python preprocess_ecg.py --subjects 2 --overwrite
"""

import argparse
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# HNA utils
from HNA.modules.utils import extract_condition_data

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Sampling rate
FS = 256.0


def preprocess_ecg_segment(ecg_signal, fs=FS):
    """
    Clean ECG signal and detect R-peaks using NeuroKit2.
    
    Parameters
    ----------
    ecg_signal : array-like
        Raw ECG signal
    fs : float
        Sampling rate in Hz
    
    Returns
    -------
    cleaned_ecg : np.ndarray
        Cleaned ECG signal
    rpeaks : np.ndarray
        Indices of detected R-peaks
    """
    try:
        # Clean ECG signal
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method='neurokit')
        
        # Detect R-peaks
        signals, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs, method='neurokit')
        rpeaks = info['ECG_R_Peaks']
        
        print(f"    Detected {len(rpeaks)} R-peaks in {len(ecg_signal)/fs:.1f}s "
              f"({len(rpeaks)/(len(ecg_signal)/fs)*60:.1f} bpm avg)")
        
        return cleaned_ecg, rpeaks
        
    except Exception as e:
        print(f"    ERROR in ECG preprocessing: {e}")
        return None, None


def process_subject(subject_id, overwrite=False):
    """
    Process ECG data for one subject across all conditions.
    
    Parameters
    ----------
    subject_id : int
        Subject ID (e.g., 2, 3, 4, ...)
    overwrite : bool
        If True, overwrite existing processed files
    """
    print(f"\n{'='*80}")
    print(f"Processing Subject {subject_id:02d}")
    print('='*80)
    
    # Paths
    subj_folder = f'sub-{subject_id:02d}'
    subj_dir = PROCESSED / subj_folder
    tables_dir = subj_dir / 'tables'
    ecg_dir = subj_dir / 'ecg_processed'
    ecg_dir.mkdir(parents=True, exist_ok=True)
    
    # Load merged data
    merged_file = tables_dir / 'merged_annotated_with_audio.csv'
    if not merged_file.exists():
        print(f"ERROR: {merged_file} not found")
        return
    
    print(f"Loading: {merged_file}")
    df = pd.read_csv(merged_file, low_memory=False)
    
    # Check for ECG column
    if 'ecg' not in df.columns:
        print(f"ERROR: No 'ecg' column in dataframe")
        return
    
    # Conditions to process
    conditions = ['RS1', 'VIZ', 'AUD', 'MULTI', 'RS2']
    
    for condition in conditions:
        print(f"\n{condition}:")
        
        # Check if already processed
        out_file = ecg_dir / f'ecg_{condition}.csv'
        if out_file.exists() and not overwrite:
            print(f"  EXISTS: {out_file.name} (use --overwrite to regenerate)")
            continue
        
        # Extract condition data
        df_condition = extract_condition_data(df, condition)
        
        if df_condition is None or len(df_condition) == 0:
            print(f"  WARNING: No data for {condition}")
            continue
        
        # Get ECG signal
        ecg_signal = df_condition['ecg'].values
        
        # Handle NaN values
        if np.any(np.isnan(ecg_signal)):
            n_nan = np.sum(np.isnan(ecg_signal))
            print(f"  WARNING: {n_nan} NaN values in ECG, interpolating...")
            ecg_signal = pd.Series(ecg_signal).interpolate(method='linear', 
                                                           limit_direction='both').values
        
        # Preprocess ECG
        cleaned_ecg, rpeaks = preprocess_ecg_segment(ecg_signal, fs=FS)
        
        if cleaned_ecg is None:
            print(f"  FAILED: Could not process ECG for {condition}")
            continue
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'time_s': df_condition['time_s'].values if 'time_s' in df_condition.columns 
                     else np.arange(len(cleaned_ecg)) / FS,
            'ecg_raw': ecg_signal,
            'ecg_clean': cleaned_ecg,
            'rpeak': 0  # Will mark R-peaks with 1
        })
        
        # Mark R-peak locations
        output_df.loc[rpeaks, 'rpeak'] = 1
        
        # Save
        output_df.to_csv(out_file, index=False)
        print(f"  Saved: {out_file.name} ({len(output_df)} samples)")
        
        # Also save R-peaks separately for convenience
        rpeaks_file = ecg_dir / f'rpeaks_{condition}.npy'
        np.save(rpeaks_file, rpeaks)
        print(f"  Saved: {rpeaks_file.name} ({len(rpeaks)} peaks)")
    
    print(f"\n{'='*80}")
    print(f"Completed Subject {subject_id:02d}")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ECG data using NeuroKit2'
    )
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject IDs to process (default: all found in processed/)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing processed files')
    
    args = parser.parse_args()
    
    # Get subject list
    if args.subjects:
        subjects = args.subjects
    else:
        # Auto-detect from processed directory
        subjects = []
        for subj_dir in sorted(PROCESSED.glob('sub-*')):
            try:
                subj_id = int(subj_dir.name.split('-')[1])
                subjects.append(subj_id)
            except (IndexError, ValueError):
                continue
    
    if not subjects:
        print("ERROR: No subjects found")
        return
    
    print(f"\nProcessing {len(subjects)} subjects: {subjects}")
    
    # Process each subject
    for subj_id in subjects:
        try:
            process_subject(subj_id, overwrite=args.overwrite)
        except Exception as e:
            print(f"\nERROR processing subject {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("ALL DONE")
    print("="*80)


if __name__ == '__main__':
    main()
