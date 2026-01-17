#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract HRV (Heart Rate Variability) features from preprocessed ECG data.

Computes temporal and spectral HRV features using NeuroKit2 with rolling windows.
Features include time-domain (RMSSD, SDNN, etc.), frequency-domain (HF, LF, etc.),
and nonlinear measures (entropy, DFA, etc.).

Usage:
    python extract_hrv_features.py --subjects 2 3 4 5 6
    python extract_hrv_features.py --subjects 2 --window 30 --overlap 0.5
"""

import argparse
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Default parameters
FS = 256.0
WIN_SEC = 30.0      # Window size in seconds
OVERLAP = 0.9       # Overlap ratio (0-1)


def compute_rolling_hrv_features(ecg_clean, rpeaks, fs, win_sec, overlap):
    """
    Compute HRV features in rolling windows.
    
    Parameters
    ----------
    ecg_clean : np.ndarray
        Cleaned ECG signal
    rpeaks : np.ndarray
        R-peak indices
    fs : float
        Sampling rate
    win_sec : float
        Window size in seconds
    overlap : float
        Overlap ratio (0-1)
    
    Returns
    -------
    features_df : pd.DataFrame
        HRV features for each window
    """
    win_samples = int(win_sec * fs)
    step_samples = int(win_samples * (1 - overlap))
    
    features_list = []
    
    for start_idx in range(0, len(ecg_clean) - win_samples + 1, step_samples):
        end_idx = start_idx + win_samples
        
        # Get R-peaks within this window
        window_rpeaks = rpeaks[(rpeaks >= start_idx) & (rpeaks < end_idx)]
        
        if len(window_rpeaks) < 5:
            # Not enough peaks for reliable HRV analysis
            continue
        
        # Adjust R-peak indices to be relative to window start
        window_rpeaks_rel = window_rpeaks - start_idx
        
        try:
            # Compute HRV features
            hrv_features = nk.hrv(
                window_rpeaks_rel,
                sampling_rate=fs,
                show=False
            )
            
            # Add time information
            time_start = start_idx / fs
            time_end = end_idx / fs
            
            feature_dict = hrv_features.iloc[0].to_dict()
            feature_dict['window_idx'] = len(features_list)
            feature_dict['time_start'] = time_start
            feature_dict['time_end'] = time_end
            feature_dict['n_peaks'] = len(window_rpeaks)
            
            features_list.append(feature_dict)
            
        except Exception as e:
            # If feature extraction fails, fill with NaN
            if features_list:
                # Use previous window's keys to maintain structure
                nan_dict = {k: np.nan for k in features_list[-1].keys()}
                nan_dict['window_idx'] = len(features_list)
                nan_dict['time_start'] = start_idx / fs
                nan_dict['time_end'] = end_idx / fs
                nan_dict['n_peaks'] = len(window_rpeaks)
                features_list.append(nan_dict)
    
    if not features_list:
        return None
    
    features_df = pd.DataFrame(features_list)
    
    # Reorder columns to put metadata first
    meta_cols = ['window_idx', 'time_start', 'time_end', 'n_peaks']
    other_cols = [c for c in features_df.columns if c not in meta_cols]
    features_df = features_df[meta_cols + other_cols]
    
    return features_df


def process_subject(subject_id, win_sec=WIN_SEC, overlap=OVERLAP, overwrite=False):
    """
    Extract HRV features for one subject across all conditions.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    win_sec : float
        Window size in seconds
    overlap : float
        Overlap ratio (0-1)
    overwrite : bool
        Overwrite existing feature files
    """
    print(f"\n{'='*80}")
    print(f"Processing Subject {subject_id:02d}")
    print('='*80)
    print(f"Window: {win_sec}s, Overlap: {overlap*100:.0f}%")
    
    # Paths
    subj_folder = f'sub-{subject_id:02d}'
    subj_dir = PROCESSED / subj_folder
    ecg_dir = subj_dir / 'ecg_processed'
    tables_dir = subj_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    if not ecg_dir.exists():
        print(f"ERROR: {ecg_dir} not found. Run preprocess_ecg.py first.")
        return
    
    # Conditions to process
    conditions = ['RS1', 'VIZ', 'AUD', 'MULTI', 'RS2']
    
    for condition in conditions:
        print(f"\n{condition}:")
        
        # Check if already processed
        out_file = tables_dir / f'hrv_features_{condition}.csv'
        if out_file.exists() and not overwrite:
            print(f"  EXISTS: {out_file.name} (use --overwrite to regenerate)")
            continue
        
        # Load cleaned ECG
        ecg_file = ecg_dir / f'ecg_{condition}.csv'
        if not ecg_file.exists():
            print(f"  WARNING: {ecg_file.name} not found")
            continue
        
        df_ecg = pd.read_csv(ecg_file)
        ecg_clean = df_ecg['ecg_clean'].values
        
        # Load R-peaks
        rpeaks_file = ecg_dir / f'rpeaks_{condition}.npy'
        if not rpeaks_file.exists():
            print(f"  WARNING: {rpeaks_file.name} not found")
            continue
        
        rpeaks = np.load(rpeaks_file)
        
        print(f"  Loaded: {len(ecg_clean)} samples, {len(rpeaks)} R-peaks")
        
        # Compute HRV features
        features_df = compute_rolling_hrv_features(
            ecg_clean, rpeaks, FS, win_sec, overlap
        )
        
        if features_df is None or len(features_df) == 0:
            print(f"  WARNING: No features extracted for {condition}")
            continue
        
        # Add subject and condition info
        features_df.insert(0, 'subject', f'sub-{subject_id:02d}')
        features_df.insert(1, 'condition', condition)
        
        # Save features
        features_df.to_csv(out_file, index=False)
        
        print(f"  Saved: {out_file.name}")
        print(f"  Features: {len(features_df)} windows × {len(features_df.columns)-6} features")
        print(f"  Duration: {features_df['time_end'].iloc[-1]:.1f}s")
        
        # Print sample features
        hrv_cols = [c for c in features_df.columns if c.startswith('HRV_')]
        if hrv_cols:
            print(f"  Sample features: {', '.join(hrv_cols[:5])}")
    
    print(f"\n{'='*80}")
    print(f"Completed Subject {subject_id:02d}")
    print('='*80)


def aggregate_hrv_features(subjects=None):
    """
    Aggregate HRV features across all subjects and conditions.
    
    Parameters
    ----------
    subjects : list of int, optional
        Subject IDs to aggregate. If None, finds all available.
    """
    print("\n" + "="*80)
    print("AGGREGATING HRV FEATURES")
    print("="*80)
    
    if subjects is None:
        # Auto-detect subjects
        subjects = []
        for subj_dir in sorted(PROCESSED.glob('sub-*')):
            try:
                subj_id = int(subj_dir.name.split('-')[1])
                subjects.append(subj_id)
            except (IndexError, ValueError):
                continue
    
    all_features = []
    conditions = ['RS1', 'VIZ', 'AUD', 'MULTI', 'RS2']
    
    for subj_id in subjects:
        subj_folder = f'sub-{subj_id:02d}'
        tables_dir = PROCESSED / subj_folder / 'tables'
        
        for condition in conditions:
            feature_file = tables_dir / f'hrv_features_{condition}.csv'
            
            if feature_file.exists():
                df = pd.read_csv(feature_file)
                all_features.append(df)
                print(f"  Loaded: sub-{subj_id:02d}, {condition} ({len(df)} windows)")
    
    if not all_features:
        print("ERROR: No feature files found")
        return
    
    # Concatenate all features
    df_all = pd.concat(all_features, ignore_index=True)
    
    # Save aggregated file
    out_file = PROCESSED / 'hrv_features_all_subjects.csv'
    df_all.to_csv(out_file, index=False)
    
    print(f"\nSaved: {out_file}")
    print(f"Total: {len(df_all)} windows across {len(subjects)} subjects")
    print(f"Features: {len([c for c in df_all.columns if c.startswith('HRV_')])} HRV metrics")
    
    # Print summary statistics
    print("\nSummary by condition:")
    summary = df_all.groupby('condition').size()
    for cond, count in summary.items():
        print(f"  {cond}: {count} windows")
    
    return df_all


def main():
    parser = argparse.ArgumentParser(
        description='Extract HRV features from preprocessed ECG'
    )
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject IDs to process (default: all)')
    parser.add_argument('--window', type=float, default=WIN_SEC,
                       help=f'Window size in seconds (default: {WIN_SEC})')
    parser.add_argument('--overlap', type=float, default=OVERLAP,
                       help=f'Overlap ratio 0-1 (default: {OVERLAP})')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing feature files')
    parser.add_argument('--aggregate', action='store_true',
                       help='After processing, aggregate all features')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.window <= 0:
        print("ERROR: Window size must be > 0")
        return
    if not 0 <= args.overlap < 1:
        print("ERROR: Overlap must be in [0, 1)")
        return
    
    # Get subject list
    if args.subjects:
        subjects = args.subjects
    else:
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
            process_subject(subj_id, 
                          win_sec=args.window,
                          overlap=args.overlap,
                          overwrite=args.overwrite)
        except Exception as e:
            print(f"\nERROR processing subject {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate if requested
    if args.aggregate:
        try:
            aggregate_hrv_features(subjects)
        except Exception as e:
            print(f"\nERROR aggregating features: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL DONE")
    print("="*80)


if __name__ == '__main__':
    main()
