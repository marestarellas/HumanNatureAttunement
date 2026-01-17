"""
Script to extract EEG features from all subjects.

This script:
1. Loads merged_annotated_with_audio.csv for each subject
2. Applies bandpass filtering (1-50 Hz)
3. Extracts data for each condition (RS1, VIZ, AUD, MULTI, RS2)
4. Computes PSD features (delta, theta, alpha, beta, gamma)
5. Computes entropy features (LZC, permutation, spectral, SVD, sample entropy)
6. Saves features_[condition].csv for each condition and subject

Usage:
    python extract_eeg_features.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from HNA.modules.eeg import filter_eeg, compute_psd_features, compute_entropy_features
from HNA.modules.utils import extract_condition_data


def process_subject(subject_folder, data_dir, window_sec=5, overlap_sec=0):
    """
    Process EEG data for a single subject.
    
    Parameters
    ----------
    subject_folder : str
        Subject folder name (e.g., 'sub-02')
    data_dir : Path
        Path to data directory
    window_sec : float
        Window length in seconds (default: 5)
    overlap_sec : float
        Overlap between windows in seconds (default: 0 for non-overlapping)
    
    Returns
    -------
    success : bool
        True if processing successful, False otherwise
    """
    
    print(f"\n{'='*60}")
    print(f"Processing {subject_folder}")
    print(f"{'='*60}")
    
    # Path to merged data
    input_file = data_dir / 'processed' / subject_folder / 'tables' / 'merged_annotated_with_audio.csv'
    
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        return False
    
    # Load data
    print(f"\nLoading data from {input_file.name}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples")
    
    # Filter EEG
    print(f"\n{'-'*60}")
    print("Step 1: Filtering EEG (1-50 Hz bandpass)")
    print(f"{'-'*60}")
    df_filtered = filter_eeg(df, sampling_rate=256, lowcut=1.0, highcut=50)
    
    # Conditions to extract
    conditions = ['RS1', 'VIZ', 'AUD', 'MULTI', 'RS2']
    
    # Output directory
    output_dir = data_dir / 'processed' / subject_folder / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each condition
    for condition in conditions:
        print(f"\n{'-'*60}")
        print(f"Step 2: Processing condition: {condition}")
        print(f"{'-'*60}")
        
        # Extract condition data
        df_condition = extract_condition_data(df_filtered, condition)
        
        if df_condition is None:
            print(f"Skipping {condition} - markers not found")
            continue
        
        # Compute PSD features
        print(f"\nComputing PSD features...")
        psd_features = compute_psd_features(
            df_condition, 
            sampling_rate=256, 
            window_sec=window_sec, 
            overlap_sec=overlap_sec
        )
        
        # Compute entropy features (LZC, permutation, spectral, SVD, sample entropy)
        print(f"\nComputing entropy features...")
        entropy_features = compute_entropy_features(
            df_condition, 
            sampling_rate=256, 
            window_sec=window_sec, 
            overlap_sec=overlap_sec
        )
        
        # Merge PSD and entropy features
        features = pd.merge(
            psd_features, 
            entropy_features[['channel', 'window_idx', 'lzc', 'perm_entropy', 
                             'spectral_entropy', 'svd_entropy', 'sample_entropy']], 
            on=['channel', 'window_idx'],
            how='outer'
        )
        
        # Add subject and condition columns
        features.insert(0, 'subject', subject_folder)
        features.insert(1, 'condition', condition)
        
        # Save features
        output_file = output_dir / f'features_{condition}.csv'
        features.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
        print(f"Features shape: {features.shape}")
        print(f"Columns: {list(features.columns)}")
    
    print(f"\n{'='*60}")
    print(f"Completed {subject_folder}")
    print(f"{'='*60}")
    
    return True


def main():
    """Main processing function."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    print("="*60)
    print("EEG FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Processed directory: {processed_dir}")
    
    # Window parameters (non-overlapping for statistical validity)
    window_sec = 5
    overlap_sec = 0
    print(f"Window parameters: {window_sec}s windows, {overlap_sec}s overlap")
    
    # Find all subject folders
    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found: {processed_dir}")
        return
    
    subject_folders = sorted([
        f.name for f in processed_dir.iterdir() 
        if f.is_dir() and f.name.startswith('sub-')
    ])
    
    if not subject_folders:
        print(f"ERROR: No subject folders found in {processed_dir}")
        return
    
    print(f"\nFound {len(subject_folders)} subject folders:")
    for subj in subject_folders:
        print(f"  - {subj}")
    
    # Process each subject
    results = {}
    for subject_folder in subject_folders:
        try:
            success = process_subject(subject_folder, data_dir, window_sec, overlap_sec)
            results[subject_folder] = 'SUCCESS' if success else 'FAILED'
        except Exception as e:
            print(f"\nERROR processing {subject_folder}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[subject_folder] = 'ERROR'
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    for subject, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {subject}: {status}")
    
    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    print(f"\nTotal: {success_count}/{len(results)} subjects processed successfully")


if __name__ == '__main__':
    main()
