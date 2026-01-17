"""
Script to cut audio files into condition-specific chunks.

This script:
1. Loads merged_annotated_with_audio.csv to get condition timings (256 Hz)
2. Loads audio_cut files from processed/sub-XX/audio/ 
3. Converts EEG sample indices to audio sample indices
4. Cuts audio into 5 chunks (RS1, VIZ, AUD, MULTI, RS2)
5. Saves condition-specific audio files

Usage:
    python cut_audio_by_conditions.py --subjects 2 3 4 5 6
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_condition_indices(merged_file, eeg_sr=256):
    """
    Load condition start/stop indices from merged CSV.
    
    Parameters
    ----------
    merged_file : Path
        Path to merged_annotated_with_audio.csv
    eeg_sr : int
        EEG sampling rate in Hz
    
    Returns
    -------
    condition_times : dict
        Dictionary with condition names as keys and (start_idx, stop_idx) tuples
    """
    print(f"\n  Loading: {merged_file.name}")
    df = pd.read_csv(merged_file)
    
    conditions = ['RS1', 'VIZ', 'AUD', 'MULTI', 'RS2']
    condition_times = {}
    
    for condition in conditions:
        start_label = f'{condition}_start'
        stop_label = f'{condition}_stop'
        
        # Find start and stop indices
        start_idx = df[df['condition_names'] == start_label].index
        stop_idx = df[df['condition_names'] == stop_label].index
        
        if len(start_idx) > 0 and len(stop_idx) > 0:
            start = start_idx[0]
            stop = stop_idx[0]
            duration_sec = (stop - start) / eeg_sr
            
            condition_times[condition] = (start, stop)
            print(f"    {condition}: samples {start}-{stop} ({duration_sec:.1f} seconds)")
        else:
            print(f"    {condition}: markers not found")
    
    return condition_times


def cut_audio_by_condition(audio_path, condition_times, eeg_sr=256):
    """
    Cut audio file into condition-specific chunks.
    
    Parameters
    ----------
    audio_path : Path
        Path to audio_cut file
    condition_times : dict
        Dictionary with EEG condition timings (start_idx, stop_idx)
    eeg_sr : int
        EEG sampling rate
    
    Returns
    -------
    audio_chunks : dict
        Dictionary with condition names as keys and audio arrays as values
    audio_sr : int
        Audio sampling rate
    """
    print(f"\n  Loading audio: {audio_path.name}")
    
    # Load audio
    audio, audio_sr = sf.read(audio_path)
    
    print(f"    Audio sampling rate: {audio_sr} Hz")
    print(f"    Audio duration: {len(audio) / audio_sr:.1f} seconds")
    print(f"    Audio samples: {len(audio)}")
    
    # Calculate conversion ratio
    sr_ratio = audio_sr / eeg_sr
    print(f"    Sampling rate ratio (audio/eeg): {sr_ratio:.4f}")
    
    # Cut audio for each condition
    audio_chunks = {}
    
    for condition, (eeg_start, eeg_stop) in condition_times.items():
        # Convert EEG indices to audio indices
        audio_start = int(eeg_start * sr_ratio)
        audio_stop = int(eeg_stop * sr_ratio)
        
        # Ensure we don't exceed audio length
        audio_stop = min(audio_stop, len(audio))
        
        if audio_start < len(audio):
            # Extract chunk
            chunk = audio[audio_start:audio_stop]
            audio_chunks[condition] = chunk
            
            duration_sec = len(chunk) / audio_sr
            print(f"    {condition}: samples {audio_start}-{audio_stop} ({duration_sec:.1f} seconds)")
        else:
            print(f"    {condition}: start index exceeds audio length, skipping")
    
    return audio_chunks, audio_sr


def process_subject(subject_id, data_dir):
    """
    Process one subject: load timings and cut audio.
    
    Parameters
    ----------
    subject_id : int
        Subject number
    data_dir : Path
        Path to data directory
    
    Returns
    -------
    success : bool
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing Subject {subject_id:02d}")
    print(f"{'='*60}")
    
    subject_folder = f'sub-{subject_id:02d}'
    processed_dir = data_dir / 'processed' / subject_folder
    
    # Check for required files
    merged_file = processed_dir / 'tables' / 'merged_annotated_with_audio.csv'
    audio_dir = processed_dir / 'audio'
    
    if not merged_file.exists():
        print(f"  ERROR: {merged_file} not found")
        return False
    
    if not audio_dir.exists():
        print(f"  ERROR: {audio_dir} not found")
        return False
    
    # Find audio_cut file
    audio_files = list(audio_dir.glob('*_cut.wav'))
    
    if len(audio_files) == 0:
        print(f"  ERROR: No *_cut.wav file found in {audio_dir}")
        return False
    
    if len(audio_files) > 1:
        print(f"  WARNING: Multiple *_cut.wav files found, using first one")
    
    audio_path = audio_files[0]
    
    # Load condition timings from EEG data
    condition_times = load_condition_indices(merged_file)
    
    if len(condition_times) == 0:
        print("  ERROR: No condition timings found")
        return False
    
    # Cut audio by conditions
    audio_chunks, audio_sr = cut_audio_by_condition(audio_path, condition_times)
    
    if len(audio_chunks) == 0:
        print("  ERROR: No audio chunks extracted")
        return False
    
    # Save audio chunks
    print(f"\n  Saving condition-specific audio files...")
    
    for condition, chunk in audio_chunks.items():
        output_file = audio_dir / f'audio_{condition}.wav'
        sf.write(output_file, chunk, audio_sr)
        print(f"    Saved: {output_file.name} ({len(chunk)} samples)")
    
    print(f"\n  ✓ Successfully processed {subject_folder}")
    return True


def main():
    """Main processing function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Cut audio files by conditions')
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='Subject IDs to process (default: 2 3 4 5 6)')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    
    print("="*60)
    print("AUDIO CUTTING BY CONDITIONS")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Subjects: {args.subjects}")
    print(f"\nConditions to extract: RS1, VIZ, AUD, MULTI, RS2")
    
    # Process each subject
    results = {}
    
    for subject_id in args.subjects:
        try:
            success = process_subject(subject_id, data_dir)
            results[subject_id] = 'SUCCESS' if success else 'FAILED'
        except Exception as e:
            print(f"\n  ERROR processing subject {subject_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[subject_id] = 'ERROR'
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    for subject_id, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} Subject {subject_id:02d}: {status}")
    
    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    print(f"\nTotal: {success_count}/{len(results)} subjects processed successfully")
    
    print("\nNext steps:")
    print("  - Audio files saved as: audio_[CONDITION].wav in each subject's audio folder")
    print("  - Ready for spectral coherence analysis between EEG and audio")


if __name__ == '__main__':
    main()
