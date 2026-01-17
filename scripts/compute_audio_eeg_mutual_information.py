"""
Script to compute audio-EEG coupling using mutual information.

Mutual information captures both linear AND non-linear relationships between signals,
making it more sensitive than correlation but also more computationally intensive.

This script computes:
1. Direct mutual information between filtered audio envelope and filtered EEG per band
2. Time-lagged mutual information (finds optimal lag accounting for neural delays)

Like the correlation analysis, both audio envelope AND EEG signal are bandpass 
filtered to the SAME frequency band before computing MI.

MI is always non-negative (0 = independent, higher = more dependent).
Unlike correlation, MI can detect non-linear coupling patterns.

Usage:
    python compute_audio_eeg_mutual_information.py --subjects 2 3 5 --conditions VIZ AUD MULTI
    python compute_audio_eeg_mutual_information.py --subjects 2 --conditions VIZ --n-bins 20
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import hilbert, butter, sosfiltfilt, resample_poly
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


def extract_audio_envelope(audio_signal, fs):
    """
    Extract amplitude envelope using Hilbert transform.
    
    Parameters:
    -----------
    audio_signal : array
        Audio waveform
    fs : int
        Sampling rate
    
    Returns:
    --------
    envelope : array
        Amplitude envelope
    """
    analytic_signal = hilbert(audio_signal)
    envelope = np.abs(analytic_signal)
    return envelope


def bandpass_filter_envelope(envelope, lowcut, highcut, fs, order=4):
    """
    Bandpass filter the envelope to isolate specific frequency range.
    Uses second-order sections for numerical stability.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz at fs={fs} Hz")
    
    sos = butter(order, [low, high], btype='band', output='sos')
    padlen = min(len(envelope) // 4, 500)
    filtered = sosfiltfilt(sos, envelope, padlen=padlen)
    
    return filtered


def downsample_envelope(envelope, original_fs, target_fs):
    """
    Downsample envelope to match EEG sampling rate.
    """
    downsample_factor = original_fs / target_fs
    
    if downsample_factor == 1:
        return envelope
    
    down = int(original_fs)
    up = int(target_fs)
    
    from math import gcd
    g = gcd(down, up)
    down = down // g
    up = up // g
    
    downsampled = resample_poly(envelope, up, down)
    
    return downsampled


def bandpass_filter_eeg(eeg_signal, lowcut, highcut, fs, order=4):
    """
    Bandpass filter EEG signal to isolate specific frequency band.
    Uses second-order sections for numerical stability.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz at fs={fs} Hz")
    
    sos = butter(order, [low, high], btype='band', output='sos')
    padlen = min(len(eeg_signal) // 4, 500)
    filtered = sosfiltfilt(sos, eeg_signal, padlen=padlen)
    
    return filtered


def load_eeg_data(subject_id, condition, data_dir):
    """
    Load EEG data for one subject and condition.
    """
    subject_folder = f'sub-{subject_id:02d}'
    merged_file = data_dir / 'processed' / subject_folder / 'tables' / 'merged_annotated_with_audio.csv'
    
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_file}")
    
    df = pd.read_csv(merged_file)
    
    # Find condition column
    if 'condition_names' in df.columns:
        marker_col = 'condition_names'
    elif 'annotation' in df.columns:
        marker_col = 'annotation'
    else:
        raise ValueError("Cannot find 'condition_names' or 'annotation' column")
    
    # Look for start/stop markers
    start_idx = df[df[marker_col] == f'{condition}_start'].index
    stop_idx = df[df[marker_col] == f'{condition}_stop'].index
    
    if len(start_idx) == 0 or len(stop_idx) == 0:
        raise ValueError(f"Condition {condition} markers not found for subject {subject_id}")
    
    start = start_idx[0]
    stop = stop_idx[0]
    df_condition = df.iloc[start:stop].copy()
    
    # Get EEG channels
    exclude_cols = ['annotation', 'condition_names', 'time_s', 'audio_sample', 'audio_time_s',
                    'ecg', 'respiration', 'physio_triggers', 'condition_triggers', 
                    'sequence', 'battery', 'flags', 'eeg_triggers',
                    'env_broad', 'env_swell_0p3', 'env_swell_0p1', 'env_splash_1_5']
    
    eeg_channels = [col for col in df_condition.columns 
                   if col.startswith('EEG-ch') or (col not in exclude_cols and col not in df_condition.columns[:10])]
    
    if len(eeg_channels) == 0:
        raise ValueError(f"No EEG channels found for subject {subject_id}")
    
    # Create dictionary
    eeg_data = {}
    for channel in eeg_channels:
        eeg_data[channel] = df_condition[channel].values
    
    return eeg_data


def discretize_signal(signal, n_bins):
    """
    Discretize continuous signal into bins for mutual information computation.
    
    Parameters:
    -----------
    signal : array
        Continuous signal
    n_bins : int
        Number of bins
    
    Returns:
    --------
    discretized : array
        Discretized signal (integer bin indices)
    """
    # Use equal-frequency binning (quantile-based)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(signal, percentiles)
    bin_edges[-1] += 1e-10  # Ensure last value is included
    
    discretized = np.digitize(signal, bin_edges) - 1
    discretized = np.clip(discretized, 0, n_bins - 1)
    
    return discretized


def compute_mutual_information(envelope, eeg_signal, n_bins=20, method='discrete'):
    """
    Compute mutual information between envelope and EEG.
    
    Parameters:
    -----------
    envelope : array
        Filtered audio envelope
    eeg_signal : array
        Filtered EEG signal
    n_bins : int
        Number of bins for discretization (if method='discrete')
    method : str
        'discrete' = use sklearn.metrics.mutual_info_score (faster, requires discretization)
        'continuous' = use sklearn.feature_selection.mutual_info_regression (slower, handles continuous)
    
    Returns:
    --------
    mi : float
        Mutual information (nats if continuous, bits if discrete)
    """
    # Ensure same length
    min_len = min(len(envelope), len(eeg_signal))
    env = envelope[:min_len]
    eeg = eeg_signal[:min_len]
    
    if method == 'discrete':
        # Discretize both signals
        env_binned = discretize_signal(env, n_bins)
        eeg_binned = discretize_signal(eeg, n_bins)
        
        # Compute MI using discrete method
        mi = mutual_info_score(env_binned, eeg_binned)
        
    elif method == 'continuous':
        # Use continuous MI estimator (k-nearest neighbors)
        # Reshape for sklearn
        X = env.reshape(-1, 1)
        y = eeg
        
        # Compute MI
        mi = mutual_info_regression(X, y, n_neighbors=3, random_state=42)[0]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mi


def compute_lagged_mutual_information(envelope, eeg_signal, fs, max_lag_ms=500, 
                                     n_bins=20, method='discrete'):
    """
    Compute mutual information with time lags to find optimal delay.
    
    Parameters:
    -----------
    envelope : array
        Filtered audio envelope
    eeg_signal : array
        Filtered EEG signal
    fs : int
        Sampling rate
    max_lag_ms : float
        Maximum lag to test in milliseconds
    n_bins : int
        Number of bins for discretization
    method : str
        MI computation method
    
    Returns:
    --------
    max_mi : float
        Maximum mutual information
    optimal_lag_ms : float
        Optimal lag in milliseconds (positive = EEG lags behind audio)
    optimal_lag_samples : int
        Optimal lag in samples
    """
    # Ensure same length
    min_len = min(len(envelope), len(eeg_signal))
    env = envelope[:min_len]
    eeg = eeg_signal[:min_len]
    
    # Define lag range
    max_lag_samples = int(max_lag_ms * fs / 1000)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    
    mi_values = []
    
    for lag in lags:
        if lag < 0:
            # EEG leads audio
            env_shifted = env[:lag]
            eeg_shifted = eeg[-lag:]
        elif lag > 0:
            # Audio leads EEG (typical for sensory processing)
            env_shifted = env[lag:]
            eeg_shifted = eeg[:-lag]
        else:
            # No lag
            env_shifted = env
            eeg_shifted = eeg
        
        # Compute MI
        try:
            mi = compute_mutual_information(env_shifted, eeg_shifted, n_bins=n_bins, method=method)
            mi_values.append(mi)
        except:
            mi_values.append(0)
    
    # Find maximum
    mi_values = np.array(mi_values)
    max_idx = np.argmax(mi_values)
    max_mi = mi_values[max_idx]
    optimal_lag_samples = lags[max_idx]
    optimal_lag_ms = optimal_lag_samples * 1000 / fs
    
    return max_mi, optimal_lag_ms, optimal_lag_samples


def process_subject_condition(subject_id, condition, data_dir, audio_dir, eeg_sr=256, 
                               frequency_bands=None, max_lag_ms=500, n_bins=20, method='discrete'):
    """
    Process one subject-condition: compute mutual information between audio and EEG.
    
    Parameters:
    -----------
    subject_id : int
        Subject number
    condition : str
        Condition name
    data_dir : Path
        Data directory
    audio_dir : Path
        Directory with cut audio files
    eeg_sr : int
        EEG sampling rate
    frequency_bands : dict
        Frequency bands to analyze
    max_lag_ms : float
        Maximum lag for cross-MI (ms)
    n_bins : int
        Number of bins for discretization
    method : str
        MI computation method ('discrete' or 'continuous')
    
    Returns:
    --------
    results : list
        List of result dictionaries
    """
    if frequency_bands is None:
        frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'low_beta': (13, 20),
            'high_beta': (20, 30),
            'gamma1': (30, 50)
        }
    
    print(f"\n  Processing subject {subject_id}, condition {condition}")
    
    # Load audio
    audio_file = audio_dir / f'sub-{subject_id:02d}' / 'audio' / f'audio_{condition}.wav'
    if not audio_file.exists():
        print(f"    Warning: Audio file not found: {audio_file}")
        return []
    
    audio_signal, audio_sr = sf.read(audio_file)
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)
    
    print(f"    Audio: {len(audio_signal)} samples at {audio_sr} Hz")
    
    # Extract envelope
    envelope = extract_audio_envelope(audio_signal, audio_sr)
    
    # Load EEG
    eeg_data = load_eeg_data(subject_id, condition, data_dir)
    print(f"    EEG: {len(eeg_data)} channels, {len(list(eeg_data.values())[0])} samples")
    
    results_list = []
    
    # Process each frequency band
    for band_name, (lowcut, highcut) in frequency_bands.items():
        print(f"  Processing {band_name} band ({lowcut}-{highcut} Hz)...")
        
        try:
            # Filter audio envelope
            filtered_envelope = bandpass_filter_envelope(envelope, lowcut, highcut, audio_sr)
            
            if np.any(np.isnan(filtered_envelope)) or np.any(np.isinf(filtered_envelope)):
                print(f"    Warning: NaN/Inf after filtering audio for {band_name}, skipping")
                continue
            
            # Downsample
            downsampled_envelope = downsample_envelope(filtered_envelope, audio_sr, eeg_sr)
            
            if np.any(np.isnan(downsampled_envelope)) or np.any(np.isinf(downsampled_envelope)):
                print(f"    Warning: NaN/Inf after downsampling for {band_name}, skipping")
                continue
            
        except Exception as e:
            print(f"    Error processing envelope for {band_name}: {e}")
            continue
        
        # Process each EEG channel
        for channel, eeg_signal in eeg_data.items():
            try:
                # Filter EEG to the same frequency band
                filtered_eeg = bandpass_filter_eeg(eeg_signal, lowcut, highcut, eeg_sr)
                
                if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
                    continue
                
                # Trim to same length
                min_len = min(len(downsampled_envelope), len(filtered_eeg))
                env = downsampled_envelope[:min_len]
                eeg = filtered_eeg[:min_len]
                
                # Check for NaN/Inf
                if np.any(np.isnan(env)) or np.any(np.isnan(eeg)):
                    continue
                if np.any(np.isinf(env)) or np.any(np.isinf(eeg)):
                    continue
                
                # Check for zero variance (bad channels)
                if np.std(eeg) < 1e-10 or np.std(env) < 1e-10:
                    continue
                
                # Direct mutual information
                mi_direct = compute_mutual_information(env, eeg, n_bins=n_bins, method=method)
                
                # Lagged mutual information
                max_mi, optimal_lag_ms, optimal_lag_samples = compute_lagged_mutual_information(
                    env, eeg, eeg_sr, max_lag_ms=max_lag_ms, n_bins=n_bins, method=method
                )
                
                # Store results
                results_list.append({
                    'subject_id': subject_id,
                    'condition': condition,
                    'channel': channel,
                    'band': band_name,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'mi_direct': mi_direct,
                    'mi_max_lagged': max_mi,
                    'optimal_lag_ms': optimal_lag_ms,
                    'optimal_lag_samples': optimal_lag_samples
                })
                
            except Exception as e:
                print(f"    Warning: Error processing {channel} in {band_name}: {e}")
                continue
        
        print(f"    Completed {band_name}: {len([r for r in results_list if r['band'] == band_name])} channel pairs")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(description='Compute audio-EEG mutual information')
    parser.add_argument('--subjects', type=int, nargs='+', required=True,
                       help='Subject IDs to process')
    parser.add_argument('--conditions', type=str, nargs='+', required=True,
                       help='Conditions to process (e.g., VIZ AUD MULTI)')
    parser.add_argument('--max-lag-ms', type=float, default=500,
                       help='Maximum lag for cross-MI in ms (default: 500)')
    parser.add_argument('--n-bins', type=int, default=20,
                       help='Number of bins for discretization (default: 20)')
    parser.add_argument('--method', type=str, default='discrete', choices=['discrete', 'continuous'],
                       help='MI computation method (default: discrete)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/audio_eeg_mutual_information)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    audio_dir = data_dir / 'processed'
    
    if args.output_dir is None:
        output_dir = project_dir / 'results' / 'audio_eeg_mutual_information'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("AUDIO-EEG MUTUAL INFORMATION ANALYSIS")
    print("="*60)
    print(f"Subjects: {args.subjects}")
    print(f"Conditions: {args.conditions}")
    print(f"Max lag: ±{args.max_lag_ms} ms")
    print(f"Method: {args.method}")
    if args.method == 'discrete':
        print(f"Bins: {args.n_bins}")
    print(f"Output: {output_dir}")
    
    # Process all subject-condition pairs
    all_results = []
    
    for subject_id in args.subjects:
        for condition in args.conditions:
            try:
                results = process_subject_condition(
                    subject_id, condition, data_dir, audio_dir,
                    max_lag_ms=args.max_lag_ms, n_bins=args.n_bins, method=args.method
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  Error processing subject {subject_id}, condition {condition}: {e}")
                continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save results
    output_file = output_dir / 'audio_eeg_mutual_information_results.csv'
    df_results.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"Total records: {len(df_results)}")
    print(f"Subjects: {sorted(df_results['subject_id'].unique())}")
    print(f"Conditions: {sorted(df_results['condition'].unique())}")
    print(f"Bands: {sorted(df_results['band'].unique())}")
    print(f"Channels: {len(df_results['channel'].unique())}")
    print(f"\nSaved to: {output_file}")


if __name__ == '__main__':
    main()
