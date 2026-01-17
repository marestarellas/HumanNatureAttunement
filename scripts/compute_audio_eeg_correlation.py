"""
Script to compute audio-EEG coupling using correlation methods.

This script computes:
1. Direct envelope correlation (Pearson r between audio envelope and EEG per band)
2. Time-lagged cross-correlation (finds optimal lag accounting for neural delays)

IMPORTANT: Both audio envelope AND EEG signal are bandpass filtered to the SAME 
frequency band before correlation. This isolates band-specific coupling and follows
standard practice in auditory neuroscience.

Example:
- Theta band (4-8 Hz): Correlate 4-8 Hz filtered audio envelope with 4-8 Hz filtered EEG
- This tests if theta-rate audio fluctuations couple with theta brain oscillations

These methods are simpler and more robust than spectral coherence, capturing
amplitude relationships without requiring phase consistency.

Usage:
    python compute_audio_eeg_correlation.py --subjects 2 3 5 --conditions VIZ AUD MULTI
    python compute_audio_eeg_correlation.py --subjects 2 --conditions VIZ
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from scipy.signal import hilbert, butter, sosfiltfilt, resample_poly, correlate
from scipy.stats import pearsonr
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
    
    Parameters:
    -----------
    envelope : array
        Audio envelope
    lowcut : float
        Low frequency cutoff (Hz)
    highcut : float
        High frequency cutoff (Hz)
    fs : int
        Sampling rate
    order : int
        Filter order
    
    Returns:
    --------
    filtered : array
        Bandpass filtered envelope
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Validate frequency bounds
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz at fs={fs} Hz")
    
    # Use second-order sections for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Apply filter with padding to reduce edge effects
    padlen = min(len(envelope) // 4, 500)
    filtered = sosfiltfilt(sos, envelope, padlen=padlen)
    
    return filtered


def downsample_envelope(envelope, original_fs, target_fs):
    """
    Downsample envelope to match EEG sampling rate.
    
    Parameters:
    -----------
    envelope : array
        Envelope at original sampling rate
    original_fs : int
        Original sampling rate
    target_fs : int
        Target sampling rate (EEG rate)
    
    Returns:
    --------
    downsampled : array
        Downsampled envelope
    """
    # Calculate downsampling ratio
    downsample_factor = original_fs / target_fs
    
    if downsample_factor == 1:
        return envelope
    
    # Use polyphase filtering for clean downsampling
    down = int(original_fs)
    up = int(target_fs)
    
    # Find GCD to simplify ratio
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
    
    Parameters:
    -----------
    eeg_signal : array
        Raw EEG signal
    lowcut : float
        Low frequency cutoff (Hz)
    highcut : float
        High frequency cutoff (Hz)
    fs : int
        Sampling rate
    order : int
        Filter order
    
    Returns:
    --------
    filtered : array
        Bandpass filtered EEG signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Validate frequency bounds
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz at fs={fs} Hz")
    
    # Use second-order sections for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Apply filter with padding to reduce edge effects
    padlen = min(len(eeg_signal) // 4, 500)
    filtered = sosfiltfilt(sos, eeg_signal, padlen=padlen)
    
    return filtered


def load_eeg_data(subject_id, condition, data_dir):
    """
    Load EEG data for one subject and condition.
    
    Parameters:
    -----------
    subject_id : int
        Subject number
    condition : str
        Condition name
    data_dir : Path
        Data directory
    
    Returns:
    --------
    eeg_data : dict
        Dictionary mapping channel names to EEG signals
    """
    # Load merged annotated file
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


def compute_correlation(envelope, eeg_signal):
    """
    Compute Pearson correlation between envelope and EEG.
    
    Parameters:
    -----------
    envelope : array
        Filtered audio envelope
    eeg_signal : array
        EEG signal
    
    Returns:
    --------
    r : float
        Pearson correlation coefficient
    p : float
        P-value
    """
    # Ensure same length
    min_len = min(len(envelope), len(eeg_signal))
    env = envelope[:min_len]
    eeg = eeg_signal[:min_len]
    
    # Compute correlation
    r, p = pearsonr(env, eeg)
    
    return r, p


def compute_lagged_correlation(envelope, eeg_signal, fs, max_lag_ms=500):
    """
    Compute cross-correlation with time lags to find optimal delay.
    
    Parameters:
    -----------
    envelope : array
        Filtered audio envelope
    eeg_signal : array
        EEG signal
    fs : int
        Sampling rate
    max_lag_ms : float
        Maximum lag to test in milliseconds
    
    Returns:
    --------
    max_corr : float
        Maximum correlation coefficient
    optimal_lag_ms : float
        Optimal lag in milliseconds (positive = EEG lags behind audio)
    optimal_lag_samples : int
        Optimal lag in samples
    """
    # Ensure same length
    min_len = min(len(envelope), len(eeg_signal))
    env = envelope[:min_len]
    eeg = eeg_signal[:min_len]
    
    # Normalize signals
    env = (env - np.mean(env)) / np.std(env)
    eeg = (eeg - np.mean(eeg)) / np.std(eeg)
    
    # Compute cross-correlation
    xcorr = correlate(eeg, env, mode='same', method='auto')
    xcorr = xcorr / len(env)  # Normalize
    
    # Define lag range
    max_lag_samples = int(max_lag_ms * fs / 1000)
    center = len(xcorr) // 2
    lag_range = slice(center - max_lag_samples, center + max_lag_samples + 1)
    
    # Extract relevant lags
    xcorr_windowed = xcorr[lag_range]
    lags_samples = np.arange(-max_lag_samples, max_lag_samples + 1)
    
    # Find maximum
    max_idx = np.argmax(np.abs(xcorr_windowed))
    max_corr = xcorr_windowed[max_idx]
    optimal_lag_samples = lags_samples[max_idx]
    optimal_lag_ms = optimal_lag_samples * 1000 / fs
    
    return max_corr, optimal_lag_ms, optimal_lag_samples


def process_subject_condition(subject_id, condition, data_dir, audio_dir, eeg_sr=256, 
                               frequency_bands=None, max_lag_ms=500):
    """
    Process one subject-condition: compute correlations between audio and EEG.
    
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
        Maximum lag for cross-correlation (ms)
    
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
            # Filter envelope
            filtered_envelope = bandpass_filter_envelope(envelope, lowcut, highcut, audio_sr)
            
            # Check for NaN/Inf after filtering
            if np.any(np.isnan(filtered_envelope)) or np.any(np.isinf(filtered_envelope)):
                print(f"    Warning: NaN/Inf after filtering for {band_name}, skipping")
                continue
            
            # Downsample
            downsampled_envelope = downsample_envelope(filtered_envelope, audio_sr, eeg_sr)
            
            # Check after downsampling
            if np.any(np.isnan(downsampled_envelope)) or np.any(np.isinf(downsampled_envelope)):
                print(f"    Warning: NaN/Inf after downsampling for {band_name}, skipping")
                continue
            
        except Exception as e:
            print(f"    Error processing envelope for {band_name}: {e}")
            continue
        
        # Process each EEG channel
        for channel, eeg_signal in eeg_data.items():
            try:
                # Filter EEG to the same frequency band as audio envelope
                filtered_eeg = bandpass_filter_eeg(eeg_signal, lowcut, highcut, eeg_sr)
                
                # Check for NaN/Inf after filtering
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
                
                # Direct correlation
                r_direct, p_direct = compute_correlation(env, eeg)
                
                # Lagged correlation
                max_corr, optimal_lag_ms, optimal_lag_samples = compute_lagged_correlation(
                    env, eeg, eeg_sr, max_lag_ms=max_lag_ms
                )
                
                # Store results
                results_list.append({
                    'subject_id': subject_id,
                    'condition': condition,
                    'channel': channel,
                    'band': band_name,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'correlation_direct': r_direct,
                    'pvalue_direct': p_direct,
                    'correlation_max_lagged': max_corr,
                    'optimal_lag_ms': optimal_lag_ms,
                    'optimal_lag_samples': optimal_lag_samples
                })
                
            except Exception as e:
                print(f"    Warning: Error processing {channel} in {band_name}: {e}")
                continue
        
        print(f"    Completed {band_name}: {len([r for r in results_list if r['band'] == band_name])} channel pairs")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(description='Compute audio-EEG correlation metrics')
    parser.add_argument('--subjects', type=int, nargs='+', required=True,
                       help='Subject IDs to process')
    parser.add_argument('--conditions', type=str, nargs='+', required=True,
                       help='Conditions to process (e.g., VIZ AUD MULTI)')
    parser.add_argument('--max-lag-ms', type=float, default=500,
                       help='Maximum lag for cross-correlation in ms (default: 500)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/audio_eeg_correlation)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data'
    audio_dir = data_dir / 'processed'
    
    if args.output_dir is None:
        output_dir = project_dir / 'results' / 'audio_eeg_correlation'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("AUDIO-EEG CORRELATION ANALYSIS")
    print("="*60)
    print(f"Subjects: {args.subjects}")
    print(f"Conditions: {args.conditions}")
    print(f"Max lag: ±{args.max_lag_ms} ms")
    print(f"Output: {output_dir}")
    
    # Process all subject-condition pairs
    all_results = []
    
    for subject_id in args.subjects:
        for condition in args.conditions:
            try:
                results = process_subject_condition(
                    subject_id, condition, data_dir, audio_dir,
                    max_lag_ms=args.max_lag_ms
                )
                all_results.extend(results)
            except Exception as e:
                print(f"  Error processing subject {subject_id}, condition {condition}: {e}")
                continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save results
    output_file = output_dir / 'audio_eeg_correlation_results.csv'
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
