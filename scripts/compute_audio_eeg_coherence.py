"""
Compute spectral coherence between audio envelopes and EEG signals.

This script:
1. Loads condition-specific audio files (cut by conditions)
2. Extracts amplitude envelope using Hilbert transform
3. Bandpass filters envelope into EEG frequency bands
4. Downsamples envelope to match EEG sampling rate (256 Hz)
5. Loads corresponding EEG data for each condition
6. Computes spectral coherence between envelope and each EEG channel
7. Saves coherence results and generates topomaps

Usage:
    python scripts/compute_audio_eeg_coherence.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI
"""

import argparse
import os
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, coherence, butter, filtfilt, resample_poly
from pathlib import Path
import mne
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def extract_audio_envelope(audio, audio_sr, method='hilbert'):
    """
    Extract amplitude envelope from audio signal.
    
    Parameters:
    -----------
    audio : array
        Audio signal
    audio_sr : int
        Audio sampling rate
    method : str
        'hilbert' for Hilbert transform or 'rms' for RMS envelope
    
    Returns:
    --------
    envelope : array
        Amplitude envelope
    """
    if method == 'hilbert':
        # Hilbert transform to get analytic signal
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
    elif method == 'rms':
        # RMS envelope with sliding window
        window_size = int(audio_sr * 0.05)  # 50ms window
        envelope = np.sqrt(np.convolve(audio**2, 
                                       np.ones(window_size)/window_size, 
                                       mode='same'))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return envelope


def bandpass_filter_envelope(envelope, audio_sr, lowcut, highcut, order=4):
    """
    Bandpass filter the envelope to isolate specific frequency band.
    
    Parameters:
    -----------
    envelope : array
        Amplitude envelope
    audio_sr : int
        Sampling rate of envelope
    lowcut : float
        Lower cutoff frequency (Hz)
    highcut : float
        Upper cutoff frequency (Hz)
    order : int
        Filter order
    
    Returns:
    --------
    filtered_envelope : array
        Bandpass filtered envelope
    """
    from scipy.signal import sosfiltfilt, butter
    
    # Use second-order sections (SOS) for better numerical stability
    nyq = 0.5 * audio_sr
    low = lowcut / nyq
    high = highcut / nyq
    
    # Check frequency bounds
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz for SR {audio_sr}")
    
    # Use SOS format for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Apply filter with padding to reduce edge effects
    filtered_envelope = sosfiltfilt(sos, envelope, padlen=min(len(envelope)//4, 500))
    
    return filtered_envelope


def downsample_envelope(envelope, original_sr, target_sr):
    """
    Downsample envelope to match EEG sampling rate.
    
    Parameters:
    -----------
    envelope : array
        Envelope to downsample
    original_sr : int
        Original sampling rate
    target_sr : int
        Target sampling rate (typically 256 Hz for EEG)
    
    Returns:
    --------
    downsampled : array
        Downsampled envelope
    """
    # Calculate decimation factor
    decimation_factor = original_sr / target_sr
    
    # Use polyphase filtering for high-quality resampling
    # resample_poly(signal, up, down) - we want to go down
    up = 1
    down = int(np.round(decimation_factor))
    
    # If the ratio isn't exact, use scipy.signal.resample
    if abs(decimation_factor - down) > 0.01:
        n_samples = int(len(envelope) * target_sr / original_sr)
        downsampled = signal.resample(envelope, n_samples)
    else:
        downsampled = resample_poly(envelope, up, down)
    
    return downsampled


def load_eeg_data(subject_id, condition, data_dir, eeg_sr=256):
    """
    Load EEG data for a specific subject and condition.
    
    Parameters:
    -----------
    subject_id : int
        Subject number
    condition : str
        Condition name (e.g., 'VIZ', 'AUD', 'MULTI')
    data_dir : str
        Base data directory
    eeg_sr : int
        EEG sampling rate
    
    Returns:
    --------
    eeg_data : dict
        Dictionary with channel names as keys and EEG timeseries as values
    """
    # Load merged data to get EEG for the condition
    subject_folder = Path(data_dir) / 'processed' / f'sub-{subject_id:02d}' / 'tables'
    merged_file = subject_folder / 'merged_annotated_with_audio.csv'
    
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_file}")
    
    df = pd.read_csv(merged_file)
    
    # Find condition start and stop markers (works for both 'condition_names' and 'annotation' columns)
    marker_col = None
    if 'condition_names' in df.columns:
        marker_col = 'condition_names'
    elif 'annotation' in df.columns:
        marker_col = 'annotation'
    else:
        raise ValueError("Cannot find 'condition_names' or 'annotation' column in merged file")
    
    # Look for start and stop markers
    start_idx = df[df[marker_col] == f'{condition}_start'].index
    stop_idx = df[df[marker_col] == f'{condition}_stop'].index
    
    if len(start_idx) == 0 or len(stop_idx) == 0:
        # Debug: show what markers are available
        unique_markers = df[marker_col].dropna().unique()
        marker_subset = [m for m in unique_markers if '_start' in str(m) or '_stop' in str(m)]
        raise ValueError(f"Condition {condition} markers not found for subject {subject_id}. "
                        f"Available markers: {marker_subset[:20]}")
    
    start = start_idx[0]
    stop = stop_idx[0]
    df_condition = df.iloc[start:stop].copy()
    
    # Get EEG channel columns (exclude non-EEG columns)
    exclude_cols = ['annotation', 'condition_names', 'time_s', 'audio_sample', 'audio_time_s',
                    'ecg', 'respiration', 'physio_triggers', 'condition_triggers', 
                    'sequence', 'battery', 'flags', 'eeg_triggers',
                    'env_broad', 'env_swell_0p3', 'env_swell_0p1', 'env_splash_1_5']
    
    # Get EEG channels (should start with 'EEG-ch')
    eeg_channels = [col for col in df_condition.columns 
                   if col.startswith('EEG-ch') or (col not in exclude_cols and col not in df_condition.columns[:10])]
    
    if len(eeg_channels) == 0:
        raise ValueError(f"No EEG channels found in merged file for subject {subject_id}")
    
    print(f"  Found {len(eeg_channels)} EEG channels, {len(df_condition)} samples")
    
    # Create dictionary of EEG data
    eeg_data = {}
    for channel in eeg_channels:
        eeg_data[channel] = df_condition[channel].values
    
    return eeg_data


def compute_coherence_spectrum(signal1, signal2, fs, nperseg=256):
    """
    Compute coherence spectrum between two signals.
    
    Parameters:
    -----------
    signal1 : array
        First signal (e.g., audio envelope)
    signal2 : array
        Second signal (e.g., EEG channel)
    fs : int
        Sampling rate
    nperseg : int
        Length of each segment for coherence computation
    
    Returns:
    --------
    freqs : array
        Frequency values
    coh : array
        Coherence values (0-1)
    """
    # Ensure signals have same length
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    # Compute coherence
    freqs, coh = coherence(signal1, signal2, fs=fs, nperseg=nperseg, 
                           noverlap=nperseg//2)
    
    return freqs, coh


def compute_band_coherence(freqs, coh, band_range):
    """
    Average coherence within a frequency band.
    
    Parameters:
    -----------
    freqs : array
        Frequency values from coherence computation
    coh : array
        Coherence values
    band_range : tuple
        (low_freq, high_freq) for the band
    
    Returns:
    --------
    mean_coh : float
        Mean coherence within the band
    """
    low, high = band_range
    band_mask = (freqs >= low) & (freqs <= high)
    if np.sum(band_mask) == 0:
        return np.nan
    mean_coh = np.mean(coh[band_mask])
    return mean_coh


def process_subject_condition(subject_id, condition, data_dir, eeg_sr=256, 
                               frequency_bands=None):
    """
    Process one subject-condition pair: compute coherence between audio envelope and EEG.
    
    Parameters:
    -----------
    subject_id : int
        Subject number
    condition : str
        Condition name
    data_dir : str
        Base data directory
    eeg_sr : int
        EEG sampling rate
    frequency_bands : dict
        Dictionary of frequency bands {name: (low, high)}
    
    Returns:
    --------
    results : pd.DataFrame
        DataFrame with coherence results for each channel and frequency band
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
    
    print(f"\nProcessing subject {subject_id}, condition {condition}...")
    
    # Load cut audio file
    audio_folder = Path(data_dir) / 'processed' / f'sub-{subject_id:02d}' / 'audio'
    audio_file = audio_folder / f'audio_{condition}.wav'
    
    if not audio_file.exists():
        print(f"  Warning: Audio file not found: {audio_file}")
        return None
    
    # Load audio
    audio, audio_sr = sf.read(audio_file)
    if len(audio.shape) > 1:  # If stereo, take first channel
        audio = audio[:, 0]
    
    print(f"  Loaded audio: {len(audio)} samples at {audio_sr} Hz")
    
    # Extract envelope
    print(f"  Extracting amplitude envelope...")
    envelope = extract_audio_envelope(audio, audio_sr, method='hilbert')
    
    # Load EEG data
    print(f"  Loading EEG data...")
    eeg_data = load_eeg_data(subject_id, condition, data_dir, eeg_sr)
    n_channels = len(eeg_data)
    print(f"  Loaded {n_channels} EEG channels")
    
    # Process each frequency band
    results_list = []
    
    for band_name, (lowcut, highcut) in frequency_bands.items():
        print(f"  Processing {band_name} band ({lowcut}-{highcut} Hz)...")
        
        # Check envelope before filtering
        if np.any(np.isnan(envelope)) or np.any(np.isinf(envelope)):
            print(f"    Warning: NaN or Inf in raw envelope, skipping {band_name}")
            continue
        
        # Bandpass filter the envelope
        try:
            filtered_envelope = bandpass_filter_envelope(envelope, audio_sr, 
                                                         lowcut, highcut, order=4)
            
            # Check for NaN after filtering
            if np.any(np.isnan(filtered_envelope)) or np.any(np.isinf(filtered_envelope)):
                print(f"    Warning: NaN or Inf after filtering envelope for {band_name}, skipping")
                continue
                
        except Exception as e:
            print(f"    Warning: Error filtering envelope for {band_name}: {e}")
            continue
        
        # Downsample envelope to EEG sampling rate
        downsampled_envelope = downsample_envelope(filtered_envelope, audio_sr, eeg_sr)
        
        # Check after downsampling
        if np.any(np.isnan(downsampled_envelope)) or np.any(np.isinf(downsampled_envelope)):
            print(f"    Warning: NaN or Inf after downsampling for {band_name}, skipping")
            continue
        
        # Compute coherence with each EEG channel
        for channel, eeg_signal in eeg_data.items():
            try:
                # Ensure signals are same length
                min_len = min(len(downsampled_envelope), len(eeg_signal))
                env_trimmed = downsampled_envelope[:min_len]
                eeg_trimmed = eeg_signal[:min_len]
                
                # Check for NaN or inf
                if np.any(np.isnan(env_trimmed)) or np.any(np.isnan(eeg_trimmed)):
                    print(f"    Warning: NaN in {channel} or envelope for {band_name}, skipping")
                    continue
                if np.any(np.isinf(env_trimmed)) or np.any(np.isinf(eeg_trimmed)):
                    print(f"    Warning: Inf in {channel} or envelope for {band_name}, skipping")
                    continue
                
                # Check for zero or very low variance (bad channels)
                eeg_std = np.std(eeg_trimmed)
                env_std = np.std(env_trimmed)
                if eeg_std < 1e-10:
                    print(f"    Warning: Zero variance in {channel} for {band_name} (std={eeg_std:.2e}), skipping")
                    continue
                if env_std < 1e-10:
                    print(f"    Warning: Zero variance in envelope for {band_name} (std={env_std:.2e}), skipping")
                    continue
                
                # Compute coherence spectrum
                freqs, coh = compute_coherence_spectrum(env_trimmed, 
                                                       eeg_trimmed, 
                                                       fs=eeg_sr, 
                                                       nperseg=256)
                
                # Check if coherence was computed
                if len(coh) == 0 or np.all(np.isnan(coh)):
                    print(f"    Warning: Empty coherence for {channel} in {band_name}")
                    print(f"      Signal lengths: env={len(env_trimmed)}, eeg={len(eeg_trimmed)}")
                    print(f"      EEG stats: mean={np.mean(eeg_trimmed):.3f}, std={np.std(eeg_trimmed):.3f}, range=[{np.min(eeg_trimmed):.3f}, {np.max(eeg_trimmed):.3f}]")
                    print(f"      Envelope stats: mean={np.mean(env_trimmed):.6f}, std={np.std(env_trimmed):.6f}, range=[{np.min(env_trimmed):.6f}, {np.max(env_trimmed):.6f}]")
                    print(f"      Coherence result: len(coh)={len(coh)}, all_nan={np.all(np.isnan(coh)) if len(coh) > 0 else 'N/A'}")
                    continue
                
                # Average coherence within the band
                mean_coh = compute_band_coherence(freqs, coh, (lowcut, highcut))
                
                # Also compute peak coherence in the band
                band_mask = (freqs >= lowcut) & (freqs <= highcut)
                if np.sum(band_mask) > 0:
                    peak_coh = np.max(coh[band_mask])
                    peak_freq = freqs[band_mask][np.argmax(coh[band_mask])]
                else:
                    peak_coh = np.nan
                    peak_freq = np.nan
                
                # Store results
                results_list.append({
                    'subject_id': subject_id,
                    'condition': condition,
                    'channel': channel,
                    'band': band_name,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'mean_coherence': mean_coh,
                    'peak_coherence': peak_coh,
                    'peak_frequency': peak_freq
                })
                
            except Exception as e:
                print(f"    Warning: Error processing {channel} in {band_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(results_list) == 0:
        return None
    
    results_df = pd.DataFrame(results_list)
    print(f"  Completed: {len(results_df)} channel-band pairs")
    
    return results_df


def plot_coherence_topomap(results_df, condition, band, metric='mean_coherence',
                           output_dir=None, alpha=0.05, vmin=None, vmax=None):
    """
    Plot topomap of coherence values for one condition and frequency band.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with coherence values
    condition : str
        Condition to plot
    band : str
        Frequency band to plot
    metric : str
        Which coherence metric to plot ('mean_coherence' or 'peak_coherence')
    output_dir : str or None
        Directory to save plot
    alpha : float
        Threshold for significance (not used here, placeholder for future stats)
    vmin, vmax : float or None
        Colormap limits
    """
    # Filter data
    data = results_df[(results_df['condition'] == condition) & 
                      (results_df['band'] == band)].copy()
    
    if len(data) == 0:
        print(f"No data for {condition} - {band}")
        return
    
    # Get unique channels and average values (in case of duplicates)
    data = data.groupby('channel')[metric].mean().reset_index()
    channels = data['channel'].values
    values = data[metric].values
    
    # Standard 32 channel names for MNE
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    # Map to standard channel names
    n_channels = len(channels)
    mne_ch_names = standard_32_channels[:n_channels]
    
    # Create MNE info for standard 1020 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=mne_ch_names, 
                          sfreq=256, 
                          ch_types='eeg')
    info.set_montage(montage)
    
    # Get positions
    pos = np.array([info['chs'][i]['loc'][:2] for i in range(len(channels))])
    
    # Plot topomap
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanmax(values)
    
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False,
                                  cmap='Reds', vmin=vmin, vmax=vmax,
                                  contours=6, sensors=True)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Coherence', rotation=270, labelpad=20)
    
    # Title
    metric_name = 'Mean' if metric == 'mean_coherence' else 'Peak'
    ax.set_title(f'{condition} - {band.upper()}\n{metric_name} Audio-EEG Coherence')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'coherence_topo_{condition}_{band}_{metric}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir / filename}")
    
    plt.close()


def plot_all_bands_topomap(results_df, condition, metric='mean_coherence',
                           output_dir=None):
    """
    Plot all frequency bands in one figure for a condition.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    condition : str
        Condition to plot
    metric : str
        Which coherence metric to plot
    output_dir : str or None
        Directory to save plot
    """
    bands = sorted(results_df['band'].unique())
    n_bands = len(bands)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Get global vmin/vmax
    vmin = 0
    vmax = results_df[results_df['condition'] == condition][metric].max()
    
    # Standard 32 channel names for MNE
    standard_32_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF8'
    ]
    
    for idx, band in enumerate(bands):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Filter data
        data = results_df[(results_df['condition'] == condition) & 
                          (results_df['band'] == band)].copy()
        
        if len(data) == 0:
            ax.axis('off')
            continue
        
        # Get frequency range for title (before groupby)
        lowcut = data['lowcut'].iloc[0]
        highcut = data['highcut'].iloc[0]
        
        # Get unique channels and values
        data = data.groupby('channel')[metric].mean().reset_index()
        channels = data['channel'].values
        values = data[metric].values
        
        # Map to standard channel names
        n_channels = len(channels)
        mne_ch_names = standard_32_channels[:n_channels]
        
        # Create MNE info
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=mne_ch_names, 
                              sfreq=256, 
                              ch_types='eeg')
        info.set_montage(montage)
        
        # Plot topomap
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False,
                                      cmap='Reds', vlim=(vmin, vmax),
                                      contours=6, sensors=True)
        
        # Set title with frequency range
        ax.set_title(f'{band.upper()}\n({lowcut}-{highcut} Hz)')
    
    # Remove unused axes
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Coherence', rotation=270, labelpad=20)
    
    # Overall title
    metric_name = 'Mean' if metric == 'mean_coherence' else 'Peak'
    fig.suptitle(f'{condition} - {metric_name} Audio-EEG Coherence\nAll Frequency Bands', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'coherence_all_bands_{condition}_{metric}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / filename}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute audio-EEG coherence')
    parser.add_argument('--subjects', nargs='+', type=int, default=[2, 3, 4, 5, 6],
                       help='Subject IDs to process')
    parser.add_argument('--conditions', nargs='+', default=['VIZ', 'AUD', 'MULTI'],
                       help='Conditions to process')
    parser.add_argument('--data-dir', default='data',
                       help='Base data directory')
    parser.add_argument('--output-dir', default='results/audio_eeg_coherence',
                       help='Output directory for results')
    parser.add_argument('--eeg-sr', type=int, default=256,
                       help='EEG sampling rate')
    parser.add_argument('--plot-individual', action='store_true',
                       help='Plot individual band topomaps (in addition to combined)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define frequency bands
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma1': (30, 50)
    }
    
    # Process all subjects and conditions
    all_results = []
    
    for subject_id in args.subjects:
        for condition in args.conditions:
            try:
                results_df = process_subject_condition(
                    subject_id, condition, args.data_dir, 
                    eeg_sr=args.eeg_sr,
                    frequency_bands=frequency_bands
                )
                
                if results_df is not None:
                    all_results.append(results_df)
                    
            except Exception as e:
                print(f"Error processing subject {subject_id}, condition {condition}: {e}")
                continue
    
    if len(all_results) == 0:
        print("No results to save!")
        return
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    results_file = output_dir / 'audio_eeg_coherence_results.csv'
    combined_results.to_csv(results_file, index=False)
    print(f"\nSaved coherence results to: {results_file}")
    print(f"Total records: {len(combined_results)}")
    
    # Create visualizations
    print("\nGenerating topomaps...")
    
    # Plot combined topomaps for each condition
    for condition in args.conditions:
        print(f"\nPlotting {condition}...")
        plot_all_bands_topomap(combined_results, condition, 
                               metric='mean_coherence',
                               output_dir=output_dir)
        
        # Individual band plots if requested
        if args.plot_individual:
            for band in frequency_bands.keys():
                plot_coherence_topomap(combined_results, condition, band,
                                      metric='mean_coherence',
                                      output_dir=output_dir)
    
    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
