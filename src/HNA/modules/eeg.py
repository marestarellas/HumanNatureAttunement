from scipy import signal
import numpy as np
import pandas as pd
import antropy as ant

def filter_eeg(df, sampling_rate=256, lowcut=1.0, highcut=50):
    """
    Simple EEG filtering without ICA artifact removal.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with EEG channels
    sampling_rate : int
        Sampling rate in Hz (default: 256)
    lowcut : float
        Low cutoff frequency for bandpass filter (default: 1.0 Hz)
    highcut : float
        High cutoff frequency for bandpass filter (default: 50 Hz)
    
    Returns
    -------
    df_filtered : pd.DataFrame
        Dataframe with bandpass filtered EEG channels
    """
    
    df_filtered = df.copy()
    
    # Get EEG channel names
    eeg_channels = [col for col in df.columns if col.startswith('EEG-ch')]
    print(f"Filtering {len(eeg_channels)} EEG channels")
    print(f"Bandpass filter: {lowcut}-{highcut} Hz")
    
    # Extract EEG data
    eeg_data = df[eeg_channels].values
    
    # Handle NaN values
    nan_mask = np.isnan(eeg_data)
    total_nans = nan_mask.sum()
    
    if total_nans > 0:
        print(f"  Found {total_nans} NaN values, interpolating...")
        eeg_data_clean = eeg_data.copy()
        for i in range(eeg_data.shape[1]):
            channel_data = eeg_data[:, i]
            nan_indices = np.where(np.isnan(channel_data))[0]
            if len(nan_indices) > 0:
                valid_indices = np.where(~np.isnan(channel_data))[0]
                if len(valid_indices) > 0:
                    eeg_data_clean[nan_indices, i] = np.interp(nan_indices, valid_indices, channel_data[valid_indices])
                else:
                    eeg_data_clean[:, i] = 0
        eeg_data = eeg_data_clean
    
    # Design bandpass filter
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter to each channel
    eeg_filtered = np.zeros_like(eeg_data)
    for i in range(len(eeg_channels)):
        eeg_filtered[:, i] = signal.filtfilt(b, a, eeg_data[:, i])
    
    # Update dataframe
    for i, ch in enumerate(eeg_channels):
        df_filtered[ch] = eeg_filtered[:, i]
    
    print("Filtering complete!")
    return df_filtered

from scipy import signal as sp_signal

def compute_psd_features(df_condition, sampling_rate=256, window_sec=5, overlap_sec=2):
    """
    Compute power spectral density (PSD) features for EEG data across time.
    
    Parameters
    ----------
    df_condition : pd.DataFrame
        Dataframe containing EEG data for one condition
    sampling_rate : int
        Sampling rate in Hz (default: 256)
    window_sec : float
        Window length in seconds (default: 5)
    overlap_sec : float
        Overlap between windows in seconds (default: 2)
    
    Returns
    -------
    features_df : pd.DataFrame
        Dataframe with PSD features for each window and channel
        Columns: channel, window_idx, time_start, time_end,
                 delta_abs, theta_abs, alpha_abs, low_beta_abs, high_beta_abs, gamma1_abs,
                 delta_rel, theta_rel, alpha_rel, low_beta_rel, high_beta_rel, gamma1_rel
    """
    
    # Define frequency bands (in Hz)
    bands = {
        'delta': (2, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'gamma1': (30, 50)
    }
    
    # Get EEG channels
    eeg_channels = [col for col in df_condition.columns if col.startswith('EEG-ch')]
    print(f"Computing PSD features for {len(eeg_channels)} channels")
    print(f"Window: {window_sec}s, Overlap: {overlap_sec}s")
    
    # Convert to samples
    window_samples = int(window_sec * sampling_rate)
    overlap_samples = int(overlap_sec * sampling_rate)
    step_samples = window_samples - overlap_samples
    
    # Extract EEG data
    eeg_data = df_condition[eeg_channels].values  # Shape: (n_samples, n_channels)
    n_samples, n_channels = eeg_data.shape
    
    # Calculate number of windows
    n_windows = (n_samples - window_samples) // step_samples + 1
    print(f"Total windows: {n_windows}")
    
    # Storage for features
    features_list = []
    
    # Process each window
    for win_idx in range(n_windows):
        start_idx = win_idx * step_samples
        end_idx = start_idx + window_samples
        
        if end_idx > n_samples:
            break
        
        # Time stamps
        time_start = start_idx / sampling_rate
        time_end = end_idx / sampling_rate
        
        # Process each channel
        for ch_idx, ch_name in enumerate(eeg_channels):
            window_data = eeg_data[start_idx:end_idx, ch_idx]
            
            # Skip if window contains NaN
            if np.any(np.isnan(window_data)):
                continue
            
            # Compute PSD using Welch's method
            freqs, psd = sp_signal.welch(
                window_data,
                fs=sampling_rate,
                nperseg=min(window_samples, 256),  # Use smaller nperseg for better freq resolution
                noverlap=min(window_samples, 256) // 2
            )
            
            # Compute absolute power for each band
            band_powers = {}
            for band_name, (low_freq, high_freq) in bands.items():
                # Find frequency indices for this band
                freq_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                
                # Integrate PSD over frequency band (trapezoidal rule)
                if len(freq_idx) > 0:
                    band_power = np.trapz(psd[freq_idx], freqs[freq_idx])
                else:
                    band_power = 0.0
                
                band_powers[f'{band_name}_abs'] = band_power
            
            # Compute total power for relative power calculation
            total_power = sum(band_powers.values())
            
            # Compute relative power
            if total_power > 0:
                for band_name in bands.keys():
                    band_powers[f'{band_name}_rel'] = band_powers[f'{band_name}_abs'] / total_power
            else:
                for band_name in bands.keys():
                    band_powers[f'{band_name}_rel'] = 0.0
            
            # Create feature row
            feature_row = {
                'channel': ch_name,
                'window_idx': win_idx,
                'time_start': time_start,
                'time_end': time_end,
                **band_powers
            }
            
            features_list.append(feature_row)
    
    # Create dataframe
    features_df = pd.DataFrame(features_list)
    
    print(f"Extracted {len(features_df)} feature rows ({len(features_df) // len(eeg_channels)} windows × {len(eeg_channels)} channels)")
    
    return features_df


def compute_entropy_features(df_condition, sampling_rate=256, window_sec=5, overlap_sec=2):
    """
    Compute entropy features for EEG data across time using antropy.
    Includes: LZC, Permutation entropy, Spectral entropy, SVD entropy, and Sample entropy.
    
    Parameters
    ----------
    df_condition : pd.DataFrame
        Dataframe containing EEG data for one condition
    sampling_rate : int
        Sampling rate in Hz (default: 256)
    window_sec : float
        Window length in seconds (default: 5)
    overlap_sec : float
        Overlap between windows in seconds (default: 2)
    
    Returns
    -------
    features_df : pd.DataFrame
        Dataframe with entropy features for each window and channel
        Columns: channel, window_idx, time_start, time_end, 
                 lzc, perm_entropy, spectral_entropy, svd_entropy, sample_entropy
    """
    
    # Get EEG channels
    eeg_channels = [col for col in df_condition.columns if col.startswith('EEG-ch')]
    print(f"Computing entropy features for {len(eeg_channels)} channels")
    print(f"Window: {window_sec}s, Overlap: {overlap_sec}s")
    
    # Convert to samples
    window_samples = int(window_sec * sampling_rate)
    overlap_samples = int(overlap_sec * sampling_rate)
    step_samples = window_samples - overlap_samples
    
    # Extract EEG data
    eeg_data = df_condition[eeg_channels].values
    n_samples, n_channels = eeg_data.shape
    
    # Calculate number of windows
    n_windows = (n_samples - window_samples) // step_samples + 1
    print(f"Total windows: {n_windows}")
    
    # Storage for features
    features_list = []
    
    # Process each window
    for win_idx in range(n_windows):
        start_idx = win_idx * step_samples
        end_idx = start_idx + window_samples
        
        if end_idx > n_samples:
            break
        
        # Time stamps
        time_start = start_idx / sampling_rate
        time_end = end_idx / sampling_rate
        
        # Process each channel
        for ch_idx, ch_name in enumerate(eeg_channels):
            window_data = eeg_data[start_idx:end_idx, ch_idx]
            
            # Skip if window contains NaN
            if np.any(np.isnan(window_data)):
                continue
            
            # Compute entropy measures
            try:
                # Lempel-Ziv Complexity (requires binarization)
                median = np.median(window_data)
                binary_seq = ''.join(['1' if x > median else '0' for x in window_data])
                lzc = ant.lziv_complexity(binary_seq, normalize=True)
            except:
                lzc = np.nan
            
            try:
                # Permutation entropy
                perm_ent = ant.perm_entropy(window_data, normalize=True)
            except:
                perm_ent = np.nan
            
            try:
                # Spectral entropy
                spectral_ent = ant.spectral_entropy(window_data, sf=sampling_rate, method='welch', normalize=True)
            except:
                spectral_ent = np.nan
            
            try:
                # Singular value decomposition entropy
                svd_ent = ant.svd_entropy(window_data, normalize=True)
            except:
                svd_ent = np.nan
            
            try:
                # Sample entropy
                samp_ent = ant.sample_entropy(window_data)
            except:
                samp_ent = np.nan
            
            # Create feature row
            feature_row = {
                'channel': ch_name,
                'window_idx': win_idx,
                'time_start': time_start,
                'time_end': time_end,
                'lzc': lzc,
                'perm_entropy': perm_ent,
                'spectral_entropy': spectral_ent,
                'svd_entropy': svd_ent,
                'sample_entropy': samp_ent
            }
            
            features_list.append(feature_row)
    
    # Create dataframe
    features_df = pd.DataFrame(features_list)
    
    print(f"Extracted {len(features_df)} entropy feature rows")
    
    return features_df


def compute_lzc_features(df_condition, sampling_rate=256, window_sec=5, overlap_sec=2, normalize=True):
    """
    Compute Lempel-Ziv Complexity (LZC) for EEG data across time using antropy.
    LZC measures the complexity/randomness of a time series.
    
    NOTE: This function is deprecated. Use compute_entropy_features() instead,
    which computes LZC along with other entropy measures.
    
    Parameters
    ----------
    df_condition : pd.DataFrame
        Dataframe containing EEG data for one condition
    sampling_rate : int
        Sampling rate in Hz (default: 256)
    window_sec : float
        Window length in seconds (default: 5)
    overlap_sec : float
        Overlap between windows in seconds (default: 2)
    normalize : bool
        If True, normalize LZC by sequence length (default: True)
    
    Returns
    -------
    features_df : pd.DataFrame
        Dataframe with LZC features for each window and channel
        Columns: channel, window_idx, time_start, time_end, lzc
    """
    
    print("WARNING: compute_lzc_features() is deprecated. Use compute_entropy_features() instead.")
    
    # Get EEG channels
    eeg_channels = [col for col in df_condition.columns if col.startswith('EEG-ch')]
    print(f"Computing LZC features for {len(eeg_channels)} channels")
    print(f"Window: {window_sec}s, Overlap: {overlap_sec}s")
    
    # Convert to samples
    window_samples = int(window_sec * sampling_rate)
    overlap_samples = int(overlap_sec * sampling_rate)
    step_samples = window_samples - overlap_samples
    
    # Extract EEG data
    eeg_data = df_condition[eeg_channels].values
    n_samples, n_channels = eeg_data.shape
    
    # Calculate number of windows
    n_windows = (n_samples - window_samples) // step_samples + 1
    print(f"Total windows: {n_windows}")
    
    # Storage for features
    features_list = []
    
    # Process each window
    for win_idx in range(n_windows):
        start_idx = win_idx * step_samples
        end_idx = start_idx + window_samples
        
        if end_idx > n_samples:
            break
        
        # Time stamps
        time_start = start_idx / sampling_rate
        time_end = end_idx / sampling_rate
        
        # Process each channel
        for ch_idx, ch_name in enumerate(eeg_channels):
            window_data = eeg_data[start_idx:end_idx, ch_idx]
            
            # Skip if window contains NaN
            if np.any(np.isnan(window_data)):
                continue
            
            # Binarize signal (above/below median)
            median = np.median(window_data)
            binary_seq = ''.join(['1' if x > median else '0' for x in window_data])
            
            # Compute LZC using antropy
            try:
                lzc = ant.lziv_complexity(binary_seq, normalize=normalize)
            except:
                lzc = np.nan
            
            # Create feature row
            feature_row = {
                'channel': ch_name,
                'window_idx': win_idx,
                'time_start': time_start,
                'time_end': time_end,
                'lzc': lzc
            }
            
            features_list.append(feature_row)
    
    # Create dataframe
    features_df = pd.DataFrame(features_list)
    
    print(f"Extracted {len(features_df)} LZC feature rows")
    
    return features_df

