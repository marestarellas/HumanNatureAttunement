"""EEG modality: signal-specific cleaning only.

Modality-agnostic EEG feature extraction (PSD band powers, entropy, fractal
exponents, FOOOF aperiodic) lives in :mod:`HNA.features` — those
helpers work on any 1-D signal, not just EEG.

This module keeps the truly EEG-specific operation: a multi-channel
zero-phase Butterworth bandpass that iterates over the conventional
``EEG-ch*`` columns of a wide-format DataFrame.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import signal


def filter_eeg(
    df: pd.DataFrame,
    sampling_rate: int = 256,
    lowcut: float = 1.0,
    highcut: float = 50.0,
    order: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bandpass-filter every ``EEG-ch*`` column of ``df``.

    Steps:

    1. Detects every column whose name starts with ``"EEG-ch"``.
    2. Linearly interpolates NaNs per channel (so the IIR filter sees a
       finite signal everywhere) and reports the count.
    3. Applies a zero-phase Butterworth bandpass at ``[lowcut, highcut]``.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format table with one column per EEG channel.
    sampling_rate : int
        Sampling rate (Hz). Default 256.
    lowcut, highcut : float
        Butterworth bandpass cutoffs in Hz. Defaults 1–50 Hz.
    order : int
        Butterworth order (default 4).
    verbose : bool
        Print summary lines (count of channels + NaN totals).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the EEG channels filtered. Non-EEG columns
        are carried through untouched.
    """
    df_filtered = df.copy()

    eeg_channels: List[str] = [c for c in df.columns if c.startswith("EEG-ch")]
    if verbose:
        print(f"Filtering {len(eeg_channels)} EEG channels")
        print(f"Bandpass filter: {lowcut}-{highcut} Hz")
    if not eeg_channels:
        return df_filtered

    eeg_data = df[eeg_channels].values

    nan_mask = np.isnan(eeg_data)
    total_nans = int(nan_mask.sum())
    if total_nans > 0:
        if verbose:
            print(f"  Found {total_nans} NaN values, interpolating...")
        eeg_data = eeg_data.copy()
        for i in range(eeg_data.shape[1]):
            ch = eeg_data[:, i]
            nan_idx = np.where(np.isnan(ch))[0]
            if nan_idx.size:
                valid_idx = np.where(~np.isnan(ch))[0]
                if valid_idx.size:
                    eeg_data[nan_idx, i] = np.interp(nan_idx, valid_idx, ch[valid_idx])
                else:
                    eeg_data[:, i] = 0

    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")

    eeg_filtered = np.zeros_like(eeg_data)
    for i in range(len(eeg_channels)):
        eeg_filtered[:, i] = signal.filtfilt(b, a, eeg_data[:, i])

    for i, ch in enumerate(eeg_channels):
        df_filtered[ch] = eeg_filtered[:, i]

    if verbose:
        print("Filtering complete!")
    return df_filtered
