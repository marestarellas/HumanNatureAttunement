"""Modality-agnostic feature extraction.

Each submodule exposes thin scalar feature functions that work on any 1-D
signal. The :mod:`HNA.features.windowed` iterator slides any
combination of them across an EEG-like wide DataFrame.

Submodules
----------
- :mod:`.psd`        Welch PSD + band-integrated power.
- :mod:`.entropy`    LZC / permutation / spectral / SVD / sample entropy.
- :mod:`.fractal`    Higuchi / Katz / Petrosian FD, DFA α, Hurst R/S.
- :mod:`.aperiodic`  FOOOF aperiodic offset/exponent/knee + R².
- :mod:`.windowed`   Generic ``windowed_channel_features`` iterator.

For convenience, three legacy-named wrappers are re-exported here so
existing call sites can migrate with a one-line import change:

    # before
    from HNA.eeg import (filter_eeg, compute_psd_features,
                                  compute_entropy_features)

    # after
    from HNA.modalities.eeg import filter_eeg
    from HNA.features import (compute_psd_features,
                                       compute_entropy_features,
                                       compute_fractal_features,
                                       compute_aperiodic_features)
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

from .windowed import windowed_channel_features
from .psd import EEG_BANDS, welch_band_powers
from .entropy import (
    all_entropies,
    lzc, perm_entropy, spectral_entropy, svd_entropy, sample_entropy,
)
from .fractal import (
    all_fractals,
    higuchi_fd, katz_fd, petrosian_fd, dfa_alpha, hurst_rs,
)
from .aperiodic import fit_aperiodic_psd, aperiodic_features


__all__ = [
    # Iterator
    "windowed_channel_features",
    # PSD
    "EEG_BANDS", "welch_band_powers",
    # Entropy
    "all_entropies", "lzc", "perm_entropy", "spectral_entropy",
    "svd_entropy", "sample_entropy",
    # Fractal
    "all_fractals", "higuchi_fd", "katz_fd", "petrosian_fd",
    "dfa_alpha", "hurst_rs",
    # Aperiodic
    "fit_aperiodic_psd", "aperiodic_features",
    # Convenience wrappers (legacy column-order DataFrames)
    "compute_psd_features", "compute_entropy_features",
    "compute_fractal_features", "compute_aperiodic_features",
]


# ----------------------------------------------------------------------
# Convenience wrappers — preserve legacy CSV column order
# ----------------------------------------------------------------------
def _eeg_chans(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("EEG-ch")]


def compute_psd_features(
    df_condition: pd.DataFrame,
    sampling_rate: int = 256,
    window_sec: float = 5,
    overlap_sec: float = 2,
    bands: Mapping[str, Tuple[float, float]] = EEG_BANDS,
    channel_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-(channel, window) PSD band powers.

    Drop-in replacement for the legacy ``compute_psd_features`` that lived
    in ``modalities/eeg.py`` — same output column order so downstream
    CSVs keep working.
    """
    chans = list(channel_columns) if channel_columns is not None else _eeg_chans(df_condition)
    return windowed_channel_features(
        df_condition, chans,
        fs=sampling_rate, win_sec=window_sec, overlap_sec=overlap_sec,
        feature_fn=lambda x: welch_band_powers(x, fs=sampling_rate, bands=bands),
    )


def compute_entropy_features(
    df_condition: pd.DataFrame,
    sampling_rate: int = 256,
    window_sec: float = 5,
    overlap_sec: float = 2,
    channel_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-(channel, window) entropy feature set.

    Drop-in replacement for the legacy ``compute_entropy_features``.
    """
    chans = list(channel_columns) if channel_columns is not None else _eeg_chans(df_condition)
    return windowed_channel_features(
        df_condition, chans,
        fs=sampling_rate, win_sec=window_sec, overlap_sec=overlap_sec,
        feature_fn=lambda x: all_entropies(x, fs=sampling_rate),
    )


def compute_fractal_features(
    df_condition: pd.DataFrame,
    sampling_rate: int = 256,
    window_sec: float = 5,
    overlap_sec: float = 2,
    higuchi_kmax: int = 10,
    channel_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-(channel, window) fractal/scaling feature set."""
    chans = list(channel_columns) if channel_columns is not None else _eeg_chans(df_condition)
    return windowed_channel_features(
        df_condition, chans,
        fs=sampling_rate, win_sec=window_sec, overlap_sec=overlap_sec,
        feature_fn=lambda x: all_fractals(x, kmax=higuchi_kmax),
    )


def compute_aperiodic_features(
    df_condition: pd.DataFrame,
    sampling_rate: int = 256,
    window_sec: float = 5,
    overlap_sec: float = 2,
    freq_range: Tuple[float, float] = (1.0, 50.0),
    aperiodic_mode: str = "fixed",
    max_n_peaks: int = 8,
    channel_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-(channel, window) FOOOF aperiodic feature set."""
    chans = list(channel_columns) if channel_columns is not None else _eeg_chans(df_condition)
    return windowed_channel_features(
        df_condition, chans,
        fs=sampling_rate, win_sec=window_sec, overlap_sec=overlap_sec,
        feature_fn=lambda x: aperiodic_features(
            x, fs=sampling_rate, freq_range=freq_range,
            max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode,
        ),
    )
