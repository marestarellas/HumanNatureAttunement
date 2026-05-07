"""Generic windowed-channel feature iterator.

Replaces the three nearly-identical loops that lived in
:mod:`HNA.modalities.eeg` (PSD, entropy) and the old top-level
``fractal.py`` (fractal scalars, FOOOF aperiodic). One iterator, any
``feature_fn`` that returns a ``dict[str, float]`` per window.

The output schema matches the legacy convention so existing CSVs stay
forward-compatible:

    channel, window_idx, time_start, time_end, <feature_fn keys ...>

Pass ``feature_fn`` as a closure / lambda to bind extra arguments
(sample rate, band table, etc.) — the iterator does not pass them in.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def windowed_channel_features(
    df: pd.DataFrame,
    channels: Sequence[str],
    fs: float,
    win_sec: float,
    overlap_sec: float,
    feature_fn: Callable[[np.ndarray], Dict[str, float]],
    skip_nan: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Slide a window over ``channels`` of ``df`` and apply ``feature_fn``.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format table with one column per channel.
    channels : sequence of str
        Column names to iterate (e.g. ``["EEG-ch01", ..., "EEG-ch32"]`` or a
        single 1-element list to feature-extract on a non-EEG modality).
    fs : float
        Sampling rate (Hz).
    win_sec, overlap_sec : float
        Window length and overlap (both in seconds).
    feature_fn : callable(x: np.ndarray) -> dict[str, float]
        Computes one row of features from a 1-D window. Use a closure to
        bind extra arguments::

            feature_fn=lambda x: welch_band_powers(x, fs=256, bands=EEG_BANDS)

    skip_nan : bool
        If True (default), windows containing any NaN are skipped silently.
    progress : bool
        If True, prints "win i/N done" lines (useful for long jobs).

    Returns
    -------
    pd.DataFrame
        Long-format with columns ``channel, window_idx, time_start,
        time_end`` plus whatever keys ``feature_fn`` returns.
    """
    channels = list(channels)
    win_samples = int(win_sec * fs)
    step_samples = max(1, win_samples - int(overlap_sec * fs))
    if win_samples <= 0:
        raise ValueError("win_sec must be > 0")

    data = df[channels].values  # shape (n_samples, n_channels)
    n_samples = data.shape[0]

    rows = []
    starts = list(range(0, n_samples - win_samples + 1, step_samples))
    for win_idx, start in enumerate(starts):
        end = start + win_samples
        for ch_idx, ch_name in enumerate(channels):
            seg = data[start:end, ch_idx]
            if skip_nan and np.any(np.isnan(seg)):
                continue
            feat = feature_fn(seg)
            row = {
                "channel": ch_name,
                "window_idx": win_idx,
                "time_start": start / fs,
                "time_end": end / fs,
                **feat,
            }
            rows.append(row)
        if progress and (win_idx + 1) % 50 == 0:
            print(f"  windowed_channel_features: {win_idx + 1}/{len(starts)} windows done")

    return pd.DataFrame(rows)
