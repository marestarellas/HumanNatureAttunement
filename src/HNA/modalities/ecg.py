"""ECG modality: ECG cleaning, R-peak detection, and HRV time-series helpers.

HRV is grouped here (rather than in a separate module) because every HRV
metric is *derived* from the ECG-cleaning + R-peak step that lives at the
top of this file — keeping them together avoids the temptation to drift
the two stages apart.

Public API
----------
ECG-level
~~~~~~~~~
- :func:`preprocess_ecg_segment` — clean a raw ECG segment with NeuroKit2
  and return ``(cleaned_ecg, rpeaks)``.

HRV time-series (windowed)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- :func:`compute_rolling_hrv_features` — slide a window over a cleaned ECG
  + R-peak set and emit a :class:`pandas.DataFrame` of HRV features per
  window (NeuroKit2's ``nk.hrv``).

HRV ↔ audio alignment helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :func:`interpolate_hrv_to_regular_grid` — resample a windowed HRV column
  onto a regular time grid (typically 4 Hz).
- :func:`match_audio_to_hrv` — interpolate an audio envelope onto an HRV
  time grid for downstream coupling.

All of the above are imported by the analysis and feature-extraction
scripts so the same code path is used everywhere.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


DEFAULT_FS = 256.0
DEFAULT_HRV_FS_TARGET = 4.0  # Hz; standard for slow cardiac modulation


# ----------------------------------------------------------------------
# ECG cleaning + R-peak detection
# ----------------------------------------------------------------------
def preprocess_ecg_segment(
    ecg_signal: np.ndarray,
    fs: float = DEFAULT_FS,
    method: str = "neurokit",
    verbose: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Clean ECG and detect R-peaks via NeuroKit2.

    Parameters
    ----------
    ecg_signal : array-like
        Raw single-channel ECG.
    fs : float
        Sampling rate in Hz.
    method : str
        Cleaning / peak-finding method passed through to NeuroKit2
        (default ``"neurokit"``).
    verbose : bool
        Print a short summary on success.

    Returns
    -------
    (cleaned_ecg, rpeaks) : tuple of np.ndarray or (None, None) on failure
        ``cleaned_ecg`` is the bandpassed/decoupled ECG at the input rate;
        ``rpeaks`` is an integer index array of detected R-peaks.
    """
    import neurokit2 as nk  # local import: optional dependency at module load

    try:
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs, method=method)
        _, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs, method=method)
        rpeaks = info["ECG_R_Peaks"]
        if verbose:
            dur = len(ecg_signal) / fs
            bpm = len(rpeaks) / dur * 60 if dur > 0 else float("nan")
            print(f"    Detected {len(rpeaks)} R-peaks in {dur:.1f}s ({bpm:.1f} bpm avg)")
        return cleaned_ecg, rpeaks
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"    ERROR in ECG preprocessing: {e}")
        return None, None


# ----------------------------------------------------------------------
# Windowed HRV features
# ----------------------------------------------------------------------
def compute_rolling_hrv_features(
    ecg_clean: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
    win_sec: float,
    overlap: float,
    min_peaks: int = 5,
) -> Optional[pd.DataFrame]:
    """Compute HRV features in a sliding window.

    Parameters
    ----------
    ecg_clean : np.ndarray
        Cleaned ECG signal (used only for length; HRV is computed from
        ``rpeaks``).
    rpeaks : np.ndarray
        Integer R-peak indices into ``ecg_clean``.
    fs : float
        Sampling rate of ``ecg_clean`` / ``rpeaks`` in Hz.
    win_sec : float
        Window length in seconds.
    overlap : float
        Overlap ratio in [0, 1).
    min_peaks : int
        Skip windows with fewer than this many R-peaks (default 5).

    Returns
    -------
    pd.DataFrame or None
        One row per window; columns include ``window_idx``, ``time_start``,
        ``time_end``, ``n_peaks``, plus all ``HRV_*`` columns from NeuroKit2.
        Returns ``None`` if no window had enough peaks.
    """
    import neurokit2 as nk

    win_samples = int(win_sec * fs)
    step_samples = max(1, int(win_samples * (1 - overlap)))

    features_list: list[dict] = []
    for start_idx in range(0, len(ecg_clean) - win_samples + 1, step_samples):
        end_idx = start_idx + win_samples
        window_rpeaks = rpeaks[(rpeaks >= start_idx) & (rpeaks < end_idx)]
        if len(window_rpeaks) < min_peaks:
            continue

        window_rpeaks_rel = window_rpeaks - start_idx
        try:
            hrv_features = nk.hrv(window_rpeaks_rel, sampling_rate=fs, show=False)
            feature_dict = hrv_features.iloc[0].to_dict()
            feature_dict["window_idx"] = len(features_list)
            feature_dict["time_start"] = start_idx / fs
            feature_dict["time_end"] = end_idx / fs
            feature_dict["n_peaks"] = len(window_rpeaks)
            features_list.append(feature_dict)
        except Exception:  # noqa: BLE001
            if features_list:
                # Pad with NaNs so downstream column ordering is preserved.
                nan_dict = {k: np.nan for k in features_list[-1].keys()}
                nan_dict["window_idx"] = len(features_list)
                nan_dict["time_start"] = start_idx / fs
                nan_dict["time_end"] = end_idx / fs
                nan_dict["n_peaks"] = len(window_rpeaks)
                features_list.append(nan_dict)

    if not features_list:
        return None

    features_df = pd.DataFrame(features_list)
    meta_cols = ["window_idx", "time_start", "time_end", "n_peaks"]
    other_cols = [c for c in features_df.columns if c not in meta_cols]
    return features_df[meta_cols + other_cols]


# ----------------------------------------------------------------------
# HRV ↔ audio alignment
# ----------------------------------------------------------------------
def interpolate_hrv_to_regular_grid(
    hrv_df: pd.DataFrame,
    feature_name: str,
    fs_target: float = DEFAULT_HRV_FS_TARGET,
    valid_ratio_min: float = 0.5,
    clip_margin: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a windowed HRV feature onto a regular time grid.

    Parameters
    ----------
    hrv_df : pd.DataFrame
        Output of :func:`compute_rolling_hrv_features` (must contain
        ``time_start``, ``time_end``, and ``feature_name``).
    feature_name : str
        Column to interpolate (e.g. ``"HRV_RMSSD"``).
    fs_target : float
        Target sampling rate in Hz (default 4 Hz).
    valid_ratio_min : float
        Minimum fraction of finite samples required (raises if below).
    clip_margin : float
        Extrapolated values are clipped to ``[min - clip_margin*range,
        max + clip_margin*range]`` to keep edge artefacts bounded.

    Returns
    -------
    (time_grid, hrv_interp) : tuple of np.ndarray
        Regular time grid (s) and interpolated HRV values.

    Raises
    ------
    ValueError
        If too few valid points or low valid ratio.
    """
    time_centers = (hrv_df["time_start"].values + hrv_df["time_end"].values) / 2
    hrv_values = hrv_df[feature_name].values

    valid_mask = np.isfinite(hrv_values)
    time_centers = time_centers[valid_mask]
    hrv_values = hrv_values[valid_mask]

    if len(time_centers) < 2:
        raise ValueError(
            f"Not enough valid {feature_name} values "
            f"(only {len(time_centers)} valid points)"
        )
    valid_ratio = valid_mask.sum() / len(valid_mask)
    if valid_ratio < valid_ratio_min:
        raise ValueError(
            f"Too many NaN/Inf in {feature_name} ({valid_ratio*100:.1f}% valid data)"
        )

    t_start = time_centers[0]
    t_end = time_centers[-1]
    time_grid = np.arange(t_start, t_end, 1.0 / fs_target)

    interp_func = interp1d(
        time_centers, hrv_values, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )
    hrv_interp = interp_func(time_grid)

    orig_min, orig_max = float(np.nanmin(hrv_values)), float(np.nanmax(hrv_values))
    margin = (orig_max - orig_min) * clip_margin
    hrv_interp = np.clip(hrv_interp, orig_min - margin, orig_max + margin)

    return time_grid, hrv_interp


def match_audio_to_hrv(
    audio_env: np.ndarray,
    audio_time: np.ndarray,
    hrv_time: np.ndarray,
    hrv_fs: Optional[float] = None,
) -> np.ndarray:
    """Interpolate an audio envelope onto an HRV time grid.

    Parameters
    ----------
    audio_env : np.ndarray
        Audio envelope samples (1-D).
    audio_time : np.ndarray
        Time vector for ``audio_env`` (seconds, monotonically increasing).
    hrv_time : np.ndarray
        Target HRV time grid (seconds).
    hrv_fs : float, optional
        Currently unused; kept for API parity with the original helper in
        ``run_hrv_audio_coupling.py``.

    Returns
    -------
    np.ndarray
        Audio envelope on the HRV time grid, with NaN edges trimmed.
    """
    interp_func = interp1d(
        audio_time, audio_env, kind="linear",
        bounds_error=False, fill_value=np.nan,
    )
    audio_matched = interp_func(hrv_time)
    valid = ~np.isnan(audio_matched)
    return audio_matched[valid]
