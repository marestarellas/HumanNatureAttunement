"""Welch PSD + band-integrated power for any 1-D signal.

This used to live inside ``modalities/eeg.py`` as ``compute_psd_features``,
hard-wired to EEG channels and EEG bands. The split here exposes the
modality-agnostic primitive (:func:`welch_band_powers`) so the same Welch
routine works for audio envelopes, respiration, HRV time series, etc.

Default frequency bands match the EEG convention used in the project; pass
your own ``bands`` mapping for other modalities.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


#: Default EEG bands used by the project (Hz).
EEG_BANDS: Mapping[str, Tuple[float, float]] = {
    "delta":     (2,  4),
    "theta":     (4,  8),
    "alpha":     (8,  13),
    "low_beta":  (13, 20),
    "high_beta": (20, 30),
    "gamma1":    (30, 50),
}


def welch_band_powers(
    x: np.ndarray,
    fs: float,
    bands: Mapping[str, Tuple[float, float]] = EEG_BANDS,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> dict:
    """Welch PSD + band-integrated absolute and relative power.

    Returns a dict with one ``<band>_abs`` key per band (in PSD-units²·Hz)
    plus one ``<band>_rel`` per band (fraction of total band power, sums
    to 1 across the keys when total > 0).

    Parameters
    ----------
    x : array-like
        1-D signal (one window).
    fs : float
        Sampling rate (Hz).
    bands : mapping name -> (lo_Hz, hi_Hz)
        Frequency bands to integrate. Defaults to :data:`EEG_BANDS`.
    nperseg, noverlap : int, optional
        Welch parameters. Defaults: ``min(len(x), 256)`` and ``nperseg // 2``.

    Returns
    -------
    dict
        ``{"delta_abs": ..., ..., "delta_rel": ..., ...}`` — keys ordered
        as bands' abs first, then bands' rel, matching the legacy CSV
        column order.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        # Degenerate window — return NaNs.
        return {**{f"{b}_abs": float("nan") for b in bands},
                **{f"{b}_rel": float("nan") for b in bands}}

    if nperseg is None:
        nperseg = int(min(x.size, 256))
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    abs_powers = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            abs_powers[f"{name}_abs"] = float(np.trapz(psd[mask], freqs[mask]))
        else:
            abs_powers[f"{name}_abs"] = 0.0

    total = sum(abs_powers.values())
    if total > 0:
        rel_powers = {f"{n}_rel": abs_powers[f"{n}_abs"] / total for n in bands}
    else:
        rel_powers = {f"{n}_rel": 0.0 for n in bands}

    return {**abs_powers, **rel_powers}
