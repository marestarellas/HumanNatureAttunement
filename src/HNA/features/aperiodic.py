"""Aperiodic 1/f spectral parameterization (FOOOF) for any 1-D signal.

Two entry points:

- :func:`fit_aperiodic_psd` — fit FOOOF to a precomputed PSD. Use when you
  already have ``(freqs, psd)`` from somewhere (e.g. a multi-window Welch
  computed once for several analyses).
- :func:`aperiodic_features` — compute Welch internally and run FOOOF on
  it. Convenience for the windowed iterator.

FOOOF is imported lazily so this module stays cheap to import.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


def fit_aperiodic_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float] = (1.0, 50.0),
    max_n_peaks: int = 8,
    aperiodic_mode: str = "fixed",
) -> dict:
    """Fit FOOOF to a precomputed PSD.

    Parameters
    ----------
    freqs, psd : np.ndarray
        Output of :func:`scipy.signal.welch` (linear PSD).
    freq_range : (float, float)
        Frequency range to fit (default 1–50 Hz).
    max_n_peaks : int
        Cap on the number of periodic peaks (default 8).
    aperiodic_mode : str
        ``"fixed"`` (offset + exponent) or ``"knee"`` (offset + knee + exponent).

    Returns
    -------
    dict
        ``aperiodic_offset, aperiodic_exponent, aperiodic_knee, r_squared,
        error, n_peaks, peaks`` (peaks is a list of dicts).
    """
    from fooof import FOOOF  # lazy

    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    out = {
        "aperiodic_offset":   float("nan"),
        "aperiodic_exponent": float("nan"),
        "aperiodic_knee":     float("nan"),
        "r_squared":          float("nan"),
        "error":              float("nan"),
        "n_peaks":            0,
        "peaks":              [],
    }

    try:
        fm = FOOOF(max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode,
                   verbose=False)
        fm.fit(freqs, psd, freq_range)
        ap = fm.aperiodic_params_
        if aperiodic_mode == "fixed":
            out["aperiodic_offset"]   = float(ap[0])
            out["aperiodic_exponent"] = float(ap[1])
        else:  # "knee"
            out["aperiodic_offset"]   = float(ap[0])
            out["aperiodic_knee"]     = float(ap[1])
            out["aperiodic_exponent"] = float(ap[2])

        out["r_squared"] = float(fm.r_squared_)
        out["error"]     = float(fm.error_)
        peaks = fm.peak_params_
        out["n_peaks"] = int(len(peaks))
        out["peaks"] = [
            {"cf": float(p[0]), "pw": float(p[1]), "bw": float(p[2])}
            for p in peaks
        ]
    except Exception:  # noqa: BLE001
        pass

    return out


def aperiodic_features(
    x: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float] = (1.0, 50.0),
    max_n_peaks: int = 8,
    aperiodic_mode: str = "fixed",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> dict:
    """Welch + FOOOF in one shot for a 1-D signal.

    Returns a dict with the same scalar keys as :func:`fit_aperiodic_psd`
    *except* the ``peaks`` list is dropped — only ``n_peaks`` is kept, so
    the result fits cleanly into a long-form CSV. Use
    :func:`fit_aperiodic_psd` directly if you need per-peak details.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return {
            "aperiodic_offset":   float("nan"),
            "aperiodic_exponent": float("nan"),
            "aperiodic_knee":     float("nan"),
            "r_squared":          float("nan"),
            "error":              float("nan"),
            "n_peaks":            0,
        }

    if nperseg is None:
        nperseg = int(min(x.size, 256))
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    res = fit_aperiodic_psd(freqs, psd, freq_range=freq_range,
                             max_n_peaks=max_n_peaks,
                             aperiodic_mode=aperiodic_mode)
    # Drop the per-peak list; long-form CSVs only carry scalar columns.
    res.pop("peaks", None)
    return res
