"""Welch-spectrum overlay helper for multi-condition / multi-signal comparisons."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal


def spectrum_overlay(
    ax: plt.Axes,
    signals: Mapping[str, np.ndarray],
    fs: float,
    fmax: Optional[float] = None,
    nperseg: Optional[int] = None,
    db_scale: bool = True,
    colors: Optional[Mapping[str, str]] = None,
    show_iqr: bool = False,
    title: Optional[str] = None,
) -> plt.Axes:
    """Draw a Welch power-spectrum overlay onto ``ax``.

    Parameters
    ----------
    ax : matplotlib Axes
    signals : mapping ``label -> 1-D array`` (or ``label -> (n_segments, n)`` for
        multi-segment cases)
    fs : float
        Sample rate (Hz).
    fmax : float, optional
        Upper x-axis limit. Defaults to ``fs / 2``.
    nperseg : int, optional
        Welch segment length in samples. Defaults to ``min(8*fs, len(x))``.
    db_scale : bool
        If True (default), plot ``10*log10(PSD)``.
    colors : mapping label -> color, optional
        Per-label color override. Falls back to the matplotlib cycle.
    show_iqr : bool
        If a value in ``signals`` is 2-D ``(n_segments, n)``, draw the
        median spectrum and a shaded IQR band rather than a single PSD.
    title : str, optional

    Returns
    -------
    The same ``ax``.
    """
    if fmax is None:
        fmax = fs / 2

    for label, sig in signals.items():
        sig = np.asarray(sig, dtype=float)
        n_default = int(min(8 * fs, sig.shape[-1]))
        nps = nperseg if nperseg is not None else max(64, n_default)
        c = colors.get(label) if colors else None

        if sig.ndim == 1 or not show_iqr:
            x = sig if sig.ndim == 1 else sig.mean(axis=0)
            f, p = sp_signal.welch(x, fs=fs, nperseg=min(nps, x.size))
            y = 10 * np.log10(p + 1e-30) if db_scale else p
            ax.plot(f, y, label=label, color=c, lw=1.3)
        else:
            psds = []
            for seg in sig:
                f, p = sp_signal.welch(seg, fs=fs, nperseg=min(nps, seg.size))
                psds.append(p)
            psds = np.asarray(psds)
            med = np.median(psds, axis=0)
            q1, q3 = np.percentile(psds, [25, 75], axis=0)
            y_med = 10 * np.log10(med + 1e-30) if db_scale else med
            y_q1 = 10 * np.log10(q1 + 1e-30) if db_scale else q1
            y_q3 = 10 * np.log10(q3 + 1e-30) if db_scale else q3
            ax.fill_between(f, y_q1, y_q3, alpha=0.20, color=c, linewidth=0)
            ax.plot(f, y_med, label=label, color=c, lw=1.4)

    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)" if db_scale else "PSD")
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best", frameon=False)
    return ax
