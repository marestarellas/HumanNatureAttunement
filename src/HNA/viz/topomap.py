"""EEG band-topomap helper.

Imports MNE lazily — calling :func:`band_topomap` without MNE installed
raises a clear ``ImportError`` with an install hint, but importing this
module on its own does not.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def band_topomap(
    ax: plt.Axes,
    values: Sequence[float],
    ch_names: Sequence[str],
    montage: str = "standard_1020",
    cmap: str = "RdBu_r",
    vlim: Optional[tuple[float, float]] = None,
    show_sig_mask: Optional[Sequence[bool]] = None,
    title: Optional[str] = None,
    sensors: bool = True,
) -> plt.Axes:
    """Draw an EEG topomap into ``ax`` from a per-channel value vector.

    Parameters
    ----------
    ax : matplotlib Axes
    values : array-like, shape (n_channels,)
    ch_names : sequence of str
        Standard 10-20 (or 10-10) channel names matching ``values``.
    montage : str
        Standard MNE montage name (default ``"standard_1020"``).
    cmap : str
        Matplotlib colormap (default ``"RdBu_r"``).
    vlim : (float, float), optional
        Color scale limits. Defaults to a symmetric range based on
        ``max(|values|)``.
    show_sig_mask : array-like of bool, shape (n_channels,), optional
        If provided, mark significant channels with a small black dot.
    title : str, optional
    sensors : bool
        Whether to plot sensor locations as dots (default True).

    Returns
    -------
    The same ``ax``.
    """
    try:
        import mne  # noqa: F401
        from mne.viz import plot_topomap
        from mne.channels import make_standard_montage
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "band_topomap() requires the 'mne' package. "
            "Install with: pip install mne"
        ) from e

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != len(ch_names):
        raise ValueError("values must be 1-D with len == len(ch_names)")

    if vlim is None:
        vmax = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
        vlim = (-vmax, vmax)

    mont = make_standard_montage(montage)
    pos = np.array([mont.get_positions()["ch_pos"][c][:2]
                    for c in ch_names])

    mask = np.asarray(show_sig_mask, dtype=bool) if show_sig_mask is not None else None
    mask_params = dict(marker="o", markerfacecolor="black",
                       markeredgecolor="black", markersize=3) if mask is not None else None

    plot_topomap(
        values, pos, axes=ax, show=False, cmap=cmap,
        vlim=vlim, sensors=sensors, mask=mask, mask_params=mask_params,
    )
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold")
    return ax
