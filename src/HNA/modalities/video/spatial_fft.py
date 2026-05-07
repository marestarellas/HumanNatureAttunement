"""
Per-frame 2-D Fourier descriptors of the visual scene: peak radial
wavenumber, horizontal/vertical anisotropy, and weighted-mean orientation.

These complement the radial-PSD slope (a single scalar that collapses the
2-D spectrum to one number) with three richer per-frame scalars that
exploit the full 2-D spectrum. All outputs are 1-D signals at frame rate.

Lives in the "whole-image" tier of the framework and the oscillatory /
spatial-pattern feature family.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ._common import iter_frames


def _spatial_fft_descriptors(gray: np.ndarray) -> Tuple[float, float, float]:
    """
    Single-frame 2-D FFT descriptors that complement the radial-PSD slope:

      peak_k        : normalised radial wavenumber of the spectral peak
                      (in [0,1] where 1 is Nyquist), excluding DC.
      anisotropy    : log-ratio of horizontal-stripe to vertical-stripe
                      band power (positive = horizontal swell crests
                      dominate; negative = vertically striped pattern).
      orientation   : dominant orientation of the 2-D spectrum, in radians
                      [0, pi); 0 = horizontal stripes, pi/2 = vertical
                      stripes.
    """
    g = gray.astype(np.float32) - gray.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    P = np.abs(F) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(P.shape)
    ky = (yy - cy)
    kx = (xx - cx)
    r = np.sqrt(kx * kx + ky * ky)
    r_max = float(min(cy, cx))
    if r_max < 4:
        return float("nan"), float("nan"), float("nan")
    mask_ring = (r > 2) & (r < 0.95 * r_max)
    if not mask_ring.any():
        return float("nan"), float("nan"), float("nan")
    Pm = P * mask_ring

    # peak wavenumber
    flat_idx = int(np.argmax(Pm))
    py, px = np.unravel_index(flat_idx, P.shape)
    peak_k = float(np.sqrt((py - cy) ** 2 + (px - cx) ** 2) / r_max)

    # Anisotropy: log-ratio of power in horizontal-stripe-content
    # (Fourier power on the ky axis, angles near +/- pi/2) over vertical-
    # stripe-content (Fourier power on the kx axis, angles near 0 or
    # +/- pi).
    ang = np.arctan2(ky, kx)  # in (-pi, pi]
    wedge = np.deg2rad(15)
    near_vaxis = np.abs(np.abs(ang) - np.pi / 2) < wedge
    near_haxis = (np.abs(ang) < wedge) | (np.abs(np.abs(ang) - np.pi) < wedge)
    band_horiz_stripes = mask_ring & near_vaxis
    band_vert_stripes = mask_ring & near_haxis
    ph = P[band_horiz_stripes].sum() + 1e-9
    pv = P[band_vert_stripes].sum() + 1e-9
    anisotropy = float(np.log(ph / pv))

    # weighted-mean orientation (moment of inertia of P over the ring)
    ang_mod = ang % np.pi  # collapse to [0, pi)
    s2 = float((np.sin(2 * ang_mod) * Pm).sum())
    c2 = float((np.cos(2 * ang_mod) * Pm).sum())
    orientation = 0.5 * float(np.arctan2(s2, c2))
    if orientation < 0:
        orientation += np.pi
    return peak_k, anisotropy, orientation


def extract_spatial_fft_signals(path: str, target_long: int = 192,
                                stride: int = 1,
                                max_frames: Optional[int] = None
                                ) -> Dict[str, np.ndarray]:
    """
    Per-frame 2-D-FFT scalars: peak wavenumber, anisotropy, orientation.
    All are 1-D signals at frame rate.
    """
    pk, ani, ori = [], [], []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        a, b, c = _spatial_fft_descriptors(gray)
        pk.append(a)
        ani.append(b)
        ori.append(c)
    return dict(spatial_fft_peak_k=np.asarray(pk),
                spatial_fft_anisotropy=np.asarray(ani),
                spatial_fft_orientation=np.asarray(ori))


__all__ = [
    "extract_spatial_fft_signals",
    "_spatial_fft_descriptors",
]
