"""
Spatio-temporal modal decomposition (DMD or POD/SVD fallback).

`extract_modal` fits one Koopman operator to the whole clip and returns
the top-k spatial modes plus their temporal coefficients (one 1-D signal
per mode). `extract_modal_windowed` re-fits DMD inside a sliding window
to track regime changes; the per-window dominant frequency / energy
become time-resolved 1-D signals at frame rate.

Lives in the "per-pixel" tier of the framework (the spatial modes are
full-resolution 2-D images) and the oscillatory feature family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as _sps

from ._common import _stack_video, _PyDMD, _HAVE_PYDMD


@dataclass
class ModalResult:
    method: str                              # "dmd" or "svd"
    fps: float
    frequencies_hz: np.ndarray               # length k
    growth_rates: np.ndarray                 # length k (real part of eigvals)
    energies: np.ndarray                     # length k, normalised [0,1]
    temporal_coeffs: np.ndarray              # (k, n_frames) -- 1-D per mode
    spatial_modes_shape: Tuple[int, int]     # (h, w) of the spatial modes


def extract_modal(path: str, fps: float, target_long: int = 96,
                  stride: int = 1, max_frames: Optional[int] = None,
                  k: int = 4, prefer: str = "dmd") -> ModalResult:
    """
    Top-k spatio-temporal modes via Dynamic Mode Decomposition (PyDMD).

    DMD assumes a linear evolution X_{t+1} = A X_t and returns *complex*
    eigenvalues, so each physical mode has a single frequency -- unlike
    real-valued POD/SVD which pairs sin/cos at the same frequency.

    `prefer`: 'dmd' (default, requires PyDMD) or 'svd' (POD-with-Welch
    fallback; kept for diagnostic comparison).
    """
    X, shape = _stack_video(path, target_long, stride, max_frames)

    if prefer == "dmd":
        if not _HAVE_PYDMD:
            raise ImportError(
                "extract_modal(prefer='dmd') requires PyDMD. "
                "Install with `pip install pydmd`, or pass prefer='svd' "
                "for the POD/SVD fallback (which pairs sin/cos at f_p).")
        dmd = _PyDMD(svd_rank=int(2 * k)).fit(X)
        eigs = np.asarray(dmd.eigs)
        log_lam = np.log(eigs + 1e-30) * float(fps)
        freqs = np.abs(log_lam.imag) / (2 * np.pi)
        growth = log_lam.real
        dyn = np.asarray(dmd.dynamics)               # (k, T) complex
        modes = np.asarray(dmd.modes)                # (P, k) complex
        amps = np.linalg.norm(modes, axis=0) * np.abs(dyn).mean(axis=1)

        # de-duplicate conjugate pairs
        keys = np.round(freqs, 4)
        seen: Dict[float, int] = {}
        for i, key in enumerate(keys):
            cur = seen.get(key)
            if cur is None or amps[i] > amps[cur]:
                seen[key] = i
        keep = np.array(sorted(seen.values()), dtype=int)
        freqs = freqs[keep]
        growth = growth[keep]
        amps = amps[keep]
        dyn = dyn[keep]

        order = np.argsort(-amps)[:k]
        freqs = freqs[order]
        growth = growth[order]
        amps = amps[order]
        dyn = dyn[order]
        energies = amps / (amps.max() + 1e-12) if amps.size else amps
        return ModalResult("dmd", float(fps),
                           freqs.astype(float), growth.astype(float),
                           energies.astype(float),
                           np.real(dyn).astype(np.float64), shape)

    # SVD/POD fallback (paired sin/cos modes at f_p; useful for diagnostics)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Vt = Vt[:k]
    S = S[:k]
    coeffs = (S[:, None] * Vt)
    energies = (S ** 2)
    energies = energies / (energies[0] + 1e-12)
    freqs = np.zeros(k)
    growth = np.zeros(k)
    for i in range(k):
        f, P = _sps.welch(coeffs[i], fs=fps,
                          nperseg=min(len(coeffs[i]), 256))
        if len(f) > 1:
            freqs[i] = float(f[np.argmax(P[1:]) + 1])
    return ModalResult("svd", fps, freqs, growth, energies, coeffs, shape)


def extract_modal_windowed(
    path: str, fps: float,
    win_sec: float = 4.0, step_sec: float = 0.5,
    target_long: int = 64,
    stride: int = 1, max_frames: Optional[int] = None,
    k: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Sliding-window DMD. Returns time-resolved 1-D series at frame rate
    (interpolated from window centres):

      modal_w_top_freq_hz(t)     : f of the most-energetic mode in window
      modal_w_top_energy(t)      : its (normalised) energy share
      modal_w_top2_freq_hz(t)    : second-most-energetic mode freq
      modal_w_top2_energy(t)     : ditto energy
      modal_w_n_modes(t)         : number of distinct modes after
                                    conjugate-pair de-duplication

    Useful for clips where stationarity is suspect (a swell building, a
    wave breaking, regime changes). Requires PyDMD.
    """
    if not _HAVE_PYDMD:
        raise ImportError("extract_modal_windowed requires PyDMD "
                          "(`pip install pydmd`).")
    X, _shape = _stack_video(path, target_long, stride, max_frames)
    P, T = X.shape
    fs = float(fps)
    W = max(8, int(round(win_sec * fs)))
    H = max(1, int(round(step_sec * fs)))
    if W > T:
        raise RuntimeError(f"Window {win_sec}s ({W} samples) > clip length "
                           f"({T} samples).")
    starts = np.arange(0, T - W + 1, H)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    centers = starts + W // 2

    n_w = starts.size
    top_f = np.full(n_w, np.nan, dtype=np.float32)
    top_e = np.full(n_w, np.nan, dtype=np.float32)
    top2_f = np.full(n_w, np.nan, dtype=np.float32)
    top2_e = np.full(n_w, np.nan, dtype=np.float32)
    n_modes = np.zeros(n_w, dtype=np.float32)

    for wi, st in enumerate(starts):
        Xw = X[:, st:st + W]
        try:
            dmd = _PyDMD(svd_rank=int(2 * k)).fit(Xw)
            eigs = np.asarray(dmd.eigs)
            if eigs.size == 0:
                continue
            log_lam = np.log(eigs + 1e-30) * fs
            freqs_w = np.abs(log_lam.imag) / (2 * np.pi)
            dyn = np.asarray(dmd.dynamics)
            modes = np.asarray(dmd.modes)
            amps = np.linalg.norm(modes, axis=0) * np.abs(dyn).mean(axis=1)

            keys = np.round(freqs_w, 4)
            seen: Dict[float, int] = {}
            for i, key in enumerate(keys):
                cur = seen.get(key)
                if cur is None or amps[i] > amps[cur]:
                    seen[key] = i
            keep = np.array(sorted(seen.values()), dtype=int)
            freqs_w = freqs_w[keep]
            amps = amps[keep]
            order = np.argsort(-amps)
            freqs_w = freqs_w[order]
            amps = amps[order]
            n_modes[wi] = float(freqs_w.size)
            if amps.size:
                e = amps / (amps.max() + 1e-12)
                top_f[wi] = float(freqs_w[0])
                top_e[wi] = float(e[0])
            if amps.size > 1:
                top2_f[wi] = float(freqs_w[1])
                top2_e[wi] = float(e[1])
        except Exception:
            continue

    t_centers = centers.astype(np.float64) / fs
    t_frames = np.arange(T) / fs

    def _to_framerate(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if not finite.any():
            return np.full(T, np.nan, dtype=np.float32)
        a = arr.astype(np.float64).copy()
        a[~finite] = np.interp(t_centers[~finite],
                               t_centers[finite], a[finite])
        return np.interp(t_frames, t_centers, a).astype(np.float32)

    return {
        "modal_w_top_freq_hz": _to_framerate(top_f),
        "modal_w_top_energy": _to_framerate(top_e),
        "modal_w_top2_freq_hz": _to_framerate(top2_f),
        "modal_w_top2_energy": _to_framerate(top2_e),
        "modal_w_n_modes": _to_framerate(n_modes),
    }


__all__ = ["ModalResult", "extract_modal", "extract_modal_windowed"]
