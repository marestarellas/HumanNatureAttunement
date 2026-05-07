"""
Per-pixel temporal extractors: every pixel's luminance time-series is
analysed independently, producing both 2-D maps (one value per pixel)
and 1-D coupling-ready summaries.

Two feature families are covered at the per-pixel tier:

  * Oscillatory : `extract_pixel_spectrum` (FFT of luminance(t) at every
                  pixel) and `extract_pixel_spectrum_windowed`
                  (sliding STFT counterpart).
  * Complexity  : `extract_pixel_complexity` (NEW) -- Higuchi fractal
                  dimension and DFA at every pixel; opt-in because
                  iterating Python complexity measures over each pixel
                  is the most expensive operation in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._common import iter_frames, _HAVE_ANTROPY, _HAVE_NK, _ant, _nk


_DEFAULT_SPEC_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("low",  0.05, 0.25),
    ("mid",  0.25, 0.50),
    ("high", 0.50, 2.00),
)


# ---------------------------------------------------------------------------
# Oscillatory: per-pixel temporal FFT
# ---------------------------------------------------------------------------

@dataclass
class PixelSpectrumResult:
    fps: float                                # effective sampling rate (fs)
    target_long: int                          # spatial resolution used
    freqs: np.ndarray                         # (n_freqs,)  frequency axis (Hz)
    peak_freq_map: np.ndarray                 # (h, w)      dominant freq per pixel (Hz)
    peak_power_map: np.ndarray                # (h, w)      power at the dominant freq
    spectral_entropy_map: np.ndarray          # (h, w)      Shannon entropy of normalised PSD
    band_power_maps: Dict[str, np.ndarray]    # band_name -> (h, w)
    mean_spectrum: np.ndarray                 # (n_freqs,)  mean power across pixels
    sync_index: float                         # how concentrated the mean spectrum is
    coherence_index: float                    # how spatially uniform the per-pixel peak freq is
    bands: Tuple[Tuple[str, float, float], ...]


def extract_pixel_spectrum(path: str, fps: float,
                           target_long: int = 96,
                           stride: int = 1,
                           max_frames: Optional[int] = None,
                           bands: Tuple[Tuple[str, float, float], ...]
                           = _DEFAULT_SPEC_BANDS,
                           detrend: bool = True,
                           ) -> PixelSpectrumResult:
    """
    For each pixel (x, y), compute the FFT of luminance(t). Returns:
      - 2-D maps: peak frequency, peak power, spectral entropy, band powers
      - 1-D mean spectrum across pixels
      - global scalars: synchronisation index (concentration of the mean
        spectrum) and coherence index (spatial uniformity of peak_freq_map)
    """
    frames: List[np.ndarray] = []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        frames.append(gray.astype(np.float32))
    if len(frames) < 8:
        raise RuntimeError("Need at least ~8 frames for a useful pixel spectrum.")
    X = np.stack(frames, axis=0)        # (T, h, w)
    T, h, w = X.shape

    if detrend:
        t_axis = np.arange(T, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = float(((t_axis - t_mean) ** 2).sum() + 1e-9)
        x_mean = X.mean(axis=0, keepdims=True)
        slope = ((t_axis - t_mean)[:, None, None] * (X - x_mean)).sum(axis=0) / t_var
        X = X - x_mean - slope[None, :, :] * (t_axis - t_mean)[:, None, None]
    else:
        X = X - X.mean(axis=0, keepdims=True)

    win = np.hanning(T).astype(np.float32)
    Xt = X * win[:, None, None]
    F = np.fft.rfft(Xt, axis=0)         # (n_freqs, h, w), complex
    P = (F.real ** 2 + F.imag ** 2)     # power
    freqs = np.fft.rfftfreq(T, d=1.0 / float(fps)).astype(np.float32)

    P_ac = P.copy()
    P_ac[0] = 0.0

    peak_idx = np.argmax(P_ac, axis=0)              # (h, w)
    peak_freq_map = freqs[peak_idx]                  # (h, w)
    peak_power_map = np.take_along_axis(P_ac, peak_idx[None, :, :], axis=0)[0]

    P_norm = P_ac / (P_ac.sum(axis=0, keepdims=True) + 1e-12)
    P_safe = np.where(P_norm > 0, P_norm, 1.0)
    log2_safe = np.where(P_norm > 0, np.log2(P_safe), 0.0)
    Hmap = -(P_norm * log2_safe).sum(axis=0)
    n_bins = P_ac.shape[0] - 1
    spectral_entropy_map = (Hmap / np.log2(max(n_bins, 2))).astype(np.float32)

    band_power_maps: Dict[str, np.ndarray] = {}
    for name, fmin, fmax in bands:
        m = (freqs >= fmin) & (freqs < fmax)
        band_power_maps[name] = (P_ac[m].sum(axis=0)
                                 if m.any() else np.zeros((h, w),
                                                          dtype=np.float32))

    mean_spectrum = P_ac.mean(axis=(1, 2))
    ms_norm = mean_spectrum / (mean_spectrum.sum() + 1e-12)
    ms_safe = np.where(ms_norm > 0, ms_norm, 1.0)
    H_ms = -(ms_norm * np.where(ms_norm > 0, np.log2(ms_safe), 0.0)).sum()
    sync_index = float(1.0 - H_ms / np.log2(max(n_bins, 2)))

    pf_flat = peak_freq_map.flatten()
    pf_finite = pf_flat[np.isfinite(pf_flat)]
    if pf_finite.size > 1:
        rng = freqs[-1] - freqs[1] if freqs.size > 2 else 1.0
        coherence_index = float(1.0 - (pf_finite.std() / max(rng, 1e-9)))
        coherence_index = max(0.0, min(1.0, coherence_index))
    else:
        coherence_index = float("nan")

    return PixelSpectrumResult(
        fps=float(fps), target_long=int(target_long),
        freqs=freqs,
        peak_freq_map=peak_freq_map.astype(np.float32),
        peak_power_map=peak_power_map.astype(np.float32),
        spectral_entropy_map=spectral_entropy_map,
        band_power_maps=band_power_maps,
        mean_spectrum=mean_spectrum.astype(np.float32),
        sync_index=sync_index,
        coherence_index=coherence_index,
        bands=bands,
    )


def extract_pixel_spectrum_windowed(
    path: str, fps: float,
    win_sec: float = 4.0, step_sec: float = 0.5,
    target_long: int = 64,
    stride: int = 1, max_frames: Optional[int] = None,
    bands: Tuple[Tuple[str, float, float], ...] = _DEFAULT_SPEC_BANDS,
    detrend_window: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Sliding-window per-pixel temporal FFT. Same procedure as
    `extract_pixel_spectrum` inside each window; reduces the maps to 1-D
    summaries that feed coupling tools at frame rate. Returns frame-rate
    series:

      pixspec_w_sync_index(t)
      pixspec_w_coherence_index(t)
      pixspec_w_peak_freq_median(t)
      pixspec_w_spectral_entropy_mean(t)
      pixspec_w_band_power_<band>(t)   for each requested band
    """
    frames: List[np.ndarray] = []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        frames.append(gray.astype(np.float32))
    if not frames:
        raise RuntimeError("No frames decoded.")
    X = np.stack(frames, axis=0)             # (T, h, w)
    T, h, w = X.shape
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
    sync = np.empty(n_w, dtype=np.float32)
    coh = np.empty(n_w, dtype=np.float32)
    pf_med = np.empty(n_w, dtype=np.float32)
    sp_ent_mean = np.empty(n_w, dtype=np.float32)
    band_p: Dict[str, np.ndarray] = {name: np.empty(n_w, dtype=np.float32)
                                     for name, _, _ in bands}

    win_kernel = np.hanning(W).astype(np.float32)

    for wi, st in enumerate(starts):
        seg = X[st:st + W]
        if detrend_window:
            seg = seg - seg.mean(axis=0, keepdims=True)
        seg_t = seg * win_kernel[:, None, None]
        F = np.fft.rfft(seg_t, axis=0)
        P = (F.real ** 2 + F.imag ** 2)
        freqs = np.fft.rfftfreq(W, d=1.0 / fs).astype(np.float32)
        P_ac = P.copy()
        P_ac[0] = 0.0
        n_bins = P_ac.shape[0] - 1

        peak_idx = np.argmax(P_ac, axis=0)
        peak_freq_map = freqs[peak_idx]
        P_norm = P_ac / (P_ac.sum(axis=0, keepdims=True) + 1e-12)
        log2_safe = np.where(P_norm > 0,
                             np.log2(np.where(P_norm > 0, P_norm, 1.0)), 0.0)
        Hmap = -(P_norm * log2_safe).sum(axis=0)
        sp_ent = (Hmap / np.log2(max(n_bins, 2))).astype(np.float32)

        ms = P_ac.mean(axis=(1, 2))
        ms_norm = ms / (ms.sum() + 1e-12)
        ms_log = np.where(ms_norm > 0,
                          np.log2(np.where(ms_norm > 0, ms_norm, 1.0)), 0.0)
        H_ms = -(ms_norm * ms_log).sum()
        sync[wi] = float(1.0 - H_ms / np.log2(max(n_bins, 2)))

        pf_flat = peak_freq_map.flatten()
        pf_flat = pf_flat[np.isfinite(pf_flat)]
        if pf_flat.size > 1:
            rng = freqs[-1] - freqs[1] if freqs.size > 2 else 1.0
            coh[wi] = float(max(0.0, min(1.0,
                                         1.0 - pf_flat.std() / max(rng, 1e-9))))
            pf_med[wi] = float(np.median(pf_flat))
        else:
            coh[wi] = float("nan")
            pf_med[wi] = float("nan")
        sp_ent_mean[wi] = float(sp_ent.mean())

        for name, fmin, fmax in bands:
            mfreq = (freqs >= fmin) & (freqs < fmax)
            if mfreq.any():
                p = float(P_ac[mfreq].sum())
                band_p[name][wi] = float(np.log10(p + 1.0))
            else:
                band_p[name][wi] = float("nan")

    t_centers = centers.astype(np.float64) / fs
    t_frames = np.arange(T) / fs

    def _to_framerate(arr: np.ndarray) -> np.ndarray:
        if arr.size == 1:
            return np.full(T, arr[0], dtype=np.float32)
        return np.interp(t_frames, t_centers, arr).astype(np.float32)

    out: Dict[str, np.ndarray] = {
        "pixspec_w_sync_index": _to_framerate(sync),
        "pixspec_w_coherence_index": _to_framerate(coh),
        "pixspec_w_peak_freq_median": _to_framerate(pf_med),
        "pixspec_w_spectral_entropy_mean": _to_framerate(sp_ent_mean),
    }
    for name, _, _ in bands:
        out[f"pixspec_w_band_power_{name}"] = _to_framerate(band_p[name])
    return out


# ---------------------------------------------------------------------------
# Complexity: per-pixel temporal complexity (NEW)
# ---------------------------------------------------------------------------

@dataclass
class PixelComplexityResult:
    fps: float
    target_long: int
    measure_maps: Dict[str, np.ndarray]   # measure name -> (h, w) float32 map
    measures: Tuple[str, ...]
    notes: List[str]


def _per_pixel_complexity_pass(X: np.ndarray, fs: float,
                               measures: Tuple[str, ...]
                               ) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    X has shape (T, h, w). For each (y, x) compute the requested
    nonlinear-dynamics measures on the column X[:, y, x] and return a
    dict of (h, w) maps. Returns also a list of notes about missing
    optional dependencies.
    """
    T, h, w = X.shape
    notes: List[str] = []
    out: Dict[str, np.ndarray] = {m: np.full((h, w), np.nan, dtype=np.float32)
                                  for m in measures}

    # Resolve per-measure callables
    callables: Dict[str, callable] = {}  # type: ignore[type-arg]
    for m in measures:
        if m == "higuchi_fd":
            if _HAVE_ANTROPY:
                callables[m] = lambda x: float(_ant.higuchi_fd(x))
            else:
                notes.append("higuchi_fd skipped (antropy missing).")
        elif m == "katz_fd":
            if _HAVE_ANTROPY:
                callables[m] = lambda x: float(_ant.katz_fd(x))
            else:
                notes.append("katz_fd skipped (antropy missing).")
        elif m == "petrosian_fd":
            if _HAVE_ANTROPY:
                callables[m] = lambda x: float(_ant.petrosian_fd(x))
            else:
                notes.append("petrosian_fd skipped (antropy missing).")
        elif m == "perm_entropy":
            if _HAVE_ANTROPY:
                callables[m] = lambda x: float(
                    _ant.perm_entropy(x, normalize=True))
            else:
                notes.append("perm_entropy skipped (antropy missing).")
        elif m == "spectral_entropy":
            if _HAVE_ANTROPY:
                callables[m] = lambda x, _fs=fs: float(
                    _ant.spectral_entropy(x, sf=_fs, normalize=True))
            else:
                notes.append("spectral_entropy skipped (antropy missing).")
        elif m == "dfa_alpha":
            if _HAVE_NK:
                def _dfa(x: np.ndarray) -> float:
                    try:
                        v, _ = _nk.fractal_dfa(x)
                        return float(v)
                    except Exception:
                        return float("nan")
                callables[m] = _dfa
            else:
                notes.append("dfa_alpha skipped (neurokit2 missing).")
        else:
            notes.append(f"unknown measure '{m}' skipped.")

    if not callables:
        return out, notes

    # Loop pixels (this is the expensive part). We pre-flatten X to (T, P)
    # so pixel-iteration is a tight numpy slice.
    flat = X.reshape(T, h * w)
    # Filter pixels with no variance to avoid wasted work
    std_per_pix = flat.std(axis=0)
    valid = std_per_pix > 1e-9
    valid_idx = np.where(valid)[0]

    for p_idx in valid_idx:
        col = flat[:, p_idx]
        finite = col[np.isfinite(col)]
        if finite.size < 32:
            continue
        y, x = divmod(int(p_idx), w)
        for m, fn in callables.items():
            try:
                out[m][y, x] = fn(finite)
            except Exception:
                out[m][y, x] = np.nan
    return out, notes


def extract_pixel_complexity(
    path: str, fps: float,
    target_long: int = 48,
    stride: int = 1, max_frames: Optional[int] = None,
    measures: Tuple[str, ...] = ("higuchi_fd", "perm_entropy"),
    detrend: bool = True,
) -> PixelComplexityResult:
    """
    Per-pixel temporal complexity. For each pixel of a downscaled grayscale
    copy, compute one or more nonlinear-dynamics measures on its luminance
    time series, then return one (h, w) map per measure.

    Defaults are aggressive about cost: `target_long=48` keeps the pixel
    count in the low thousands, and the default `measures` are the cheapest
    informative pair (Higuchi FD + permutation entropy). Add `dfa_alpha`
    only on short clips or coarse grids -- DFA on 100k pixels can take
    minutes.

    Returns a `PixelComplexityResult` with one map per measure and a list
    of notes about any optional dependencies that were missing.
    """
    frames: List[np.ndarray] = []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        frames.append(gray.astype(np.float32))
    if len(frames) < 32:
        raise RuntimeError(
            "extract_pixel_complexity needs at least ~32 frames "
            "for stable estimates.")
    X = np.stack(frames, axis=0)                # (T, h, w)
    T, h, w = X.shape

    if detrend:
        t_axis = np.arange(T, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = float(((t_axis - t_mean) ** 2).sum() + 1e-9)
        x_mean = X.mean(axis=0, keepdims=True)
        slope = ((t_axis - t_mean)[:, None, None]
                 * (X - x_mean)).sum(axis=0) / t_var
        X = X - x_mean - slope[None, :, :] * (t_axis - t_mean)[:, None, None]
    else:
        X = X - X.mean(axis=0, keepdims=True)

    maps, notes = _per_pixel_complexity_pass(X, float(fps), measures)
    return PixelComplexityResult(
        fps=float(fps), target_long=int(target_long),
        measure_maps=maps, measures=tuple(measures), notes=notes,
    )


__all__ = [
    "PixelSpectrumResult", "PixelComplexityResult",
    "extract_pixel_spectrum", "extract_pixel_spectrum_windowed",
    "extract_pixel_complexity",
    "_DEFAULT_SPEC_BANDS",
]
