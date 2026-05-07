"""
High-level orchestrator: `quantify_video(path, ...)` runs the relevant
extractors across all three spatial tiers (whole-image, per-patch,
per-pixel) and the three feature families (raw, oscillatory,
complexity), then returns a single `VideoFeatures` container.

Every 1-D signal in `feats.signals` is sampled at the frame rate and is
a drop-in input to `HNA.modules.coupling`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._common import (
    VideoMeta, probe_video,
    _HAVE_ANTROPY, _HAVE_NK, _HAVE_PYDMD,
)
from .whole_image import (
    extract_global_signals,
    extract_optical_flow_signals,
    extract_spatial_complexity,
)
from .spatial_fft import extract_spatial_fft_signals
from .per_patch import (
    extract_spatial_field, SpatialFieldResult, _DEFAULT_FIELD_MEASURES,
)
from .per_pixel import (
    extract_pixel_spectrum, extract_pixel_spectrum_windowed,
    extract_pixel_complexity, PixelSpectrumResult, PixelComplexityResult,
)
from .modal import extract_modal, extract_modal_windowed, ModalResult
from .timestack import (
    extract_timestack, extract_timestack_windowed, TimestackResult,
)
from .temporal_complexity import complexity_summary, windowed_complexity


@dataclass
class VideoFeatures:
    meta: VideoMeta
    fs: float                              # effective sampling rate of 1-D signals
    signals: Dict[str, np.ndarray]         # 1-D arrays, all length n_used
    complexity: Dict[str, Dict[str, float]]  # per-signal nonlinear summaries
    timestack: Optional[TimestackResult] = None
    modal: Optional[ModalResult] = None
    pixel_spectrum: Optional[PixelSpectrumResult] = None
    pixel_complexity: Optional[PixelComplexityResult] = None
    spatial_field: Optional[SpatialFieldResult] = None
    notes: List[str] = field(default_factory=list)

    def as_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(self.signals)
        df.insert(0, "t_s", np.arange(len(df)) / self.fs)
        return df


def quantify_video(path: str,
                   target_long: int = 192,
                   stride: int = 1,
                   max_frames: Optional[int] = None,
                   include_flow: bool = True,
                   include_spatial: bool = True,
                   include_spatial_fft: bool = True,
                   include_pixel_spectrum: bool = False,
                   pixel_spectrum_target_long: int = 96,
                   include_modal: bool = True,
                   include_timestack: bool = True,
                   include_complexity: bool = True,
                   include_windowed_complexity: bool = False,
                   windowed_complexity_signals: Tuple[str, ...] = (
                       "luminance", "frame_diff", "flow_mag_mean",
                       "patch_entropy"),
                   windowed_complexity_measures: Tuple[str, ...] = (
                       "perm_entropy", "hjorth_complexity",
                       "higuchi_fd", "spectral_entropy"),
                   windowed_complexity_win_sec: float = 4.0,
                   windowed_complexity_step_sec: float = 0.25,
                   include_pixel_spectrum_windowed: bool = False,
                   include_modal_windowed: bool = False,
                   include_timestack_windowed: bool = False,
                   windowed_win_sec: float = 4.0,
                   windowed_step_sec: float = 0.5,
                   include_spatial_field: bool = False,
                   spatial_field_patch_size: int = 24,
                   spatial_field_measures: Tuple[str, ...]
                       = _DEFAULT_FIELD_MEASURES,
                   include_pixel_complexity: bool = False,
                   pixel_complexity_target_long: int = 48,
                   pixel_complexity_measures: Tuple[str, ...] = (
                       "higuchi_fd", "perm_entropy"),
                   modal_k: int = 4,
                   modal_target_long: int = 96) -> VideoFeatures:
    """
    Full pipeline on a single video. All extractors share the same frame
    iteration pattern and are run sequentially (cheapest first).

    The returned `signals` dict contains 1-D arrays of length `n_used`
    sampled at `fs = fps / stride` -- pass these directly to
    `HNA.modules.coupling`.

    Three new opt-in flags expose work added at the per-patch and
    per-pixel tiers of the framework:

      * `include_spatial_field=True` runs `extract_spatial_field` and
        attaches `feats.spatial_field` (per-patch maps + per-frame
        spatial-mean signals already in `feats.signals` via the existing
        `extract_spatial_complexity`).

      * `include_pixel_complexity=True` runs `extract_pixel_complexity`
        and attaches `feats.pixel_complexity` (per-pixel temporal Higuchi
        FD / permutation entropy / DFA maps; see that function's docstring
        for cost notes).
    """
    meta = probe_video(path)
    fs = meta.fps / max(1, stride)
    notes: List[str] = []
    if not _HAVE_PYDMD:
        notes.append("pydmd not installed -> SVD/POD fallback used for modal decomposition.")
    if not _HAVE_ANTROPY:
        notes.append("antropy not installed -> permutation/sample entropy unavailable.")
    if not _HAVE_NK:
        notes.append("neurokit2 not installed -> DFA/Hurst/LZ unavailable.")

    signals: Dict[str, np.ndarray] = {}
    signals.update(extract_global_signals(path, target_long, stride, max_frames))
    if include_flow:
        signals.update(extract_optical_flow_signals(
            path, target_long, stride, max_frames))
    if include_spatial:
        signals.update(extract_spatial_complexity(
            path, target_long, stride, max_frames))
    if include_spatial_fft:
        signals.update(extract_spatial_fft_signals(
            path, target_long, stride, max_frames))

    n_used = min(len(v) for v in signals.values())
    signals = {k: v[:n_used] for k, v in signals.items()}

    timestack = None
    if include_timestack:
        timestack = extract_timestack(path, fs, target_long=target_long,
                                      stride=stride, max_frames=max_frames)

    pixel_spectrum: Optional[PixelSpectrumResult] = None
    if include_pixel_spectrum:
        try:
            pixel_spectrum = extract_pixel_spectrum(
                path, fps=fs, target_long=pixel_spectrum_target_long,
                stride=stride, max_frames=max_frames)
        except Exception as e:
            notes.append(f"pixel-spectrum skipped: {e}")

    modal = None
    if include_modal:
        try:
            modal = extract_modal(path, fps=fs, target_long=modal_target_long,
                                  stride=stride, max_frames=max_frames, k=modal_k)
            for i, coeff in enumerate(modal.temporal_coeffs[:modal_k]):
                signals[f"modal_{i+1}"] = coeff[:n_used]
        except Exception as e:
            notes.append(f"modal decomposition skipped: {e}")

    # Optional spatial-field (per-patch maps): adds the per-frame spatial
    # mean of each patch-grid measure to `signals` under the prefix
    # `field_<measure>` so they don't collide with the whole-image
    # versions of the same names. The maps themselves live in
    # `feats.spatial_field`.
    spatial_field: Optional[SpatialFieldResult] = None
    if include_spatial_field:
        try:
            spatial_field = extract_spatial_field(
                path, fps=fs,
                target_long=target_long,
                patch_size=spatial_field_patch_size,
                stride=stride, max_frames=max_frames,
                measures=spatial_field_measures)
            for k, v in spatial_field.signals.items():
                v = np.asarray(v)
                if v.size >= n_used:
                    signals[f"field_{k}"] = v[:n_used]
        except Exception as e:
            notes.append(f"spatial field skipped: {e}")

    # Per-pixel temporal complexity (NEW): expensive, opt-in. Only the
    # full-resolution maps are returned (no frame-rate signal -- this is
    # an oscillatory-vs-complexity *snapshot* of the whole clip).
    pixel_complexity: Optional[PixelComplexityResult] = None
    if include_pixel_complexity:
        try:
            pixel_complexity = extract_pixel_complexity(
                path, fps=fs,
                target_long=pixel_complexity_target_long,
                stride=stride, max_frames=max_frames,
                measures=pixel_complexity_measures)
            notes.extend(pixel_complexity.notes)
        except Exception as e:
            notes.append(f"pixel complexity skipped: {e}")

    # Windowed spectral / modal / timestack: time-resolved versions of the
    # global summaries above. All three return 1-D series at frame rate, so
    # they slot directly into `signals` and are interpolated/cropped to
    # n_used like everything else.
    def _add_windowed(extra: Dict[str, np.ndarray]) -> None:
        for k, v in extra.items():
            v = np.asarray(v)
            if v.size >= n_used:
                signals[k] = v[:n_used]
            elif v.size >= 2:
                t_src = np.linspace(0.0, 1.0, v.size)
                t_dst = np.linspace(0.0, 1.0, n_used)
                signals[k] = np.interp(t_dst, t_src, v).astype(np.float32)

    if include_pixel_spectrum_windowed:
        try:
            extra = extract_pixel_spectrum_windowed(
                path, fps=fs,
                win_sec=windowed_win_sec, step_sec=windowed_step_sec,
                target_long=pixel_spectrum_target_long,
                stride=stride, max_frames=max_frames)
            _add_windowed(extra)
        except Exception as e:
            notes.append(f"pixel-spectrum windowed skipped: {e}")

    if include_modal_windowed:
        try:
            extra = extract_modal_windowed(
                path, fps=fs,
                win_sec=windowed_win_sec, step_sec=windowed_step_sec,
                target_long=modal_target_long,
                stride=stride, max_frames=max_frames, k=modal_k)
            _add_windowed(extra)
        except Exception as e:
            notes.append(f"modal windowed skipped: {e}")

    if include_timestack_windowed:
        try:
            extra = extract_timestack_windowed(
                path, fps=fs,
                win_sec=windowed_win_sec, step_sec=windowed_step_sec,
                target_long=target_long,
                stride=stride, max_frames=max_frames)
            _add_windowed(extra)
        except Exception as e:
            notes.append(f"timestack windowed skipped: {e}")

    complexity: Dict[str, Dict[str, float]] = {}
    if include_complexity:
        for k, v in signals.items():
            complexity[k] = complexity_summary(v, fs=fs)

    if include_windowed_complexity:
        wc_targets = [s for s in windowed_complexity_signals if s in signals]
        for s_name in wc_targets:
            wc = windowed_complexity(
                signals[s_name], fs=fs,
                win_sec=windowed_complexity_win_sec,
                step_sec=windowed_complexity_step_sec,
                measures=windowed_complexity_measures)
            t_target = np.arange(n_used) / fs
            for m in windowed_complexity_measures:
                y = wc[m]; t = wc["times_s"]
                if y.size >= 2 and np.isfinite(y).any():
                    y_filled = y.copy()
                    bad = ~np.isfinite(y_filled)
                    if bad.any() and (~bad).any():
                        y_filled[bad] = np.interp(
                            t[bad], t[~bad], y_filled[~bad])
                    signals[f"wc_{s_name}__{m}"] = np.interp(
                        t_target, t, y_filled,
                        left=y_filled[0], right=y_filled[-1])

    return VideoFeatures(
        meta=meta, fs=fs, signals=signals,
        complexity=complexity, timestack=timestack,
        modal=modal, pixel_spectrum=pixel_spectrum,
        pixel_complexity=pixel_complexity,
        spatial_field=spatial_field,
        notes=notes,
    )


__all__ = ["VideoFeatures", "quantify_video"]
