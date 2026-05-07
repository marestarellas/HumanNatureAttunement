"""
Video-derived oscillation & complexity features for coupling with
EEG / HRV / EMG.

Subpackage layout (axes of the project framework):

    spatial scale  ───►  whole-image       per-patch        per-pixel
    feature family
    │ raw          ───►  whole_image       per_patch        --
    │                    (envelopes,       (per-patch
    │                     optical flow)     frame_diff)
    │ oscillatory  ───►  spatial_fft       --                per_pixel
    │                    timestack                           (FFT, modal)
    │                    modal
    │ complexity   ───►  whole_image       per_patch         per_pixel
    │                    (fractal_dim,     (patch entropy    (Higuchi /
    │                     lacunarity,       grid, GLCM        DFA per
    │                     GLCM, ...)        per patch)        pixel)
    └─ temporal_complexity : nonlinear-dynamics on ANY 1-D signal
       pipeline             : `quantify_video`, the orchestrator

For most users the public API is unchanged: keep importing from
`HNA.modalities.video` directly.
"""

from __future__ import annotations

# Frame I/O & metadata
from ._common import (
    VideoMeta, probe_video, iter_frames,
    _stack_video, _spatial_entropy, _resize_keep_aspect,
    _HAVE_ANTROPY, _HAVE_NK, _HAVE_PYDMD,
)

# Whole-image extractors
from .whole_image import (
    extract_global_signals,
    extract_optical_flow_signals,
    extract_optical_flow_multiscale,
    extract_spatial_complexity,
)
from .spatial_fft import extract_spatial_fft_signals

# Per-patch extractors
from .per_patch import (
    extract_spatial_field, SpatialFieldResult,
)

# Per-pixel extractors
from .per_pixel import (
    PixelSpectrumResult, PixelComplexityResult,
    extract_pixel_spectrum, extract_pixel_spectrum_windowed,
    extract_pixel_complexity,
)

# Modal & timestack
from .modal import (
    ModalResult, extract_modal, extract_modal_windowed,
)
from .timestack import (
    TimestackResult, extract_timestack, extract_timestack_windowed,
)

# Temporal complexity & pipeline
from .temporal_complexity import complexity_summary, windowed_complexity
from .pipeline import VideoFeatures, quantify_video


__all__ = [
    # I/O
    "VideoMeta", "probe_video", "iter_frames",
    # Whole-image
    "extract_global_signals",
    "extract_optical_flow_signals", "extract_optical_flow_multiscale",
    "extract_spatial_complexity", "extract_spatial_fft_signals",
    # Per-patch
    "SpatialFieldResult", "extract_spatial_field",
    # Per-pixel
    "PixelSpectrumResult", "PixelComplexityResult",
    "extract_pixel_spectrum", "extract_pixel_spectrum_windowed",
    "extract_pixel_complexity",
    # Modal & timestack
    "ModalResult", "extract_modal", "extract_modal_windowed",
    "TimestackResult", "extract_timestack", "extract_timestack_windowed",
    # Temporal complexity
    "complexity_summary", "windowed_complexity",
    # Pipeline
    "VideoFeatures", "quantify_video",
]
