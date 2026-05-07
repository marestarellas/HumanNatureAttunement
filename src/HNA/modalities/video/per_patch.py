"""
Per-patch (low-resolution map) extractors. The frame is tiled into a
fixed-size grid; for each tile we compute a battery of within-tile
descriptors. Results are returned both as patch-grid maps (one
(n_patches_y, n_patches_x, T) array per descriptor) and as the
clip-mean reduction (a 1-D scalar per frame), so the same call serves
both visualisation (the maps) and coupling (the scalars).

This file fills the "per-patch" tier of the framework (between the
whole-image scalars in `whole_image.py` and the full-resolution
per-pixel analyses in `per_pixel.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ._common import iter_frames
from .whole_image import (
    _patch_shannon_entropy_field,
    _box_count_dim,
    _lacunarity,
    _glcm_contrast_homogeneity,
)


@dataclass
class SpatialFieldResult:
    fps: float
    target_long: int
    patch_size: int
    grid_shape: Tuple[int, int]                  # (n_patches_y, n_patches_x)
    maps: Dict[str, np.ndarray]                  # name -> (Gy, Gx, T) float32
    signals: Dict[str, np.ndarray]               # name -> (T,) float32
                                                 #   (mean over patches per frame)
    measures: Tuple[str, ...]


_DEFAULT_FIELD_MEASURES: Tuple[str, ...] = (
    "patch_entropy",       # Shannon entropy of the intensity histogram
    "edge_density",        # fraction of Canny edges
    "fractal_dim_grad",    # box-counting D on binarised gradient
    "lacunarity_grad",     # log lacunarity on binarised gradient
    "glcm_contrast",       # GLCM contrast
    "glcm_homogeneity",    # GLCM homogeneity
    "frame_diff",          # per-patch absolute frame difference
)


def _patches_view(gray: np.ndarray, patch: int) -> Tuple[np.ndarray,
                                                         Tuple[int, int]]:
    """Return a (Gy, Gx, patch, patch) view + (Gy, Gx) of the grid."""
    h, w = gray.shape
    Gy, Gx = h // patch, w // patch
    if Gy == 0 or Gx == 0:
        return np.empty((0, 0, patch, patch), dtype=gray.dtype), (0, 0)
    H = Gy * patch
    W = Gx * patch
    g = gray[:H, :W].reshape(Gy, patch, Gx, patch).swapaxes(1, 2)
    return g, (Gy, Gx)


def _per_patch_frame(gray: np.ndarray, prev_gray: Optional[np.ndarray],
                     patch: int, measures: Tuple[str, ...],
                     canny_low: int, canny_high: int
                     ) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    """Compute requested per-patch maps for a single frame."""
    out: Dict[str, np.ndarray] = {}

    # patch entropy uses the optimised whole-frame routine
    if "patch_entropy" in measures:
        field = _patch_shannon_entropy_field(gray, patch=patch)
        out["patch_entropy"] = field
        gshape = field.shape
    else:
        # need a grid shape anyway for sizing the other maps
        Gy, Gx = gray.shape[0] // patch, gray.shape[1] // patch
        gshape = (Gy, Gx)

    if gshape[0] == 0 or gshape[1] == 0:
        return {m: np.zeros((0, 0), dtype=np.float32) for m in measures}, gshape
    Gy, Gx = gshape

    # tile views of frame, edge map, gradient binarisation, frame-diff
    edges_full = cv2.Canny(gray, canny_low, canny_high) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    gthr = float(np.percentile(gmag, 60.0))
    grad_bin_full = gmag > max(gthr, 1.0)

    g_view, _ = _patches_view(gray, patch)
    e_view, _ = _patches_view(edges_full.astype(np.uint8), patch)
    gb_view, _ = _patches_view(grad_bin_full.astype(np.uint8), patch)

    if "edge_density" in measures:
        ed_map = e_view.mean(axis=(2, 3)).astype(np.float32)
        out["edge_density"] = ed_map
    if "fractal_dim_grad" in measures:
        fd = np.zeros((Gy, Gx), dtype=np.float32)
        for j in range(Gy):
            for i in range(Gx):
                fd[j, i] = _box_count_dim(gb_view[j, i].astype(bool))
        out["fractal_dim_grad"] = fd
    if "lacunarity_grad" in measures:
        lac = np.zeros((Gy, Gx), dtype=np.float32)
        for j in range(Gy):
            for i in range(Gx):
                lac[j, i] = _lacunarity(gb_view[j, i].astype(bool))
        out["lacunarity_grad"] = lac
    if "glcm_contrast" in measures or "glcm_homogeneity" in measures:
        glc = np.zeros((Gy, Gx), dtype=np.float32)
        glh = np.zeros((Gy, Gx), dtype=np.float32)
        for j in range(Gy):
            for i in range(Gx):
                c, hgm = _glcm_contrast_homogeneity(g_view[j, i])
                glc[j, i] = c
                glh[j, i] = hgm
        if "glcm_contrast" in measures:
            out["glcm_contrast"] = glc
        if "glcm_homogeneity" in measures:
            out["glcm_homogeneity"] = glh
    if "frame_diff" in measures:
        if prev_gray is None:
            out["frame_diff"] = np.zeros((Gy, Gx), dtype=np.float32)
        else:
            d = np.abs(gray.astype(np.int16)
                       - prev_gray.astype(np.int16)).astype(np.float32)
            d_view, _ = _patches_view(d, patch)
            out["frame_diff"] = d_view.mean(axis=(2, 3)).astype(np.float32)

    return out, gshape


def extract_spatial_field(
    path: str, fps: float,
    target_long: int = 192,
    patch_size: int = 24,
    stride: int = 1, max_frames: Optional[int] = None,
    measures: Tuple[str, ...] = _DEFAULT_FIELD_MEASURES,
    canny_low: int = 35, canny_high: int = 110,
) -> SpatialFieldResult:
    """
    Per-patch spatial field analysis.

    For each requested measure, returns:
      - a (Gy, Gx, T) float32 *map stack* (`result.maps[name]`), suitable
        for visualisation as a low-resolution movie of the descriptor
        across the frame and time;
      - the spatial mean per frame as a 1-D coupling-ready signal
        (`result.signals[name]`).

    Default measures cover the within-frame complexity / texture battery
    used elsewhere in the pipeline: patch entropy, edge density, gradient-
    binarised fractal dimension, gradient-binarised lacunarity, GLCM
    contrast / homogeneity, and per-patch frame-difference.

    The patch grid is whatever fits in the downscaled frame at
    `patch_size` px (default 24 px on a 192-px-long-axis frame -> a
    coarse 8 x ~8 grid).
    """
    measures = tuple(measures)
    map_lists: Dict[str, List[np.ndarray]] = {m: [] for m in measures}
    signal_lists: Dict[str, List[float]] = {m: [] for m in measures}

    prev_gray: Optional[np.ndarray] = None
    grid_shape: Tuple[int, int] = (0, 0)

    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        per_frame, gshape = _per_patch_frame(
            gray, prev_gray, patch=patch_size, measures=measures,
            canny_low=canny_low, canny_high=canny_high)
        if gshape[0] and gshape[1]:
            grid_shape = gshape
        for m in measures:
            map_lists[m].append(per_frame[m])
            if per_frame[m].size:
                v = float(np.nanmean(per_frame[m]))
            else:
                v = float("nan")
            signal_lists[m].append(v)
        prev_gray = gray

    if not signal_lists[measures[0]]:
        raise RuntimeError("No frames decoded.")

    maps: Dict[str, np.ndarray] = {}
    for m in measures:
        if map_lists[m] and map_lists[m][0].size:
            maps[m] = np.stack(map_lists[m], axis=-1).astype(np.float32)
        else:
            maps[m] = np.zeros((0, 0, 0), dtype=np.float32)
    signals: Dict[str, np.ndarray] = {
        m: np.asarray(signal_lists[m], dtype=np.float32) for m in measures
    }

    return SpatialFieldResult(
        fps=float(fps), target_long=int(target_long),
        patch_size=int(patch_size), grid_shape=grid_shape,
        maps=maps, signals=signals, measures=measures,
    )


__all__ = [
    "SpatialFieldResult", "extract_spatial_field",
    "_DEFAULT_FIELD_MEASURES",
]
