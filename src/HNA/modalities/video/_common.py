"""
Shared low-level helpers for the `HNA.modalities.video` subpackage:
frame I/O, downscaled-grayscale iteration, and the (pixels x time)
flat stack used by the modal and per-pixel extractors.

Importable as `HNA.modalities.video._common`, but most users will reach
these symbols through `HNA.modalities.video` directly (re-exported in
`__init__.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "HNA.modalities.video requires opencv-python "
        "(`pip install opencv-python`)") from e

# --- Optional dependency probes ------------------------------------------
# Re-exported so other submodules can branch on availability without each
# of them re-doing the try/except.
try:
    import antropy as _ant            # noqa: F401
    _HAVE_ANTROPY = True
except ImportError:                   # pragma: no cover
    _ant = None                       # type: ignore
    _HAVE_ANTROPY = False

try:
    import neurokit2 as _nk           # noqa: F401
    _HAVE_NK = True
except ImportError:                   # pragma: no cover
    _nk = None                        # type: ignore
    _HAVE_NK = False

try:
    from pydmd import DMD as _PyDMD   # noqa: F401
    _HAVE_PYDMD = True
except ImportError:                   # pragma: no cover
    _PyDMD = None                     # type: ignore
    _HAVE_PYDMD = False


# ---------------------------------------------------------------------------
# Metadata + I/O
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int
    duration_s: float


def probe_video(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoMeta(path=str(path), fps=fps, n_frames=n,
                     width=w, height=h,
                     duration_s=(n / fps if fps > 0 else 0.0))


def _resize_keep_aspect(img: np.ndarray, target_long: int) -> np.ndarray:
    h, w = img.shape[:2]
    long = max(h, w)
    if long <= target_long:
        return img
    scale = target_long / long
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def iter_frames(path: str, target_long: int = 192,
                stride: int = 1, max_frames: Optional[int] = None):
    """Yield (idx, gray_uint8, bgr_resized) tuples."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    i = -1
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % stride != 0:
                continue
            small = _resize_keep_aspect(frame, target_long)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            yield i, gray, small
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Spatial Shannon entropy of a grayscale frame (used by global envelope)
# ---------------------------------------------------------------------------

def _spatial_entropy(gray: np.ndarray, bins: int = 64) -> float:
    h, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


# ---------------------------------------------------------------------------
# Flat (pixels x time) stack used by modal + per-pixel extractors
# ---------------------------------------------------------------------------

def _stack_video(path: str, target_long: int, stride: int,
                 max_frames: Optional[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Decode a video and return X of shape (P, T) with P = h'*w' pixels of the
    downscaled grayscale frame and T = number of decoded frames. The temporal
    mean is removed per pixel. Also returns the (h', w') spatial shape.
    """
    frames: List[np.ndarray] = []
    shape: Optional[Tuple[int, int]] = None
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        if shape is None:
            shape = gray.shape
        frames.append(gray.astype(np.float32).ravel())
    if not frames:
        raise RuntimeError("No frames decoded.")
    X = np.stack(frames, axis=1)        # (pixels, time)
    X -= X.mean(axis=1, keepdims=True)  # remove temporal mean per pixel
    return X, shape  # type: ignore[return-value]


__all__ = [
    "VideoMeta", "probe_video", "iter_frames",
    "_resize_keep_aspect", "_spatial_entropy", "_stack_video",
    "_HAVE_ANTROPY", "_HAVE_NK", "_HAVE_PYDMD",
    "_ant", "_nk", "_PyDMD",
]
