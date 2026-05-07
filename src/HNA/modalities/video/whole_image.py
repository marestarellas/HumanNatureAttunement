"""
Whole-image extractors: per-frame scalar reductions of the visual scene.

Every function in this file returns a `Dict[str, np.ndarray]` of 1-D
signals at frame rate (length = number of decoded frames). The reductions
fall into three feature families per the project framework:

  * Raw          : luminance, color means, frame-difference, optical flow
                   magnitude / curl / divergence / direction entropy.
  * Oscillatory  : (no whole-image-only oscillatory descriptors live here;
                   timestack and modal cover that axis.)
  * Complexity   : edge density, radial 2-D-FFT slope, box-counting fractal
                   dimensions, lacunarity, mean patch entropy, GLCM
                   contrast / homogeneity.

These signals slot directly into `HNA.modules.coupling`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ._common import iter_frames, _spatial_entropy


# ---------------------------------------------------------------------------
# 1) Global temporal envelope ("video envelope")
# ---------------------------------------------------------------------------

def extract_global_signals(path: str, target_long: int = 192,
                           stride: int = 1, max_frames: Optional[int] = None
                           ) -> Dict[str, np.ndarray]:
    """
    Per-frame scalars: the visual analogue of an audio envelope.

    Returns a dict of 1-D arrays (length = number of frames yielded), keys:
      luminance, blue_mean, red_mean, green_mean,
      spatial_std, spatial_entropy, frame_diff
    `frame_diff[0]` is set to 0 (no previous frame).
    """
    lum, bm, gm, rm, sstd, sent, fdiff = [], [], [], [], [], [], []
    prev_gray = None
    for _, gray, bgr in iter_frames(path, target_long=target_long,
                                    stride=stride, max_frames=max_frames):
        b, g, r = cv2.split(bgr)
        lum.append(float(gray.mean()))
        bm.append(float(b.mean()))
        gm.append(float(g.mean()))
        rm.append(float(r.mean()))
        sstd.append(float(gray.std()))
        sent.append(_spatial_entropy(gray))
        if prev_gray is None:
            fdiff.append(0.0)
        else:
            fdiff.append(float(np.abs(gray.astype(np.int16)
                                      - prev_gray.astype(np.int16)).mean()))
        prev_gray = gray
    return dict(luminance=np.asarray(lum), blue_mean=np.asarray(bm),
                green_mean=np.asarray(gm), red_mean=np.asarray(rm),
                spatial_std=np.asarray(sstd), spatial_entropy=np.asarray(sent),
                frame_diff=np.asarray(fdiff))


# ---------------------------------------------------------------------------
# 2) Optical flow (Farneback)
# ---------------------------------------------------------------------------

def _flow_curl_div(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    du_dx = np.gradient(u, axis=1)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    dv_dy = np.gradient(v, axis=0)
    div = du_dx + dv_dy
    curl = dv_dx - du_dy
    return curl, div


def _direction_entropy(u: np.ndarray, v: np.ndarray, bins: int = 16,
                       mag_floor: float = 0.1) -> float:
    mag = np.hypot(u, v)
    mask = mag > mag_floor
    if not mask.any():
        return 0.0
    ang = np.arctan2(v[mask], u[mask])  # (-pi, pi]
    h, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi), density=True)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def _spatial_lowpass_flow(uv: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian low-pass on the 2-channel flow field (h, w, 2)."""
    if sigma_px <= 0.5:
        return uv
    out = np.empty_like(uv)
    out[..., 0] = cv2.GaussianBlur(uv[..., 0], (0, 0), sigmaX=sigma_px)
    out[..., 1] = cv2.GaussianBlur(uv[..., 1], (0, 0), sigmaX=sigma_px)
    return out


def extract_optical_flow_signals(path: str, target_long: int = 192,
                                 stride: int = 1,
                                 max_frames: Optional[int] = None,
                                 pyr_scale: float = 0.5, levels: int = 4,
                                 winsize: int = 21, iterations: int = 5,
                                 poly_n: int = 7, poly_sigma: float = 1.5
                                 ) -> Dict[str, np.ndarray]:
    """
    Per-frame summaries of the dense Farneback flow field. First entry is 0
    (no previous frame). Returns dict keys:
      flow_mag_mean, flow_mag_p95, flow_curl_abs_mean, flow_div_mean,
      flow_dir_entropy, flow_u_mean, flow_v_mean
    """
    keys = ["flow_mag_mean", "flow_mag_p95", "flow_curl_abs_mean",
            "flow_div_mean", "flow_dir_entropy",
            "flow_u_mean", "flow_v_mean"]
    out: Dict[str, List[float]] = {k: [] for k in keys}
    prev_gray = None
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        if prev_gray is None:
            for k in keys:
                out[k].append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
            u, v = flow[..., 0], flow[..., 1]
            mag = np.hypot(u, v)
            curl, div = _flow_curl_div(u, v)
            out["flow_mag_mean"].append(float(mag.mean()))
            out["flow_mag_p95"].append(float(np.percentile(mag, 95)))
            out["flow_curl_abs_mean"].append(float(np.abs(curl).mean()))
            out["flow_div_mean"].append(float(div.mean()))
            out["flow_dir_entropy"].append(_direction_entropy(u, v))
            out["flow_u_mean"].append(float(u.mean()))
            out["flow_v_mean"].append(float(v.mean()))
        prev_gray = gray
    return {k: np.asarray(v) for k, v in out.items()}


def extract_optical_flow_multiscale(
    path: str, target_long: int = 192,
    stride: int = 1, max_frames: Optional[int] = None,
    temporal_strides: Tuple[int, ...] = (1, 3, 9),
    spatial_lowpass_sigma_px: float = 6.0,
    pyr_scale: float = 0.5, levels: int = 4,
    winsize: int = 21, iterations: int = 5,
    poly_n: int = 7, poly_sigma: float = 1.5,
) -> Dict[str, np.ndarray]:
    """
    Multi-scale optical flow features at frame rate.

    Two granularity dimensions are produced:
      * Temporal strides (default 1, 3, 9 frames). Each dt picks a different
        motion timescale: dt=1 -> ripple/foam-wiggle, dt=3 -> ~0.1 s motion,
        dt=9 -> ~0.4 s motion (close to a swell quarter-period).
      * Spatial decomposition of the flow field. After computing flow, the
        (u, v) field is split into a low-pass component (Gaussian sigma
        `spatial_lowpass_sigma_px`) and the high-pass residual.

    Returned keys: `flow_dt{N}_{full|large|small}_{stat}` where
    {stat} is one of mag_mean, curl_abs_mean, div_mean, dir_entropy.
    """
    frames: List[np.ndarray] = []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        frames.append(gray)
    if len(frames) < max(temporal_strides) + 1:
        raise RuntimeError("Not enough frames for the largest temporal stride.")

    n_frames = len(frames)
    out: Dict[str, List[float]] = {}
    keys_template = ("mag_mean", "curl_abs_mean", "div_mean", "dir_entropy")
    for dt in temporal_strides:
        for scale in ("full", "large", "small"):
            for k in keys_template:
                out[f"flow_dt{dt}_{scale}_{k}"] = [0.0] * n_frames

    # Down-sample for the dense flow (cost) — same convention as the single-
    # scale extractor.
    h_small, w_small = 192, 192
    smalls = [cv2.resize(f, (w_small, h_small), interpolation=cv2.INTER_AREA)
              for f in frames]

    for t in range(n_frames):
        for dt in temporal_strides:
            if t < dt:
                continue
            flow = cv2.calcOpticalFlowFarneback(
                smalls[t - dt], smalls[t], None,
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
            uv_full = flow.astype(np.float32)
            uv_large = _spatial_lowpass_flow(
                uv_full, sigma_px=spatial_lowpass_sigma_px)
            uv_small = uv_full - uv_large

            for scale, uv in (("full", uv_full),
                              ("large", uv_large),
                              ("small", uv_small)):
                u, v = uv[..., 0], uv[..., 1]
                mag = np.hypot(u, v)
                curl, divg = _flow_curl_div(u, v)
                out[f"flow_dt{dt}_{scale}_mag_mean"][t] = float(mag.mean())
                out[f"flow_dt{dt}_{scale}_curl_abs_mean"][t] = float(np.abs(curl).mean())
                out[f"flow_dt{dt}_{scale}_div_mean"][t] = float(divg.mean())
                out[f"flow_dt{dt}_{scale}_dir_entropy"][t] = _direction_entropy(u, v)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 3) Per-frame spatial complexity
#
# Reductions of within-frame structure to one number per frame. They live
# in the "whole-image" file because every output is a single scalar per
# frame; the patch-grid maps that *underlie* some of them are exposed by
# `per_patch.extract_spatial_field`.
# ---------------------------------------------------------------------------

def _radial_psd_slope(gray: np.ndarray, k_min_frac: float = 0.05,
                      k_max_frac: float = 0.5) -> float:
    """Slope of log10(P(k)) vs log10(k) for the radially-averaged 2-D power."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) - gray.mean()))
    p = np.abs(f) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(p.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), p.ravel())
    nr = np.bincount(r_int.ravel())
    radial = tbin / np.maximum(nr, 1)
    k = np.arange(len(radial))
    k_max = min(cy, cx)
    lo, hi = int(k_min_frac * k_max), int(k_max_frac * k_max)
    if hi <= lo + 3:
        return float("nan")
    kk = k[lo:hi]
    pp = radial[lo:hi]
    pp = np.where(pp > 0, pp, np.nan)
    if np.isnan(pp).all():
        return float("nan")
    valid = ~np.isnan(pp)
    slope, _ = np.polyfit(np.log10(kk[valid]), np.log10(pp[valid]), 1)
    return float(slope)


def _box_count_dim(binary: np.ndarray,
                   sizes: Optional[List[int]] = None) -> float:
    """Box-counting fractal dimension of a binary mask."""
    if sizes is None:
        s = min(binary.shape)
        sizes = [s // d for d in (2, 3, 4, 6, 8, 12, 16) if s // d >= 2]
        sizes = sorted(set(sizes))
    counts = []
    for sz in sizes:
        h = (binary.shape[0] // sz) * sz
        w = (binary.shape[1] // sz) * sz
        if h == 0 or w == 0:
            continue
        b = binary[:h, :w]
        S = b.reshape(h // sz, sz, w // sz, sz).any(axis=(1, 3))
        counts.append(int(S.sum()))
    if len(counts) < 3:
        return float("nan")
    sizes = sizes[:len(counts)]
    slope, _ = np.polyfit(np.log(1.0 / np.asarray(sizes, float)),
                          np.log(np.asarray(counts, float)), 1)
    return float(slope)


def _lacunarity(binary: np.ndarray,
                sizes: Optional[List[int]] = None) -> float:
    """
    Mean lacunarity over multiple box sizes (Plotnick 1993; Allgood 1991).
    Lambda(r) = 1 + Var[N] / (E[N])^2; returns log(mean(Lambda)).
    """
    b = binary.astype(np.float32)
    if sizes is None:
        s = min(binary.shape)
        sizes = [s // d for d in (4, 6, 8, 12, 16, 24)
                 if 2 <= s // d <= s // 2]
        sizes = sorted(set(sizes))
    if not sizes:
        return float("nan")
    lambdas = []
    for sz in sizes:
        h = (binary.shape[0] // sz) * sz
        w = (binary.shape[1] // sz) * sz
        if h == 0 or w == 0:
            continue
        bb = b[:h, :w].reshape(h // sz, sz, w // sz, sz)
        counts = bb.sum(axis=(1, 3))
        m = float(counts.mean())
        v = float(counts.var())
        if m > 1e-9:
            lambdas.append(1.0 + v / (m * m))
    if not lambdas:
        return float("nan")
    return float(np.log(np.mean(lambdas)))


def _patch_shannon_entropy_field(gray: np.ndarray, patch: int = 16,
                                 bins: int = 32) -> np.ndarray:
    """Per-patch Shannon entropy as a 2-D map (rows in y, cols in x)."""
    h, w = gray.shape
    H = (h // patch) * patch
    W = (w // patch) * patch
    if H == 0 or W == 0:
        return np.zeros((0, 0), dtype=np.float32)
    g = gray[:H, :W].reshape(H // patch, patch, W // patch, patch).swapaxes(1, 2)
    g = g.reshape(g.shape[0], g.shape[1], -1)
    out = np.zeros(g.shape[:2], dtype=np.float32)
    edges = np.linspace(0, 256, bins + 1)
    for j in range(g.shape[0]):
        for i in range(g.shape[1]):
            hist, _ = np.histogram(g[j, i], bins=edges, density=True)
            hist = hist[hist > 0]
            out[j, i] = float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0
    return out


def _patch_shannon_entropy(gray: np.ndarray, patch: int = 16,
                           bins: int = 32) -> float:
    """Backward-compatible scalar wrapper around `_patch_shannon_entropy_field`."""
    field = _patch_shannon_entropy_field(gray, patch=patch, bins=bins)
    return float(field.mean()) if field.size else float("nan")


def _glcm_contrast_homogeneity(gray: np.ndarray, levels: int = 16,
                               d: int = 1) -> Tuple[float, float]:
    """
    Lightweight GLCM (no scikit-image): mean contrast and homogeneity across
    horizontal, vertical and the two diagonal offsets at pixel distance `d`.
    """
    q = (gray.astype(np.int32) * (levels - 1) // 255).clip(0, levels - 1)
    H, W = q.shape
    contrasts: List[float] = []
    homogs: List[float] = []
    for dy, dx in ((0, d), (d, 0), (d, d), (d, -d)):
        if abs(dy) >= H or abs(dx) >= W:
            continue
        ya0, ya1 = (0, H - dy) if dy >= 0 else (-dy, H)
        yb0, yb1 = (dy, H) if dy >= 0 else (0, H + dy)
        xa0, xa1 = (0, W - dx) if dx >= 0 else (-dx, W)
        xb0, xb1 = (dx, W) if dx >= 0 else (0, W + dx)
        a = q[ya0:ya1, xa0:xa1].ravel()
        b = q[yb0:yb1, xb0:xb1].ravel()
        gl = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(gl, (a, b), 1.0)
        gl /= max(gl.sum(), 1.0)
        ii, jj = np.indices((levels, levels))
        contrasts.append(float(((ii - jj) ** 2 * gl).sum()))
        homogs.append(float((gl / (1.0 + (ii - jj) ** 2)).sum()))
    if not contrasts:
        return float("nan"), float("nan")
    return float(np.mean(contrasts)), float(np.mean(homogs))


def extract_spatial_complexity(path: str, target_long: int = 192,
                               stride: int = 1,
                               max_frames: Optional[int] = None,
                               canny_low: int = 35, canny_high: int = 110,
                               patch_size: int = 24
                               ) -> Dict[str, np.ndarray]:
    """
    Returns dict keys (1-D, frame-rate):
      edge_density, spatial_psd_slope,
      fractal_dim_edges, fractal_dim_grad,
      lacunarity_edges, lacunarity_grad,
      patch_entropy, patch_entropy_std, patch_entropy_p95,
      patch_entropy_spread,
      glcm_contrast, glcm_homogeneity

    `fractal_dim_*` are box-counting dimensions on two binarisations: the
    Canny edge map (sharp foam contours) and the gradient-magnitude image
    above the 60th percentile (graded turbulent structure). `lacunarity_*`
    is Plotnick / Allgood gappiness on the same two inputs.
    """
    ed, slope = [], []
    fd_edges, fd_grad = [], []
    lac_edges, lac_grad = [], []
    pe_mean, pe_std, pe_p95, pe_spread = [], [], [], []
    gc, gh = [], []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        edges = cv2.Canny(gray, canny_low, canny_high) > 0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.hypot(gx, gy)
        gthr = float(np.percentile(gmag, 60.0))
        grad_bin = gmag > max(gthr, 1.0)

        ed.append(float(edges.mean()))
        slope.append(_radial_psd_slope(gray))
        fd_edges.append(_box_count_dim(edges))
        fd_grad.append(_box_count_dim(grad_bin))
        lac_edges.append(_lacunarity(edges))
        lac_grad.append(_lacunarity(grad_bin))

        field = _patch_shannon_entropy_field(gray, patch=patch_size)
        if field.size:
            pe_mean.append(float(field.mean()))
            pe_std.append(float(field.std()))
            pe_p95.append(float(np.percentile(field, 95)))
            pe_spread.append(float(field.max() - field.min()))
        else:
            pe_mean.append(np.nan)
            pe_std.append(np.nan)
            pe_p95.append(np.nan)
            pe_spread.append(np.nan)
        c, h = _glcm_contrast_homogeneity(gray)
        gc.append(c)
        gh.append(h)
    return dict(edge_density=np.asarray(ed),
                spatial_psd_slope=np.asarray(slope),
                # Backward-compatible alias for the edge-map fractal dim.
                fractal_dim=np.asarray(fd_edges),
                fractal_dim_edges=np.asarray(fd_edges),
                fractal_dim_grad=np.asarray(fd_grad),
                lacunarity_edges=np.asarray(lac_edges),
                lacunarity_grad=np.asarray(lac_grad),
                patch_entropy=np.asarray(pe_mean),
                patch_entropy_std=np.asarray(pe_std),
                patch_entropy_p95=np.asarray(pe_p95),
                patch_entropy_spread=np.asarray(pe_spread),
                glcm_contrast=np.asarray(gc),
                glcm_homogeneity=np.asarray(gh))


__all__ = [
    "extract_global_signals",
    "extract_optical_flow_signals",
    "extract_optical_flow_multiscale",
    "extract_spatial_complexity",
    # private helpers re-exported for the per_patch module to reuse
    "_patch_shannon_entropy_field", "_patch_shannon_entropy",
    "_box_count_dim", "_lacunarity",
    "_glcm_contrast_homogeneity", "_radial_psd_slope",
    "_flow_curl_div", "_direction_entropy", "_spatial_lowpass_flow",
]
