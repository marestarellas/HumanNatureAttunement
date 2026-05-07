"""
Video-derived oscillation & complexity features for coupling with EEG / HRV / EMG.

Produces 1-D signals sampled at the video frame rate (`fs = fps`) plus a few
scalar summaries. The 1-D signals are drop-in inputs for the coupling tools in
`HNA.modules.coupling` (`windowed_xcorr`, `band_coherence`, `plv_phase_sync`,
`wpli_phase_sync`, ...).

Feature families implemented (see Bellemare/Estarellas project notes for the
literature scaffold):

  - Global temporal envelope        : luminance / blue / spatial-std / frame-diff
  - Optical flow (Farneback)        : mean |v|, mean |curl|, mean div, dir entropy
  - Per-frame spatial complexity    : edge density, spatial Shannon entropy,
                                      radial 2D-FFT slope, box-counting D,
                                      patch entropy (mean / std / p95 / spread)
  - Per-frame 2D FFT signals        : peak radial wavenumber, anisotropy,
                                      dominant orientation (richer than slope alone)
  - Per-pixel temporal spectrum     : FFT of luminance(t) at every pixel ->
                                      maps of peak frequency, band power,
                                      spectral entropy + 1-D synchronization /
                                      coherence aggregates
  - Spatio-temporal modal           : DMD top modes (optional, needs `pydmd`)
                                      + plain SVD/POD fallback
  - Oceanographic                   : timestack 1-D-FFT dominant period
  - Nonlinear-dynamics summaries    : permutation entropy + DFA + 1/f slope
                                      + dominant-frequency / band-power on every
                                      signal (uses antropy / neurokit2 if available)

All heavy operations run on a downscaled grayscale copy of the frame (default
192px on the long axis) so a minute of HD footage fits in memory and runs in
seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("video.py requires opencv-python (`pip install opencv-python`)") from e

try:
    import antropy as _ant
    _HAVE_ANTROPY = True
except ImportError:
    _HAVE_ANTROPY = False

try:
    import neurokit2 as _nk
    _HAVE_NK = True
except ImportError:
    _HAVE_NK = False

try:
    from pydmd import DMD as _PyDMD
    _HAVE_PYDMD = True
except ImportError:
    _HAVE_PYDMD = False

from scipy import signal as _sps


# ---------------------------------------------------------------------------
# Frame I/O
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
                     width=w, height=h, duration_s=(n / fps if fps > 0 else 0.0))


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
# 1) Global temporal envelope (cheap, frame-rate sampled)
# ---------------------------------------------------------------------------

def _spatial_entropy(gray: np.ndarray, bins: int = 64) -> float:
    h, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def extract_global_signals(path: str, target_long: int = 192,
                           stride: int = 1, max_frames: Optional[int] = None
                           ) -> Dict[str, np.ndarray]:
    """
    Per-frame scalars: the 'video envelope' analogues to an audio envelope.

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
        bm.append(float(b.mean())); gm.append(float(g.mean())); rm.append(float(r.mean()))
        sstd.append(float(gray.std()))
        sent.append(_spatial_entropy(gray))
        if prev_gray is None:
            fdiff.append(0.0)
        else:
            fdiff.append(float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean()))
        prev_gray = gray
    return dict(luminance=np.asarray(lum), blue_mean=np.asarray(bm),
                green_mean=np.asarray(gm), red_mean=np.asarray(rm),
                spatial_std=np.asarray(sstd), spatial_entropy=np.asarray(sent),
                frame_diff=np.asarray(fdiff))


# ---------------------------------------------------------------------------
# 2) Optical flow (Farneback) — wave-motion descriptors
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
        dt=9 -> ~0.4 s motion (close to a swell quarter-period). The flow is
        between frame t and frame t-dt.

      * Spatial decomposition of the flow field. After computing flow, the
        (u, v) field is split into a low-pass component (Gaussian sigma
        `spatial_lowpass_sigma_px`) and the high-pass residual. The
        large-scale component captures swell direction/speed; the small-
        scale component captures turbulent motion.

    Returned keys (one set per dt + scale):

      flow_dtN_<scale>_mag_mean    : mean of |flow|
      flow_dtN_<scale>_curl_abs_mean : mean of |curl|
      flow_dtN_<scale>_div_mean    : mean of div
      flow_dtN_<scale>_dir_entropy : direction entropy

    where N is the temporal stride and <scale> is one of {full, large, small}.
    All series are 1-D, length = number of decoded frames; the first N
    samples of each dt-N series are 0 (no past frame).
    """
    # Decode all frames at target_long, keep grayscale stack
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
            uv_large = _spatial_lowpass_flow(uv_full, sigma_px=spatial_lowpass_sigma_px)
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


def extract_optical_flow_signals(path: str, target_long: int = 192,
                                 stride: int = 1, max_frames: Optional[int] = None,
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
            "flow_div_mean", "flow_dir_entropy", "flow_u_mean", "flow_v_mean"]
    out = {k: [] for k in keys}
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


# ---------------------------------------------------------------------------
# 3) Per-frame spatial complexity descriptors
# ---------------------------------------------------------------------------

def _radial_psd_slope(gray: np.ndarray, k_min_frac: float = 0.05,
                      k_max_frac: float = 0.5) -> float:
    """
    Slope of log10(P(k)) vs log10(k) for radially-averaged 2D power spectrum.
    Sea/foam textures typically give ~ -1 to -3.
    """
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
    kk = k[lo:hi]; pp = radial[lo:hi]
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


def _lacunarity(binary: np.ndarray, sizes: Optional[List[int]] = None) -> float:
    """
    Mean lacunarity over multiple box sizes (Plotnick 1993; Allgood 1991).
    Lambda(r) = E[N^2] / (E[N])^2, where N is the count of occupied pixels
    in a box of size r. Returns log(mean(Lambda)) - useful as a single
    "gappiness" scalar; high values indicate a clumpy / heterogeneous
    occupation pattern at multiple scales.
    """
    b = binary.astype(np.float32)
    if sizes is None:
        s = min(binary.shape)
        sizes = [s // d for d in (4, 6, 8, 12, 16, 24) if 2 <= s // d <= s // 2]
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
    """
    Per-patch Shannon entropy as a 2-D map (rows = patches in y, cols = in x).
    Useful both as a per-frame scalar (mean) and as a 2-D complexity image.
    """
    h, w = gray.shape
    H = (h // patch) * patch; W = (w // patch) * patch
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
                               stride: int = 1, max_frames: Optional[int] = None,
                               canny_low: int = 35, canny_high: int = 110,
                               patch_size: int = 24
                               ) -> Dict[str, np.ndarray]:
    """
    Returns dict keys (1-D, frame-rate):
      edge_density, spatial_psd_slope,
      fractal_dim_edges, fractal_dim_grad, lacunarity_edges, lacunarity_grad,
      patch_entropy, patch_entropy_std, patch_entropy_p95, patch_entropy_spread,
      glcm_contrast, glcm_homogeneity

    `fractal_dim_*` are box-counting dimensions applied to two binarisations
    of the frame: the Canny edge map (sharp foam contours) and a thresholded
    gradient-magnitude image (graded turbulent structure, captures more than
    edges). `lacunarity_*` is Plotnick/Allgood gappiness on the same two
    inputs. All are 1-D coupling-ready time series.
    """
    ed, slope = [], []
    fd_edges, fd_grad = [], []
    lac_edges, lac_grad = [], []
    pe_mean, pe_std, pe_p95, pe_spread = [], [], [], []
    gc, gh = [], []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        edges = cv2.Canny(gray, canny_low, canny_high) > 0
        # gradient-magnitude binarisation (Otsu-ish: threshold at 60th pct)
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
            pe_mean.append(np.nan); pe_std.append(np.nan)
            pe_p95.append(np.nan); pe_spread.append(np.nan)
        c, h = _glcm_contrast_homogeneity(gray)
        gc.append(c); gh.append(h)
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


# ---------------------------------------------------------------------------
# 3b) Richer per-frame 2D-FFT signals (peak wavenumber / anisotropy / orientation)
# ---------------------------------------------------------------------------

def _spatial_fft_descriptors(gray: np.ndarray) -> Tuple[float, float, float]:
    """
    Single-frame 2-D FFT descriptors that complement the radial-PSD slope:

      peak_k        : normalized radial wavenumber of the spectral peak
                      (in [0,1] where 1 is Nyquist), excluding DC
      anisotropy    : log-ratio of horizontal to vertical band power
                      (positive = horizontally striped pattern, e.g. distant
                      swell crests; negative = vertically striped)
      orientation   : dominant orientation of the 2-D spectrum, in radians
                      [0, pi); 0 = horizontal stripes, pi/2 = vertical stripes
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
    # mask out DC neighborhood
    mask_ring = (r > 2) & (r < 0.95 * r_max)
    if not mask_ring.any():
        return float("nan"), float("nan"), float("nan")
    Pm = P * mask_ring

    # peak wavenumber
    flat_idx = int(np.argmax(Pm))
    py, px = np.unravel_index(flat_idx, P.shape)
    peak_k = float(np.sqrt((py - cy) ** 2 + (px - cx) ** 2) / r_max)

    # Anisotropy: log-ratio of power in horizontal-stripe-content (Fourier
    # power on the ky axis, i.e. angles near +/- pi/2) over vertical-stripe-
    # content (Fourier power on the kx axis, i.e. angles near 0 or +/- pi).
    ang = np.arctan2(ky, kx)  # in (-pi, pi]
    wedge = np.deg2rad(15)
    near_vaxis = np.abs(np.abs(ang) - np.pi / 2) < wedge        # ky-axis
    near_haxis = (np.abs(ang) < wedge) | (np.abs(np.abs(ang) - np.pi) < wedge)  # kx-axis
    band_horiz_stripes = mask_ring & near_vaxis
    band_vert_stripes  = mask_ring & near_haxis
    ph = P[band_horiz_stripes].sum() + 1e-9
    pv = P[band_vert_stripes].sum() + 1e-9
    anisotropy = float(np.log(ph / pv))

    # weighted-mean orientation (using the moment of inertia of P over the ring)
    ang_mod = ang % np.pi  # collapse to [0, pi)
    w_pow = Pm.copy()
    s2 = float((np.sin(2 * ang_mod) * w_pow).sum())
    c2 = float((np.cos(2 * ang_mod) * w_pow).sum())
    orientation = 0.5 * float(np.arctan2(s2, c2))
    if orientation < 0:
        orientation += np.pi
    return peak_k, anisotropy, orientation


def extract_spatial_fft_signals(path: str, target_long: int = 192,
                                stride: int = 1,
                                max_frames: Optional[int] = None
                                ) -> Dict[str, np.ndarray]:
    """
    Per-frame summaries of the 2-D Fourier spectrum that go beyond the radial
    slope: dominant spatial wavenumber, horizontal/vertical anisotropy, and
    weighted-mean orientation. All are 1-D signals at frame rate.
    """
    pk, ani, ori = [], [], []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        a, b, c = _spatial_fft_descriptors(gray)
        pk.append(a); ani.append(b); ori.append(c)
    return dict(spatial_fft_peak_k=np.asarray(pk),
                spatial_fft_anisotropy=np.asarray(ani),
                spatial_fft_orientation=np.asarray(ori))


# ---------------------------------------------------------------------------
# 3c) Per-pixel temporal spectrum (FFT of luminance(t) at every pixel)
# ---------------------------------------------------------------------------

@dataclass
class PixelSpectrumResult:
    fps: float                                # effective sampling rate (fs)
    target_long: int                          # spatial resolution used
    freqs: np.ndarray                         # (n_freqs,)  frequency axis (Hz)
    peak_freq_map: np.ndarray                 # (h, w)      dominant frequency per pixel (Hz)
    peak_power_map: np.ndarray                # (h, w)      power at the dominant frequency
    spectral_entropy_map: np.ndarray          # (h, w)      Shannon entropy of normalized spectrum
    band_power_maps: Dict[str, np.ndarray]    # band_name -> (h, w)  band-integrated power
    mean_spectrum: np.ndarray                 # (n_freqs,)  mean power spectrum across pixels
    sync_index: float                         # how concentrated the *mean* spectrum is
    coherence_index: float                    # how spatially uniform the per-pixel peak freq is
    bands: Tuple[Tuple[str, float, float], ...]


_DEFAULT_SPEC_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("low",  0.05, 0.25),
    ("mid",  0.25, 0.50),
    ("high", 0.50, 2.00),
)


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
      - global scalars: synchronization index (concentration of the mean
        spectrum) and coherence index (spatial uniformity of peak_freq_map)

    The 2-D maps are themselves spatial images that can be inspected
    visually -- for sea waves the `peak_freq_map` typically reveals which
    parts of the frame oscillate at the swell period vs. at finer foam
    timescales.

    Memory: builds a (T, h, w) float32 stack (~ T * target_long^2 * 4 bytes).
    """
    # 1) stack frames at target_long
    frames: List[np.ndarray] = []
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        frames.append(gray.astype(np.float32))
    if len(frames) < 8:
        raise RuntimeError("Need at least ~8 frames for a useful pixel spectrum.")
    X = np.stack(frames, axis=0)        # (T, h, w)
    T, h, w = X.shape

    # 2) detrend per pixel (remove linear trend or just the mean)
    if detrend:
        # linear detrend along time axis using least-squares slope/intercept
        t_axis = np.arange(T, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = float(((t_axis - t_mean) ** 2).sum() + 1e-9)
        x_mean = X.mean(axis=0, keepdims=True)
        slope = ((t_axis - t_mean)[:, None, None] * (X - x_mean)).sum(axis=0) / t_var
        X = X - x_mean - slope[None, :, :] * (t_axis - t_mean)[:, None, None]
    else:
        X = X - X.mean(axis=0, keepdims=True)

    # 3) Hann taper to reduce spectral leakage, then real-input FFT along time
    win = np.hanning(T).astype(np.float32)
    Xt = X * win[:, None, None]
    F = np.fft.rfft(Xt, axis=0)         # (n_freqs, h, w), complex
    P = (F.real ** 2 + F.imag ** 2)     # (n_freqs, h, w), power
    freqs = np.fft.rfftfreq(T, d=1.0 / float(fps)).astype(np.float32)

    # ignore DC bin in all derived statistics
    P_ac = P.copy()
    P_ac[0] = 0.0

    # 4) per-pixel maps
    peak_idx = np.argmax(P_ac, axis=0)              # (h, w)
    peak_freq_map = freqs[peak_idx]                  # (h, w)
    peak_power_map = np.take_along_axis(P_ac, peak_idx[None, :, :], axis=0)[0]

    P_norm = P_ac / (P_ac.sum(axis=0, keepdims=True) + 1e-12)
    P_safe = np.where(P_norm > 0, P_norm, 1.0)
    log2_safe = np.where(P_norm > 0, np.log2(P_safe), 0.0)
    H = -(P_norm * log2_safe).sum(axis=0)
    n_bins = P_ac.shape[0] - 1            # excluding DC
    spectral_entropy_map = (H / np.log2(max(n_bins, 2))).astype(np.float32)

    band_power_maps: Dict[str, np.ndarray] = {}
    for name, fmin, fmax in bands:
        m = (freqs >= fmin) & (freqs < fmax)
        band_power_maps[name] = P_ac[m].sum(axis=0) if m.any() else \
            np.zeros((h, w), dtype=np.float32)

    # 5) 1-D aggregates
    mean_spectrum = P_ac.mean(axis=(1, 2))
    ms_norm = mean_spectrum / (mean_spectrum.sum() + 1e-12)
    ms_safe = np.where(ms_norm > 0, ms_norm, 1.0)
    H_ms = -(ms_norm * np.where(ms_norm > 0, np.log2(ms_safe), 0.0)).sum()
    sync_index = float(1.0 - H_ms / np.log2(max(n_bins, 2)))   # 1 = perfectly peaked

    # spatial uniformity of peak_freq_map: 1 - normalized std
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


# ---------------------------------------------------------------------------
# 4) Spatio-temporal modal decomposition (DMD / SVD-POD fallback)
# ---------------------------------------------------------------------------

@dataclass
class ModalResult:
    method: str                              # "dmd" or "svd"
    fps: float
    frequencies_hz: np.ndarray               # length k
    growth_rates: np.ndarray                 # length k (real part of eigvals)
    energies: np.ndarray                     # length k, normalized [0,1]
    temporal_coeffs: np.ndarray              # (k, n_frames) -- 1-D per mode
    spatial_modes_shape: Tuple[int, int]     # (h, w) of the spatial modes


def _stack_video(path: str, target_long: int, stride: int,
                 max_frames: Optional[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    frames = []
    shape = None
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        if shape is None:
            shape = gray.shape
        frames.append(gray.astype(np.float32).ravel())
    if not frames:
        raise RuntimeError("No frames decoded.")
    X = np.stack(frames, axis=1)        # (pixels, time)
    X -= X.mean(axis=1, keepdims=True)  # remove temporal mean per pixel
    return X, shape


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
        # PyDMD svd_rank=2k so we have headroom for conjugate pairs;
        # we'll de-duplicate and keep top-k unique frequencies after.
        dmd = _PyDMD(svd_rank=int(2 * k)).fit(X)
        eigs = np.asarray(dmd.eigs)
        log_lam = np.log(eigs + 1e-30) * float(fps)
        freqs = np.abs(log_lam.imag) / (2 * np.pi)
        growth = log_lam.real
        dyn = np.asarray(dmd.dynamics)               # (k, T) complex
        modes = np.asarray(dmd.modes)                # (P, k) complex
        amps = np.linalg.norm(modes, axis=0) * np.abs(dyn).mean(axis=1)

        # De-duplicate conjugate pairs: each unique |f| keeps its highest-amp
        # representative (with non-negative imag of eig if available).
        keys = np.round(freqs, 4)
        seen: dict = {}
        for i, key in enumerate(keys):
            cur = seen.get(key)
            if cur is None or amps[i] > amps[cur]:
                seen[key] = i
        keep = np.array(sorted(seen.values()), dtype=int)
        freqs = freqs[keep]; growth = growth[keep]
        amps = amps[keep]; dyn = dyn[keep]

        order = np.argsort(-amps)[:k]
        freqs = freqs[order]; growth = growth[order]
        amps = amps[order]; dyn = dyn[order]
        energies = amps / (amps.max() + 1e-12) if amps.size else amps
        return ModalResult("dmd", float(fps),
                           freqs.astype(float), growth.astype(float),
                           energies.astype(float),
                           np.real(dyn).astype(np.float64), shape)

    # SVD/POD fallback (paired sin/cos modes at f_p; useful for diagnostics)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Vt = Vt[:k]; S = S[:k]
    coeffs = (S[:, None] * Vt)
    energies = (S ** 2); energies = energies / (energies[0] + 1e-12)
    freqs = np.zeros(k); growth = np.zeros(k)
    for i in range(k):
        f, P = _sps.welch(coeffs[i], fs=fps,
                          nperseg=min(len(coeffs[i]), 256))
        if len(f) > 1:
            freqs[i] = float(f[np.argmax(P[1:]) + 1])
    return ModalResult("svd", fps, freqs, growth, energies, coeffs, shape)


# ---------------------------------------------------------------------------
# 5) Timestack — oceanographic dominant wave period
# ---------------------------------------------------------------------------

@dataclass
class TimestackResult:
    fps: float
    column_index: int
    dominant_freq_hz: float
    dominant_period_s: float
    psd_freqs: np.ndarray
    psd_power: np.ndarray
    timestack_image: np.ndarray            # (n_frames, height_small)


def extract_timestack(path: str, fps: float, target_long: int = 192,
                      column_frac: float = 0.5,
                      stride: int = 1, max_frames: Optional[int] = None,
                      f_min: float = 0.05, f_max: float = 2.0
                      ) -> TimestackResult:
    """
    Sample one vertical pixel column over time; FFT its row-mean to find the
    dominant wave frequency. `column_frac` selects which column (0..1).
    """
    rows = []
    h_small = None
    for _, gray, _ in iter_frames(path, target_long=target_long,
                                  stride=stride, max_frames=max_frames):
        if h_small is None:
            h_small = gray.shape[0]
            col_idx = int(np.clip(column_frac * gray.shape[1], 0, gray.shape[1] - 1))
        rows.append(gray[:, col_idx].astype(np.float32))
    ts = np.stack(rows, axis=0)            # (n_frames, height)
    sig = ts.mean(axis=1)
    sig = sig - sig.mean()
    nper = min(len(sig), int(round(fps * 8)))
    if nper < 16:
        nper = len(sig)
    f, P = _sps.welch(sig, fs=fps, nperseg=nper)
    band = (f >= f_min) & (f <= f_max)
    if band.any():
        i = np.argmax(P[band])
        peak_f = float(f[band][i])
    else:
        peak_f = float("nan")
    return TimestackResult(
        fps=fps, column_index=col_idx,
        dominant_freq_hz=peak_f,
        dominant_period_s=(1.0 / peak_f) if peak_f and np.isfinite(peak_f) else float("nan"),
        psd_freqs=f, psd_power=P, timestack_image=ts)


# ---------------------------------------------------------------------------
# 6) Nonlinear-dynamics summaries on any 1-D signal
# ---------------------------------------------------------------------------

def _hjorth(x: np.ndarray) -> Tuple[float, float, float]:
    """Hjorth (1970) activity, mobility, complexity."""
    x = np.asarray(x, float)
    dx = np.diff(x); ddx = np.diff(dx)
    var_x = np.var(x); var_dx = np.var(dx); var_ddx = np.var(ddx)
    activity = float(var_x)
    mobility = float(np.sqrt(var_dx / var_x)) if var_x > 0 else np.nan
    if var_dx > 0 and mobility > 0:
        complexity = float(np.sqrt(var_ddx / var_dx) / mobility)
    else:
        complexity = float("nan")
    return activity, mobility, complexity


def complexity_summary(x: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Nonlinear-dynamics summary of a 1-D signal. Returned keys:

      perm_entropy, sample_entropy, approx_entropy, spectral_entropy,
      svd_entropy, higuchi_fd, katz_fd, petrosian_fd,
      hjorth_mobility, hjorth_complexity,
      dfa_alpha, hurst, lz_complexity,
      mse_area, psd_slope

    Missing optional libraries (antropy / neurokit2) -> NaN for those keys.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    keys = ("perm_entropy sample_entropy approx_entropy spectral_entropy "
            "svd_entropy higuchi_fd katz_fd petrosian_fd "
            "hjorth_mobility hjorth_complexity "
            "dfa_alpha hurst lz_complexity mse_area psd_slope "
            "peak_freq peak_power spectral_centroid spectral_bandwidth "
            "band_power_low band_power_mid band_power_high").split()
    out: Dict[str, float] = {k: float("nan") for k in keys}
    if x.size < 32 or np.std(x) < 1e-12:
        return out

    if _HAVE_ANTROPY:
        try: out["perm_entropy"] = float(_ant.perm_entropy(x, normalize=True))
        except Exception: pass
        try: out["sample_entropy"] = float(_ant.sample_entropy(x))
        except Exception: pass
        try: out["approx_entropy"] = float(_ant.app_entropy(x))
        except Exception: pass
        try: out["spectral_entropy"] = float(_ant.spectral_entropy(x, sf=fs, normalize=True))
        except Exception: pass
        try: out["svd_entropy"] = float(_ant.svd_entropy(x, normalize=True))
        except Exception: pass
        try: out["higuchi_fd"] = float(_ant.higuchi_fd(x))
        except Exception: pass
        try: out["katz_fd"] = float(_ant.katz_fd(x))
        except Exception: pass
        try: out["petrosian_fd"] = float(_ant.petrosian_fd(x))
        except Exception: pass

    _, mob, cmpl = _hjorth(x)
    out["hjorth_mobility"] = mob
    out["hjorth_complexity"] = cmpl

    if _HAVE_NK:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                v, _ = _nk.fractal_dfa(x); out["dfa_alpha"] = float(v)
            except Exception: pass
            try:
                v, _ = _nk.fractal_hurst(x); out["hurst"] = float(v)
            except Exception: pass
            try:
                v, _ = _nk.complexity_lempelziv(x); out["lz_complexity"] = float(v)
            except Exception: pass
            # Multiscale entropy area-under-curve (single scalar summary)
            try:
                v, _ = _nk.entropy_multiscale(x, scale="default", method="MSEn")
                out["mse_area"] = float(v)
            except Exception: pass

    # Spectral features: peak frequency / power, band powers, spectral
    # centroid, bandwidth, and the 1/f aperiodic slope.
    f, P = _sps.welch(x, fs=fs, nperseg=min(len(x), 256))
    m = (f > 0) & np.isfinite(P)
    if m.sum() > 4:
        f_pos = f[m]; P_pos = P[m]
        slope, _ = np.polyfit(np.log10(f_pos[P_pos > 0]),
                              np.log10(P_pos[P_pos > 0]), 1)
        out["psd_slope"] = float(slope)
        i_pk = int(np.argmax(P_pos))
        out["peak_freq"] = float(f_pos[i_pk])
        out["peak_power"] = float(P_pos[i_pk])
        # spectral centroid & bandwidth (proxy for "where the energy is")
        Pn = P_pos / (P_pos.sum() + 1e-12)
        centroid = float((f_pos * Pn).sum())
        out["spectral_centroid"] = centroid
        out["spectral_bandwidth"] = float(np.sqrt(((f_pos - centroid) ** 2 * Pn).sum()))
        # band powers in physiologically interesting bands (HRV-ish for video)
        for name, lo, hi in (("low", 0.05, 0.25),
                             ("mid", 0.25, 0.50),
                             ("high", 0.50, 2.00)):
            sel = (f_pos >= lo) & (f_pos < hi)
            out[f"band_power_{name}"] = float(P_pos[sel].sum()) if sel.any() else float("nan")
    else:
        for k in ("peak_freq", "peak_power", "spectral_centroid",
                  "spectral_bandwidth",
                  "band_power_low", "band_power_mid", "band_power_high"):
            out[k] = float("nan")
    return out


def windowed_complexity(x: np.ndarray, fs: float,
                        win_sec: float = 4.0, step_sec: float = 0.25,
                        measures: Tuple[str, ...] = (
                            "perm_entropy", "hjorth_complexity",
                            "higuchi_fd", "spectral_entropy"),
                        ) -> Dict[str, np.ndarray]:
    """
    Time-resolved complexity: slide a window of `win_sec` seconds with hop
    `step_sec` and compute the requested measures at each step. Returns a
    dict with keys 'times_s' plus one 1-D array per measure -- so complexity
    *itself* becomes a signal that can be cross-correlated with EEG/HRV/EMG.

    `measures` is a subset of the keys returned by `complexity_summary`.
    """
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, np.nan)
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    if W < 16:
        raise ValueError("win_sec * fs must be >= 16 samples")
    starts = np.arange(0, max(len(x) - W + 1, 0), H)
    out: Dict[str, List[float]] = {m: [] for m in measures}
    times: List[float] = []
    for st in starts:
        seg = x[st:st + W]
        seg = seg[np.isfinite(seg)]
        if seg.size < 16 or np.std(seg) < 1e-12:
            for m in measures:
                out[m].append(np.nan)
        else:
            s = complexity_summary(seg, fs=fs)
            for m in measures:
                out[m].append(s.get(m, np.nan))
        times.append((st + W / 2) / fs)
    res: Dict[str, np.ndarray] = {"times_s": np.asarray(times)}
    for m in measures:
        res[m] = np.asarray(out[m])
    return res


# ---------------------------------------------------------------------------
# 7) High-level pipeline
# ---------------------------------------------------------------------------

@dataclass
class VideoFeatures:
    meta: VideoMeta
    fs: float                              # effective sampling rate of 1-D signals
    signals: Dict[str, np.ndarray]         # 1-D arrays, all length n_used
    complexity: Dict[str, Dict[str, float]]  # per-signal nonlinear summaries
    timestack: Optional[TimestackResult] = None
    modal: Optional[ModalResult] = None
    pixel_spectrum: Optional["PixelSpectrumResult"] = None
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
                   modal_k: int = 4,
                   modal_target_long: int = 96) -> VideoFeatures:
    """
    Full pipeline on a single video. All extractors share the same frame
    iteration pattern and are run sequentially (ordering picked so the
    cheapest run first).

    The returned `signals` dict contains 1-D arrays of length `n_used` sampled
    at `fs = fps / stride` — pass these directly to the coupling tools.
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
        signals.update(extract_optical_flow_signals(path, target_long, stride, max_frames))
    if include_spatial:
        signals.update(extract_spatial_complexity(path, target_long, stride, max_frames))
    if include_spatial_fft:
        signals.update(extract_spatial_fft_signals(path, target_long, stride, max_frames))

    # Truncate everything to the shortest length (defensive — should already match)
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
            # Resample each complexity time series to the frame grid so it
            # lines up with the other signals (constant fs).
            t_target = np.arange(n_used) / fs
            for m in windowed_complexity_measures:
                y = wc[m]; t = wc["times_s"]
                if y.size >= 2 and np.isfinite(y).any():
                    y_filled = y.copy()
                    bad = ~np.isfinite(y_filled)
                    if bad.any() and (~bad).any():
                        y_filled[bad] = np.interp(t[bad], t[~bad], y_filled[~bad])
                    signals[f"wc_{s_name}__{m}"] = np.interp(
                        t_target, t, y_filled,
                        left=y_filled[0], right=y_filled[-1])

    return VideoFeatures(meta=meta, fs=fs, signals=signals,
                         complexity=complexity, timestack=timestack,
                         modal=modal, pixel_spectrum=pixel_spectrum,
                         notes=notes)


__all__ = [
    "VideoMeta", "VideoFeatures", "ModalResult", "TimestackResult",
    "PixelSpectrumResult",
    "probe_video", "iter_frames",
    "extract_global_signals",
    "extract_optical_flow_signals", "extract_optical_flow_multiscale",
    "extract_spatial_complexity", "extract_spatial_fft_signals",
    "extract_pixel_spectrum",
    "extract_modal", "extract_timestack",
    "complexity_summary", "windowed_complexity", "quantify_video",
]
