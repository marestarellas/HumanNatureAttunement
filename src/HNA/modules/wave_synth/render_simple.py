"""
Tier-1 NumPy renderer: per-pixel raymarched seascape with perspective camera,
analytical sun + sky, Schlick-Fresnel water shading, and foam compositing.

The render loop is fully vectorised over the output image, so a 5-second
640x360 clip renders in seconds on a modern CPU. The same parameter
dataclasses (`Camera`, `Sun`, `Sky`, `WaterSpec`) feed Tier 2 and Tier 3.
"""
from __future__ import annotations

import sys
import time
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from . import Camera, Sun, Sky, WaterSpec
    from .fft_ocean import FFTOcean

import cv2


# ---------------------------------------------------------------------------
# Camera basis
# ---------------------------------------------------------------------------

def _camera_basis(yaw_deg: float, pitch_deg: float):
    """Right-handed world basis: x=east(right), y=north(forward), z=up."""
    p = np.deg2rad(pitch_deg); y = np.deg2rad(yaw_deg)
    forward = np.array([np.cos(p) * np.sin(y),
                         np.cos(p) * np.cos(y),
                         np.sin(p)], dtype=np.float64)
    right = np.array([np.cos(y), -np.sin(y), 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    return forward, right, up


def _ray_directions(W: int, H: int, fov_deg: float, forward, right, up):
    """Return (H, W, 3) unit ray directions for a perspective camera."""
    aspect = W / H
    tan_y = np.tan(np.deg2rad(fov_deg) / 2.0)
    tan_x = tan_y * aspect

    # Pixel centers in normalized device coords (-1..+1)
    xs = (np.arange(W, dtype=np.float64) + 0.5) / W * 2.0 - 1.0
    ys = (np.arange(H, dtype=np.float64) + 0.5) / H * 2.0 - 1.0
    ys = -ys  # image y points down; we want camera-up positive
    Xs, Ys = np.meshgrid(xs, ys, indexing="xy")

    rd = (forward[None, None, :]
          + (Xs * tan_x)[..., None] * right[None, None, :]
          + (Ys * tan_y)[..., None] * up[None, None, :])
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True) + 1e-12
    return rd  # (H, W, 3)


# ---------------------------------------------------------------------------
# Sky model
# ---------------------------------------------------------------------------

def _sky_color(ray_dir: np.ndarray, sky: "Sky", sun_dir: np.ndarray,
               sun_color: np.ndarray, sun_intensity: float) -> np.ndarray:
    """
    Procedural sky: gradient between horizon and zenith colors, plus a
    soft sun disk + halo.
    ray_dir : (..., 3)  unit vectors
    """
    z = np.clip(ray_dir[..., 2], 0.0, 1.0)        # 0 at horizon, 1 at zenith
    horizon = np.asarray(sky.horizon_color, dtype=np.float64)
    zenith = np.asarray(sky.zenith_color, dtype=np.float64)
    base = (1.0 - z[..., None]) * horizon + z[..., None] * zenith
    # Rayleigh-ish darkening near zenith for high turbidity
    base = base * (0.85 + 0.15 / max(sky.turbidity, 1e-3))

    # Sun: soft Gaussian + small bright core
    cos_sun = np.clip((ray_dir * sun_dir).sum(axis=-1), -1.0, 1.0)
    halo = np.exp(-((1.0 - cos_sun) * 60.0)) * 0.6
    core = (cos_sun > 0.9995).astype(np.float64) * 4.0
    sun = (halo + core)[..., None] * sun_color * sun_intensity
    return np.clip(base + sun, 0.0, 8.0)


# ---------------------------------------------------------------------------
# Water shading
# ---------------------------------------------------------------------------

def _shade_water(hit_xyz: np.ndarray, ray_dir: np.ndarray, normal: np.ndarray,
                 foam_intensity: np.ndarray, water: "WaterSpec",
                 sun_dir: np.ndarray, sun_color: np.ndarray, sun_intensity: float,
                 sky: "Sky") -> np.ndarray:
    """
    Vectorised water shading: deep water color + Schlick-Fresnel blend
    with the reflected sky + sun specular highlight + foam.
    All array shapes broadcast.
    """
    n = normal
    v = -ray_dir                                         # view direction (away from surface)
    cos_theta = np.clip((n * v).sum(axis=-1), 0.0, 1.0)  # angle between view and normal

    # Schlick Fresnel (water F0 ~ 0.02)
    F0 = 0.02
    F = F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5

    # Reflected ray
    r = ray_dir - 2.0 * (ray_dir * n).sum(axis=-1, keepdims=True) * n
    sky_refl = _sky_color(r, sky, sun_dir, sun_color, sun_intensity)

    # Diffuse component (mostly contributes via Fresnel transmission below water,
    # but cheaply approximate as a darkened water color)
    deep = np.asarray(water.water_color, dtype=np.float64)
    n_dot_l = np.clip((n * sun_dir).sum(axis=-1), 0.0, 1.0)[..., None]
    diffuse = deep * (0.4 + 0.6 * n_dot_l)

    # Sun specular (Blinn-Phong, sharp)
    half = (sun_dir + v); half /= np.linalg.norm(half, axis=-1, keepdims=True) + 1e-12
    spec_factor = np.clip((n * half).sum(axis=-1), 0.0, 1.0) ** 220
    spec = spec_factor[..., None] * sun_color * sun_intensity * 6.0

    # Composite water = (1-F)*diffuse + F*sky_refl + spec
    water_rgb = (1.0 - F)[..., None] * diffuse + F[..., None] * sky_refl + spec

    # Foam: soft white wash where Jacobian collapsed
    f = np.clip(foam_intensity * water.foam_coverage * 1.6 - water.foam_threshold, 0.0, 1.0)
    foam_base = np.array([0.92, 0.94, 0.96], dtype=np.float64)[None, None, :]
    foam_rgb = foam_base * (0.7 + 0.3 * n_dot_l)   # shape (H, W, 3)
    out = water_rgb * (1 - f[..., None]) + foam_rgb * f[..., None]

    return np.clip(out, 0.0, 8.0)


# ---------------------------------------------------------------------------
# Single-frame render
# ---------------------------------------------------------------------------

def render_frame(ocean: "FFTOcean", t: float,
                 camera: "Camera", sun: "Sun", sky: "Sky",
                 water: "WaterSpec", resolution: Tuple[int, int],
                 max_distance_m: float = 4000.0,
                 march_iters: int = 2,
                 weights: list = None) -> np.ndarray:
    """
    Render one frame at time `t` (seconds). Returns an HxWx3 uint8 RGB image.

    `weights` (length = ocean.n_components) overrides the per-component
    weights for this frame -- enables time-varying amplitude envelopes.
    """
    W, H = resolution
    forward, right, up = _camera_basis(camera.yaw_deg, camera.pitch_deg)
    rd = _ray_directions(W, H, camera.fov_deg, forward, right, up)  # (H, W, 3)
    cam_pos = np.array([0.0, 0.0, float(camera.height_m)])

    # Sun direction (world)
    saz = np.deg2rad(sun.azimuth_deg)
    sel = np.deg2rad(sun.elevation_deg)
    sun_dir = np.array([np.cos(sel) * np.sin(saz),
                        np.cos(sel) * np.cos(saz),
                        np.sin(sel)], dtype=np.float64)
    sun_color = np.asarray(sun.color, dtype=np.float64)

    # Compute the ocean state once (full N x N), then sample bilinearly.
    state = ocean.state_at(t, weights=weights)

    # Sky vs water classification.
    rz = rd[..., 2]
    sky_mask = rz >= -1e-3   # rays going up or flat -> sky

    # ----- water rays: solve heightfield intersection -----
    # Initial guess: intersection with z = 0.
    rz_safe = np.where(rz < 0, rz, -1e-3)
    t_hit = -cam_pos[2] / rz_safe                # may be huge for nearly-horizontal rays
    t_hit = np.clip(t_hit, 0.0, max_distance_m)
    hit_x = cam_pos[0] + t_hit * rd[..., 0]
    hit_y = cam_pos[1] + t_hit * rd[..., 1]

    # Refine using sampled height: t' = (cam.z - h) / -rz
    for _ in range(march_iters):
        h_s, _, _, _ = ocean.sample(state, hit_x, hit_y)
        t_hit = (cam_pos[2] - h_s) / np.maximum(-rz, 1e-6)
        t_hit = np.clip(t_hit, 0.0, max_distance_m)
        hit_x = cam_pos[0] + t_hit * rd[..., 0]
        hit_y = cam_pos[1] + t_hit * rd[..., 1]

    h_s, gx_s, gy_s, foam_s = ocean.sample(state, hit_x, hit_y)
    # Surface normal n = normalize((-dh/dx, -dh/dy, 1))
    nv = np.stack([-gx_s, -gy_s, np.ones_like(h_s)], axis=-1)
    nv /= np.linalg.norm(nv, axis=-1, keepdims=True) + 1e-12

    # Distance attenuation (atmospheric haze) for water rays.
    dist = np.minimum(t_hit, max_distance_m)
    haze = np.clip(dist / max_distance_m, 0.0, 1.0)[..., None]

    # Shade
    img_water = _shade_water(
        hit_xyz=np.stack([hit_x, hit_y, h_s], axis=-1),
        ray_dir=rd, normal=nv, foam_intensity=foam_s,
        water=water, sun_dir=sun_dir, sun_color=sun_color,
        sun_intensity=sun.intensity, sky=sky)

    img_sky = _sky_color(rd, sky, sun_dir, sun_color, sun.intensity)

    # Distance haze blends water toward sky color near the horizon.
    img_water_hazed = (1.0 - haze) * img_water + haze * img_sky

    # Composite: sky where rays go up; water otherwise. Smooth blend at horizon.
    horizon_blend = np.clip((-rz / 0.05), 0.0, 1.0)[..., None]   # 0 at horizon, 1 well below
    img = horizon_blend * img_water_hazed + (1 - horizon_blend) * img_sky

    # Tone map and output
    img = np.clip(img, 0.0, 1.0)
    rgb = (img * 255.0).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Video render
# ---------------------------------------------------------------------------

def render_video(ocean: "FFTOcean", out_mp4: str, n_frames: int, fps: float,
                 resolution: Tuple[int, int],
                 water: "WaterSpec", sun: "Sun", sky: "Sky",
                 camera: "Camera",
                 weights_at=None,
                 schedule_eval=None,
                 progress: bool = True) -> None:
    """
    Render `n_frames` frames at `fps` to `out_mp4` (mp4v codec).

    `weights_at(t)` -> list[float], optional
        Per-frame override of per-component weights. Length must equal
        ocean.n_components.

    `schedule_eval(spec, t)` -> dataclass, optional
        Hook to evaluate Schedule fields on the input dataclasses
        (water/sun/sky/camera) at each frame's time. If None, the
        dataclasses are used as-is for every frame.
    """
    W, H = resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_mp4}")

    t_start = time.time()
    try:
        for k in range(n_frames):
            t = k / float(fps)
            cam_t = schedule_eval(camera, t) if schedule_eval else camera
            sun_t = schedule_eval(sun, t) if schedule_eval else sun
            sky_t = schedule_eval(sky, t) if schedule_eval else sky
            water_t = schedule_eval(water, t) if schedule_eval else water
            w_t = weights_at(t) if weights_at else None
            rgb = render_frame(ocean, t, cam_t, sun_t, sky_t, water_t,
                               resolution, weights=w_t)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if progress and (k % max(1, n_frames // 30) == 0 or k == n_frames - 1):
                elapsed = time.time() - t_start
                eta = (elapsed / max(k + 1, 1)) * (n_frames - k - 1)
                sys.stdout.write(f"\r  rendering {k + 1:4d}/{n_frames} "
                                 f"  elapsed {elapsed:6.1f}s  eta {eta:6.1f}s")
                sys.stdout.flush()
    finally:
        writer.release()
        if progress:
            sys.stdout.write("\n")
