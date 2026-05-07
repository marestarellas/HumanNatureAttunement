"""
Tier-2 backend stub: Taichi GPU shader.

The plan is a Taichi-lang kernel that does the same Tessendorf FFT ocean
on-GPU and a fragment-shader-style raymarched water surface.

Until that lands we raise NotImplementedError; the entry point in
__init__.py forwards a clear message. The reason this file is kept (rather
than left absent) is to make the planned API surface explicit:

    def render_video(ocean: FFTOcean, out_mp4: str,
                     n_frames: int, fps: float,
                     resolution: Tuple[int, int],
                     water, sun, sky, camera,
                     progress: bool = True) -> None:

Mirrors the signature of `render_simple.render_video` so swapping tiers
costs nothing at the call site.
"""
from __future__ import annotations


def render_video(*args, **kwargs):
    raise NotImplementedError(
        "Tier-2 (Taichi) wave synthesis is a planned backend. Use "
        "tier='numpy' for an in-process render or tier='blender' for "
        "near-photorealistic output via Cycles.")
