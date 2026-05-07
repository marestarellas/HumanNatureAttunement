"""
CLI driver for `HNA.modules.wave_synth.synthesise`.

Examples
--------
    # Tier-1 NumPy render, target peak frequency 0.25 Hz, 5 seconds
    python scripts/synthesise_ocean.py --out stimuli/swell_025Hz.mp4 \\
           --fp 0.25 --duration 5 --resolution 640 360 --tier numpy

    # Tier-3 Blender script (run with: blender --background --python <script>)
    python scripts/synthesise_ocean.py --out stimuli/storm.mp4 \\
           --preset stormy --duration 60 --tier blender

    # Apply a preset and validate the round-trip
    python scripts/synthesise_ocean.py --preset light_swell \\
           --out stimuli/light_swell.mp4 --duration 8 --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.wave_synth import (   # noqa: E402
    synthesise, validate_clip,
    WindSpec, WaterSpec, Sun, Sky, Camera,
)
from HNA.modules.wave_synth.presets import PRESETS, get as get_preset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output mp4 path.")
    p.add_argument("--fp", type=float, default=None,
                   help="Target peak frequency (Hz). Overrides --U10 / preset wind.")
    p.add_argument("--U10", type=float, default=None, help="Wind speed at 10 m, m/s.")
    p.add_argument("--duration", type=float, default=8.0, help="Clip duration (s).")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--resolution", nargs=2, type=int, default=[640, 360],
                   metavar=("W", "H"))
    p.add_argument("--tier", choices=["numpy", "taichi", "blender"], default="numpy")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Pick a named preset for wind/water/sun/sky.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-N", type=int, default=256)
    p.add_argument("--patch-L", type=float, default=200.0,
                   help="Patch size in meters.")
    p.add_argument("--cam-height", type=float, default=2.0)
    p.add_argument("--cam-pitch", type=float, default=-5.0)
    p.add_argument("--cam-yaw", type=float, default=0.0)
    p.add_argument("--cam-fov", type=float, default=55.0)
    p.add_argument("--validate", action="store_true",
                   help="After rendering, route the mp4 back through "
                        "HNA.modules.video and report f_p match.")
    args = p.parse_args()

    wind = water = sun = sky = None
    if args.preset:
        wind, water, sun, sky = get_preset(args.preset)
    if args.U10 is not None and wind is None:
        wind = WindSpec(speed_U10=args.U10)
    elif args.U10 is not None:
        wind = WindSpec(**{**wind.__dict__, "speed_U10": args.U10})

    camera = Camera(height_m=args.cam_height, pitch_deg=args.cam_pitch,
                    yaw_deg=args.cam_yaw, fov_deg=args.cam_fov)

    res = synthesise(out_mp4=args.out,
                     f_p_hz=args.fp,
                     duration_s=args.duration,
                     fps=args.fps,
                     resolution=tuple(args.resolution),
                     tier=args.tier,
                     wind=wind, water=water, sun=sun, sky=sky,
                     camera=camera,
                     seed=args.seed,
                     grid_N=args.grid_N,
                     patch_L_m=args.patch_L)

    print()
    print("synthesise() returned:")
    print(json.dumps({
        "out_mp4": res.out_mp4,
        "tier": res.tier,
        "f_p_hz_target": res.f_p_hz_target,
        "f_p_hz_resolved": res.f_p_hz_resolved,
        "U10_used": res.U10_used,
        "duration_s": res.duration_s,
        "resolution": list(res.resolution),
        "blender_script": res.blender_script,
        "notes": res.notes,
    }, indent=2))

    if args.validate:
        if args.tier != "numpy":
            print("(--validate only meaningful when an mp4 was actually rendered)")
            return 0
        if args.fp is None:
            print("(--validate requires --fp)")
            return 0
        v = validate_clip(args.out, args.fp)
        print()
        print("validate_clip() returned:")
        print(json.dumps(v, indent=2))
        return 0 if v["passes"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
