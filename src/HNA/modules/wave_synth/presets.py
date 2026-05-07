"""
Named parameter presets for `synthesise(...)` calls.

A preset returns the four dataclass instances (wind, water, sun, sky)
ready to feed into the entry point. The user typically still chooses a
camera and supplies `f_p_hz` (which overrides the wind speed).
"""
from __future__ import annotations

from typing import Dict, Tuple

from . import WindSpec, WaterSpec, Sun, Sky


def calm_morning() -> Tuple[WindSpec, WaterSpec, Sun, Sky]:
    """Glassy dawn sea, low sun, faint swell."""
    return (
        WindSpec(speed_U10=3.0, direction_deg=15, fetch_km=120, depth_m=200),
        WaterSpec(choppiness=0.4, foam_coverage=0.0, foam_threshold=0.7,
                  water_color=(0.05, 0.14, 0.22)),
        Sun(azimuth_deg=85, elevation_deg=8,
            color=(1.0, 0.78, 0.55), intensity=1.1),
        Sky(model="horizon_zenith", turbidity=2.0,
            horizon_color=(0.95, 0.78, 0.62), zenith_color=(0.50, 0.62, 0.78)),
    )


def light_swell() -> Tuple[WindSpec, WaterSpec, Sun, Sky]:
    """Mid-day, moderate breeze, the canonical sea-wave clip."""
    return (
        WindSpec(speed_U10=5.1, direction_deg=20, fetch_km=80, depth_m=200),
        WaterSpec(choppiness=0.9, foam_coverage=0.25, foam_threshold=0.5,
                  water_color=(0.06, 0.18, 0.28)),
        Sun(azimuth_deg=120, elevation_deg=35,
            color=(1.0, 0.95, 0.86), intensity=1.0),
        Sky(model="horizon_zenith", turbidity=2.5,
            horizon_color=(0.78, 0.86, 0.94), zenith_color=(0.36, 0.55, 0.78)),
    )


def stormy() -> Tuple[WindSpec, WaterSpec, Sun, Sky]:
    """Heavy weather: long-period swell, heavy foam."""
    return (
        WindSpec(speed_U10=12.7, direction_deg=10, fetch_km=300, depth_m=200),
        WaterSpec(choppiness=1.4, foam_coverage=0.85, foam_threshold=0.3,
                  water_color=(0.05, 0.12, 0.20)),
        Sun(azimuth_deg=200, elevation_deg=12,
            color=(0.90, 0.92, 0.96), intensity=0.6),
        Sky(model="horizon_zenith", turbidity=5.0,
            horizon_color=(0.55, 0.58, 0.62), zenith_color=(0.30, 0.34, 0.40)),
    )


def small_chop() -> Tuple[WindSpec, WaterSpec, Sun, Sky]:
    """Short-period chop (~0.5 Hz) — entrainment band upper edge."""
    return (
        WindSpec(speed_U10=2.5, direction_deg=20, fetch_km=20, depth_m=50),
        WaterSpec(choppiness=0.7, foam_coverage=0.05, foam_threshold=0.6,
                  water_color=(0.06, 0.18, 0.28)),
        Sun(azimuth_deg=140, elevation_deg=40,
            color=(1.0, 0.95, 0.86), intensity=1.0),
        Sky(model="horizon_zenith", turbidity=3.0,
            horizon_color=(0.78, 0.86, 0.94), zenith_color=(0.36, 0.55, 0.78)),
    )


PRESETS: Dict[str, callable] = {
    "calm_morning": calm_morning,
    "light_swell": light_swell,
    "stormy": stormy,
    "small_chop": small_chop,
}


def get(name: str):
    """Lookup a preset by name."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Choose from {list(PRESETS.keys())}.")
    return PRESETS[name]()
