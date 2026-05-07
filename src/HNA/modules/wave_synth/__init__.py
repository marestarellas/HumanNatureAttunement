"""
HNA.modules.wave_synth — generate parameter-controlled, reproducible sea-wave
clips for neurophysiology entrainment studies.

Three rendering tiers share one parameter API:

  Tier 1 (NumPy/CuPy)        — fastest, fully procedural, bundled here.
  Tier 2 (Taichi GPU shader) — interactive, requires `taichi`. Stub.
  Tier 3 (Blender bpy + Cycles) — near-photorealistic; emits a script you
                                  run with Blender headless.

The high-level entry point is `synthesise(...)`. All three tiers consume
the same dataclasses (`WindSpec`, `WaterSpec`, `Sun`, `Sky`, `Camera`),
so swapping `tier="numpy" -> "blender"` on the same call simply increases
realism while preserving the spectral parameters that determine the
dominant frequency f_p.

A round-trip helper (`validate_clip`) routes a generated mp4 back through
`HNA.modules.video` to confirm the delivered timestack peak matches the
requested f_p.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .spectra import GRAVITY, fp_to_U10, U10_to_fp
from .fft_ocean import FFTOcean
from .render_simple import render_video as _render_video_numpy
from .schedule import (Schedule, evaluate, evaluate_dataclass,
                       schedule_from_keyframes)
from .audio import render_audio as _render_audio, write_wav, mux_audio_into_mp4


__all__ = [
    "GRAVITY", "fp_to_U10", "U10_to_fp",
    "WindSpec", "WaterSpec", "Sun", "Sky", "Camera",
    "WaveComponent", "mix_components",
    "Schedule", "schedule_from_keyframes",
    "SynthResult", "TierLimitation",
    "FFTOcean",
    "synthesise",
    "validate_clip",
]


class TierLimitation(UserWarning):
    """Raised (as a warning) when a knob cannot be honored by the chosen tier."""


# ---------------------------------------------------------------------------
# Parameter dataclasses (all tiers consume these)
# ---------------------------------------------------------------------------

@dataclass
class WindSpec:
    speed_U10: float = 6.0          # m/s at 10 m height
    direction_deg: float = 20.0     # 0 = wind blowing toward +y (north)
    fetch_km: float = 80.0          # JONSWAP only
    depth_m: float = 200.0          # finite-depth dispersion
    spreading: str = "cosine_squared"  # 'cosine_squared' | 'mitsuyasu' | 'donelan'


@dataclass
class WaveComponent:
    """
    One wave system in a possibly multi-modal sea. The full ocean is the
    superposition of independent components, e.g. a long-period swell from
    one storm + a short-period local chop from a different direction.

    Either `f_p_hz` (peak frequency) or `speed_U10` may be given; if both
    are given, `f_p_hz` wins via the Pierson-Moskowitz inverse.

    Two mutually-exclusive ways to set the component's amplitude:

      - `weight` (default = 1.0):  relative linear amplitude; the component
        contributes `weight^2` of its natural Phillips power to the spectrum.
        Natural Phillips means the component's Hs follows U10 via PM.

      - `amplitude_m`:  override the natural PM Hs and force this component
        to have a specific significant wave height (meters). When set,
        `weight` is rescaled internally so the actual delivered Hs matches.

    Use `amplitude_m` to balance a multi-component sea (so each component
    is equally visible) or to dial the dose of a stimulus directly.
    """
    f_p_hz: Optional[float] = None
    speed_U10: float = 6.0
    direction_deg: float = 0.0
    weight: float = 1.0
    amplitude_m: Optional[float] = None
    spreading: str = "cosine_squared"

    def resolved_U10(self) -> float:
        if self.f_p_hz is not None and self.f_p_hz > 0:
            return fp_to_U10(self.f_p_hz)
        return float(self.speed_U10)

    def natural_Hs(self) -> float:
        """PM-derived significant wave height for this component's U10."""
        return 0.21 * (self.resolved_U10() ** 2) / GRAVITY

    def effective_weight(self) -> float:
        """
        Final weight used inside FFTOcean. If `amplitude_m` was given,
        rescale so the component's actual Hs matches the override.
        """
        if self.amplitude_m is not None and self.amplitude_m > 0:
            Hs_pm = max(self.natural_Hs(), 1e-9)
            return float(self.amplitude_m) / Hs_pm * float(self.weight)
        return float(self.weight)


def mix_components(specs, mode: str = "equal_height",
                   Hs_m: float = 1.0,
                   relative_powers: Optional[list] = None,
                   spreading: str = "cosine_squared") -> list:
    """
    Convenience factory for multi-component seas.

    Parameters
    ----------
    specs : list of (f_p_hz, direction_deg) tuples
        The components' peak frequencies and propagation directions.
    mode : {'equal_height', 'equal_power', 'relative_power', 'pm_natural'}
        How to set component amplitudes:
          - 'equal_height'   : every component has the same Hs = `Hs_m`.
          - 'equal_power'    : every component contributes equal variance.
                                Equivalent to equal_height with Hs_m derived
                                from the natural-Hs midpoint.
          - 'relative_power' : `relative_powers[i]` sets the relative variance
                                contribution; each component's amplitude is
                                rescaled to deliver that fraction.
          - 'pm_natural'     : each component follows its own PM Hs (no
                                rescaling) -- this is the default raw mode.
    Hs_m : float
        Target Hs in meters (used by 'equal_height').
    relative_powers : list of float, optional
        Required for mode='relative_power'. Length must match `specs`.
    spreading : str
        Directional spreading model applied to every component.

    Returns
    -------
    components : list[WaveComponent]
        Ready to pass to `synthesise(components=...)`.
    """
    n = len(specs)
    if n == 0:
        return []

    if mode == "equal_height":
        return [WaveComponent(f_p_hz=fp, direction_deg=d,
                              amplitude_m=float(Hs_m),
                              spreading=spreading) for fp, d in specs]

    if mode == "equal_power":
        # Pick Hs that gives total variance = mean(natural Hs^2)
        nat_Hs2 = [(0.21 * (fp_to_U10(fp) ** 2) / GRAVITY) ** 2 for fp, _ in specs]
        target = float(np.sqrt(np.mean(nat_Hs2)))
        return [WaveComponent(f_p_hz=fp, direction_deg=d,
                              amplitude_m=target, spreading=spreading)
                for fp, d in specs]

    if mode == "relative_power":
        if relative_powers is None or len(relative_powers) != n:
            raise ValueError("mode='relative_power' requires `relative_powers` "
                             "with length equal to specs.")
        rp = np.asarray(relative_powers, float)
        rp = rp / rp.sum()                   # normalise to sum 1
        # Total Hs of the mix targets the largest component's natural Hs.
        nat_Hs = np.array([0.21 * (fp_to_U10(fp) ** 2) / GRAVITY for fp, _ in specs])
        Hs_total = float(nat_Hs.max())
        Hs_per = Hs_total * np.sqrt(rp)      # variance share -> amplitude share
        return [WaveComponent(f_p_hz=fp, direction_deg=d,
                              amplitude_m=float(Hs_per[i]),
                              spreading=spreading)
                for i, (fp, d) in enumerate(specs)]

    if mode == "pm_natural":
        return [WaveComponent(f_p_hz=fp, direction_deg=d, weight=1.0,
                              spreading=spreading) for fp, d in specs]

    raise ValueError(f"Unknown mode {mode!r}. Use 'equal_height', "
                     f"'equal_power', 'relative_power', or 'pm_natural'.")


@dataclass
class WaterSpec:
    choppiness: float = 1.0         # Tessendorf lambda; 0 = no horizontal disp
    foam_coverage: float = 0.4      # global foam multiplier
    foam_threshold: float = 0.4     # Jacobian threshold below which foam appears
    water_color: Tuple[float, float, float] = (0.06, 0.18, 0.28)  # deep BGR/RGB


@dataclass
class Sun:
    azimuth_deg: float = 120.0      # 0 = north, 90 = east
    elevation_deg: float = 25.0     # 0 = horizon, 90 = zenith
    color: Tuple[float, float, float] = (1.0, 0.97, 0.85)
    intensity: float = 1.0


@dataclass
class Sky:
    model: str = "horizon_zenith"   # 'horizon_zenith' (Tier 1) | 'hosek_wilkie' (Tier 2/3)
    turbidity: float = 2.5
    horizon_color: Tuple[float, float, float] = (0.78, 0.86, 0.94)
    zenith_color: Tuple[float, float, float] = (0.36, 0.55, 0.78)
    hdri_path: Optional[str] = None   # Tier 2/3 only


@dataclass
class Camera:
    height_m: float = 2.0           # camera height above mean water level
    pitch_deg: float = -5.0         # negative = looking down
    yaw_deg: float = 0.0
    fov_deg: float = 55.0
    focal_mm: Optional[float] = None        # Tier 3 only
    aperture_fstop: Optional[float] = None  # Tier 3 only
    focus_distance_m: Optional[float] = None  # Tier 3 only


@dataclass
class SynthResult:
    out_mp4: str
    duration_s: float
    fps: float
    resolution: Tuple[int, int]
    f_p_hz_target: Optional[float]
    f_p_hz_resolved: float           # what was actually used after fp -> U10
    U10_used: float
    seed: int
    tier: str
    notes: list = field(default_factory=list)
    blender_script: Optional[str] = None  # populated when tier='blender'
    out_wav: Optional[str] = None         # populated when out_wav was requested
    out_mp4_av: Optional[str] = None      # populated when audio was muxed


# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------

def _resolve_wind(wind: WindSpec, f_p_hz: Optional[float],
                  notes: list) -> Tuple[WindSpec, float]:
    """If f_p is given, override wind.speed_U10. Returns (wind, f_p_used)."""
    if f_p_hz is not None and f_p_hz > 0:
        U10 = fp_to_U10(f_p_hz)
        if abs(wind.speed_U10 - U10) > 1e-3:
            notes.append(
                f"f_p={f_p_hz:.3f} Hz overrode wind.speed_U10 "
                f"(was {wind.speed_U10:.2f}, set to {U10:.2f} m/s).")
        wind = WindSpec(**{**asdict(wind), "speed_U10": U10})
        return wind, float(f_p_hz)
    return wind, U10_to_fp(wind.speed_U10)


def _check_tier_capabilities(tier: str, sky: Sky, camera: Camera) -> list:
    """Warn about knobs the chosen tier cannot honor."""
    warns = []
    if tier == "numpy":
        if sky.hdri_path:
            warnings.warn("Tier 1 (numpy) ignores hdri_path; using analytical sky.",
                          TierLimitation)
            warns.append("hdri_path ignored (tier=numpy)")
        if sky.model not in ("horizon_zenith",):
            warnings.warn(f"Tier 1 (numpy) does not implement sky.model='{sky.model}'; "
                          "falling back to 'horizon_zenith'.", TierLimitation)
            warns.append(f"sky.model='{sky.model}' unsupported (tier=numpy)")
        if camera.focal_mm or camera.aperture_fstop or camera.focus_distance_m:
            warnings.warn("Tier 1 (numpy) ignores focal_mm/aperture/focus_distance; "
                          "use tier='blender' for depth-of-field.", TierLimitation)
            warns.append("focal_mm/aperture/focus_distance ignored (tier=numpy)")
    return warns


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def synthesise(
    out_mp4: str,
    f_p_hz: Optional[float] = None,
    duration_s: float = 10.0,
    fps: float = 30.0,
    resolution: Tuple[int, int] = (640, 360),
    tier: str = "numpy",
    wind: Optional[WindSpec] = None,
    components: Optional[list] = None,
    water: Optional[WaterSpec] = None,
    sun: Optional[Sun] = None,
    sky: Optional[Sky] = None,
    camera: Optional[Camera] = None,
    seed: int = 42,
    grid_N: int = 256,
    patch_L_m: float = 200.0,
    progress: bool = True,
    out_wav: Optional[str] = None,
    audio_listener_xy: Tuple[float, float] = (0.0, 0.0),
    audio_sample_rate: int = 44100,
    mux_av: bool = True,
) -> SynthResult:
    """
    Generate a sea-wave video clip with reproducible, parametric control.

    Parameters
    ----------
    out_mp4 : str
        Destination path for the rendered video.
    f_p_hz : float, optional
        Desired dominant wave frequency. If given, overrides
        `wind.speed_U10` via the Pierson-Moskowitz inverse:
        U10 = 0.13 * g / f_p_hz. None means use `wind.speed_U10` directly.
    duration_s, fps, resolution
        Output video timing and image size.
    tier : {'numpy','taichi','blender'}
        Rendering backend. Tier 1 ('numpy') runs in-process. Tier 3
        ('blender') generates a `.py` script next to the requested mp4
        which you run with `blender --background --python`.
    wind, water, sun, sky, camera
        Optional overrides for the parameter dataclasses. Defaults give
        a calm-to-moderate sea on a low-camera afternoon.
    seed : int
        RNG seed; fixes the wave realisation.
    grid_N, patch_L_m
        Spatial discretisation of the FFT ocean (N x N grid spanning L meters).
        Larger N = finer ripples; larger L = wider ocean tile (less repetition).
    progress : bool
        Print per-frame progress (Tier 1 only).
    """
    out_mp4_p = Path(out_mp4)
    out_mp4_p.parent.mkdir(parents=True, exist_ok=True)

    wind = wind or WindSpec()
    water = water or WaterSpec()
    sun = sun or Sun()
    sky = sky or Sky()
    camera = camera or Camera()

    notes: list = []
    wind, f_p_used = _resolve_wind(wind, f_p_hz, notes)
    notes.extend(_check_tier_capabilities(tier, sky, camera))

    if tier == "numpy":
        comp_tuples = None
        weights_at = None
        if components:
            # Resolve static descriptors (U10, direction, spreading); the
            # spectrum bank is built from these. weights/amplitude_m are
            # allowed to vary in time and are evaluated each frame below.
            comp_tuples = [
                (c.resolved_U10(), np.deg2rad(c.direction_deg),
                 1.0,                   # placeholder; weights_at provides the real values
                 c.spreading)
                for c in components
            ]
            # Pre-compute natural Hs per component for the weights_at closure.
            natural_Hs = [c.natural_Hs() for c in components]

            def weights_at(t, _comps=components, _Hs=natural_Hs):
                ws = []
                for c, Hs_pm in zip(_comps, _Hs):
                    w_raw = float(evaluate(c.weight, t))
                    amp = c.amplitude_m
                    if amp is None:
                        ws.append(w_raw)
                    else:
                        amp_t = float(evaluate(amp, t))
                        ws.append(amp_t / max(Hs_pm, 1e-9) * w_raw)
                return ws

            notes.append(
                f"using {len(comp_tuples)} wave components: " +
                ", ".join(
                    f"f_p={c.f_p_hz}Hz/U10={c.resolved_U10():.2f} "
                    f"@dir={c.direction_deg}deg "
                    f"{'amp_m=' + str(c.amplitude_m) if c.amplitude_m is not None else 'w=' + str(c.weight)}"
                    f"{' [time-varying]' if (callable(c.amplitude_m) or callable(c.weight)) else ''}"
                    for c in components))

        ocean = FFTOcean(N=grid_N, L=patch_L_m,
                         U10=wind.speed_U10,
                         wind_dir_rad=np.deg2rad(wind.direction_deg),
                         depth_m=wind.depth_m,
                         choppiness=water.choppiness,
                         seed=seed,
                         components=comp_tuples)
        n_frames = int(round(duration_s * fps))
        _render_video_numpy(
            ocean=ocean,
            out_mp4=str(out_mp4_p),
            n_frames=n_frames,
            fps=float(fps),
            resolution=resolution,
            water=water,
            sun=sun,
            sky=sky,
            camera=camera,
            weights_at=weights_at,
            schedule_eval=evaluate_dataclass,
            progress=progress,
        )

        # ---- Optional: paired audio synthesis -------------------------------
        out_wav_path: Optional[str] = None
        out_mp4_av_path: Optional[str] = None
        if out_wav is not None:
            out_wav_path = str(Path(out_wav))
            audio, sr = _render_audio(
                ocean,
                duration_s=float(duration_s),
                fps_video=float(fps),
                weights_at=weights_at,
                listener_xy=tuple(audio_listener_xy),
                sample_rate=int(audio_sample_rate),
                seed=int(seed),
            )
            write_wav(out_wav_path, audio, sr)
            notes.append(f"audio written to {out_wav_path}")
            if mux_av:
                muxed = out_mp4_p.with_name(out_mp4_p.stem + "_av.mp4")
                ok = mux_audio_into_mp4(str(out_mp4_p), out_wav_path, str(muxed))
                if ok:
                    out_mp4_av_path = str(muxed)
                    notes.append(f"AV-muxed mp4 at {muxed}")
                else:
                    notes.append("ffmpeg unavailable - skipped AV mux "
                                 "(WAV+MP4 still produced separately).")

        return SynthResult(
            out_mp4=str(out_mp4_p),
            duration_s=float(duration_s),
            fps=float(fps),
            resolution=tuple(resolution),
            f_p_hz_target=f_p_hz,
            f_p_hz_resolved=f_p_used,
            U10_used=wind.speed_U10,
            seed=seed,
            tier="numpy",
            notes=notes,
            out_wav=out_wav_path,
            out_mp4_av=out_mp4_av_path,
        )

    if tier == "taichi":
        from .render_taichi import render_video as _render_video_taichi  # noqa: F401
        raise NotImplementedError(
            "Tier 2 (taichi) is a planned backend. The reference shader and "
            "PR scaffolding live in render_taichi.py.")

    if tier == "blender":
        from .render_blender import build_script
        script = build_script(
            out_mp4=str(out_mp4_p),
            duration_s=duration_s, fps=fps, resolution=resolution,
            wind=wind, water=water, sun=sun, sky=sky, camera=camera,
            seed=seed, patch_L_m=patch_L_m, grid_N=grid_N,
        )
        script_path = out_mp4_p.with_suffix(".blend.py")
        script_path.write_text(script, encoding="utf-8")
        notes.append(
            f"Blender script written to {script_path}. Run with: "
            f"blender --background --python \"{script_path}\"")
        return SynthResult(
            out_mp4=str(out_mp4_p),
            duration_s=float(duration_s),
            fps=float(fps),
            resolution=tuple(resolution),
            f_p_hz_target=f_p_hz,
            f_p_hz_resolved=f_p_used,
            U10_used=wind.speed_U10,
            seed=seed,
            tier="blender",
            notes=notes,
            blender_script=str(script_path),
        )

    raise ValueError(f"Unknown tier {tier!r}. Use 'numpy', 'taichi' or 'blender'.")


# ---------------------------------------------------------------------------
# Round-trip validation against HNA.modules.video
# ---------------------------------------------------------------------------

def validate_clip(mp4_path: str,
                  f_p_hz_target: Optional[float] = None,
                  expected_peaks_hz: Optional[list] = None,
                  tolerance_hz: float = 0.03,
                  return_spectrum: bool = False) -> dict:
    """
    Re-quantify a synthesised clip with `HNA.modules.video` and check that
    the delivered dominant frequency matches the requested f_p (single-peak
    case) OR that all `expected_peaks_hz` are detectable in the rendered
    clip's mean pixel spectrum (multi-component case).

    Parameters
    ----------
    f_p_hz_target : float, optional
        Single expected peak. Equivalent to expected_peaks_hz=[f_p_hz_target].
    expected_peaks_hz : list of float, optional
        Multiple expected peaks (for multi-component synthesis). For each
        injected peak, we report the closest detected peak in the mean
        pixel spectrum and the absolute error.
    tolerance_hz : float
        A peak passes if its closest detected match is within this
        distance (Hz).
    return_spectrum : bool
        If True, additionally return the mean pixel spectrum (`freqs`,
        `mean_spectrum`) for plotting.
    """
    from HNA.modules.video import quantify_video
    import scipy.signal as _sps
    import numpy as _np

    feats = quantify_video(mp4_path,
                           target_long=192,
                           include_pixel_spectrum=True,
                           pixel_spectrum_target_long=64,
                           include_windowed_complexity=False,
                           modal_k=2)
    ts_peak = float(feats.timestack.dominant_freq_hz) if feats.timestack else float("nan")
    modal_peak = (float(feats.modal.frequencies_hz[0])
                  if feats.modal is not None and feats.modal.frequencies_hz.size
                  else float("nan"))
    coherence = (float(feats.pixel_spectrum.coherence_index)
                 if feats.pixel_spectrum is not None else float("nan"))

    # Build the multi-peak target list (deduped, sorted ascending)
    targets = []
    if expected_peaks_hz:
        targets.extend(float(p) for p in expected_peaks_hz)
    if f_p_hz_target is not None:
        targets.append(float(f_p_hz_target))
    targets = sorted(set(round(t, 6) for t in targets))

    # Detect peaks on the mean pixel spectrum (most robust spectral signature)
    detected = []
    freqs_arr = mean_arr = None
    if feats.pixel_spectrum is not None:
        freqs_arr = _np.asarray(feats.pixel_spectrum.freqs, float)
        mean_arr = _np.asarray(feats.pixel_spectrum.mean_spectrum, float)
        # ignore DC, restrict to the band of interest
        m = (freqs_arr > 0.02) & (freqs_arr < 2.0)
        if m.any():
            f_band = freqs_arr[m]; P_band = mean_arr[m]
            # Smooth a touch and call find_peaks with a prominence floor
            prom = float(_np.max(P_band) * 0.04)
            idx, _ = _sps.find_peaks(P_band, prominence=prom)
            for i in idx:
                detected.append({"f_hz": float(f_band[i]),
                                 "power": float(P_band[i])})
            detected.sort(key=lambda d: -d["power"])

    # Match each requested peak to the closest detected peak. ALSO compute
    # band-power around each target -- multi-component seas often produce
    # one broad PM hump per component that overlaps in the mean spectrum,
    # so band-power is a more sensitive indicator than peak-picking.
    matches = []
    band_half = 0.05   # +/- Hz around each target for band-power check
    total_band_power = None
    if freqs_arr is not None and mean_arr is not None:
        m_pos = (freqs_arr > 0.02) & (freqs_arr < 2.0)
        total_band_power = float(mean_arr[m_pos].sum()) + 1e-12

    for t in targets:
        # Closest peak
        if detected:
            best = min(detected, key=lambda d: abs(d["f_hz"] - t))
            closest = best["f_hz"]; err = abs(best["f_hz"] - t)
        else:
            closest = float("nan"); err = float("nan")
        # Band power around the target
        bp = None; bp_frac = None
        if freqs_arr is not None and mean_arr is not None:
            sel = (freqs_arr >= t - band_half) & (freqs_arr <= t + band_half)
            if sel.any():
                bp = float(mean_arr[sel].sum())
                bp_frac = float(bp / total_band_power) if total_band_power else None
        matches.append({"target_hz": t,
                        "closest_peak_hz": closest, "err_hz": float(err),
                        "band_power": bp, "band_power_frac": bp_frac,
                        "peak_passes": bool(err <= tolerance_hz),
                        # Counts as 'present' if either a peak is within tolerance
                        # OR the local band carries >5% of the total in-band power
                        "present": bool((err <= tolerance_hz)
                                        or (bp_frac is not None and bp_frac >= 0.05))})
    all_pass = (len(matches) > 0 and all(m["present"] for m in matches))

    out = dict(
        targets_hz=targets,
        timestack_peak_hz=ts_peak,
        modal_peak_hz=modal_peak,
        pixel_spectrum_coherence=coherence,
        detected_peaks=detected[:10],
        matches=matches,
        passes=all_pass,
        tolerance_hz=float(tolerance_hz),
    )
    if return_spectrum and freqs_arr is not None:
        out["spectrum"] = {"freqs": freqs_arr.tolist(),
                            "mean_spectrum": mean_arr.tolist()}
    return out
