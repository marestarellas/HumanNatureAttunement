"""
Companion audio for the FFT ocean.

Stripped to the essentials: a band-limited pink-noise wash whose amplitude
is modulated by the slow envelope of the heightfield at a virtual
hydrophone, plus a sub-bass rumble derived from the heightfield itself.
The "wave vibe" comes from the slow amplitude swell at the wave period --
no discrete break events, no bubble synthesis, no LFO bank.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _bandpass(x, fs, lo, hi, order=4):
    if not _HAVE_SCIPY:
        return x
    nyq = fs * 0.5
    sos = butter(order, [max(1e-3, lo / nyq), min(0.999, hi / nyq)],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _lowpass(x, fs, cutoff, order=4):
    if not _HAVE_SCIPY:
        return x
    nyq = fs * 0.5
    sos = butter(order, max(1e-4, cutoff / nyq), btype="lowpass", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _pink_noise(n: int, fs: int, rng) -> np.ndarray:
    """1/f pink noise via frequency-domain shaping. Zero mean, unit peak."""
    n_pos = n // 2 + 1
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    f_safe = np.where(f > 0, f, f[1] if f.size > 1 else 1.0)
    spec = (rng.standard_normal(n_pos) + 1j * rng.standard_normal(n_pos))
    spec = spec / np.sqrt(f_safe)
    spec[0] = 0.0
    out = np.fft.irfft(spec, n=n).astype(np.float32)
    out -= out.mean()
    pk = np.abs(out).max()
    return out / pk if pk > 1e-12 else out


def render_audio(ocean,
                 duration_s: float,
                 fps_video: float,
                 weights_at: Optional[Callable] = None,
                 listener_xy: Tuple[float, float] = (0.0, 0.0),
                 sample_rate: int = 44100,
                 # Bandwidth of the noise wash
                 noise_band_hz: Tuple[float, float] = (200.0, 3000.0),
                 # Sub-bass rumble derived from h(t) itself
                 rumble_cutoff_hz: float = 80.0,
                 # Mix levels
                 mix_wash: float = 0.85,
                 mix_rumble: float = 0.35,
                 # Envelope smoothing (Hz). Should be a few times the wave
                 # peak frequency so the wave-by-wave swell stays visible.
                 envelope_smoothing_hz: float = 1.5,
                 # Floor under the envelope (0 = full silence between waves,
                 # 0.2 = always some background hiss).
                 envelope_floor: float = 0.15,
                 # Stereo
                 stereo_decorrelation: float = 0.6,
                 # Determinism
                 seed: int = 0,
                 ) -> Tuple[np.ndarray, int]:
    """
    Synthesize a stereo wave audio paired with the FFT ocean rendered video.

    Architecture (very small, very on-purpose):

      noise (pink, bandpassed)  *  envelope(t)  +  rumble(t)
                                |
                                +-- envelope = lowpass(|h(t)|) at ~1.5 Hz
                                |   so it follows the wave-by-wave swell
                                |   exactly at the wave's peak frequency
                                +-- L and R use independent noise streams
                                    when stereo_decorrelation > 0
    """
    n_video = max(2, int(round(duration_s * fps_video)))
    h_t = np.zeros(n_video, dtype=np.float32)
    lx = np.array([float(listener_xy[0])])
    ly = np.array([float(listener_xy[1])])
    for k in range(n_video):
        t = k / float(fps_video)
        w = weights_at(t) if weights_at is not None else None
        st = ocean.state_at(t, weights=w)
        h, _, _, _ = ocean.sample(st, lx, ly)
        h_t[k] = float(h[0])

    n_audio = int(round(duration_s * sample_rate))
    t_video = np.arange(n_video) / float(fps_video)
    t_audio = np.arange(n_audio) / float(sample_rate)
    h_audio = np.interp(t_audio, t_video, h_t).astype(np.float32)

    # ---- Slow envelope from |h(t)| ----------------------------------------
    env = np.abs(h_audio)
    env = _lowpass(env, sample_rate, cutoff=envelope_smoothing_hz, order=2)
    env_max = float(env.max())
    env = env / env_max if env_max > 1e-9 else env
    env = float(envelope_floor) + (1.0 - float(envelope_floor)) * env

    # ---- Pink-noise wash, band-limited -----------------------------------
    rng = np.random.default_rng(seed)
    pink_l = _pink_noise(n_audio, sample_rate, rng)
    pink_r = _pink_noise(n_audio, sample_rate, rng)
    wash_l = _bandpass(pink_l, sample_rate, *noise_band_hz, order=3)
    wash_r = _bandpass(pink_r, sample_rate, *noise_band_hz, order=3)
    for v in (wash_l, wash_r):
        pk = np.abs(v).max()
        if pk > 1e-9:
            v /= pk

    # Optional stereo correlation pull-toward-mid
    if stereo_decorrelation < 1.0:
        mid = 0.5 * (wash_l + wash_r)
        wash_l = stereo_decorrelation * wash_l + (1.0 - stereo_decorrelation) * mid
        wash_r = stereo_decorrelation * wash_r + (1.0 - stereo_decorrelation) * mid

    wash_l = wash_l * env
    wash_r = wash_r * env

    # ---- Sub-bass rumble (h(t), lowpassed) -------------------------------
    rumble = _lowpass(h_audio - h_audio.mean(), sample_rate,
                      cutoff=rumble_cutoff_hz, order=3)
    pk = np.abs(rumble).max()
    if pk > 1e-9:
        rumble = rumble / pk

    # ---- Mix and normalise ------------------------------------------------
    left = mix_wash * wash_l + mix_rumble * rumble
    right = mix_wash * wash_r + mix_rumble * rumble
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    pk = float(np.abs(stereo).max())
    if pk > 1e-9:
        stereo = stereo * (0.88 / pk)
    return stereo, int(sample_rate)


def write_wav(path, audio, sample_rate):
    import soundfile as sf
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def mux_audio_into_mp4(video_in_mp4, audio_wav, out_mp4, ffmpeg="ffmpeg"):
    import shutil, subprocess
    if shutil.which(ffmpeg) is None:
        return False
    cmd = [ffmpeg, "-y",
           "-i", str(video_in_mp4),
           "-i", str(audio_wav),
           "-c:v", "copy", "-c:a", "aac", "-shortest",
           str(out_mp4)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0
