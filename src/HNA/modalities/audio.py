"""
Audio modality helpers for the HNA toolbox.

The core function is :func:`decompose_envelope`, which produces a dictionary
of band-organized amplitude envelopes from a raw audio waveform. The
envelope set covers slow swells (sub-Hz) up through canonical EEG bands
(0.5-50 Hz on the *envelope*, not the carrier).

Filter design / processing rationale
------------------------------------
- Audio is high-passed (>20 Hz default) and resampled to 22.05 kHz to
  remove DC and bandwidth headroom outside the audible carrier.
- A Hilbert envelope is computed at audio rate, low-passed at ``lpf_broad``
  (60 Hz default) to set the maximum envelope frequency we will resolve.
- Two processing rates are used for numerical stability:
    * ``ENV_FS_SLOW`` (50 Hz) for sub-Hz / mid-band filters (HRV, swell, delta..alpha)
    * ``ENV_FS_FAST`` (200 Hz) for fast bands (low-beta, high-beta, gamma1)
- Each band is bandpass- or low-passed on the appropriate buffer, then
  log-normalized for scale invariance and resampled to ``output_fs`` (256 Hz)
  so all output columns share a common time axis with EEG/physio tables.

The resulting envelopes can be coupled with any other 1-D physiological
signal (respiration, HRV, EEG) via :mod:`HNA.coupling`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

from ..dsp import (
    bandpass as _bandpass,
    lowpass as _lowpass,
    highpass as _highpass,
    hilbert_envelope as _hilbert_envelope,
    resample_to as _resample_1d,
)


# Defaults shared with scripts/preprocessing/02_compute_audio_envelopes.py
TARGET_SR_AUDIO = 22050
ENV_FS_SLOW = 50
ENV_FS_FAST = 200
OUTPUT_FS = 256
HP_CUT = 20.0
LPF_BROAD = 60.0

#: Band specifications used by :func:`decompose_envelope`.
#: Each entry is (column_name, kind, params, processing_fs).
DEFAULT_BANDS: Sequence[Tuple[str, str, dict, int]] = (
    ("env_broad",      "lowpass",  {"cut": LPF_BROAD},        ENV_FS_FAST),
    ("env_swell_0p2",  "lowpass",  {"cut": 0.2},              ENV_FS_SLOW),
    ("env_swell_0p1",  "lowpass",  {"cut": 0.1},              ENV_FS_SLOW),
    ("env_hrv_lf",     "bandpass", {"lo": 0.04, "hi": 0.15},  ENV_FS_SLOW),
    ("env_hrv_hf",     "bandpass", {"lo": 0.15, "hi": 0.40},  ENV_FS_SLOW),
    ("env_splash_1_5", "bandpass", {"lo": 1.0,  "hi": 5.0},   ENV_FS_SLOW),
    ("env_delta",      "bandpass", {"lo": 0.5,  "hi": 4.0},   ENV_FS_SLOW),
    ("env_theta",      "bandpass", {"lo": 4.0,  "hi": 8.0},   ENV_FS_SLOW),
    ("env_alpha",      "bandpass", {"lo": 8.0,  "hi": 13.0},  ENV_FS_SLOW),
    ("env_beta_low",   "bandpass", {"lo": 13.0, "hi": 20.0},  ENV_FS_FAST),
    ("env_beta_high",  "bandpass", {"lo": 20.0, "hi": 30.0},  ENV_FS_FAST),
    ("env_gamma1",     "bandpass", {"lo": 30.0, "hi": 50.0},  ENV_FS_FAST),
)


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _safe_log_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    y = np.log10(1e-8 + x)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return y.astype(np.float32)


def decompose_envelope(
    wav_path: Path | str,
    target_sr_audio: int = TARGET_SR_AUDIO,
    output_fs: int = OUTPUT_FS,
    hp_cut: float | None = HP_CUT,
    lpf_broad: float = LPF_BROAD,
    bands: Sequence[Tuple[str, str, dict, int]] = DEFAULT_BANDS,
) -> Tuple[pd.DataFrame, int]:
    """Decompose an audio file into a band-organized set of envelopes.

    Parameters
    ----------
    wav_path : path-like
        Audio file to read (any format supported by ``soundfile``).
    target_sr_audio : int
        Audio rate after the optional resample (default 22050 Hz).
    output_fs : int
        Output sample rate for all envelope columns (default 256 Hz; matches
        the typical EEG/physio merged-table rate).
    hp_cut : float or None
        High-pass cutoff applied to the raw audio before envelope extraction.
        Pass ``None`` to disable.
    lpf_broad : float
        Low-pass cutoff applied to the Hilbert envelope at audio rate. Sets
        the upper frequency limit usable by any band-pass below.
    bands : sequence
        Band specs of the form (column_name, kind, params, processing_fs)
        with ``kind in {"lowpass", "bandpass"}``. See :data:`DEFAULT_BANDS`.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``time_s`` plus one column per band (each log-normalized).
    env_fs : int
        Output sample rate (== ``output_fs``).
    """
    x, fs = sf.read(str(wav_path), always_2d=False)
    x = _to_mono(np.asarray(x, dtype=np.float64))

    if hp_cut is not None and hp_cut > 0:
        x = _highpass(x, fs=fs, cut=hp_cut, order=2)

    if int(fs) != int(target_sr_audio):
        x = _resample_1d(x, fs_from=fs, fs_to=target_sr_audio)
        fs = target_sr_audio

    env = _hilbert_envelope(x)
    env_broad_audio = _lowpass(env, fs=fs, cut=lpf_broad, order=4)

    proc_fs_set = sorted(set(spec[3] for spec in bands))
    env_at_fs = {pfs: _resample_1d(env_broad_audio, fs_from=fs, fs_to=pfs)
                 for pfs in proc_fs_set}

    out_cols = {}
    for name, kind, params, proc_fs in bands:
        src = env_at_fs[proc_fs]
        if kind == "lowpass":
            sig = _lowpass(src, fs=proc_fs, cut=params["cut"], order=4)
        elif kind == "bandpass":
            sig = _bandpass(src, fs=proc_fs, lo=params["lo"], hi=params["hi"], order=4)
        else:
            raise ValueError(f"Unknown band kind {kind!r} for {name}")
        sig_n = _safe_log_norm(np.abs(sig))
        if int(proc_fs) != int(output_fs):
            sig_n = _resample_1d(sig_n, fs_from=proc_fs, fs_to=output_fs)
        out_cols[name] = sig_n

    n = min(len(v) for v in out_cols.values())
    out_cols = {k: v[:n].astype(np.float32, copy=False) for k, v in out_cols.items()}
    t = (np.arange(n, dtype=np.float64) / float(output_fs)).astype(np.float32, copy=False)
    df = pd.DataFrame({"time_s": t, **out_cols})
    return df, int(output_fs)
