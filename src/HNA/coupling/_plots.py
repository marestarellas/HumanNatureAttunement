"""Shared plotting helpers across coupling families.

These visualisations combine results from multiple coupling methods
(xcorr + coherence + PLV summary, coherence spectrum + windowed band-avg,
3-panel signal alignment), so they live in a shared module rather than
inside any one family file.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ..dsp import interpolate_nan as _nan_interp, zscore as _zscore


def plot_signal_alignment_validation(
    resp,
    env,
    fs,
    cond_label: str = "",
    env_label: str = "env_swell_0p2",
    preview_sec: float = 60.0,
    max_lag_sec: float = 10.0,
    win_sec: float = 30.0,
    step_sec: float = 5.0,
    signal1_label: str = "Respiration",
):
    """Three-panel sanity-check figure for any 1-D signal vs an envelope.

    Panel 1: z-scored overlay of ``signal1_label`` vs envelope over the
    first ``preview_sec`` seconds.
    Panel 2: full-signal cross-correlation as a function of lag.
    Panel 3: time-varying Pearson r in ``win_sec``-second sliding windows.

    Despite the parameter names ``resp`` / ``env`` (kept for back-compat),
    this works for any pair of 1-D signals at the same sample rate.
    """
    resp = _nan_interp(np.asarray(resp, float))
    env = _nan_interp(np.asarray(env, float))
    n = min(len(resp), len(env))
    resp = resp[:n]
    env = env[:n]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # 1) Preview overlay (first preview_sec seconds)
    ax = axes[0]
    n_prev = min(int(preview_sec * fs), n)
    t = np.arange(n_prev) / fs
    ax.plot(t, _zscore(resp[:n_prev]), lw=1.4, color="tab:blue", label=signal1_label)
    ax.plot(t, _zscore(env[:n_prev]), lw=1.4, color="tab:red", alpha=0.85,
            label=f"Audio ({env_label})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 2) Lag analysis (full signal)
    ax = axes[1]
    rz = _zscore(resp)
    ez = _zscore(env)
    xcorr = signal.correlate(rz, ez, mode="full") / n
    lags = signal.correlation_lags(len(rz), len(ez), mode="full") / fs
    sel = (lags >= -max_lag_sec) & (lags <= max_lag_sec)
    L = lags[sel]
    X = xcorr[sel]
    peak_idx = int(np.argmax(np.abs(X)))
    peak_lag = L[peak_idx]
    peak_r = X[peak_idx]
    ax.plot(L, X, color="tab:orange", lw=2)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.6)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)
    ax.scatter([peak_lag], [peak_r], color="red", s=70, zorder=5,
               label=f"Peak: r={peak_r:.3f} at {peak_lag:.2f}s")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Lag Analysis (full signal)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # 3) Time-varying correlation
    ax = axes[2]
    W = int(round(win_sec * fs))
    H = int(round(step_sec * fs))
    starts = np.arange(0, max(n - W + 1, 0), H)
    times, rs = [], []
    for st in starts:
        a = resp[st:st + W]
        b = env[st:st + W]
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
        else:
            r = np.nan
        times.append((st + W / 2) / fs)
        rs.append(r)
    ax.plot(times, rs, "-o", color="tab:green", lw=1.4, ms=4)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Time-varying correlation ({int(win_sec)}s windows, {int(step_sec)}s steps)")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_coupling_over_time(xc, coh, plv_win):
    """Three-panel summary: xcorr peak/lag, band-avg coherence, PLV — all over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1) cross-corr peak + lag
    ax = axes[0]
    ax.plot(xc.times_s, xc.peak_r, lw=1.6, label="XCorr peak r")
    ax.set_ylabel("peak r")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xc.times_s, xc.peak_lag_s, lw=1.1, ls="--", label="XCorr lag")
    ax2.set_ylabel("lag (s)")
    ax.set_title("Windowed cross-correlation")

    # 2) band-avg coherence (if windowed series available)
    ax = axes[1]
    if getattr(coh, "times_s", None) is not None and getattr(coh, "band_avg_coh_win", None) is not None:
        ax.plot(coh.times_s, coh.band_avg_coh_win, lw=1.6)
    ax.set_ylabel("band-avg coherence")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Welch coherence (band avg ~ {getattr(coh, 'band_avg_coh', np.nan):.2f}, "
        f"global peak {coh.peak_coh:.2f} @ {coh.peak_f:.3f} Hz)"
    )

    # 3) PLV
    ax = axes[2]
    ax.plot(plv_win["times_s"], plv_win["plv"], lw=1.6, label="PLV")
    ax.set_ylabel("PLV")
    ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def plot_coherence_results(coh, band=None, title: str = "Coherence summary"):
    """Plot global coherence spectrum + windowed band-avg coherence over time.

    ``coh`` is either a dict or an attribute-bearing object (dataclass) with
    keys/attrs: ``f, Cxy, peak_f, peak_coh, band_avg_coh, times_s,
    band_avg_coh_win``.
    """
    def get(name, default=None):
        if isinstance(coh, dict):
            return coh.get(name, default)
        return getattr(coh, name, default)

    f = get("f")
    Cxy = get("Cxy")
    peak_f = get("peak_f", np.nan)
    peak_coh = get("peak_coh", np.nan)
    band_avg = get("band_avg_coh", np.nan)
    times = get("times_s", None)
    band_ts = get("band_avg_coh_win", None)

    if f is None or Cxy is None:
        raise ValueError("`coh` must contain 'f' and 'Cxy'.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.suptitle(title)

    # --- Top: global spectrum
    ax = axes[0]
    ax.plot(f, Cxy, lw=1.8)
    ax.set_ylabel("Coherence")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    if band is not None:
        fmin, fmax = band
        ax.axvspan(fmin, fmax, color="0.9", zorder=0)
    if np.isfinite(peak_f) and np.isfinite(peak_coh):
        ax.plot([peak_f], [peak_coh], "o")
        ax.text(peak_f, min(1.0, peak_coh + 0.03),
                f"peak {peak_coh:.2f}@{peak_f:.3f} Hz",
                ha="center", va="bottom")
    if np.isfinite(band_avg):
        ax.text(0.01, 0.95, f"band avg ~ {band_avg:.2f}", transform=ax.transAxes,
                ha="left", va="top")

    # --- Bottom: windowed band-avg over time (if available)
    ax = axes[1]
    if times is not None and band_ts is not None:
        ax.plot(times, band_ts, lw=1.8)
        ax.set_ylabel("Band-avg coherence")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No windowed series in `coh`.\nProvide times_s & band_avg_coh_win.",
                ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
