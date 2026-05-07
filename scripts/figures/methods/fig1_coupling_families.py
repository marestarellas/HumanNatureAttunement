#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 1 - Four coupling families on one synthetic signal pair.

Builds a controlled audio-vs-respiration synthetic pair with known
properties (slow 0.2 Hz coupling, 0.5 s phase lag, shared 1/f component)
and visualises it through one canonical estimator from each of the four
coupling families:

  A) Linear        - windowed cross-correlation peak vs lag
  B) Oscillatory   - PLV polar histogram + mean-vector arrow at 0.2 Hz
  C) Information   - joint scatter + raw MI vs effective (bias-corrected) MI
  D) Complexity    - log F(s) fluctuation curves + matching score

Output
------
reports/methods/figures/Methods1_coupling_families.{png,pdf}

Usage
-----
    python scripts/figures/methods_fig1_coupling_families.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from HNA.dsp import bandpass
from HNA.coupling import (
    windowed_xcorr,
    plv_phase_sync,
    effective_mi,
    fluctuation_curve,
    fluctuation_matching,
)
from HNA.viz import use_paper_style, save_figure


# --------------------------------------------------------------------- #
# Synthetic signal pair: slow swell + 1/f, lagged respiration
# --------------------------------------------------------------------- #
def _pink(n: int, rng: np.random.Generator, fs: float) -> np.ndarray:
    """1/f-amplitude noise of length ``n``."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = np.where(f > 0, 1.0 / np.sqrt(f), 0.0)
    P = rng.uniform(0, 2 * np.pi, A.size)
    s = np.fft.irfft(A * np.exp(1j * P), n=n)
    return (s - s.mean()) / (s.std() + 1e-12)


def make_pair(fs: float = 256.0, dur_s: float = 90.0,
              lag_s: float = 0.5, rng_seed: int = 0):
    """Construct (audio, respiration) with strong slow coupling + lag.

    Recipe: audio = 0.20 Hz swell + a touch of 0.40 Hz + 1/f background.
    Respiration shares a *lagged copy of the audio time series* (not just
    the spectrum) plus its own additive noise — this gives a coupling
    that survives the phase-shuffle bias correction in :func:`effective_mi`,
    so all four panels of the figure fire visibly.
    """
    rng = np.random.default_rng(rng_seed)
    n = int(dur_s * fs)
    t = np.arange(n) / fs

    # Audio: 0.2 Hz swell + a touch of 0.4 Hz + 1/f background
    audio = (np.sin(2 * np.pi * 0.20 * t)
             + 0.35 * np.sin(2 * np.pi * 0.40 * t + 1.1)
             + 0.6 * _pink(n, rng, fs))

    # Respiration: lagged copy of the *audio waveform itself* plus noise.
    # This means phase-shuffling audio (which preserves spectrum but
    # destroys the per-sample time-domain match) genuinely reduces the
    # effective MI — so panel C separates from null.
    lag_samples = int(lag_s * fs)
    audio_lagged = np.roll(audio, lag_samples)
    resp = audio_lagged + 0.4 * rng.standard_normal(n)

    # Mild bandpass to mimic the project's resp/swell preprocessing
    audio = bandpass(audio, fs=fs, lo=0.05, hi=1.0)
    resp = bandpass(resp, fs=fs, lo=0.05, hi=1.0)
    audio = (audio - audio.mean()) / audio.std()
    resp = (resp - resp.mean()) / resp.std()
    return t, audio, resp


# --------------------------------------------------------------------- #
# Panel painters
# --------------------------------------------------------------------- #
def _panel_linear(ax, t, x, y, fs):
    """A — windowed xcorr peak r vs lag, with marker at the true lag."""
    # Use windowed_xcorr with return_matrix=True so we can plot the lag profile.
    xc = windowed_xcorr(x, y, fs=fs, win_sec=30.0, step_sec=5.0,
                        max_lag_sec=5.0, return_matrix=True)
    # Mean across windows -> single lag profile.
    lag_profile = np.nanmean(xc.xcorr, axis=0)
    peak_idx = int(np.argmax(np.abs(lag_profile)))
    peak_lag = float(xc.lags_s[peak_idx])
    peak_r = float(lag_profile[peak_idx])

    ax.axvline(0, color="#aaaaaa", lw=0.8, ls=":")
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":")
    ax.plot(xc.lags_s, lag_profile, color="#3B7DD8", lw=1.6)
    ax.scatter([peak_lag], [peak_r], color="#C9325F", s=70, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.annotate(f"peak r = {peak_r:.2f}\n"
                f"at lag {peak_lag:+.2f} s",
                xy=(peak_lag, peak_r),
                xytext=(0.62, 0.85), textcoords="axes fraction",
                fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="-", color="#888",
                                lw=0.8, shrinkB=8))
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("A.  Linear   (windowed xcorr)", fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3)


def _panel_oscillatory(ax, t, x, y, fs):
    """B — PLV polar histogram of Δφ at the dominant slow frequency."""
    # We force f0 = 0.20 Hz here (the coupling frequency) since that's the
    # method's design parameter; the general API lets _dominant_freq pick it.
    res = plv_phase_sync(x, y, fs=fs, f0=0.20, bw_hz=0.10)
    plv = res.plv
    mu = res.mean_phase_diff

    # Compute the per-sample phase difference for the polar histogram.
    sos = sps.butter(4, [0.15 / (fs / 2), 0.25 / (fs / 2)],
                     btype="band", output="sos")
    bx = sps.sosfiltfilt(sos, x.astype(float))
    by = sps.sosfiltfilt(sos, y.astype(float))
    dphi = np.angle(np.exp(1j * (np.angle(sps.hilbert(bx)) -
                                  np.angle(sps.hilbert(by)))))

    n_bins = 24
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(dphi, bins=edges)
    hist = hist / hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 2 * np.pi / n_bins

    ax.bar(centers, hist, width=width, color="#5DA399",
           edgecolor="white", linewidth=0.8, alpha=0.85)
    # Mean-vector arrow at radius PLV (within-axis annotation)
    arrow_r = float(hist.max()) * (0.55 + 0.45 * plv)
    ax.annotate("",
                xy=(mu, arrow_r), xytext=(mu, 0),
                arrowprops=dict(arrowstyle="-|>", color="#222",
                                lw=2.0, mutation_scale=14))
    # Cardinal-only ticks (avoid 90/270 collision with title)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "", "180", ""])
    ax.set_yticklabels([])
    ax.set_title(f"B.  Oscillatory   (PLV at 0.20 Hz)\n"
                 f"PLV = {plv:.2f}, preferred phase = {np.rad2deg(mu):.0f}",
                 fontsize=11.5, fontweight="bold", loc="left", pad=14)


def _panel_information(ax, t, x, y, fs):
    """C — joint scatter + MI and Pearson r annotated.

    The scatter is plotted at full sample rate. MI is estimated on a
    decorrelated subsample (~4 Hz) so the kNN bias from autocorrelation is
    contained; the figure compares it to a Pearson correlation on the same
    subsample to make the linear-vs-information distinction explicit.
    """
    from sklearn.feature_selection import mutual_info_regression

    # Decorrelating downsample for the MI estimator.
    step = int(round(fs / 4.0))   # ~4 Hz
    xd = x[::step]; yd = y[::step]

    # Visual scatter at higher density.
    vis_step = max(1, x.size // 3000)
    ax.scatter(x[::vis_step], y[::vis_step], s=4, alpha=0.35,
               color="#7B5BA6", edgecolor="none")
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.axvline(0, color="#bbb", lw=0.5)

    mi = float(mutual_info_regression(
        xd.reshape(-1, 1), yd, n_neighbors=3, random_state=42)[0])
    r = float(np.corrcoef(xd, yd)[0, 1])
    txt = (f"Pearson r = {r:+.3f}\n"
           f"MI        = {mi:+.3f} nats")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="#f5f5f5", ec="#cccccc", lw=0.6))
    ax.set_xlabel("audio (z)")
    ax.set_ylabel("respiration (z)")
    ax.set_title("C.  Information   (joint dependence)",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3)


def _panel_complexity(ax, t, x, y, fs):
    """D — log-log F(s) curves for each signal + matching score."""
    n = x.size
    scales = np.unique(np.logspace(np.log10(8),
                                   np.log10(n // 4), 25).astype(int))
    cx = fluctuation_curve(x, scales=scales)
    cy = fluctuation_curve(y, scales=scales)

    valid = np.isfinite(cx["F"]) & np.isfinite(cy["F"])
    s_valid = cx["scales"][valid]
    fx = cx["F"][valid]
    fy = cy["F"][valid]

    ax.loglog(s_valid, fx, "-o", ms=3.5, color="#3B7DD8", lw=1.6,
              label=f"audio   (alpha = {cx['alpha']:.2f})")
    ax.loglog(s_valid, fy, "-s", ms=3.5, color="#E08E1A", lw=1.6,
              label=f"resp    (alpha = {cy['alpha']:.2f})")

    fm = fluctuation_matching(x, y, scales=scales)
    txt = (f"matching r(F) = {fm['r']:+.3f}\n"
           f"|delta alpha|  = {abs(fm['delta_alpha']):.2f}")
    ax.text(0.04, 0.04, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="#f5f5f5", ec="#cccccc", lw=0.6))

    ax.set_xlabel("scale s (samples)")
    ax.set_ylabel("F(s)")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("D.  Complexity   (DFA fluctuation matching)",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.grid(True, which="both", alpha=0.3)


# --------------------------------------------------------------------- #
# Top strip: signal preview
# --------------------------------------------------------------------- #
def _strip_signals(ax, t, x, y, preview_s: float = 30.0):
    n_prev = int(preview_s * (1.0 / (t[1] - t[0])))
    ax.plot(t[:n_prev], x[:n_prev], color="#3B7DD8", lw=1.2,
            label="audio (z)")
    ax.plot(t[:n_prev], y[:n_prev], color="#E08E1A", lw=1.2, alpha=0.9,
            label="respiration (z)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (z)")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2)
    ax.set_title("Synthetic signal pair  (0.20 Hz coupling, 0.5 s lag, shared 1/f)",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "methods" / "figures" / "Methods1_coupling_families")
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--dur-s", type=float, default=90.0)
    p.add_argument("--lag-s", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    use_paper_style()
    t, x, y = make_pair(fs=args.fs, dur_s=args.dur_s,
                        lag_s=args.lag_s, rng_seed=args.seed)

    fig = plt.figure(figsize=(11.0, 8.6))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[0.85, 1.0, 1.0],
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    ax_strip = fig.add_subplot(gs[0, :])
    _strip_signals(ax_strip, t, x, y, preview_s=30.0)

    ax_a = fig.add_subplot(gs[1, 0])
    _panel_linear(ax_a, t, x, y, args.fs)

    ax_b = fig.add_subplot(gs[1, 1], projection="polar")
    _panel_oscillatory(ax_b, t, x, y, args.fs)

    ax_c = fig.add_subplot(gs[2, 0])
    _panel_information(ax_c, t, x, y, args.fs)

    ax_d = fig.add_subplot(gs[2, 1])
    _panel_complexity(ax_d, t, x, y, args.fs)

    save_figure(fig, args.out)
    plt.close()
    print(f"  Saved: {args.out.name}.png (+ pdf)")


if __name__ == "__main__":
    main()
