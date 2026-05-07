#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 6 v2 - Worked example linked to the framework cells.

Same content as Methods 6 v1 (sub-02, MULTI condition, audio swell
envelope vs cleaned respiration, four coupling-family panels) but each
panel title now carries the explicit (feature x coupling) cell tag of
the framework matrix in Figure 5. Panel D is reframed as "complexity
feature x linear coupling: alpha(t) traces, then Pearson r" so that the
two-step structure (windowed_exponent -> Pearson) is visible.

Default cell: subject 02, MULTI condition. Override with --subject /
--condition / --data-dir.

Output
------
figures/report/Methods6v2_worked_example.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sps_sig
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.utils import get_condition_segments
from HNA.modalities.respiration import clean_respiration
from HNA.coupling import (
    windowed_xcorr,
    plv_phase_sync,
    windowed_exponent,
)
from HNA.viz import use_paper_style, save_figure


# Signal colours (kept distinct from the four family colours).
SIG_X_COLOR = "#34495E"   # slate
SIG_Y_COLOR = "#C0392B"   # brick
FAMILY_COLORS = {
    "linear":      "#3B7DD8",
    "oscillatory": "#5DA399",
    "information": "#7B5BA6",
    "complexity":  "#E08E1A",
}

FS = 256.0   # merged-CSV sample rate


# --------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------- #
def load_pair(data_dir: Path, subject: int, condition: str,
              env_col: str = "env_swell_0p2"):
    """Return ``(audio_env, respiration_clean, t_seconds)`` for one cell."""
    sub = f"sub-{subject:02d}"
    merged = data_dir / "processed" / sub / "tables" / "merged_annotated_with_audio.csv"
    if not merged.exists():
        raise FileNotFoundError(f"Missing merged CSV: {merged}")

    df = pd.read_csv(merged, low_memory=False)
    if env_col not in df.columns:
        raise KeyError(f"Envelope column {env_col} not in merged CSV")
    indices = get_condition_segments(df, df["condition_names"].unique())
    s_key, e_key = f"{condition}_start", f"{condition}_stop"
    if s_key not in indices or e_key not in indices:
        raise KeyError(f"Condition {condition} not found for {sub}")

    s, e = int(indices[s_key]), int(indices[e_key])
    seg = df.iloc[s:e].copy()
    audio_env = seg[env_col].to_numpy(float)
    resp = clean_respiration(seg["respiration"], fs=FS)

    # Standardise both to z-score for visual comparability.
    audio_env = (audio_env - audio_env.mean()) / audio_env.std()
    resp = (resp - resp.mean()) / resp.std()
    t = np.arange(seg.shape[0]) / FS
    return audio_env, resp, t


# --------------------------------------------------------------------- #
# Panel painters
# --------------------------------------------------------------------- #
def _strip_signals(ax, t, env, resp, preview_s: float = 60.0):
    n = min(len(t), int(preview_s * FS))
    ax.plot(t[:n], env[:n], color=SIG_X_COLOR, lw=1.2,
            label="audio swell envelope (z)")
    ax.plot(t[:n], resp[:n], color=SIG_Y_COLOR, lw=1.2, alpha=0.9,
            label="respiration (z)")
    ax.set_xlabel("Time within condition (s)")
    ax.set_ylabel("Amplitude (z)")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2)
    ax.grid(True, alpha=0.30)


def _panel_linear(ax, env, resp, fs):
    """A - windowed_xcorr peak |r| over time + the average lag profile."""
    xc = windowed_xcorr(env, resp, fs=fs, win_sec=30.0, step_sec=5.0,
                        max_lag_sec=10.0, return_matrix=True)
    mean_lag_profile = np.nanmean(xc.xcorr, axis=0)
    peak_idx = int(np.argmax(np.abs(mean_lag_profile)))
    peak_lag = float(xc.lags_s[peak_idx])
    peak_r = float(mean_lag_profile[peak_idx])

    ax.axvline(0, color="#aaaaaa", lw=0.7, ls=":")
    ax.axhline(0, color="#aaaaaa", lw=0.7, ls=":")
    ax.plot(xc.lags_s, mean_lag_profile, color=FAMILY_COLORS["linear"],
            lw=1.6)
    ax.scatter([peak_lag], [peak_r], color="#C0392B", s=70, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.text(0.04, 0.96,
            f"mean peak |r| = {np.nanmean(np.abs(xc.peak_r)):.2f}\n"
            f"avg-window peak: r = {peak_r:.2f} at lag {peak_lag:+.1f} s",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="#f5f5f5",
                      ec="#cccccc", lw=0.5))
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("A.  Linear   (windowed xcorr)\n"
                 "raw signal x linear coupling",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.30)


def _panel_oscillatory(ax, env, resp, fs):
    """B - PLV polar at the swell-band dominant frequency."""
    res = plv_phase_sync(env, resp, fs=fs, bw_hz=0.10,
                          fmin_search=0.05, fmax_search=0.5)
    plv = res.plv
    mu = res.mean_phase_diff
    f0 = res.f0

    # Compute per-sample phase difference (for the polar histogram visual).
    sos = sps_sig.butter(4, [max(1e-3, f0 - 0.05) / (fs / 2),
                              (f0 + 0.05) / (fs / 2)],
                         btype="band", output="sos")
    bx = sps_sig.sosfiltfilt(sos, env.astype(float))
    by = sps_sig.sosfiltfilt(sos, resp.astype(float))
    dphi = np.angle(np.exp(1j * (np.angle(sps_sig.hilbert(bx)) -
                                  np.angle(sps_sig.hilbert(by)))))

    n_bins = 24
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(dphi, bins=edges)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 2 * np.pi / n_bins

    ax.bar(centers, hist, width=width,
           color=FAMILY_COLORS["oscillatory"],
           edgecolor="white", linewidth=0.8, alpha=0.85)
    arrow_r = float(hist.max()) * (0.55 + 0.45 * plv) if hist.max() > 0 else 0.0
    ax.annotate("",
                xy=(mu, arrow_r), xytext=(mu, 0),
                arrowprops=dict(arrowstyle="-|>", color="#222",
                                lw=2.0, mutation_scale=14))
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "", "180", ""])
    ax.set_yticklabels([])
    ax.set_title(f"B.  Oscillatory   (PLV at {f0:.2f} Hz)\n"
                 f"oscillatory feature x oscillatory coupling\n"
                 f"PLV = {plv:.2f}, preferred phase = {np.rad2deg(mu):.0f} deg",
                 fontsize=10.5, fontweight="bold", loc="left", pad=14)


def _panel_information(ax, env, resp, fs):
    """C - joint scatter + Pearson r + MI on a decorrelated subsample."""
    step = max(1, int(fs / 4))
    xd, yd = env[::step], resp[::step]
    vis_step = max(1, env.size // 3000)
    ax.scatter(env[::vis_step], resp[::vis_step], s=4, alpha=0.30,
               color=FAMILY_COLORS["information"], edgecolor="none")
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.axvline(0, color="#bbb", lw=0.5)

    mi = float(mutual_info_regression(
        xd.reshape(-1, 1), yd, n_neighbors=3, random_state=42)[0])
    r = float(np.corrcoef(xd, yd)[0, 1])
    txt = (f"Pearson r = {r:+.3f}\n"
           f"MI (4 Hz subsample) = {mi:+.3f} nats")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="#f5f5f5",
                      ec="#cccccc", lw=0.5))
    ax.set_xlabel("audio swell envelope (z)")
    ax.set_ylabel("respiration (z)")
    ax.set_title("C.  Information   (joint dependence)\n"
                 "raw signal x information coupling",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.30)


def _panel_complexity(ax, env, resp, fs):
    """D - windowed DFA-alpha traces for both signals + Pearson r."""
    rx = windowed_exponent(env, fs=fs, win_sec=30.0, step_sec=5.0)
    ry = windowed_exponent(resp, fs=fs, win_sec=30.0, step_sec=5.0)
    n = min(rx["exponent"].size, ry["exponent"].size)
    a = rx["exponent"][:n]; b = ry["exponent"][:n]
    times = rx["times_s"][:n]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() >= 3:
        rcoup = float(np.corrcoef(a[valid], b[valid])[0, 1])
    else:
        rcoup = float("nan")

    ax.plot(times, a, "-o", ms=4, color=SIG_X_COLOR, lw=1.4,
            label="alpha(env)")
    ax.plot(times, b, "-s", ms=4, color=SIG_Y_COLOR, lw=1.4, alpha=0.9,
            label="alpha(resp)")
    ax.set_xlabel("Window centre (s)")
    ax.set_ylabel("DFA alpha")
    ax.legend(loc="upper right", frameon=False, fontsize=9.0, ncol=2)
    ax.text(0.04, 0.04,
            f"complexity_coupling r = {rcoup:+.2f}\n"
            f"mean alpha(env)  = {np.nanmean(a):.2f}\n"
            f"mean alpha(resp) = {np.nanmean(b):.2f}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.0, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="#f5f5f5",
                      ec="#cccccc", lw=0.5))
    ax.set_title("D.  Complexity feature x linear coupling\n"
                 "windowed alpha(t) per signal, then Pearson r",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.30)


# --------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------- #
def make_figure(env, resp, t, label: str, out_path: Path):
    use_paper_style()

    fig = plt.figure(figsize=(11.5, 9.4))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[0.78, 1.0, 1.0],
        hspace=0.75, wspace=0.30,
        left=0.07, right=0.97, top=0.91, bottom=0.07,
    )

    ax_strip = fig.add_subplot(gs[0, :])
    _strip_signals(ax_strip, t, env, resp, preview_s=60.0)
    ax_strip.set_title(label, fontsize=11.5, fontweight="bold", loc="left")

    _panel_linear(fig.add_subplot(gs[1, 0]), env, resp, FS)
    _panel_oscillatory(fig.add_subplot(gs[1, 1], projection="polar"),
                        env, resp, FS)
    _panel_information(fig.add_subplot(gs[2, 0]), env, resp, FS)
    _panel_complexity(fig.add_subplot(gs[2, 1]), env, resp, FS)

    fig.text(0.07, 0.97,
             "Worked example: audio swell envelope - respiration on real pilot data",
             fontsize=13.5, fontweight="bold", ha="left", va="bottom",
             color="#1F2A37")
    fig.text(0.07, 0.952,
             "Each panel labelled by its (feature x coupling) cell of the framework.",
             fontsize=10.0, color="#5A6470", ha="left", va="bottom",
             style="italic")

    save_figure(fig, out_path)
    plt.close()
    print(f"  Saved: {out_path.name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=2,
                   help="Subject ID (default: 2).")
    p.add_argument("--condition", default="MULTI",
                   help="Condition label (default: MULTI).")
    p.add_argument("--env-col", default="env_swell_0p2",
                   help="Audio envelope column (default: env_swell_0p2).")
    p.add_argument("--data-dir", type=Path,
                   default=ROOT.parents[2] / "data",
                   help="Pilot data root (default: <repo>/data, falling back "
                        "to its sibling outside the worktree).")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures" / "report" / "Methods6v2_worked_example")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env, resp, t = load_pair(args.data_dir, args.subject, args.condition,
                              env_col=args.env_col)
    label = f"sub-{args.subject:02d}, condition {args.condition}  ({len(env) / FS:.0f} s)"
    make_figure(env, resp, t, label, args.out)


if __name__ == "__main__":
    main()
