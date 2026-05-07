#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 4 - Five coupling cases with PSD + FOOOF middle column.

Same five cases as Methods Figure 3, but with an extra middle column
showing each signal's Welch power spectrum and the FOOOF-fit aperiodic
slope + detected peaks. This is especially informative for the
complexity-matching case (row 5), where the signals have NO time-domain
or phase coupling, only matched aperiodic exponents.

Layout
------
  signals (15 s preview)  |  PSD with FOOOF aperiodic + peaks  |  method bars

For each row, the middle panel reports the FOOOF aperiodic exponent for
both signals, so the reader can read off "delta alpha" - the spectral
similarity that complexity matching turns into a coupling score.

Output
------
figures/report/Methods4_cases_with_psd.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal as sps_sig
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.dsp import bandpass
from HNA.coupling import (
    plv_phase_sync,
    canolty_mvl,
    complexity_coupling,
)
from HNA.features.aperiodic import fit_aperiodic_psd
from HNA.viz import use_paper_style, save_figure


FAMILY_COLORS = {
    "linear":      "#3B7DD8",
    "oscillatory": "#5DA399",
    "information": "#7B5BA6",
    "complexity":  "#E08E1A",
}

# Signal colours kept DISTINCT from every family colour above so the family
# legend cannot be mistaken for a signal legend. Modern desaturated pair:
# charcoal slate + warm brick red.
SIG_X_COLOR = "#34495E"
SIG_Y_COLOR = "#C0392B"


# --------------------------------------------------------------------- #
# Cases (identical recipes to methods_fig3_coupling_cases.py)
# --------------------------------------------------------------------- #
def _pink(n, rng, fs, alpha=1.0):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = np.zeros_like(f)
    A[f > 0] = 1.0 / np.power(f[f > 0], alpha / 2.0)
    P = rng.uniform(0, 2 * np.pi, A.size)
    s = np.fft.irfft(A * np.exp(1j * P), n=n)
    return (s - s.mean()) / (s.std() + 1e-12)


def case_linear(fs, n, rng):
    x = _pink(n, rng, fs)
    y = 0.95 * x + 0.30 * rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    return x, y


def case_phase_only(fs, n, rng):
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.20 * t) + 0.15 * _pink(n, rng, fs)
    y = np.sin(2 * np.pi * 0.20 * t + np.pi / 2) + 0.15 * _pink(n, rng, fs)
    x = bandpass(x, fs=fs, lo=0.05, hi=1.0, order=8)
    y = bandpass(y, fs=fs, lo=0.05, hi=1.0, order=8)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


def case_pac(fs, n, rng):
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.20 * t) + 0.10 * rng.standard_normal(n)
    modulator = (1 + 0.99 * np.cos(2 * np.pi * 0.20 * t)) / 2
    y = modulator * np.sin(2 * np.pi * 15.0 * t) + 0.02 * rng.standard_normal(n)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


def case_nonlinear(fs, n, rng):
    x = _pink(n, rng, fs)
    y = np.abs(x) + 0.05 * rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    return x, y


def case_complexity(fs, n, rng):
    half = n // 2
    rng_x1 = np.random.default_rng(rng.integers(1_000_000))
    rng_y1 = np.random.default_rng(rng.integers(1_000_000))
    rng_x2 = np.random.default_rng(rng.integers(1_000_000))
    rng_y2 = np.random.default_rng(rng.integers(1_000_000))
    x = np.concatenate([rng_x1.standard_normal(half),
                        _pink(half, rng_x2, fs, alpha=1.0)])
    y = np.concatenate([rng_y1.standard_normal(half),
                        _pink(half, rng_y2, fs, alpha=1.0)])
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


CASES = [
    ("Linear amplitude coupling",
     "y = 0.95 x + noise",
     case_linear,
     "Both amplitudes co-vary linearly"),
    ("Phase coupling, Pearson r ~ 0",
     "shared 0.20 Hz phase, 90 deg offset, independent 1/f noise",
     case_phase_only,
     "Same beat, 90 deg apart -> waveforms uncorrelated"),
    ("Cross-frequency (PAC)",
     "slow phase of x modulates fast amplitude of y",
     case_pac,
     "Slow signal gates fast oscillation amplitude"),
    ("Nonlinear (info only)",
     "y = |x|",
     case_nonlinear,
     "Deterministic but Pearson r approx 0"),
    ("Complexity matching",
     "shared white -> pink scaling change; otherwise independent",
     case_complexity,
     "Both signals' DFA alpha(t) ramp up together at midpoint"),
]


# --------------------------------------------------------------------- #
# Methods (same as Fig 3)
# --------------------------------------------------------------------- #
def m_pearson_lagzero(x, y, fs):
    win = int(10.0 * fs); step = int(2.0 * fs)
    rs = []
    for s in range(0, len(x) - win + 1, step):
        a = x[s:s + win]; b = y[s:s + win]
        if a.std() > 0 and b.std() > 0:
            rs.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.mean(rs)) if rs else 0.0


def m_plv(x, y, fs):
    return float(plv_phase_sync(x, y, fs=fs, bw_hz=0.10).plv)


def m_pac(x, y, fs):
    return float(canolty_mvl(x, y, fs=fs,
                              low_band=(0.10, 0.40),
                              high_band=(10.0, 30.0)).mvl)


def m_mi(x, y, fs):
    step = max(1, int(fs / 4))
    return float(mutual_info_regression(
        x[::step].reshape(-1, 1), y[::step], n_neighbors=3,
        random_state=42)[0])


def m_complexity(x, y, fs):
    res = complexity_coupling(x, y, fs=fs, win_sec=20.0, step_sec=4.0,
                               method="pearson")
    val = res.get("coupling", float("nan"))
    return abs(val) if np.isfinite(val) else 0.0


METHODS = [
    ("Pearson |r|",   "linear",      m_pearson_lagzero),
    ("PLV",           "oscillatory", m_plv),
    ("PAC (Canolty)", "oscillatory", m_pac),
    ("MI",            "information", m_mi),
    ("complexity",    "complexity",  m_complexity),
]


# --------------------------------------------------------------------- #
# Compute results
# --------------------------------------------------------------------- #
def compute_all(fs: float, dur_s: float, rng_seed: int = 0):
    n = int(dur_s * fs)
    n_cases = len(CASES); n_methods = len(METHODS)
    V = np.full((n_cases, n_methods), np.nan)
    pairs = []
    for i, (name, _, gen, _) in enumerate(CASES):
        rng = np.random.default_rng(rng_seed + 17 * i)
        x, y = gen(fs, n, rng)
        pairs.append((x, y))
        for j, (mn, _, fn) in enumerate(METHODS):
            try:
                V[i, j] = fn(x, y, fs)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<32s} | {mn:<14s}  FAILED: {e}")
                V[i, j] = np.nan
    return V, pairs


# --------------------------------------------------------------------- #
# PSD + FOOOF middle panel
# --------------------------------------------------------------------- #
def _panel_psd(ax, x, y, fs, fmax_plot=30.0):
    """Plot Welch PSD of both signals + FOOOF aperiodic fit + peaks."""
    nper = int(min(len(x), fs * 8))      # 8 s segments
    fx, px = sps_sig.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    fy, py = sps_sig.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2)

    # Restrict plot range
    sel_x = (fx > 0) & (fx <= fmax_plot)
    sel_y = (fy > 0) & (fy <= fmax_plot)

    ax.semilogy(fx[sel_x], px[sel_x], color=SIG_X_COLOR, lw=1.2, label="x")
    ax.semilogy(fy[sel_y], py[sel_y], color=SIG_Y_COLOR, lw=1.2,
                alpha=0.9, label="y")

    # FOOOF fit on a broadband range above 1 Hz so the steep low-pass
    # rolloff present in some cases (e.g., bandpassed sinusoids) does not
    # dominate the aperiodic slope estimate. Narrow oscillatory peaks
    # within the range are absorbed into the peak component, leaving a
    # meaningful 1/f slope.
    fit_range = (1.0, min(fmax_plot, fs / 2 - 1))
    fres_x = fit_aperiodic_psd(fx, px, freq_range=fit_range, max_n_peaks=6)
    fres_y = fit_aperiodic_psd(fy, py, freq_range=fit_range, max_n_peaks=6)

    # Overlay aperiodic fit lines (10^(offset - exponent*log10(f)))
    def _ap_line(freqs, offset, exponent):
        return 10.0 ** (offset - exponent * np.log10(freqs))

    f_grid = np.geomspace(fit_range[0], fit_range[1], 200)
    R2_THRESH = 0.5   # below this the aperiodic fit is unreliable
    has_x = (np.isfinite(fres_x["aperiodic_offset"])
             and fres_x.get("r_squared", 0) >= R2_THRESH)
    has_y = (np.isfinite(fres_y["aperiodic_offset"])
             and fres_y.get("r_squared", 0) >= R2_THRESH)

    if has_x:
        ax.plot(f_grid,
                _ap_line(f_grid, fres_x["aperiodic_offset"],
                         fres_x["aperiodic_exponent"]),
                color=SIG_X_COLOR, lw=1.4, ls="--", alpha=0.8)
    if has_y:
        ax.plot(f_grid,
                _ap_line(f_grid, fres_y["aperiodic_offset"],
                         fres_y["aperiodic_exponent"]),
                color=SIG_Y_COLOR, lw=1.4, ls="--", alpha=0.8)

    # Mark detected peaks (vertical dotted lines).
    for peak in fres_x["peaks"][:6]:
        cf = peak["cf"]
        if 0 < cf <= fmax_plot:
            ax.axvline(cf, color=SIG_X_COLOR, ls=":", lw=0.7, alpha=0.55)
    for peak in fres_y["peaks"][:6]:
        cf = peak["cf"]
        if 0 < cf <= fmax_plot:
            ax.axvline(cf, color=SIG_Y_COLOR, ls=":", lw=0.7, alpha=0.55)

    # Annotate exponents — show only when fit is reliable.
    if has_x and has_y:
        txt = (f"alpha (x) = {fres_x['aperiodic_exponent']:.2f}\n"
               f"alpha (y) = {fres_y['aperiodic_exponent']:.2f}\n"
               f"|delta alpha| = "
               f"{abs(fres_x['aperiodic_exponent'] - fres_y['aperiodic_exponent']):.2f}")
    elif has_x:
        txt = f"alpha (x) = {fres_x['aperiodic_exponent']:.2f}\n(y: narrow-band)"
    elif has_y:
        txt = f"alpha (y) = {fres_y['aperiodic_exponent']:.2f}\n(x: narrow-band)"
    else:
        txt = "narrow-band signal\n(aperiodic fit not\nmeaningful)"
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", fc="#f8f8f8",
                      ec="#bbbbbb", lw=0.5))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (log)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2,
              handlelength=1.3)
    ax.grid(True, which="both", alpha=0.30)
    ax.set_xlim(0, fmax_plot)


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #
def plot_cases_psd(V, pairs, fs: float, preview_s: float, output_path: Path):
    use_paper_style()

    n_cases = len(CASES); n_methods = len(METHODS)

    col_max = np.nanmax(np.abs(V), axis=0)
    col_max = np.where(col_max > 0, col_max, 1.0)
    Vn = np.abs(V) / col_max[None, :]

    fig = plt.figure(figsize=(15.5, 13.0))
    gs = fig.add_gridspec(
        nrows=n_cases, ncols=3,
        width_ratios=[1.55, 1.05, 0.95],
        hspace=0.65, wspace=0.27,
        left=0.05, right=0.985, top=0.91, bottom=0.05,
    )

    n_prev = int(preview_s * fs)
    bar_x = np.arange(n_methods)
    bar_colors = [FAMILY_COLORS[fam] for _, fam, _ in METHODS]
    method_labels = [m[0] for m in METHODS]

    for i, ((title, formula, _, ann), (x, y)) in enumerate(zip(CASES, pairs)):
        # --- Left: signals
        ax_sig = fig.add_subplot(gs[i, 0])
        t_prev = np.arange(n_prev) / fs
        ax_sig.plot(t_prev, x[:n_prev], color=SIG_X_COLOR, lw=1.1, label="x")
        ax_sig.plot(t_prev, y[:n_prev], color=SIG_Y_COLOR, lw=1.1, alpha=0.9,
                    label="y")
        ax_sig.set_xlim(0, preview_s)
        if i == n_cases - 1:
            ax_sig.set_xlabel("Time (s)")
        ax_sig.set_ylabel("Amplitude (z)")
        ax_sig.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
        ax_sig.set_title(f"{i+1}. {title}    ({formula})",
                         fontsize=11.0, fontweight="bold", loc="left")
        ax_sig.text(0.01, -0.02, ann, transform=ax_sig.transAxes,
                    fontsize=8.5, color="#666", ha="left", va="top",
                    style="italic")
        ax_sig.grid(True, alpha=0.3)

        # --- Middle: PSD + FOOOF
        ax_psd = fig.add_subplot(gs[i, 1])
        _panel_psd(ax_psd, x, y, fs, fmax_plot=30.0)
        if i == 0:
            ax_psd.set_title("Welch PSD + FOOOF aperiodic (dashed) + peaks",
                              fontsize=10, loc="left")

        # --- Right: method bars
        ax_bar = fig.add_subplot(gs[i, 2])
        heights = Vn[i, :]
        bars = ax_bar.bar(bar_x, heights, color=bar_colors, alpha=0.92,
                           edgecolor="white", linewidth=0.8)
        for j, b in enumerate(bars):
            raw = V[i, j]
            if not np.isfinite(raw):
                continue
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.02,
                        f"{raw:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#222")
        ax_bar.set_xticks(bar_x)
        ax_bar.set_xticklabels(method_labels, rotation=22, ha="right",
                               fontsize=9)
        for tick, fam in zip(ax_bar.get_xticklabels(),
                              [m[1] for m in METHODS]):
            tick.set_color(FAMILY_COLORS[fam])
        ax_bar.set_ylim(0, 1.18)
        ax_bar.set_yticks([0, 0.5, 1.0])
        ax_bar.set_yticklabels(["0", "0.5", "1.0"])
        if i < n_cases - 1:
            ax_bar.set_xticklabels([])
        ax_bar.grid(True, axis="y", alpha=0.30)

    # Big top legend
    handles = [
        Patch(facecolor=FAMILY_COLORS[f], edgecolor="white", linewidth=1.5,
              label=f)
        for f in ("linear", "oscillatory", "information", "complexity")
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.975), ncol=4,
        frameon=False, fontsize=12.5, handlelength=1.5,
        handleheight=1.5, columnspacing=2.2, labelspacing=0.4,
    )

    save_figure(fig, output_path)
    plt.close()
    print(f"  Saved: {output_path.name}.png (+ pdf)")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--dur-s", type=float, default=60.0)
    p.add_argument("--preview-s", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures" / "report" / "Methods4_cases_with_psd")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    V, pairs = compute_all(fs=args.fs, dur_s=args.dur_s, rng_seed=args.seed)
    plot_cases_psd(V, pairs, fs=args.fs, preview_s=args.preview_s,
                    output_path=args.out)


if __name__ == "__main__":
    main()
