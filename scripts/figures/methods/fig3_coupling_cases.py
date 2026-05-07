#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 3 - Five coupling cases, side-by-side method comparison.

Each row shows ONE clear case of coupling between two synthetic signals,
with the signals on the left (so the reader can see what kind of
relationship is built in) and a bar chart on the right showing what each
of the four coupling families reports. The figure demonstrates that
**different families pick up different structures** — there is no single
"best" coupling metric.

Cases (top to bottom)
---------------------
1.  Linear amplitude coupling      y = 0.7*x + noise  (xcorr / MI fire)
2.  Phase coupling, no amplitude   shared 0.20 Hz phase, INDEPENDENT amplitude
                                   envelopes  (PLV fires; xcorr is muted)
3.  Cross-frequency (PAC)          slow phase of x modulates fast amplitude
                                   of y  (PAC fires; xcorr / PLV at the same
                                   frequency would not see this)
4.  Nonlinear, Pearson r approx 0  y = |x|  (MI fires; xcorr cannot)
5.  Complexity-matched only        two INDEPENDENT 1/f signals with the
                                   same DFA alpha  (only fluctuation
                                   matching fires)

Methods plotted (one canonical estimator per family)
----------------------------------------------------
    linear        : windowed_xcorr peak |r|
    oscillatory   : PLV  + Tort PAC modulation index
    information   : MI (kNN, decorrelating downsample to 4 Hz)
    complexity    : fluctuation matching r

Per-method values are normalized to that method's max across the five
cases, so the bar heights are visually comparable within each case (the
raw value is annotated on each bar).

Output
------
reports/methods/figures/Methods3_coupling_cases.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from HNA.dsp import bandpass
from HNA.coupling import (
    windowed_xcorr,
    plv_phase_sync,
    canolty_mvl,
    complexity_coupling,
)
from HNA.viz import use_paper_style, save_figure


FAMILY_COLORS = {
    "linear":      "#3B7DD8",
    "oscillatory": "#5DA399",
    "information": "#7B5BA6",
    "complexity":  "#E08E1A",
}

# Signal colours kept DISTINCT from every family colour above so the family
# legend cannot be mistaken for a signal legend. A modern desaturated pair:
# charcoal slate + warm brick red.
SIG_X_COLOR = "#34495E"
SIG_Y_COLOR = "#C0392B"


# --------------------------------------------------------------------- #
# Synthetic signal generators
# --------------------------------------------------------------------- #
def _pink(n, rng, fs, alpha=1.0):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = np.zeros_like(f)
    A[f > 0] = 1.0 / np.power(f[f > 0], alpha / 2.0)
    P = rng.uniform(0, 2 * np.pi, A.size)
    s = np.fft.irfft(A * np.exp(1j * P), n=n)
    return (s - s.mean()) / (s.std() + 1e-12)


def case_linear(fs, n, rng):
    """Strong linear coupling: y is a noisy linear copy of x."""
    x = _pink(n, rng, fs)
    y = 0.95 * x + 0.30 * rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    return x, y


def case_phase_only(fs, n, rng):
    """Same-frequency phase lock at PHASE OFFSET pi/2.

    Both signals oscillate at 0.20 Hz with a fixed 90 deg phase offset.
    Broadband background noise is kept LOW (0.15 amplitude) and 1-Hz
    low-pass filtered after construction so virtually no energy leaks
    into the PAC test high band (3-8 Hz) -> no spurious PAC.
    PLV stays near 1.0 because the phase relationship is constant.
    """
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.20 * t) + 0.15 * _pink(n, rng, fs)
    y = np.sin(2 * np.pi * 0.20 * t + np.pi / 2) + 0.15 * _pink(n, rng, fs)
    x = bandpass(x, fs=fs, lo=0.05, hi=1.0, order=8)   # steeper rolloff
    y = bandpass(y, fs=fs, lo=0.05, hi=1.0, order=8)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


def case_pac(fs, n, rng):
    """Phase of x (slow) modulates amplitude of y (fast 15 Hz carrier).

    Carrier at 15 Hz is well inside the PAC test high band (10-30 Hz),
    far from the 0.20 Hz dominant of x so spectral leakage cannot inflate
    the test. Modulation depth approaches 1.0 -> large Tort MI.
    """
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.20 * t) + 0.10 * rng.standard_normal(n)
    modulator = (1 + 0.99 * np.cos(2 * np.pi * 0.20 * t)) / 2
    fast = np.sin(2 * np.pi * 15.0 * t)
    y = modulator * fast + 0.02 * rng.standard_normal(n)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


def case_nonlinear(fs, n, rng):
    """y = |x|: deterministic, Pearson r approx 0 for symmetric x."""
    x = _pink(n, rng, fs)
    y = np.abs(x) + 0.05 * rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    return x, y


def case_complexity(fs, n, rng):
    """Shared scaling-change between two otherwise INDEPENDENT signals.

    Signals are uncorrelated in time, phase, and amplitude. The only thing
    they have in common is a coordinated transition from white noise
    (DFA alpha approx 0.5) to pink noise (alpha approx 1.0) at the midpoint,
    so their windowed scaling-exponent traces alpha(t) ramp up together.
    This is the classical Marmelat-Delignieres "complexity matching" case.
    """
    half = n // 2
    rng_x1 = np.random.default_rng(rng.integers(1_000_000))
    rng_y1 = np.random.default_rng(rng.integers(1_000_000))
    rng_x2 = np.random.default_rng(rng.integers(1_000_000))
    rng_y2 = np.random.default_rng(rng.integers(1_000_000))
    x_white = rng_x1.standard_normal(half)
    y_white = rng_y1.standard_normal(half)
    x_pink = _pink(half, rng_x2, fs, alpha=1.0)
    y_pink = _pink(half, rng_y2, fs, alpha=1.0)
    x = np.concatenate([x_white, x_pink])
    y = np.concatenate([y_white, y_pink])
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
# Method runners
# --------------------------------------------------------------------- #
def m_xcorr(x, y, fs):
    """Lag-zero Pearson |r| (windowed mean).

    We deliberately do NOT use the lag-search version of xcorr here:
    when two signals are phase-locked at the same frequency, a lag-search
    finds the alignment lag and reports a high correlation, which conflates
    "linear amplitude coupling" with "phase coupling". Lag-zero Pearson
    isolates linear coupling per the case 1 vs case 2 distinction.
    """
    win = int(10.0 * fs)
    step = int(2.0 * fs)
    rs = []
    for s in range(0, len(x) - win + 1, step):
        a = x[s:s + win]; b = y[s:s + win]
        if a.std() > 0 and b.std() > 0:
            rs.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.mean(rs)) if rs else 0.0


def m_plv(x, y, fs):
    return float(plv_phase_sync(x, y, fs=fs, bw_hz=0.10).plv)


def m_pac(x, y, fs):
    """Canolty mean-vector-length (raw, amplitude-aware).

    Tort MI normalises out the absolute amplitude in the high band, which
    makes it report large values when filter residuals at the high band
    happen to phase-align even though their amplitude is tiny. Canolty MVL
    is the magnitude of mean(amp * exp(i*phase)), so it is naturally zero
    when the high-band amplitude is small. We use the raw MVL here (not
    the amplitude-normalised variant) so it directly reflects "how much
    aligned amplitude is there".
    """
    return float(canolty_mvl(
        x, y, fs=fs, low_band=(0.10, 0.40), high_band=(10.0, 30.0)).mvl)


def m_mi(x, y, fs):
    step = max(1, int(fs / 4))
    return float(mutual_info_regression(
        x[::step].reshape(-1, 1), y[::step], n_neighbors=3,
        random_state=42)[0])


def m_complexity(x, y, fs):
    """Pearson r between the two windowed DFA alpha(t) traces.

    Picks up *coordinated* scaling-structure change without requiring
    sample-by-sample co-variation. For stationary signals the alpha(t)
    traces are flat (+ noise) so this returns approx 0; only signals
    whose scaling fluctuates *together* over time give a strong score.
    """
    res = complexity_coupling(x, y, fs=fs, win_sec=20.0, step_sec=4.0,
                               method="pearson")
    val = res.get("coupling", float("nan"))
    return abs(val) if np.isfinite(val) else 0.0


METHODS = [
    ("Pearson |r|",     "linear",      m_xcorr),
    ("PLV",             "oscillatory", m_plv),
    ("PAC (Canolty)",   "oscillatory", m_pac),
    ("MI",              "information", m_mi),
    ("complexity",      "complexity",  m_complexity),
]


# --------------------------------------------------------------------- #
# Compute the (case x method) value matrix
# --------------------------------------------------------------------- #
def compute_matrix(fs: float, dur_s: float, rng_seed: int = 0):
    n = int(dur_s * fs)
    n_cases = len(CASES)
    n_methods = len(METHODS)
    V = np.full((n_cases, n_methods), np.nan)
    signal_pairs = []
    for i, (name, _, gen, _) in enumerate(CASES):
        rng = np.random.default_rng(rng_seed + 17 * i)
        x, y = gen(fs, n, rng)
        signal_pairs.append((x, y))
        for j, (mn, _, fn) in enumerate(METHODS):
            try:
                V[i, j] = fn(x, y, fs)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<32s} | {mn:<14s}  FAILED: {e}")
                V[i, j] = np.nan
    return V, signal_pairs


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #
def plot_cases(V: np.ndarray, signal_pairs, fs: float,
               preview_s: float, output_path: Path):
    use_paper_style()

    n_cases = len(CASES)
    n_methods = len(METHODS)

    # Per-method normalisation for visual comparison: each method's bar
    # heights are divided by its own max across cases. Annotations show
    # the raw value.
    col_max = np.nanmax(np.abs(V), axis=0)
    col_max = np.where(col_max > 0, col_max, 1.0)
    Vn = np.abs(V) / col_max[None, :]

    fig = plt.figure(figsize=(11.5, 13.0))
    gs = fig.add_gridspec(
        nrows=n_cases, ncols=2,
        width_ratios=[1.55, 1.0],
        hspace=0.65, wspace=0.18,
        left=0.07, right=0.97, top=0.91, bottom=0.05,
    )

    n_prev = int(preview_s * fs)
    bar_x = np.arange(n_methods)
    bar_colors = [FAMILY_COLORS[fam] for _, fam, _ in METHODS]
    method_labels = [m[0] for m in METHODS]

    for i, ((title, formula, _, ann), (x, y)) in enumerate(zip(CASES, signal_pairs)):
        # Left: signals
        ax_sig = fig.add_subplot(gs[i, 0])
        t_prev = np.arange(n_prev) / fs
        ax_sig.plot(t_prev, x[:n_prev], color=SIG_X_COLOR, lw=1.2, label="x")
        ax_sig.plot(t_prev, y[:n_prev], color=SIG_Y_COLOR, lw=1.2, alpha=0.95, label="y")
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

        # Right: method bars
        ax_bar = fig.add_subplot(gs[i, 1])
        heights = Vn[i, :]
        bars = ax_bar.bar(bar_x, heights, color=bar_colors, alpha=0.92,
                          edgecolor="white", linewidth=0.8)
        # Annotate each bar with the raw value
        for j, b in enumerate(bars):
            raw = V[i, j]
            if not np.isfinite(raw):
                continue
            label = f"{raw:.2f}"
            ymax = max(0.05, b.get_height())
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.02,
                        label, ha="center", va="bottom",
                        fontsize=9, fontweight="bold",
                        color="#222")

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

    # Big, prominent family-color legend at the top — no title.
    handles = [
        Patch(facecolor=FAMILY_COLORS[f], edgecolor="white", linewidth=1.5,
              label=f"{f}")
        for f in ("linear", "oscillatory", "information", "complexity")
    ]
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.975), ncol=4,
        frameon=False, fontsize=12.5, handlelength=1.5,
        handleheight=1.5, columnspacing=2.2, labelspacing=0.4,
        title=None,
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
    p.add_argument("--preview-s", type=float, default=15.0,
                   help="Time-series preview length in seconds.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "methods" / "figures" / "Methods3_coupling_cases")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    V, pairs = compute_matrix(fs=args.fs, dur_s=args.dur_s,
                               rng_seed=args.seed)
    plot_cases(V, pairs, fs=args.fs,
               preview_s=args.preview_s, output_path=args.out)


if __name__ == "__main__":
    main()
