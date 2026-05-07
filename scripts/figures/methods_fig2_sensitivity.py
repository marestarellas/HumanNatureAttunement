#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 2 - Method x signal-type sensitivity matrix.

For each of six synthetic signal-pair types with KNOWN coupling structure,
runs every coupling-family estimator and reports the raw metric value,
column-normalized so each method's bar / cell occupies [0, 1] regardless
of its native range. The diagonal-ish pattern shows which method peaks on
which kind of coupling — a quick "use method M when you suspect coupling
type T" reference.

We deliberately do NOT use surrogate-test z-scores: phase-shuffle null is
too strong for phase / PAC / complexity methods (it preserves the very
spectral structure those methods rely on), and time-shift / sample-shuffle
nulls are not appropriate for amplitude / information methods. The plain
column-normalized raw value gives an honest, surrogate-free view.

Signal types (rows)
-------------------
1. Independent             two independent 1/f-noise signals
2. Linear                  y = 0.95*x + noise
3. Phase-locked (90 deg)   shared 0.20 Hz phase, 90 deg offset, indep 1/f
4. Cross-frequency (PAC)   slow phase of x modulates fast amplitude of y
5. Nonlinear (info)        y = |x|  (deterministic, Pearson r approx 0)
6. Complexity-matched      shared white -> pink scaling change at midpoint;
                           otherwise independent

Methods (columns) — colour-coded by family
------------------------------------------
linear:        Pearson |r| (windowed lag-zero), windowed xcorr peak |r|
oscillatory:   coherence band-avg, PLV, wPLI, PAC (Canolty MVL)
information:   MI, |Granger score|
complexity:    fluctuation matching r, complexity_coupling

Output
------
figures/report/Methods2_sensitivity_matrix.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.dsp import bandpass
from HNA.coupling import (
    windowed_xcorr,
    band_coherence,
    plv_phase_sync,
    wpli_phase_sync,
    canolty_mvl,
    granger_score,
    fluctuation_matching,
    complexity_coupling,
)
from HNA.viz import use_paper_style, save_figure


FAMILY_COLORS = {
    "linear":      "#3B7DD8",
    "oscillatory": "#5DA399",
    "information": "#7B5BA6",
    "complexity":  "#E08E1A",
}


# --------------------------------------------------------------------- #
# Signal generators (same recipes as Fig 3)
# --------------------------------------------------------------------- #
def _pink(n, rng, fs, alpha=1.0):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = np.zeros_like(f)
    A[f > 0] = 1.0 / np.power(f[f > 0], alpha / 2.0)
    P = rng.uniform(0, 2 * np.pi, A.size)
    s = np.fft.irfft(A * np.exp(1j * P), n=n)
    return (s - s.mean()) / (s.std() + 1e-12)


def gen_independent(fs, n, rng):
    return _pink(n, rng, fs), _pink(n, rng, fs)


def gen_linear(fs, n, rng):
    x = _pink(n, rng, fs)
    y = 0.95 * x + 0.30 * rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    return x, y


def gen_phase(fs, n, rng):
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.20 * t) + 0.15 * _pink(n, rng, fs)
    y = np.sin(2 * np.pi * 0.20 * t + np.pi / 2) + 0.15 * _pink(n, rng, fs)
    x = bandpass(x, fs=fs, lo=0.05, hi=1.0, order=8)
    y = bandpass(y, fs=fs, lo=0.05, hi=1.0, order=8)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


def gen_pac(fs, n, rng):
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 0.20 * t) + 0.10 * rng.standard_normal(n)
    modulator = (1 + 0.99 * np.cos(2 * np.pi * 0.20 * t)) / 2
    y = modulator * np.sin(2 * np.pi * 15.0 * t) + 0.02 * rng.standard_normal(n)
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y


def gen_nonlinear(fs, n, rng):
    x = _pink(n, rng, fs)
    y = np.abs(x) + 0.05 * rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    return x, y


def gen_complexity_match(fs, n, rng):
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


SIGNAL_GENS = [
    ("Independent",          gen_independent),
    ("Linear",               gen_linear),
    ("Phase-locked (90 deg)",gen_phase),
    ("Cross-frequency (PAC)",gen_pac),
    ("Nonlinear (info)",     gen_nonlinear),
    ("Complexity-matched",   gen_complexity_match),
]


# --------------------------------------------------------------------- #
# Method runners (raw scalar value per pair)
# --------------------------------------------------------------------- #
def m_pearson_lagzero(x, y, fs):
    """Lag-zero Pearson |r| in 10 s windows, mean across windows."""
    win = int(10.0 * fs); step = int(2.0 * fs)
    rs = []
    for s in range(0, len(x) - win + 1, step):
        a = x[s:s + win]; b = y[s:s + win]
        if a.std() > 0 and b.std() > 0:
            rs.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.mean(rs)) if rs else 0.0


def m_xcorr_lag(x, y, fs):
    """Windowed cross-correlation peak |r| over a +/-2 s lag search."""
    xc = windowed_xcorr(x, y, fs=fs, win_sec=10.0, step_sec=2.0,
                        max_lag_sec=2.0)
    return float(np.nanmean(np.abs(xc.peak_r)))


def m_coh(x, y, fs):
    nper = int(min(len(x) // 4, fs * 8))
    return float(band_coherence(x, y, fs=fs, fmin=0.05, fmax=0.5,
                                 windowed=False, nperseg=nper).band_avg_coh)


def m_plv(x, y, fs):
    return float(plv_phase_sync(x, y, fs=fs, bw_hz=0.10).plv)


def m_wpli(x, y, fs):
    return float(wpli_phase_sync(x, y, fs=fs, bw_hz=0.10).wpli)


def m_pac(x, y, fs):
    """Canolty MVL (raw) — amplitude-aware, suppresses leakage false positives."""
    return float(canolty_mvl(x, y, fs=fs,
                              low_band=(0.10, 0.40),
                              high_band=(10.0, 30.0)).mvl)


def m_mi(x, y, fs):
    step = max(1, int(fs / 4))
    return float(mutual_info_regression(
        x[::step].reshape(-1, 1), y[::step], n_neighbors=3,
        random_state=42)[0])


def m_granger(x, y, fs):
    return abs(granger_score(x, y, max_lag=10))


def m_fluct(x, y, fs):
    return float(fluctuation_matching(x, y)["r"])


def m_complexity(x, y, fs):
    res = complexity_coupling(x, y, fs=fs, win_sec=20.0, step_sec=4.0,
                               method="pearson")
    val = res.get("coupling", float("nan"))
    return abs(val) if np.isfinite(val) else 0.0


METHODS = [
    ("Pearson |r|",      "linear",      m_pearson_lagzero),
    ("xcorr peak |r|",   "linear",      m_xcorr_lag),
    ("coherence",        "oscillatory", m_coh),
    ("PLV",              "oscillatory", m_plv),
    ("wPLI",             "oscillatory", m_wpli),
    ("PAC (Canolty)",    "oscillatory", m_pac),
    ("MI",               "information", m_mi),
    ("|Granger score|",  "information", m_granger),
    ("complexity coup.", "complexity",  m_complexity),
]
# Note: ``fluctuation_matching r`` is intentionally omitted from this matrix
# because the log F(s) curves of any pair of broadband signals are strongly
# correlated, making the metric saturated and non-discriminative across
# rows (it is more useful as a within-pair detail metric, see Fig 1 panel D).


# --------------------------------------------------------------------- #
# Build matrix
# --------------------------------------------------------------------- #
def build_matrix(fs: float, dur_s: float, rng_seed: int = 0):
    n = int(dur_s * fs)
    V = np.full((len(SIGNAL_GENS), len(METHODS)), np.nan)
    for i, (sig_name, gen) in enumerate(SIGNAL_GENS):
        rng = np.random.default_rng(rng_seed + 13 * i)
        x, y = gen(fs, n, rng)
        for j, (m_name, _, fn) in enumerate(METHODS):
            try:
                V[i, j] = fn(x, y, fs)
            except Exception as e:  # noqa: BLE001
                print(f"  {sig_name:<24s} | {m_name:<18s}  FAILED: {e}")
                V[i, j] = np.nan
        line = "  ".join(f"{v:6.2f}" if np.isfinite(v) else "  nan"
                          for v in V[i, :])
        print(f"  {sig_name:<24s} | {line}")
    return V


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #
def plot_matrix(V: np.ndarray, output_path: Path):
    use_paper_style()

    n_rows, n_cols = V.shape

    # Column-wise normalization: each method's column is divided by its own
    # max absolute value across the rows. Independence row stays near 0;
    # the diagonal-ish pattern lights up.
    col_max = np.nanmax(np.abs(V), axis=0)
    col_max = np.where(col_max > 0, col_max, 1.0)
    Vn = np.abs(V) / col_max[None, :]

    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    # Single-hue colormap that emphasises peak detection.
    im = ax.imshow(Vn, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # Axes
    method_labels = [m[0] for m in METHODS]
    method_families = [m[1] for m in METHODS]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(method_labels, rotation=35, ha="right", fontsize=9.5)
    for tick, fam in zip(ax.get_xticklabels(), method_families):
        tick.set_color(FAMILY_COLORS[fam])

    signal_labels = [s[0] for s in SIGNAL_GENS]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(signal_labels, fontsize=10)

    # Annotate every cell with its raw value (not the normalized one).
    for i in range(n_rows):
        for j in range(n_cols):
            v_raw = V[i, j]; v_norm = Vn[i, j]
            if not np.isfinite(v_raw):
                continue
            label = f"{v_raw:.2f}"
            ax.text(j, i, label,
                    ha="center", va="center",
                    fontsize=8.5,
                    color="white" if v_norm > 0.55 else "#222",
                    fontweight="bold" if v_norm > 0.7 else "normal")

    ax.set_xlabel("Coupling method  (color = family)")
    ax.set_ylabel("Synthetic signal-pair type")
    ax.set_title(
        "Method x signal-type sensitivity matrix\n"
        "raw metric value, column-normalized to each method's max  "
        "(annotation = actual raw value)",
        fontsize=12, fontweight="bold", loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("normalized response  (value / column max)", fontsize=10)

    handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=f)
               for f in ("linear", "oscillatory", "information", "complexity")]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=4,
              frameon=False, fontsize=9.5, title="Coupling family")

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close()
    print(f"\n  Saved: {output_path.name}.png (+ pdf)")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--dur-s", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures" / "report" / "Methods2_sensitivity_matrix")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    V = build_matrix(fs=args.fs, dur_s=args.dur_s, rng_seed=args.seed)
    plot_matrix(V, args.out)


if __name__ == "__main__":
    main()
