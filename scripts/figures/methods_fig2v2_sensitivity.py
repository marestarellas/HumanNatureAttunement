#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 2 v2 - Sensitivity matrix linked to the features x coupling framework.

Same content as Methods 2 v1 (raw metric value, column-normalised per
method) but the columns are explicitly **grouped by feature axis**
(raw / oscillatory / complexity) and **coloured by coupling axis**
(linear / oscillatory / information). Each column thus reads as a
specific (feature, coupling) cell of the framework matrix in
Figure 5 (methods_fig5_framework).

Layout
------
- Row groups: 6 synthetic signal-pair types
- Column groups: 3 feature groups separated by gaps, each containing the
  methods whose feature transformation lives in that row of Figure 5
- Method-name colour: the coupling-method axis (3 colours)
- Two-line column header: method name (top) + feature tag (italic, below)
- Bracket header above the column labels showing the feature grouping

Output
------
figures/report/Methods2v2_sensitivity_matrix.{png,pdf}
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
    dcca_rho,
    cross_sample_entropy,
    transfer_entropy_binned,
    phi_id,
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


def m_te(x, y, fs):
    """Binned Schreiber transfer entropy, max of both directions (bits).

    Decorrelating downsample to ~2000 samples to keep pyinform fast and
    avoid penalising heavily oversampled traces.
    """
    cap = 2000
    if len(x) > cap:
        step = max(1, len(x) // cap)
        x = x[::step][:cap]
        y = y[::step][:cap]
    try:
        res = transfer_entropy_binned(x, y, history=1, n_bins=6)
    except Exception:  # noqa: BLE001
        return 0.0
    return float(max(res["te_x_to_y"], res["te_y_to_x"]))


def m_phi_id_synergy(x, y, fs):
    """Phi-ID synergy atom (sts): information in joint dynamics only."""
    cap = 2000
    if len(x) > cap:
        step = max(1, len(x) // cap)
        x = x[::step][:cap]
        y = y[::step][:cap]
    try:
        res = phi_id(x, y, tau=1, kind="gaussian", redundancy="MMI")
    except Exception:  # noqa: BLE001
        return 0.0
    val = res.get("synergy", 0.0)
    return float(val) if np.isfinite(val) else 0.0


def m_phi_id_transfer(x, y, fs):
    """Phi-ID transfer-to-Y atom group (rty + xty + sty), max of both dirs."""
    cap = 2000
    if len(x) > cap:
        step = max(1, len(x) // cap)
        x = x[::step][:cap]
        y = y[::step][:cap]
    try:
        res = phi_id(x, y, tau=1, kind="gaussian", redundancy="MMI")
    except Exception:  # noqa: BLE001
        return 0.0
    val = max(res.get("transfer_x_to_y", 0.0),
              res.get("transfer_y_to_x", 0.0))
    return float(val) if np.isfinite(val) else 0.0


def m_dcca_rho(x, y, fs):
    """Mean absolute DCCA correlation coefficient over scales."""
    res = dcca_rho(x, y)
    rho = res["rho"]
    valid = np.isfinite(rho)
    return float(np.nanmean(np.abs(rho[valid]))) if valid.any() else 0.0


def m_cross_sampen(x, y, fs):
    """Cross-sample entropy (lower => more regular joint dynamics).

    Cross-sample-entropy is O(N^2); cap input to 2000 samples so the
    sensitivity-matrix grid stays interactive. Returned as
    ``2.5 - cross_sampen`` so larger values consistently indicate
    stronger coupling, matching the rest of the matrix.
    """
    cap = 2000
    if len(x) > cap:
        # Decorrelating downsample to ~cap samples
        step = max(1, len(x) // cap)
        x = x[::step][:cap]
        y = y[::step][:cap]
    cse = cross_sample_entropy(x, y, m=2, r_factor=0.15)
    if not np.isfinite(cse):
        return 0.0
    return float(max(0.0, 2.5 - cse))


# Each method now carries TWO axes of the framework: the feature it
# operates on (or extracts internally) and the coupling-method family it
# implements. Methods are ordered for the v2 layout: grouped by feature,
# then by coupling within each feature group.
#   (label, feature, coupling, function)
METHODS = [
    # raw signal feature -------------------------------------------
    ("Pearson |r|",      "raw",         "linear",      m_pearson_lagzero),
    ("xcorr peak |r|",   "raw",         "linear",      m_xcorr_lag),
    ("MI",               "raw",         "information", m_mi),
    ("|Granger score|",  "raw",         "information", m_granger),
    ("TE (binned)",      "raw",         "information", m_te),
    ("$\\Phi$-ID transfer","raw",       "information", m_phi_id_transfer),
    ("$\\Phi$-ID synergy","raw",        "information", m_phi_id_synergy),
    ("DCCA-$\\rho$",     "raw",         "complexity",  m_dcca_rho),
    ("cross-sampen",     "raw",         "complexity",  m_cross_sampen),
    # oscillatory feature ------------------------------------------
    ("coherence",        "oscillatory", "oscillatory", m_coh),
    ("PLV",              "oscillatory", "oscillatory", m_plv),
    ("wPLI",             "oscillatory", "oscillatory", m_wpli),
    ("PAC (Canolty)",    "oscillatory", "oscillatory", m_pac),
    # complexity feature -------------------------------------------
    ("complexity coup.", "complexity",  "linear",      m_complexity),
]
# ``fluctuation_matching r`` is intentionally omitted (saturated across
# broadband signal pairs; see Fig 1 panel D for a within-pair view).
#
# Color coding follows the COUPLING axis (4 colours: linear / oscillatory /
# information / complexity). The FEATURE axis is encoded by column
# grouping + bracket headers, not by colour, to avoid ambiguity.

FEATURE_COLORS = {
    "raw":         "#2D3748",   # dark slate
    "oscillatory": "#5DA399",   # teal
    "complexity":  "#E08E1A",   # amber
}


# --------------------------------------------------------------------- #
# Build matrix
# --------------------------------------------------------------------- #
def build_matrix(fs: float, dur_s: float, rng_seed: int = 0):
    n = int(dur_s * fs)
    V = np.full((len(SIGNAL_GENS), len(METHODS)), np.nan)
    for i, (sig_name, gen) in enumerate(SIGNAL_GENS):
        rng = np.random.default_rng(rng_seed + 13 * i)
        x, y = gen(fs, n, rng)
        for j, (m_name, _, _, fn) in enumerate(METHODS):
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
def _feature_groups(methods):
    """Return [(feature_name, start_idx, end_idx_exclusive), ...] preserving order."""
    out = []
    cur = methods[0][1]
    start = 0
    for j, m in enumerate(methods[1:], start=1):
        if m[1] != cur:
            out.append((cur, start, j))
            cur = m[1]
            start = j
    out.append((cur, start, len(methods)))
    return out


def plot_matrix(V: np.ndarray, output_path: Path):
    use_paper_style()

    n_rows, n_cols = V.shape

    # Column-wise normalisation as in v1.
    col_max = np.nanmax(np.abs(V), axis=0)
    col_max = np.where(col_max > 0, col_max, 1.0)
    Vn = np.abs(V) / col_max[None, :]

    # More vertical room for the bracket header above the column labels.
    fig, ax = plt.subplots(figsize=(16.5, 6.6))
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.10, right=0.95)

    im = ax.imshow(Vn, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    # Two-line column headers: method name (top) + feature tag (bottom).
    method_labels = [m[0] for m in METHODS]
    method_features = [m[1] for m in METHODS]
    method_couplings = [m[2] for m in METHODS]
    two_line = [f"{lab}\n({feat})" for lab, feat in zip(method_labels, method_features)]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(two_line, rotation=35, ha="right", fontsize=9.3)
    # Colour the method-name line by its COUPLING family. The (feat) line
    # below is grey because feature is encoded by bracket grouping above.
    for tick, coup in zip(ax.get_xticklabels(), method_couplings):
        tick.set_color(FAMILY_COLORS[coup])

    signal_labels = [s[0] for s in SIGNAL_GENS]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(signal_labels, fontsize=10)

    # Annotate every cell with its raw value (not the normalised one).
    for i in range(n_rows):
        for j in range(n_cols):
            v_raw = V[i, j]; v_norm = Vn[i, j]
            if not np.isfinite(v_raw):
                continue
            ax.text(j, i, f"{v_raw:.2f}",
                    ha="center", va="center",
                    fontsize=8.5,
                    color="white" if v_norm > 0.55 else "#222",
                    fontweight="bold" if v_norm > 0.7 else "normal")

    # ---- Feature-group brackets above the column headers ---------- #
    groups = _feature_groups(METHODS)
    y_bracket = -0.85   # in data coords (above row 0)
    y_label   = -1.20
    for feat, j0, j1 in groups:
        # Horizontal bracket line
        ax.plot([j0 - 0.4, j1 - 0.6], [y_bracket, y_bracket],
                color=FEATURE_COLORS[feat], lw=2.2, clip_on=False)
        # Short tick down at each end
        for x in (j0 - 0.4, j1 - 0.6):
            ax.plot([x, x], [y_bracket, y_bracket + 0.10],
                    color=FEATURE_COLORS[feat], lw=2.2, clip_on=False)
        # Label
        ax.text((j0 - 0.4 + j1 - 0.6) / 2.0, y_label,
                f"{feat} feature", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold",
                color=FEATURE_COLORS[feat], clip_on=False)

    ax.set_ylabel("Synthetic signal-pair type")

    # Title at the top of the figure. The brackets + coloured column labels
    # below already communicate the "grouped by feature, coloured by
    # coupling" scheme, so no subtitle line is needed.
    fig.text(0.10, 0.965,
             "Method x signal-type sensitivity matrix  (feature-grouped)",
             fontsize=12.5, fontweight="bold", ha="left", va="top")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("normalised response  (value / column max)", fontsize=10)

    # Legend: COUPLING axis (4 colours, matches the framework's 3 x 4 columns).
    handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=f)
               for f in ("linear", "oscillatory", "information", "complexity")]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.40), ncol=4,
              frameon=False, fontsize=10.0,
              title="coupling-method family  (column colour)",
              title_fontsize=9.5)

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
                   default=ROOT / "figures" / "report" / "Methods2v2_sensitivity_matrix")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    V = build_matrix(fs=args.fs, dur_s=args.dur_s, rng_seed=args.seed)
    plot_matrix(V, args.out)


if __name__ == "__main__":
    main()
