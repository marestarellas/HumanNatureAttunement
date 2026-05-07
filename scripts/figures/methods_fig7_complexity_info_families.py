#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods Figure 7 - The cross-complexity and information coupling families.

A panorama of the methods that fill the previously-empty cells of the
3x4 features-by-coupling framework: genuinely scale-aware bivariate
complexity estimators (DCCA, multiscale cross-entropy) and the modern
information-theoretic decompositions (TE, PID, Phi-ID).

Layout
------
Row 1 -- Cross-complexity coupling (each panel uses a synthetic pair
where the complexity structure is engineered, not just inherited)
    A. signals: shared slow + independent fast (scale-dependent coupling)
    B. DCCA correlation rho(s) -- coupling at each scale
    C. multiscale cross-entropy -- joint regularity vs coarse-grain factor

Row 2 -- Information coupling
    D. transfer entropy: binned TE (pyinform) and Phi-ID transfer atoms,
       in both directions, on a triplet of synthetic systems
       (independent / x-causes-y / bidirectional)
    E. PID atoms (Williams-Beer / BROJA) for the four canonical 3-variable
       systems (RDN / XOR / COPY / AND) -- the textbook PID reference
    F. Phi-ID 16-atom decomposition (heatmap, source x target lattice)
       for the x-causes-y bivariate system

Output
------
reports/methods/figures/Methods7_complexity_info_families.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from HNA.coupling import (
    dcca_rho, multiscale_cross_entropy,
    transfer_entropy_binned, pid_2source, phi_id,
    granger_score,
)
from HNA.viz import use_paper_style, save_figure


# Modern palette -- distinct from the four "coupling family" colours.
SIG_X_COLOR = "#34495E"        # slate
SIG_Y_COLOR = "#C0392B"        # brick
ACCENT_COMPLEXITY = "#E08E1A"  # amber (the row's family hint)
ACCENT_INFORMATION = "#7B5BA6" # purple
PID_COLORS = {
    "redundant":   "#5DA399",
    "unique_x1":   "#3B7DD8",
    "unique_x2":   "#C49A6C",
    "synergistic": "#C9325F",
}


# --------------------------------------------------------------------- #
# Synthetic systems
# --------------------------------------------------------------------- #
def _pink(n, rng, fs):
    f = np.fft.rfftfreq(n, 1 / fs)
    A = np.zeros_like(f)
    A[f > 0] = 1.0 / np.sqrt(f[f > 0])
    P = rng.uniform(0, 2 * np.pi, A.size)
    s = np.fft.irfft(A * np.exp(1j * P), n=n)
    return (s - s.mean()) / (s.std() + 1e-12)


def make_scale_dependent_pair(fs=50.0, dur_s=120.0, rng_seed=0):
    """Two signals that share a slow component but have INDEPENDENT fast.

    DCCA rho(s) should be near 1 at large s (slow scales) and near 0 at
    small s (fast scales) -- the canonical scale-dependent coupling
    test.
    """
    rng = np.random.default_rng(rng_seed)
    n = int(dur_s * fs)
    sos_lp = butter(4, [1.0 / (fs / 2)], btype="lowpass", output="sos")
    sos_hp = butter(4, [1.0 / (fs / 2)], btype="highpass", output="sos")
    slow_shared = sosfiltfilt(sos_lp, _pink(n, rng, fs))
    slow_shared = (slow_shared - slow_shared.mean()) / slow_shared.std()
    fast_x = sosfiltfilt(sos_hp, _pink(n, rng, fs))
    fast_y = sosfiltfilt(sos_hp, _pink(n, rng, fs))
    fast_x = (fast_x - fast_x.mean()) / fast_x.std()
    fast_y = (fast_y - fast_y.mean()) / fast_y.std()
    x = slow_shared + 0.7 * fast_x
    y = slow_shared + 0.7 * fast_y
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return x, y, fs


def make_directional_systems(n=4000, rng_seed=0):
    """Three bivariate systems: independent / x->y / bidirectional."""
    rng = np.random.default_rng(rng_seed)

    # Independent.
    xi = rng.standard_normal(n)
    yi = rng.standard_normal(n)

    # x -> y causal.
    xc = rng.standard_normal(n)
    yc = np.zeros(n)
    for t in range(1, n):
        yc[t] = 0.7 * xc[t - 1] + 0.3 * rng.standard_normal()
    yc = (yc - yc.mean()) / yc.std()

    # Bidirectional with shared driver pattern.
    xb = np.zeros(n); yb = np.zeros(n)
    xb[0], yb[0] = rng.standard_normal(), rng.standard_normal()
    for t in range(1, n):
        xb[t] = 0.5 * xb[t - 1] + 0.3 * yb[t - 1] + 0.3 * rng.standard_normal()
        yb[t] = 0.3 * xb[t - 1] + 0.5 * yb[t - 1] + 0.3 * rng.standard_normal()
    xb = (xb - xb.mean()) / xb.std(); yb = (yb - yb.mean()) / yb.std()

    return {
        "independent":   (xi, yi),
        "x to y":        (xc, yc),
        "bidirectional": (xb, yb),
    }


def make_pid_systems(n=4000, rng_seed=0):
    """Four canonical PID 3-variable systems (binary)."""
    rng = np.random.default_rng(rng_seed)
    x1 = rng.integers(0, 2, n).astype(float)
    x2 = rng.integers(0, 2, n).astype(float)
    return {
        "RDN  (Y=X1=X2)": (x1, x1.copy(), x1.copy()),
        "XOR":             (x1, x2, (x1.astype(int) ^ x2.astype(int)).astype(float)),
        "COPY (Y=X1)":     (x1, x2, x1.copy()),
        "AND":             (x1, x2, (x1.astype(int) & x2.astype(int)).astype(float)),
    }


# --------------------------------------------------------------------- #
# Panel painters
# --------------------------------------------------------------------- #
def _panel_signals_complexity(ax, x, y, fs, preview_s=20.0):
    n = min(int(preview_s * fs), len(x))
    t = np.arange(n) / fs
    ax.plot(t, x[:n], color=SIG_X_COLOR, lw=1.2, label="x")
    ax.plot(t, y[:n], color=SIG_Y_COLOR, lw=1.2, alpha=0.9, label="y")
    ax.set_xlim(0, preview_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (z)")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    ax.set_title("A.  Synthetic pair: shared slow + independent fast",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.text(0.01, -0.02,
            "constructed to have scale-dependent coupling",
            transform=ax.transAxes, fontsize=8.5, color="#666",
            ha="left", va="top", style="italic")
    ax.grid(True, alpha=0.30)


def _panel_dcca(ax, x, y, fs):
    """B. DCCA rho(s): coupling at each scale."""
    n = min(len(x), len(y))
    scales = np.unique(np.logspace(np.log10(8), np.log10(n // 4), 28).astype(int))
    res = dcca_rho(x, y, scales=scales)
    rho = res["rho"]; s = res["scales"]
    valid = np.isfinite(rho)

    ax.axhline(0, color="#bbb", lw=0.7)
    ax.semilogx(s[valid], rho[valid], "-o", ms=4,
                color=ACCENT_COMPLEXITY, lw=1.8)
    ax.fill_between(s[valid], 0, rho[valid],
                    where=(rho[valid] > 0), color=ACCENT_COMPLEXITY,
                    alpha=0.22, step="mid")
    ax.set_xlabel("Scale s (samples)")
    ax.set_ylabel(r"DCCA $\rho_{xy}(s)$")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title("B.  DCCA correlation - coupling per scale",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.grid(True, which="both", alpha=0.30)
    # Annotate fast vs slow regimes.
    s_mid = int(np.median(s[valid]))
    ax.axvline(s_mid, color="#cccccc", ls=":", lw=0.8)
    ax.text(0.04, 0.08, "fast scales\n(small s)", transform=ax.transAxes,
            fontsize=8.5, color="#666", ha="left", va="bottom", style="italic")
    ax.text(0.96, 0.08, "slow scales\n(large s)", transform=ax.transAxes,
            fontsize=8.5, color="#666", ha="right", va="bottom", style="italic")


def _panel_mse(ax, x, y, fs):
    """C. Multiscale cross-entropy: regularity of joint dynamics vs scale."""
    res = multiscale_cross_entropy(x, y, scales=(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20))
    s = res["scales"]; cse = res["cross_sampen"]
    valid = np.isfinite(cse)
    ax.plot(s[valid], cse[valid], "-s", ms=5, color=ACCENT_COMPLEXITY,
            lw=1.6, alpha=0.95)
    ax.set_xlabel(r"Coarse-graining factor $\tau$")
    ax.set_ylabel("cross sample entropy")
    ax.set_title("C.  Multiscale cross-entropy",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.30)


def _panel_te(ax, systems):
    """D. TE bars: binned TE + Phi-ID transfer atoms in both directions."""
    labels = list(systems.keys())
    te_x_to_y_bin = []; te_y_to_x_bin = []
    te_x_to_y_phi = []; te_y_to_x_phi = []
    granger_xy = []; granger_yx = []
    for k in labels:
        x, y = systems[k]
        # binned TE
        b = transfer_entropy_binned(x, y, history=1, n_bins=6)
        te_x_to_y_bin.append(b["te_x_to_y"]); te_y_to_x_bin.append(b["te_y_to_x"])
        # Phi-ID transfer atoms
        p = phi_id(x, y, tau=1)
        te_x_to_y_phi.append(p["transfer_x_to_y"]); te_y_to_x_phi.append(p["transfer_y_to_x"])
        # Granger F (signed; use abs for magnitude)
        try:
            from HNA.coupling import granger_bivariate
            r = granger_bivariate(x, y, max_lag=5)
            granger_xy.append(np.log10(max(r.x_to_y_F, 1.0)))
            granger_yx.append(np.log10(max(r.y_to_x_F, 1.0)))
        except Exception:
            granger_xy.append(0); granger_yx.append(0)

    n = len(labels)
    x_pos = np.arange(n)
    bar_w = 0.13
    offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * bar_w

    ax.bar(x_pos + offsets[0], te_x_to_y_bin, bar_w,
           color=ACCENT_INFORMATION, alpha=0.95, label=r"TE binned: $x{\to}y$")
    ax.bar(x_pos + offsets[1], te_y_to_x_bin, bar_w,
           color=ACCENT_INFORMATION, alpha=0.45, label=r"TE binned: $y{\to}x$")
    ax.bar(x_pos + offsets[2], te_x_to_y_phi, bar_w,
           color="#5DA399", alpha=0.95, label=r"$\Phi$-ID: $x{\to}y$")
    ax.bar(x_pos + offsets[3], te_y_to_x_phi, bar_w,
           color="#5DA399", alpha=0.45, label=r"$\Phi$-ID: $y{\to}x$")
    ax.bar(x_pos + offsets[4], granger_xy, bar_w,
           color="#3B7DD8", alpha=0.95, label=r"$\log_{10}$(Granger F): $x{\to}y$")
    ax.bar(x_pos + offsets[5], granger_yx, bar_w,
           color="#3B7DD8", alpha=0.45, label=r"$\log_{10}$(Granger F): $y{\to}x$")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("coupling magnitude")
    ax.set_title("D.  Directional information: TE / $\\Phi$-ID / Granger",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.axhline(0, color="#aaa", lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2,
              handlelength=1.4)
    ax.grid(True, axis="y", alpha=0.30)


def _panel_pid(ax, systems):
    """E. PID 4-atom decomposition for the 4 canonical systems."""
    labels = list(systems.keys())
    redund = []; uniq1 = []; uniq2 = []; syn = []; total = []
    for k in labels:
        x1, x2, y = systems[k]
        p = pid_2source(x1, x2, y, n_bins=2, measure="broja")
        redund.append(p["redundant"])
        uniq1.append(p["unique_x1"])
        uniq2.append(p["unique_x2"])
        syn.append(p["synergistic"])
        total.append(p["mi_joint"])

    n = len(labels)
    x_pos = np.arange(n)
    # Stacked bars showing decomposition; each system's column sums to mi_joint.
    bottom = np.zeros(n)
    for vals, name, color in [
        (redund, "redundant",   PID_COLORS["redundant"]),
        (uniq1,  "unique X1",   PID_COLORS["unique_x1"]),
        (uniq2,  "unique X2",   PID_COLORS["unique_x2"]),
        (syn,    "synergistic", PID_COLORS["synergistic"]),
    ]:
        ax.bar(x_pos, vals, 0.55, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.8, label=name)
        bottom = bottom + np.array(vals)

    # Annotate joint-MI total above each bar.
    for i, t in enumerate(total):
        ax.text(x_pos[i], bottom[i] + 0.03, f"I_joint = {t:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#444")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=10)
    ax.set_ylabel("partial information (bits)")
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.set_title("E.  PID atoms - canonical 3-variable systems",
                 fontsize=11.0, fontweight="bold", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2,
              handlelength=1.4)
    ax.grid(True, axis="y", alpha=0.25)


def _panel_phiid(ax, systems):
    """F. Phi-ID 16-atom heatmap on the bidirectional system.

    The bidirectional coupled system produces a much richer atom
    pattern than a pure x->y system (which only fires the
    unique-X -> unique-Y cell). Here we see directional transfer in
    BOTH directions, plus storage on the diagonal and synergistic
    information flow on the off-diagonal corner cells.
    """
    x, y = systems["bidirectional"]
    res = phi_id(x, y, tau=1, kind="gaussian", redundancy="MMI")
    atoms = res["atoms"]

    src_axis = ("r", "x", "y", "s")
    tgt_axis = ("r", "x", "y", "s")
    M = np.zeros((4, 4), dtype=float)
    for i, sa in enumerate(src_axis):
        for j, ta in enumerate(tgt_axis):
            M[i, j] = atoms.get(sa + "t" + ta, np.nan)

    vmax = max(0.05, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    labels = ["redundant", "unique-X", "unique-Y", "synergistic"]
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("target atom")
    ax.set_ylabel("source atom")

    for i in range(4):
        for j in range(4):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > 0.55 * vmax else "#222"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8.5, color=color,
                    fontweight="bold" if abs(v) > 0.5 * vmax else "normal")

    ax.set_title("F.  $\\Phi$-ID 16-atom decomposition\n"
                 "    (bidirectional system)",
                 fontsize=11.0, fontweight="bold", loc="left")

    # Outline conceptual aggregates: transfer x->y (target column 'unique-Y')
    # and the synergy atom (src=s, tgt=s).
    from matplotlib.patches import Rectangle
    # Transfer x->y aggregates: cells in target column j=2 (unique-Y) from
    # source rows that include X (rows 0, 1, 3 -> red, unq-X, syn).
    for i in (0, 1, 3):
        ax.add_patch(Rectangle((2 - 0.45, i - 0.45), 0.9, 0.9, fill=False,
                               edgecolor="#1F2A37", linewidth=1.2, linestyle=":"))
    # Synergy atom (source-syn -> target-syn) in solid frame.
    ax.add_patch(Rectangle((3 - 0.45, 3 - 0.45), 0.9, 0.9, fill=False,
                           edgecolor="#C9325F", linewidth=1.6))


# --------------------------------------------------------------------- #
# Main figure
# --------------------------------------------------------------------- #
def make_figure(out_path: Path):
    use_paper_style()

    fig = plt.figure(figsize=(15.0, 9.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 1.05],
        hspace=0.55, wspace=0.30,
        left=0.06, right=0.985, top=0.91, bottom=0.06,
    )

    # ---- Row 1: cross-complexity ----
    x, y, fs_complex = make_scale_dependent_pair(fs=50.0, dur_s=120.0, rng_seed=0)
    _panel_signals_complexity(fig.add_subplot(gs[0, 0]), x, y, fs_complex)
    _panel_dcca(fig.add_subplot(gs[0, 1]), x, y, fs_complex)
    _panel_mse(fig.add_subplot(gs[0, 2]), x, y, fs_complex)

    # ---- Row 2: information ----
    info_systems = make_directional_systems(n=4000, rng_seed=0)
    pid_systems = make_pid_systems(n=4000, rng_seed=1)
    _panel_te(fig.add_subplot(gs[1, 0]), info_systems)
    _panel_pid(fig.add_subplot(gs[1, 1]), pid_systems)
    _panel_phiid(fig.add_subplot(gs[1, 2]), info_systems)

    # ---- Title strip ----
    fig.text(0.06, 0.965,
             "Cross-complexity and information coupling families",
             fontsize=14.0, fontweight="bold", ha="left", va="top",
             color="#1F2A37")

    save_figure(fig, out_path)
    plt.close()
    print(f"  Saved: {out_path.name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "methods" / "figures" / "Methods7_complexity_info_families")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.out)


if __name__ == "__main__":
    main()
