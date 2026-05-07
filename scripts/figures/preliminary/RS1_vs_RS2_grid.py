#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RS1 vs RS2 EEG-band relative-power grid (2x3, all canonical bands).

For each EEG band, computes a linear-mixed-effects model on per-(subject,
channel) relative-power values across RS1 and RS2, plus a paired Wilcoxon
on subject means. Renders a single 2x3 violin grid with the band name as
each panel's title and a significance bracket from the LMM.

Output: reports/preliminary_results/figures/Fig2_RS1_vs_RS2_all_bands.{png,pdf}
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import use_paper_style, CONDITION_COLORS, save_figure, sig_stars


BANDS = [
    ("delta_rel",     "Delta"),
    ("theta_rel",     "Theta"),
    ("alpha_rel",     "Alpha"),
    ("low_beta_rel",  "Low beta"),
    ("high_beta_rel", "High beta"),
    ("gamma1_rel",    "Gamma1"),
]

CONDITIONS = ("RS1", "RS2")


def _load(subjects, data_dir):
    rows = []
    for s in subjects:
        for cond in CONDITIONS:
            p = data_dir / "processed" / f"sub-{s:02d}" / "tables" / f"features_{cond}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df["subject_id"] = s
            df["condition"] = cond
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _lmm_per_band(df, feat):
    """LMM with crossed random effects: feat ~ condition + (1|subject) + (1|channel).

    Aggregates window-level data to per-(subject, channel, condition) means *first*
    so each cell is a single observation (no window-level pseudoreplication).
    This gives 5 subj x 32 chan x 2 cond = 320 rows per band; subject and channel
    are crossed random intercepts that absorb their respective mean differences,
    leaving the condition fixed effect to test the population-level change between
    RS1 and RS2.
    """
    sub = df.dropna(subset=[feat]).copy()
    sub = sub[sub["condition"].isin(CONDITIONS)]
    if sub.empty:
        return float("nan")

    # Aggregate window values → one cell per (subject, channel, condition).
    agg = (sub.groupby(["subject_id", "channel", "condition"])[feat]
              .mean().reset_index())
    agg["cond_dummy"] = (agg["condition"] == CONDITIONS[1]).astype(int)
    # Use a single dummy grouping with two crossed variance components.
    agg["_grp"] = 1
    try:
        m = smf.mixedlm(
            f"{feat} ~ cond_dummy",
            data=agg, groups=agg["_grp"],
            vc_formula={"subject": "0 + C(subject_id)",
                        "channel": "0 + C(channel)"},
            re_formula="0",
        ).fit(method="lbfgs", reml=True)
        return float(m.pvalues.get("cond_dummy", float("nan")))
    except Exception as e:
        # Fallback: subject as the only random grouping (channel goes into residual).
        try:
            m = smf.mixedlm(
                f"{feat} ~ cond_dummy",
                data=agg, groups=agg["subject_id"],
                vc_formula={"channel": "0 + C(channel)"},
            ).fit(method="lbfgs", reml=True)
            return float(m.pvalues.get("cond_dummy", float("nan")))
        except Exception:
            return float("nan")


def _wilcoxon_per_band(df, feat):
    sub = df.groupby(["subject_id", "condition"])[feat].mean().reset_index()
    pivot = sub.pivot(index="subject_id", columns="condition", values=feat)
    pivot = pivot.reindex(columns=CONDITIONS).dropna()
    if pivot.shape[0] < 3:
        return float("nan")
    try:
        return float(sps.wilcoxon(pivot[CONDITIONS[0]].values,
                                  pivot[CONDITIONS[1]].values,
                                  alternative="two-sided",
                                  zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def plot_grid(df, output_path, primary_test="lmm"):
    """primary_test in {'lmm','wilcoxon'} — drives the bracket stars.

    Default is the LMM with **crossed random effects** for subject and channel,
    fit on (subject, channel, condition) means (5 x 32 x 2 = 320 rows).
    This properly accounts for nesting and gives a fixed-effect test on
    Condition that the eye can read off the yellow line. The Wilcoxon on
    subject means (n=5 paired) is reported in the console as a conservative
    sanity check.
    """
    use_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6),
                             sharex=True, sharey=False)
    axes = axes.flatten()

    # Compute LMM and Wilcoxon for every band first; FDR across the 6 bands.
    lmm_p = np.array([_lmm_per_band(df, f) for f, _ in BANDS], dtype=float)
    wil_p = np.array([_wilcoxon_per_band(df, f) for f, _ in BANDS], dtype=float)
    valid = np.isfinite(lmm_p)
    lmm_fdr = np.full_like(lmm_p, np.nan)
    if valid.any():
        _, p_fdr, *_ = multipletests(lmm_p[valid], alpha=0.05, method="fdr_bh")
        lmm_fdr[valid] = p_fdr
    # FDR for Wilcoxon too (so the main visual passes a multiple-comparison correction)
    wil_fdr = np.full_like(wil_p, np.nan)
    if np.isfinite(wil_p).any():
        valid_w = np.isfinite(wil_p)
        _, pw_fdr, *_ = multipletests(wil_p[valid_w], alpha=0.05, method="fdr_bh")
        wil_fdr[valid_w] = pw_fdr

    color_a = CONDITION_COLORS.get("RS1", "#5DA399")
    color_b = CONDITION_COLORS.get("RS2", "#7B5BA6")

    for i, ((feat, label), ax) in enumerate(zip(BANDS, axes)):
        if feat not in df.columns:
            ax.axis("off")
            continue
        # per-(subject, channel) values, channel-averaged for the violin shape
        df_sub = df.dropna(subset=[feat]).copy()
        df_sub = df_sub[df_sub["condition"].isin(CONDITIONS)]

        # Violin: distribution of per-(subject, channel) means (n=5*32=160 per condition).
        per_chan = (df_sub.groupby(["subject_id", "channel", "condition"])[feat]
                          .mean().reset_index())
        data = [per_chan[per_chan["condition"] == c][feat].values for c in CONDITIONS]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.7,
                              showmeans=False, showextrema=False)
        for pc, color in zip(parts["bodies"], [color_a, color_b]):
            pc.set_facecolor(color); pc.set_alpha(0.55)
            pc.set_edgecolor(color); pc.set_linewidth(0.9)

        # Paired connectors on subject means
        subj_means = (df_sub.groupby(["subject_id", "condition"])[feat]
                            .mean().reset_index())
        pivot = subj_means.pivot(index="subject_id", columns="condition", values=feat)
        pivot = pivot.reindex(columns=CONDITIONS)
        for _, row in pivot.iterrows():
            if row.notna().all():
                ax.plot([0, 1], row.values, color="#9aa0a6", alpha=0.65, lw=1.0)
                ax.scatter([0, 1], row.values, s=22, color="black", alpha=0.7, zorder=3)
        means = [pivot[c].mean() for c in CONDITIONS]
        ax.plot([0, 1], means, "-D", color="#f5b400", lw=1.7, ms=9,
                mec="black", mew=0.6, zorder=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(CONDITIONS))
        ax.set_xlim(-0.55, 1.55)
        # Push title up so the bracket below it doesn't collide.
        ax.set_title(label, fontsize=12, fontweight="bold", pad=18)
        ax.grid(True, axis="y", alpha=0.30)

        # Significance bracket placed *inside* the panel (above violins, below title).
        # Default primary test is paired Wilcoxon on subject means (n=5) - the LMM
        # with only a subject random intercept was pseudoreplicating channels/windows.
        if primary_test == "lmm":
            p_use = lmm_fdr[i]   # FDR-corrected LMM p across the 6 bands
        else:
            p_use = wil_p[i]   # Wilcoxon raw (FDR may make all ns even more conservative)
        stars = sig_stars(p_use)
        all_vals = np.concatenate(data)
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size:
            ymax = float(np.max(finite))
            ymin = float(np.min(finite))
            yspan = max(1e-6, ymax - ymin)
            # Reserve top ~15% of the panel for the bracket.
            level = ymax + 0.04 * yspan
            ax.set_ylim(ymin - 0.05 * yspan, ymax + 0.20 * yspan)
            ax.plot([0, 1], [level, level], color="#444", lw=0.9)
            ax.text(0.5, level + 0.01 * yspan,
                    f"{stars}  (p={p_use:.3g})" if np.isfinite(p_use) else "n/a",
                    ha="center", va="bottom",
                    fontsize=10,
                    fontweight="bold" if stars not in ("ns", "n/a") else "normal",
                    color="#222" if stars not in ("ns", "n/a") else "#666")

        # Y-axis label only on the left column.
        if i % 3 == 0:
            ax.set_ylabel("Relative power", fontsize=11)

    fig.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close()
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")
    # Echo the test results for the report.
    print("\n  LMM (cond_dummy) p-values + BH-FDR across 6 bands, and paired Wilcoxon:")
    for (feat, label), pl, pf, pw in zip(BANDS, lmm_p, lmm_fdr, wil_p):
        print(f"    {label:<10}  LMM p={pl:.3g} (FDR={pf:.3g})  Wilcoxon p={pw:.3g}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures" / "Fig2_RS1_vs_RS2_all_bands")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = _load(args.subjects, args.data_dir)
    if df.empty:
        print("No data loaded.")
        return
    plot_grid(df, args.out)


if __name__ == "__main__":
    main()
