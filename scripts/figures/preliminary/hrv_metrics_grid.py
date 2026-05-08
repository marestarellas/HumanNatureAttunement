#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 5 for HRV - PLV + wPLI grid for HRV <-> audio coupling.

Reads ``hrv_audio_coupling_<COND>_<HRV_FEATURE>.json`` files produced by
``scripts/analysis/run_hrv_audio_coupling.py``.

Layout: rows = HRV features (HF / MeanNN / RMSSD), cols = metrics (PLV, wPLI).
Each panel is a violin per condition with paired connectors and a Friedman
+ pairwise-Wilcoxon stat overlay (trend marker for p<0.10).

Output: reports/preliminary_results/figures/hrv_audio_metrics_grid.{png,pdf}

Usage:
    python scripts/figures/hrv_metrics_grid.py \\
        --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI \\
        --hrv-features HRV_HF HRV_MeanNN HRV_RMSSD
"""
from __future__ import annotations
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import use_paper_style, CONDITION_COLORS, save_figure, sig_stars


METRIC_LABELS = {"plv": "PLV", "wpli": "wPLI", "coh_band_avg": "Band-avg coherence"}
HRV_LABEL = {
    "HRV_HF":     "HRV-HF",
    "HRV_MeanNN": "HRV-MeanNN",
    "HRV_RMSSD":  "HRV-RMSSD",
    "HRV_SDNN":   "HRV-SDNN",
    "HRV_LnHF":   "HRV-LnHF",
}


def _scalar(d, key):
    v = d.get(key)
    if isinstance(v, dict):
        return float(v.get(key, np.nan))
    return float(v) if v is not None else np.nan


def load_coupling(subjects, conditions, hrv_features, data_dir):
    """Read coupling values written by ``compute_hrv_audio_coupling.py``.

    The new schema (after the instantaneous-HR refactor) splits PLV /
    wPLI / coherence (oscillatory metrics, one set per condition,
    computed on instantaneous HR @ 4 Hz) into a separate file
    ``hrv_audio_coupling_<COND>_hr_instantaneous.json``, leaving the
    per-feature ``hrv_audio_coupling_<COND>_<HRV_FEAT>.json`` files to
    carry only the slow-trend metrics (xcorr). For each (subject,
    condition, hrv_feature) row we attach the same per-condition PLV /
    wPLI / coherence triplet --- it does not depend on which HRV
    feature is under test.
    """
    rows = []
    for s in subjects:
        for c in conditions:
            tables_dir = (data_dir / "processed" /
                           f"sub-{s:02d}" / "tables")
            # Per-condition oscillatory payload (shared across features)
            osc_path = tables_dir / f"hrv_audio_coupling_{c}_hr_instantaneous.json"
            osc_plv = osc_wpli = osc_coh = float("nan")
            if osc_path.exists():
                try:
                    with open(osc_path) as f:
                        osc = json.load(f)
                    p_block = osc.get("plv") or {}
                    w_block = osc.get("wpli") or {}
                    c_block = osc.get("coherence") or {}
                    osc_plv = float(p_block.get("plv", float("nan")))
                    osc_wpli = float(w_block.get("wpli", float("nan")))
                    osc_coh = float(c_block.get("band_avg_coh", float("nan")))
                except Exception:  # noqa: BLE001
                    pass

            for feat in hrv_features:
                p = tables_dir / f"hrv_audio_coupling_{c}_{feat}.json"
                if not p.exists():
                    continue
                # The per-feature JSON carries xcorr; PLV / wPLI / coh
                # have moved to the per-condition oscillatory file. Older
                # JSONs (pre-refactor) still keep all three fields, so
                # the loader prefers the per-condition file when
                # available and falls back to the per-feature file
                # otherwise (so this loader stays back-compatible).
                with open(p) as f:
                    d = json.load(f)
                fb_coh = d.get("coherence")
                fb_coh_avg = (float(fb_coh.get("band_avg_coh", np.nan))
                               if isinstance(fb_coh, dict) else float("nan"))
                fb_plv = _scalar(d, "plv")
                fb_wpli = _scalar(d, "wpli")
                rows.append({
                    "subject_id": s, "condition": c, "hrv_feature": feat,
                    "plv": osc_plv if np.isfinite(osc_plv) else fb_plv,
                    "wpli": osc_wpli if np.isfinite(osc_wpli) else fb_wpli,
                    "coh_band_avg": (osc_coh if np.isfinite(osc_coh)
                                      else fb_coh_avg),
                })
    return pd.DataFrame(rows)


def friedman_with_posthoc(df, conditions, metric, hrv_feature):
    sub = df[df["hrv_feature"] == hrv_feature].copy()
    pivot = sub.pivot(index="subject_id", columns="condition", values=metric)
    pivot = pivot.reindex(columns=conditions).dropna()
    out = {"friedman_p": np.nan, "posthoc": []}
    if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
        try:
            chi2, p = sps.friedmanchisquare(*[pivot[c].values for c in conditions])
            out["friedman_p"] = float(p)
        except Exception:
            pass
        for a, b in combinations(conditions, 2):
            try:
                res = sps.wilcoxon(pivot[a].values, pivot[b].values,
                                   alternative="two-sided", zero_method="wilcox",
                                   nan_policy="omit")
                out["posthoc"].append({"a": a, "b": b, "p": float(res.pvalue)})
            except Exception:
                out["posthoc"].append({"a": a, "b": b, "p": float("nan")})
    return out


def plot_grid(df, conditions, hrv_features, output_path,
              metrics=("plv", "wpli", "coh_band_avg")):
    """Plot a 1 x n_metrics violin grid for the oscillatory metrics.

    After the instantaneous-HR refactor, PLV / wPLI / coherence are
    computed once per condition on instantaneous HR @ 4 Hz and do
    **not** depend on which HRV feature is under test. The figure is
    therefore collapsed to a single row; the ``hrv_features`` argument
    is kept for API back-compat but only its first element is used to
    label the row (the slow-trend row is no longer produced here ---
    that's xcorr's domain, in the nature-vs-rest figure).
    """
    use_paper_style()
    metrics = list(metrics)
    feat = hrv_features[0] if hrv_features else "HRV_MeanNN"
    n_rows = 1
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols + 0.6, 3.2),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for r in range(1):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            sub = df[df["hrv_feature"] == feat].dropna(subset=[metric])
            data = [sub[sub["condition"] == cnd][metric].values for cnd in conditions]

            parts = ax.violinplot(data, positions=range(len(conditions)),
                                  widths=0.7, showmeans=False, showextrema=False)
            for pc, cnd in zip(parts["bodies"], conditions):
                pc.set_facecolor(CONDITION_COLORS.get(cnd, "#666"))
                pc.set_alpha(0.55)
                pc.set_edgecolor(CONDITION_COLORS.get(cnd, "#666"))
                pc.set_linewidth(0.9)

            # Paired connectors + subject points
            pivot = (sub.pivot(index="subject_id", columns="condition", values=metric)
                       .reindex(columns=conditions))
            for _, row in pivot.iterrows():
                if row.notna().all():
                    ax.plot(range(len(conditions)), row.values,
                            color="#9aa0a6", alpha=0.55, lw=0.9)
            rng = np.random.default_rng(0)
            for k, cnd in enumerate(conditions):
                vals = sub[sub["condition"] == cnd][metric].values
                xj = rng.normal(k, 0.045, len(vals))
                ax.scatter(xj, vals, s=22, color="black", alpha=0.7, zorder=6)

            # Mean line
            means = [sub[sub["condition"] == cnd][metric].mean() for cnd in conditions]
            ax.plot(range(len(conditions)), means, "-D", color="#f5b400",
                    lw=1.7, ms=8, mec="black", mew=0.6, zorder=10)

            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions)
            ax.grid(True, axis="y", alpha=0.30)
            ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel("instantaneous HR @ 4 Hz",
                              fontsize=11.5, fontweight="bold")

            # Stats: Friedman + pairwise Wilcoxon brackets
            stats = friedman_with_posthoc(df, conditions, metric, feat)
            fp = stats["friedman_p"]
            if np.isfinite(fp):
                ax.text(0.97, 0.04, f"Friedman p={fp:.3g} {sig_stars(fp)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8.5, color="#444",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.7, pad=1.2))
            # All 3 pairwise brackets, stacked vertically (matches Fig 5 RESP).
            ymax = float(np.nanmax([np.nanmax(d) if len(d) else np.nan for d in data]))
            ymin = float(np.nanmin([np.nanmin(d) if len(d) else np.nan for d in data]))
            yspan = max(1e-6, ymax - ymin)
            level = ymax + 0.06 * yspan
            for entry in stats["posthoc"]:
                p_val = entry["p"]
                if not np.isfinite(p_val):
                    continue
                a, b = entry["a"], entry["b"]
                if a not in conditions or b not in conditions:
                    continue
                ia = conditions.index(a); ib = conditions.index(b)
                stars = sig_stars(p_val)
                ax.plot([ia, ib], [level, level], color="#444", lw=0.8)
                ax.text((ia + ib) / 2.0, level, stars,
                        ha="center", va="bottom",
                        fontsize=11 if stars not in ("ns", "n/a") else 9,
                        fontweight="bold" if stars not in ("ns", "n/a") else "normal",
                        color="#222" if stars not in ("ns", "n/a") else "#888")
                level += 0.10 * yspan

    # Title removed (in caption).
    fig.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close()
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--conditions", nargs="+", default=["VIZ", "AUD", "MULTI"])
    p.add_argument("--hrv-features", nargs="+",
                   default=["HRV_HF", "HRV_MeanNN", "HRV_RMSSD"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures" / "hrv_audio_metrics_grid")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = load_coupling(args.subjects, args.conditions, args.hrv_features, args.data_dir)
    if df.empty:
        print("No HRV coupling JSONs found.")
        return
    plot_grid(df, args.conditions, args.hrv_features, args.out)


if __name__ == "__main__":
    main()
