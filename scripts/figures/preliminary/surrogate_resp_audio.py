#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surrogate test for respiration-audio oscillatory coupling.

Phase-shuffle null for coherence / PLV / wPLI between cleaned
respiration (256 Hz) and the audio swell envelope ``env_swell_0p2``
(256 Hz), per (subject, condition). Counterpart to
``surrogate_hrv_audio.py``.

Output:
    reports/preliminary_results/diagnostics/Fig_surrogate_resp_audio_oscillatory.{png,pdf}
    results/surrogate_resp_audio/surrogate_resp_audio.csv

Diagnostic; not in the report unless added after inspection.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import use_paper_style, save_figure
from HNA.utils import get_condition_segments
from HNA.modalities.respiration import clean_respiration
from HNA.surrogates import surrogate_test
from HNA.coupling import (
    plv_phase_sync, wpli_phase_sync, band_coherence_windowed,
)


FS = 256.0  # respiration and env_swell_0p2 both live on the merged CSV grid
COH_FMIN, COH_FMAX = 0.05, 0.5
PLV_BW = 0.12

CONDITIONS = ["RS1", "VIZ", "AUD", "MULTI", "RS2"]
COND_COLORS = {
    "RS1":   "#C9325F",
    "RS2":   "#C9325F",
    "VIZ":   "#3B7DD8",
    "AUD":   "#E08E1A",
    "MULTI": "#5DA399",
}

N_SURROGATES = 200


def _build_signals(df: pd.DataFrame, cond: str,
                    env_col: str = "env_swell_0p2"):
    """Return (resp, env) for one condition or (None, None)."""
    indices = get_condition_segments(df, df["condition_names"].unique())
    s = indices.get(f"{cond}_start"); e = indices.get(f"{cond}_stop")
    if s is None or e is None:
        return None, None
    s, e = int(s), int(e)
    r = df.iloc[s:e]
    if env_col not in r.columns:
        return None, None
    if "respiration_clean" in r.columns:
        resp = r["respiration_clean"].to_numpy(float)
    else:
        if "respiration" not in r.columns:
            return None, None
        resp = clean_respiration(r["respiration"].to_numpy(float), fs=FS)
    env = r[env_col].to_numpy(float)
    m = np.isfinite(resp) & np.isfinite(env)
    if (m.sum() < int(FS * 60)
            or float(np.std(resp[m])) < 1e-9
            or float(np.std(env[m])) < 1e-9):
        return None, None
    return resp[m], env[m]


def _coh_metric(x, y):
    res = band_coherence_windowed(x, y, fs=FS, fmin=COH_FMIN, fmax=COH_FMAX,
                                   win_sec=120.0, step_sec=10.0)
    return float(res["band_avg_coh"])


def _plv_metric(x, y):
    return float(plv_phase_sync(x, y, fs=FS, bw_hz=PLV_BW).plv)


def _wpli_metric(x, y):
    return float(wpli_phase_sync(x, y, fs=FS, bw_hz=PLV_BW).wpli)


METRICS = [
    ("coh_band_avg", "Band-avg coherence", _coh_metric),
    ("plv",          "PLV",                _plv_metric),
    ("wpli",         "wPLI",               _wpli_metric),
]


def _run_subject(subj: int, data_dir: Path, n_surrogates: int):
    rows = []
    sub = f"sub-{subj:02d}"
    sdir = data_dir / "processed" / sub
    merged = sdir / "tables" / "merged_annotated_with_audio.csv"
    if not merged.exists():
        print(f"  SKIP {sub}: no merged CSV")
        return rows
    df = pd.read_csv(merged, low_memory=False)

    for cond in CONDITIONS:
        resp, env = _build_signals(df, cond)
        if resp is None or env is None:
            print(f"    {cond}: skipped (missing signals)")
            continue
        for key, _, fn in METRICS:
            try:
                obs, null, p, z = surrogate_test(
                    fn, resp, env, n=n_surrogates,
                    method="phase_shuffle",
                    surrogate_target="y",
                    higher_is_better=True,
                    rng_seed=int(subj * 1000 + hash(key + cond) % 1000),
                )
            except Exception as e:  # noqa: BLE001
                print(f"    {sub}/{cond}/{key}: surrogate failed ({e})")
                continue
            rows.append({
                "subject_id": subj, "condition": cond, "metric": key,
                "observed": float(obs),
                "null_mean": float(np.nanmean(null)),
                "null_std": float(np.nanstd(null)),
                "null_p2_5": float(np.nanpercentile(null, 2.5)),
                "null_p97_5": float(np.nanpercentile(null, 97.5)),
                "p_one_sided": float(p),
                "z": float(z),
            })
        print(f"    {cond}: ok ({n_surrogates} surrogates / 3 metrics)")
    return rows


def _plot(df: pd.DataFrame, output_path: Path):
    use_paper_style()
    metrics = [m[0] for m in METRICS]
    metric_labels = {m[0]: m[1] for m in METRICS}

    fig, axes = plt.subplots(2, len(metrics),
                              figsize=(4.4 * len(metrics) + 0.6, 6.4),
                              sharex=True)
    cond_x = {c: i for i, c in enumerate(CONDITIONS)}

    for j, metric in enumerate(metrics):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue

        ax = axes[0, j]
        for _, row in sub.iterrows():
            x = cond_x[row["condition"]] + (
                np.random.default_rng(int(row["subject_id"] * 7 + j)).uniform(-0.18, 0.18)
            )
            ax.plot([x, x],
                     [row["null_p2_5"], row["null_p97_5"]],
                     color="#9aa0a6", lw=1.2, alpha=0.75, zorder=1)
            ax.scatter(x, row["observed"],
                        s=46, color=COND_COLORS.get(row["condition"], "#000"),
                        edgecolors="black", linewidths=0.6, zorder=4)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel("observed\n(grey bar = null 95% CI)",
                           fontsize=10.5)

        ax = axes[1, j]
        rng = np.random.default_rng(0)
        for _, row in sub.iterrows():
            x = cond_x[row["condition"]] + rng.uniform(-0.18, 0.18)
            ax.scatter(x, row["z"], s=46,
                        color=COND_COLORS.get(row["condition"], "#000"),
                        edgecolors="black", linewidths=0.6, zorder=4)
        ax.axhline(1.96, color="#444", ls="--", lw=0.9, zorder=1)
        ax.axhline(0.0, color="#444", ls=":", lw=0.6, alpha=0.6, zorder=1)
        ax.text(len(CONDITIONS) - 0.1, 1.96, "  z=1.96 (one-sided p<.05)",
                 va="bottom", ha="right", fontsize=8.5, color="#444")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel("z = (obs − null mean) / null std",
                           fontsize=10.5)

    fig.suptitle(
        "Respiration–audio oscillatory coupling vs. phase-shuffle null "
        f"(n_surr={N_SURROGATES})",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close(fig)
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+",
                   default=[2, 3, 4, 5, 6])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--n-surrogates", type=int, default=N_SURROGATES)
    p.add_argument("--out", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures"
                                / "Fig_surrogate_resp_audio_oscillatory")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "surrogate_resp_audio")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in args.subjects:
        print(f"\n[sub-{s:02d}]")
        rows.extend(_run_subject(s, args.data_dir, args.n_surrogates))
    if not rows:
        print("No rows -- nothing to plot.")
        return
    df = pd.DataFrame(rows)
    csv_path = args.results_dir / "surrogate_resp_audio.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")
    _plot(df, args.out)


if __name__ == "__main__":
    main()
