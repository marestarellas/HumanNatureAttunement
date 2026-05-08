#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature vs Rest contrast for body-audio coupling.

Pools the 5 conditions into 2 blocks per subject:
  - REST   = RS1 + RS2
  - NATURE = VIZ + AUD + MULTI

Three complementary analyses (one figure per modality):

A) Paired Wilcoxon on per-subject means (n=5 paired)
B) Window-level mixed model:  metric ~ block + (1|subject)
   on every coupling-window value pooled across conditions
C) Per-subject Delta = mean(NATURE) - mean(REST), forest plot

Outputs:
  reports/preliminary_results/figures/nature_vs_rest_<modality>.{png,pdf}
  results/nature_vs_rest/nature_vs_rest_stats.csv

Usage:
    python scripts/figures/analysis_nature_vs_rest.py \\
        --subjects 2 3 4 5 6 --modalities resp hrv_meannn
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from HNA.viz import use_paper_style, save_figure, sig_stars
from HNA.utils import get_condition_segments
from HNA.coupling import (
    windowed_xcorr, band_coherence_windowed,
    plv_phase_sync, windowed_plv,
    wpli_phase_sync, windowed_wpli,
)
from HNA.modalities.ecg import instantaneous_hr_signal
from scipy.interpolate import interp1d


REST_SET = ("RS1", "RS2")
NATURE_SET = ("VIZ", "AUD", "MULTI")
BLOCK_COLORS = {"REST": "#C9325F", "NATURE": "#5DA399"}  # rest=red, nature=green

METRICS = [
    # (key, pretty label, (json_block, windowed_value_key, windowed_time_key))
    ("xcorr_peak_r", "XCorr peak |r|",      ("xcorr",     "peak_r",          "times_s")),
    ("plv",          "PLV",                 ("plv",       "win_plv",         "win_times_s")),
    ("wpli",         "wPLI",                ("wpli",      "win_wpli",        "win_times_s")),
    ("coh_band_avg", "Band-avg coherence",  ("coherence", "band_avg_coh_win", "times_s")),
]


# --------------------------- IO ---------------------------
def _resp_json(data_dir: Path, subj: int, cond: str) -> Path:
    return data_dir / "processed" / f"sub-{subj:02d}" / "tables" / f"coupling_{cond}.json"


def _hrv_json(data_dir: Path, subj: int, cond: str, hrv_feature: str) -> Path:
    return (data_dir / "processed" / f"sub-{subj:02d}" / "tables"
            / f"hrv_audio_coupling_{cond}_{hrv_feature}.json")


def _scalar_metric(d: dict, key: str) -> float:
    """Whole-segment metric value."""
    if key == "coh_band_avg":
        coh = d.get("coherence", {})
        return float(coh.get("band_avg_coh", np.nan)) if isinstance(coh, dict) else float("nan")
    if key == "xcorr_peak_r":
        xc = d.get("xcorr", {})
        return float(xc.get("mean_peak_r", np.nan)) if isinstance(xc, dict) else float("nan")
    v = d.get(key)
    if isinstance(v, dict):
        return float(v.get(key, np.nan))
    return float(v) if v is not None else float("nan")


def _windowed_metric(d: dict, metric_key: str, json_path: tuple) -> np.ndarray:
    """Return the windowed value list for this metric, as a 1-D array."""
    block, val_key, _ = json_path
    section = d.get(block, {})
    if not isinstance(section, dict):
        return np.array([])
    arr = section.get(val_key, [])
    return np.asarray(arr, float)


# --------------------------- on-the-fly HRV coupling ---------------------------
FS_AUDIO = 256.0
FS_HRV = 4.0


def _compute_hrv_for_condition(df_merged, cond, hrv_features_csv, env_col,
                               hrv_feature="HRV_MeanNN",
                               rpeaks_file: Path = None):
    """Compute scalar + windowed HRV-audio coupling on the fly.

    Two cardiac derivations, each used for the metrics it can support:

    - **Slow trend (xcorr)**: uses the windowed HRV-feature trace
      (``hrv_signal``, e.g.\\ HRV_MeanNN) interpolated onto a 4 Hz grid.
      Effective bandwidth ~0--0.017 Hz.
    - **Oscillatory (PLV / wPLI / coherence)**: uses the
      **instantaneous heart-rate trace** at 4 Hz, built from R-peak
      indices via 1/RR cubic interpolation (effective bandwidth
      ~0--2 Hz). Required because the windowed HRV-feature trace's
      bandwidth lies below the audio swell band, making narrowband-
      Hilbert phase analyses on it meaningless.

    Returns dict with keys ``scalar`` and ``windowed`` (each a dict of
    metric -> value/array).
    """
    indices = get_condition_segments(df_merged, df_merged["condition_names"].unique())
    starts = indices.get(f"{cond}_start"); stops = indices.get(f"{cond}_stop")
    if starts is None or stops is None:
        return None
    start_idx, stop_idx = int(starts), int(stops)
    r = df_merged.iloc[start_idx:stop_idx]
    if env_col not in r.columns:
        return None
    audio_time = r["time_s"].to_numpy(float)
    audio_time = audio_time - audio_time[0]   # condition-relative
    env_full = r[env_col].to_numpy(float)
    if not np.all(np.isfinite(env_full)) or len(env_full) < int(FS_AUDIO * 30):
        return None

    # ----- Build windowed HRV-feature trace at 4 Hz (slow-trend) -----
    hrv_df = pd.read_csv(hrv_features_csv)
    if hrv_feature not in hrv_df.columns:
        return None
    hrv_centers = (hrv_df["time_start"].values + hrv_df["time_end"].values) / 2.0
    hrv_vals = hrv_df[hrv_feature].values
    valid = np.isfinite(hrv_vals)
    if valid.sum() < 4:
        return None
    f_hrv = interp1d(hrv_centers[valid], hrv_vals[valid], kind="linear",
                     bounds_error=False, fill_value="extrapolate")
    hrv_grid = np.arange(hrv_centers[valid].min(), hrv_centers[valid].max(), 1.0 / FS_HRV)
    if len(hrv_grid) < int(FS_HRV * 60):
        return None
    hrv_signal = f_hrv(hrv_grid)
    f_aud = interp1d(audio_time, env_full, kind="linear",
                     bounds_error=False, fill_value=np.nan)
    env_on_hrv = f_aud(hrv_grid)
    m = np.isfinite(hrv_signal) & np.isfinite(env_on_hrv)
    if m.sum() < int(FS_HRV * 60):
        return None
    h_slow = hrv_signal[m]; e_slow = env_on_hrv[m]

    # ----- Build instantaneous-HR trace at 4 Hz (oscillatory) -----
    hr_inst = None
    env_at_4hz = None
    if rpeaks_file is not None and rpeaks_file.exists():
        rpeaks_seg = np.load(rpeaks_file)  # condition-relative indices
        if len(rpeaks_seg) >= 4:
            seg_duration_s = (stop_idx - start_idx) / FS_AUDIO
            n_target = int(round(seg_duration_s * FS_HRV))
            try:
                hr_inst = instantaneous_hr_signal(
                    rpeaks_seg, fs_in=FS_AUDIO, fs_target=FS_HRV,
                    n_samples=n_target,
                )
                t_target = np.arange(n_target) / FS_HRV
                env_at_4hz = np.interp(
                    t_target,
                    audio_time[np.isfinite(env_full)],
                    env_full[np.isfinite(env_full)],
                )
                m_osc = np.isfinite(hr_inst) & np.isfinite(env_at_4hz)
                if (m_osc.sum() < int(FS_HRV * 60)
                        or float(np.std(hr_inst[m_osc])) < 1e-9
                        or float(np.std(env_at_4hz[m_osc])) < 1e-9):
                    hr_inst = env_at_4hz = None
                else:
                    hr_inst = hr_inst[m_osc]
                    env_at_4hz = env_at_4hz[m_osc]
            except Exception:  # noqa: BLE001
                hr_inst = env_at_4hz = None

    # ----- Slow-trend metric on HRV-feature trace -----
    xc = windowed_xcorr(h_slow, e_slow, fs=FS_HRV,
                         win_sec=120.0, step_sec=10.0, max_lag_sec=30.0)

    # ----- Oscillatory metrics on instantaneous HR -----
    if hr_inst is not None and env_at_4hz is not None:
        coh = band_coherence_windowed(hr_inst, env_at_4hz, fs=FS_HRV,
                                       fmin=0.01, fmax=0.5,
                                       win_sec=120.0, step_sec=30.0)
        plv_w = windowed_plv(hr_inst, env_at_4hz, fs=FS_HRV,
                              win_sec=120.0, step_sec=10.0, bw_hz=0.10,
                              fmin_search=0.02, fmax_search=0.5)
        wpli_w = windowed_wpli(hr_inst, env_at_4hz, fs=FS_HRV,
                                win_sec=120.0, step_sec=10.0, bw_hz=0.10,
                                fmin_search=0.02, fmax_search=0.5)
        plv_g = plv_phase_sync(hr_inst, env_at_4hz, fs=FS_HRV,
                                bw_hz=0.10, fmin_search=0.02, fmax_search=0.5)
        wpli_g = wpli_phase_sync(hr_inst, env_at_4hz, fs=FS_HRV,
                                  bw_hz=0.10, fmin_search=0.02, fmax_search=0.5)
        coh_band_avg_scalar = float(coh["band_avg_coh"])
        coh_band_avg_win = np.asarray(coh["band_avg_coh_win"], float)
        plv_scalar = float(plv_g.plv)
        wpli_scalar = float(wpli_g.wpli)
        plv_win = np.asarray(plv_w["plv"], float)
        wpli_win = np.asarray(wpli_w["wpli"], float)
    else:
        coh_band_avg_scalar = float("nan")
        coh_band_avg_win = np.array([], dtype=float)
        plv_scalar = float("nan")
        wpli_scalar = float("nan")
        plv_win = np.array([], dtype=float)
        wpli_win = np.array([], dtype=float)

    return {
        "scalar": {
            "xcorr_peak_r": float(np.nanmean(xc.peak_r)),
            "coh_band_avg": coh_band_avg_scalar,
            "plv": plv_scalar,
            "wpli": wpli_scalar,
        },
        "windowed": {
            "xcorr_peak_r": np.asarray(xc.peak_r, float),
            "coh_band_avg": coh_band_avg_win,
            "plv": plv_win,
            "wpli": wpli_win,
        },
    }


def collect_hrv_envelope(subjects, env_col, data_dir,
                         hrv_feature="HRV_MeanNN"):
    """On-the-fly HRV coupling against a chosen envelope column.

    Yields the same (scalar_df, window_df) shape as ``collect()``.
    """
    scalar_rows, window_rows = [], []
    for s in subjects:
        sub = f"sub-{s:02d}"
        merged = data_dir / "processed" / sub / "tables" / "merged_annotated_with_audio.csv"
        if not merged.exists():
            continue
        df = pd.read_csv(merged, low_memory=False)
        if "time_s" not in df.columns:
            df["time_s"] = np.arange(len(df)) / FS_AUDIO
        for cond in (*REST_SET, *NATURE_SET):
            hrv_csv = data_dir / "processed" / sub / "tables" / f"hrv_features_{cond}.csv"
            if not hrv_csv.exists():
                continue
            rpeaks_file = (data_dir / "processed" / sub / "ecg_processed"
                            / f"rpeaks_{cond}.npy")
            res = _compute_hrv_for_condition(df, cond, hrv_csv, env_col,
                                             hrv_feature=hrv_feature,
                                             rpeaks_file=rpeaks_file)
            if res is None:
                continue
            for mk, val in res["scalar"].items():
                if np.isfinite(val):
                    scalar_rows.append({"subject_id": s, "condition": cond,
                                         "metric": mk, "value": float(val)})
            for mk, arr in res["windowed"].items():
                for v in arr:
                    if np.isfinite(v):
                        window_rows.append({"subject_id": s, "condition": cond,
                                             "metric": mk, "value": float(v)})
        print(f"  {sub}: HRV-{hrv_feature} <-> {env_col} done")
    return pd.DataFrame(scalar_rows), pd.DataFrame(window_rows)


# --------------------------- aggregation ---------------------------
def collect(subjects, modality: str, data_dir: Path, hrv_feature: str = "HRV_MeanNN"):
    """Returns:
        scalar_df : long-form (subject, condition, metric, value)
        window_df : long-form (subject, condition, metric, window_value)

    For HRV (post-refactor) the metric values come from two files:
        * xcorr_peak_r       <- per-feature JSON (windowed HRV-feature trace)
        * plv / wpli / coh   <- per-condition `_hr_instantaneous.json`
                                (instantaneous HR @ 4 Hz)
    """
    scalar_rows, window_rows = [], []

    for s in subjects:
        for c in (*REST_SET, *NATURE_SET):
            if modality == "resp":
                p = _resp_json(data_dir, s, c)
                if not p.exists():
                    continue
                with open(p) as f:
                    d = json.load(f)
                primary = d
                osc = d   # resp JSON keeps all metrics in one file
            else:
                # HRV: per-feature JSON has xcorr; per-condition file has
                # plv / wpli / coh on instantaneous HR
                p = _hrv_json(data_dir, s, c, hrv_feature)
                if not p.exists():
                    continue
                with open(p) as f:
                    primary = json.load(f)
                osc_path = (data_dir / "processed" / f"sub-{s:02d}" / "tables"
                             / f"hrv_audio_coupling_{c}_hr_instantaneous.json")
                if osc_path.exists():
                    with open(osc_path) as f:
                        osc = json.load(f)
                else:
                    osc = {}

            for metric_key, _, json_path in METRICS:
                # Choose the file: xcorr_peak_r from `primary`; oscillatory
                # metrics from `osc`. (For resp these are the same dict.)
                src = primary if metric_key == "xcorr_peak_r" else osc
                val = _scalar_metric(src, metric_key)
                if np.isfinite(val):
                    scalar_rows.append({"subject_id": s, "condition": c,
                                         "metric": metric_key, "value": float(val)})
                arr = _windowed_metric(src, metric_key, json_path)
                for v in arr:
                    if np.isfinite(v):
                        window_rows.append({"subject_id": s, "condition": c,
                                             "metric": metric_key, "value": float(v)})
    return pd.DataFrame(scalar_rows), pd.DataFrame(window_rows)


def _block(condition: str) -> str:
    if condition in REST_SET:
        return "REST"
    if condition in NATURE_SET:
        return "NATURE"
    return "OTHER"


# --------------------------- A: paired Wilcoxon ---------------------------
def per_subject_block_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean across conditions within each block, per (subject, metric, block)."""
    df = df.copy()
    df["block"] = df["condition"].apply(_block)
    df = df[df["block"].isin(("REST", "NATURE"))]
    return (df.groupby(["subject_id", "metric", "block"])["value"]
              .mean().reset_index())


def paired_wilcoxon(block_means: pd.DataFrame, metric: str) -> dict:
    sub = block_means[block_means["metric"] == metric]
    pivot = sub.pivot(index="subject_id", columns="block", values="value").dropna()
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return {"n": int(pivot.shape[0]), "p": float("nan"), "stat": float("nan"),
                "delta_mean": float("nan")}
    try:
        res = sps.wilcoxon(pivot["NATURE"].values, pivot["REST"].values,
                           alternative="two-sided", zero_method="wilcox",
                           nan_policy="omit")
        p = float(res.pvalue); stat = float(res.statistic)
    except Exception:
        p = float("nan"); stat = float("nan")
    delta = (pivot["NATURE"] - pivot["REST"]).values
    return {"n": int(pivot.shape[0]), "p": p, "stat": stat,
            "delta_mean": float(np.mean(delta)),
            "deltas": delta, "subject_ids": pivot.index.values,
            "rest": pivot["REST"].values, "nature": pivot["NATURE"].values}


# --------------------------- B: window-level mixed model ---------------------------
def window_lmm(window_df: pd.DataFrame, metric: str) -> dict:
    """value ~ block + (1|subject) on the windowed values, REST vs NATURE.

    Returns {n_windows, n_subj, beta_block, p, model_str}.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return {"n_windows": 0, "p": float("nan"), "beta_block": float("nan"),
                "note": "statsmodels not available"}
    sub = window_df[window_df["metric"] == metric].copy()
    sub["block"] = sub["condition"].apply(_block)
    sub = sub[sub["block"].isin(("REST", "NATURE"))].dropna(subset=["value"])
    if sub.empty or sub["block"].nunique() < 2 or sub["subject_id"].nunique() < 3:
        return {"n_windows": int(len(sub)), "p": float("nan"),
                "beta_block": float("nan"), "note": "insufficient data"}
    sub["block_dummy"] = (sub["block"] == "NATURE").astype(int)
    try:
        model = smf.mixedlm("value ~ block_dummy", data=sub,
                            groups=sub["subject_id"]).fit(method="lbfgs", reml=True)
        beta = float(model.params.get("block_dummy", float("nan")))
        p = float(model.pvalues.get("block_dummy", float("nan")))
    except Exception as e:
        return {"n_windows": int(len(sub)), "p": float("nan"),
                "beta_block": float("nan"), "note": f"LMM failed: {e}"}
    return {"n_windows": int(len(sub)),
            "n_subj": int(sub["subject_id"].nunique()),
            "beta_block": beta, "p": p,
            "model_str": "value ~ block_dummy + (1|subject)"}


# --------------------------- plotting ---------------------------
def plot_modality(scalar_df, window_df, modality: str, output_path: Path,
                  title_prefix: str):
    use_paper_style()
    n_metrics = len(METRICS)
    # `sharex='col'` keeps the violin (left) and forest (right) panels aligned
    # within their column, with x-tick labels only on the bottom row.
    fig, axes = plt.subplots(n_metrics, 2, figsize=(7.4, 2.4 * n_metrics),
                             gridspec_kw={"width_ratios": [1.0, 1.0]},
                             sharex='col')
    axes = np.atleast_2d(axes)

    block_means = per_subject_block_means(scalar_df)
    stats_rows = []

    for i, (metric_key, label, _) in enumerate(METRICS):
        # ---------- A: paired violin ----------
        ax_v = axes[i, 0]
        wilc = paired_wilcoxon(block_means, metric_key)
        lmm = window_lmm(window_df, metric_key)

        if "deltas" in wilc:
            rest_vals = wilc["rest"]; nature_vals = wilc["nature"]
            data = [rest_vals, nature_vals]
            parts = ax_v.violinplot(data, positions=[0, 1],
                                    widths=0.7, showmeans=False, showextrema=False)
            for pc, blk in zip(parts["bodies"], ["REST", "NATURE"]):
                pc.set_facecolor(BLOCK_COLORS[blk]); pc.set_alpha(0.5)
                pc.set_edgecolor(BLOCK_COLORS[blk]); pc.set_linewidth(0.9)
            for r, n in zip(rest_vals, nature_vals):
                ax_v.plot([0, 1], [r, n], color="#9aa0a6", alpha=0.6, lw=1.0)
                ax_v.scatter([0, 1], [r, n], s=24, color="black", alpha=0.75, zorder=4)
            means = [np.mean(rest_vals), np.mean(nature_vals)]
            ax_v.plot([0, 1], means, "-D", color="#f5b400", lw=1.7, ms=9,
                      mec="black", mew=0.6, zorder=10)

            # Bracket with stars driven by the windowed LMM (more powerful than n=5).
            stars = sig_stars(lmm.get("p", float("nan")))
            ymax = float(np.nanmax(np.concatenate(data)))
            ymin = float(np.nanmin(np.concatenate(data)))
            yspan = max(1e-6, ymax - ymin)
            # Reserve generous headroom above the data so the bracket+stars never collide
            # with the panel top / column header.
            ax_v.set_ylim(ymin - 0.05 * yspan, ymax + 0.30 * yspan)
            level = ymax + 0.10 * yspan
            ax_v.plot([0, 1], [level, level], color="#444", lw=0.9)
            ax_v.text(0.5, level + 0.02 * yspan, stars,
                      ha="center", va="bottom",
                      fontsize=14 if stars not in ("ns", "n/a") else 11,
                      fontweight="bold" if stars not in ("ns", "n/a") else "normal",
                      color="#222" if stars not in ("ns", "n/a") else "#888")

        ax_v.set_xticks([0, 1])
        ax_v.set_xticklabels(["REST\n(RS1+RS2)", "NATURE\n(VIZ+AUD+MULTI)"],
                             fontsize=9.5)
        ax_v.set_xlim(-0.55, 1.55)
        ax_v.set_ylabel(label, fontsize=11, fontweight="bold")
        ax_v.grid(True, axis="y", alpha=0.3)

        # ---------- C: per-subject Δ forest ----------
        ax_f = axes[i, 1]
        if "deltas" in wilc:
            ds = wilc["deltas"]
            sids = wilc["subject_ids"]
            y_positions = np.arange(len(ds))[::-1]
            colors = [BLOCK_COLORS["NATURE"] if d > 0 else BLOCK_COLORS["REST"] for d in ds]
            ax_f.scatter(ds, y_positions, s=70, c=colors, edgecolor="black",
                         linewidth=0.7, zorder=4)
            for d, y, c in zip(ds, y_positions, colors):
                ax_f.plot([0, d], [y, y], color=c, lw=1.5, alpha=0.7, zorder=2)
            ax_f.axvline(0, color="#888", lw=0.8, ls="--")
            grand_mean = float(np.mean(ds))
            ax_f.scatter(grand_mean, -0.7, s=120, marker="D",
                         color="#f5b400", edgecolor="black", linewidth=1.0, zorder=5)
            ax_f.plot([0, grand_mean], [-0.7, -0.7], color="#f5b400",
                      lw=2.2, alpha=0.85, zorder=4)
            ax_f.set_yticks(list(y_positions) + [-0.7])
            ax_f.set_yticklabels([f"S{s:02d}" for s in sids] + ["Mean"],
                                 fontsize=9)
            ax_f.set_xlabel("Δ = NATURE − REST", fontsize=10)
            ax_f.set_ylim(-1.4, len(ds) - 0.5)
            ax_f.grid(True, axis="x", alpha=0.3)

        if i == 0:
            axes[0, 0].set_title("Paired comparison\n(per-subject means)",
                                 fontsize=11, fontweight="bold")
            axes[0, 1].set_title("Per-subject Δ\n(forest plot)",
                                 fontsize=11, fontweight="bold")

        stats_rows.append({
            "modality": modality, "metric": metric_key,
            "wilcoxon_p": wilc.get("p", float("nan")),
            "wilcoxon_stat": wilc.get("stat", float("nan")),
            "delta_mean": wilc.get("delta_mean", float("nan")),
            "lmm_p": lmm.get("p", float("nan")),
            "lmm_beta": lmm.get("beta_block", float("nan")),
            "lmm_n_windows": lmm.get("n_windows", 0),
            "n_subjects": wilc.get("n", 0),
        })

    # Title removed (in caption).
    fig.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(""))
    plt.close()
    print(f"  Saved: {Path(output_path).name}.png (+ pdf)")
    return stats_rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--modalities", nargs="+",
                   default=["resp", "hrv_meannn", "hrv_meannn_swell_0p1"],
                   choices=["resp", "hrv_meannn", "hrv_meannn_swell_0p1"])
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--report-dir", type=Path,
                   default=ROOT / "reports" / "preliminary_results" / "figures")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "nature_vs_rest")
    return p.parse_args()


def main():
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for mod in args.modalities:
        if mod == "resp":
            sca, win = collect(args.subjects, "resp", args.data_dir)
            title = "Respiration ↔ audio swell"
            out = args.report_dir / "nature_vs_rest_resp"
        elif mod == "hrv_meannn":
            sca, win = collect(args.subjects, "hrv", args.data_dir,
                               hrv_feature="HRV_MeanNN")
            title = "HRV-MeanNN ↔ env_swell_0p2"
            out = args.report_dir / "nature_vs_rest_hrv_meannn"
        elif mod == "hrv_meannn_swell_0p1":
            sca, win = collect_hrv_envelope(args.subjects, "env_swell_0p1",
                                             args.data_dir, hrv_feature="HRV_MeanNN")
            title = "HRV-MeanNN ↔ env_swell_0p1"
            out = args.report_dir / "nature_vs_rest_hrv_meannn_swell_0p1"
        else:
            continue
        if sca.empty:
            print(f"  No data for modality '{mod}'; skipping")
            continue
        stats_rows = plot_modality(sca, win, mod, out, title)
        all_stats.extend(stats_rows)

    if all_stats:
        df = pd.DataFrame(all_stats)
        out_csv = args.results_dir / "nature_vs_rest_stats.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  Stats CSV: {out_csv}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
