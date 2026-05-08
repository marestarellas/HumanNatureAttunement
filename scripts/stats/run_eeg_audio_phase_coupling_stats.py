#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMM-based stats for EEG-audio band-matched coupling (PLV / wPLI / direct
correlation).

Per (band, metric) we fit a linear mixed-effects model with crossed
random effects for subject and channel:

    value ~ C(condition, Treatment(reference="VIZ"))
            + (1 | subject)
            + (1 | channel)

Reported per (band, metric):
  - omnibus_p  : two-coefficient Wald F across condition dummies
                 (i.e. ``does condition matter at all?``)
  - p_AUD_vs_VIZ, p_MULTI_vs_VIZ, p_AUD_vs_MULTI : pairwise Wald p-values
                 from t-tests on the corresponding contrasts
  - mean_VIZ, mean_AUD, mean_MULTI : per-condition raw group means
                 (subject- and channel-pooled).

Inputs:
  --plv-csv     results/eeg_audio_phase_coupling/eeg_audio_phase_coupling.csv
  --corr-csv    results/audio_eeg_correlation/audio_eeg_correlation_results.csv
                (or band-matched variant; auto-detected)

Output:
  results/eeg_audio_phase_coupling/eeg_audio_phase_coupling_stats.csv
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# statsmodels MixedLM with two random effects (crossed) is awkward to
# express directly; we use the ``vc_formula`` approach: subject is the
# main grouping level, channel is a variance component.
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]


CONDITIONS = ("VIZ", "AUD", "MULTI")  # reference VIZ
BANDS = ("delta", "theta", "alpha", "low_beta", "high_beta", "gamma1")


def _fit_lmm(df: pd.DataFrame, value_col: str):
    """Fit ``value ~ condition + (1|subject) + (1|channel)`` and return
    a dict with omnibus_p, three pairwise p-values, and means.

    Crossed random effects via vc_formula: subject grouping +
    channel variance component. Reference level is VIZ.
    """
    sub = df.dropna(subset=[value_col]).copy()
    sub = sub[sub["condition"].isin(CONDITIONS)]
    if sub.empty or sub["subject_id"].nunique() < 2 \
            or sub["condition"].nunique() < 2:
        return None
    sub["condition"] = pd.Categorical(
        sub["condition"], categories=list(CONDITIONS), ordered=False,
    )
    sub["subject_str"] = sub["subject_id"].astype(str)
    sub["channel"] = sub["channel"].astype(str)

    formula = (f"{value_col} ~ "
               f'C(condition, Treatment(reference="VIZ"))')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(
                formula, data=sub, groups=sub["subject_str"],
                vc_formula={"channel": "0 + C(channel)"},
                re_formula="~1",
            )
            fit = model.fit(method="lbfgs", reml=True)
        except Exception:  # noqa: BLE001
            return None

    params = fit.params
    bse = fit.bse
    aud_name = 'C(condition, Treatment(reference="VIZ"))[T.AUD]'
    multi_name = 'C(condition, Treatment(reference="VIZ"))[T.MULTI]'

    if aud_name not in params.index or multi_name not in params.index:
        return None

    # Build numeric contrast vectors over FIXED-EFFECTS coefficients
    # (statsmodels MixedLM.t_test only accepts FE-sized contrasts).
    fe_names = list(fit.fe_params.index)
    n_fe = len(fe_names)
    i_aud = fe_names.index(aud_name)
    i_multi = fe_names.index(multi_name)

    def _contrast(c: dict) -> np.ndarray:
        v = np.zeros((1, n_fe))
        for k, w in c.items():
            v[0, k] = w
        return v

    def _wald_p(c: np.ndarray) -> float:
        try:
            ct = fit.t_test(c)
            return float(np.asarray(ct.pvalue).item())
        except Exception:  # noqa: BLE001
            return float("nan")

    # Omnibus: joint F-test of both condition dummies = 0.
    # f_test wants a contrast over the FULL param vector (FE + RE), so
    # rebuild the row with the FE indices preserved within all-params space.
    try:
        all_names = list(params.index)
        n_all = len(all_names)
        i_aud_all = all_names.index(aud_name)
        i_multi_all = all_names.index(multi_name)
        c1_all = np.zeros(n_all); c1_all[i_aud_all] = 1.0
        c2_all = np.zeros(n_all); c2_all[i_multi_all] = 1.0
        omnibus_p = float(fit.f_test(np.vstack([c1_all, c2_all])).pvalue)
    except Exception:  # noqa: BLE001
        omnibus_p = float("nan")

    p_aud_viz = _wald_p(_contrast({i_aud: 1.0}))
    p_multi_viz = _wald_p(_contrast({i_multi: 1.0}))
    p_aud_multi = _wald_p(_contrast({i_aud: 1.0, i_multi: -1.0}))

    # Per-condition raw group means (channel- and subject-pooled)
    means = (sub.groupby("condition", observed=True)[value_col]
                .mean().reindex(list(CONDITIONS)))

    return {
        "omnibus_p": omnibus_p,
        "p_AUD_vs_VIZ": p_aud_viz,
        "p_MULTI_vs_VIZ": p_multi_viz,
        "p_AUD_vs_MULTI": p_aud_multi,
        "beta_AUD": float(params.get(aud_name, np.nan)),
        "beta_MULTI": float(params.get(multi_name, np.nan)),
        "se_AUD": float(bse.get(aud_name, np.nan)),
        "se_MULTI": float(bse.get(multi_name, np.nan)),
        "mean_VIZ": float(means.get("VIZ", np.nan)),
        "mean_AUD": float(means.get("AUD", np.nan)),
        "mean_MULTI": float(means.get("MULTI", np.nan)),
        "n_subjects": int(sub["subject_id"].nunique()),
        "n_channels": int(sub["channel"].nunique()),
        "n_obs": int(len(sub)),
    }


def run_phase_coupling(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    rows = []
    for band in BANDS:
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        for metric in ("plv", "wpli"):
            res = _fit_lmm(sub, metric)
            if res is None:
                continue
            rows.append({"band": band, "metric": metric, **res})
    return pd.DataFrame(rows)


def run_correlation(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    # The band-matched correlation file uses ``correlation_direct`` (raw
    # Pearson r). Other variants may use ``correlation_max_lagged``.
    candidates = ["correlation_direct"]
    available = [c for c in candidates if c in df.columns]
    if not available:
        return pd.DataFrame()
    metric = available[0]
    rows = []
    for band in BANDS:
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        res = _fit_lmm(sub, metric)
        if res is None:
            continue
        rows.append({"band": band, "metric": metric, **res})
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plv-csv", type=Path,
                   default=ROOT / "results" / "eeg_audio_phase_coupling"
                                / "eeg_audio_phase_coupling.csv")
    p.add_argument("--corr-csv", type=Path,
                   default=ROOT / "results" / "audio_eeg_correlation"
                                / "audio_eeg_correlation_results.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "eeg_audio_phase_coupling"
                                / "eeg_audio_phase_coupling_stats.csv")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []

    if args.plv_csv.exists():
        df_phase = run_phase_coupling(args.plv_csv)
        out_rows.append(df_phase)
        print(f"  PLV / wPLI stats: {len(df_phase)} rows")
    else:
        print(f"  WARNING: {args.plv_csv} missing; PLV/wPLI skipped")

    if args.corr_csv.exists():
        df_corr = run_correlation(args.corr_csv)
        out_rows.append(df_corr)
        print(f"  correlation stats: {len(df_corr)} rows")
    else:
        print(f"  WARNING: {args.corr_csv} missing; correlation skipped")

    if not out_rows:
        print("No stats produced.")
        return
    df = pd.concat(out_rows, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"\n  Saved: {args.out}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
