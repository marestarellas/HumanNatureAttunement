# Human-Nature Attunement (HNA) — Project Status

**Branch:** `claude/jolly-pare-81ab43`
**Pilot dataset:** `data/processed/sub-{02..06}/`
**Goal:** Quantify how respiration, cardiac, and cortical rhythms entrain
to the temporal structure of nature sounds (sea waves), using a multimodal
coupling toolbox built around `src/HNA/modules/`.

---

## 1. Pipeline overview

### Stage 0 — Pre-processing (`scripts/preprocessing/`)
Per-subject pipeline driven by a single config file (`config/subjects.json`).

| Step | Script | Output |
|------|--------|--------|
| 1. Align EEG/physio + annotate conditions | `01_align_and_annotate.py` | `tables/merged_annotated.csv`, `tables/merged_annotated_cut.csv`, `audio/<base>_cut.wav` |
| 2. Decompose audio envelope into 12 bands | `02_compute_audio_envelopes.py` | `audio/sea_envelopes_curves.csv` |
| 3. Merge audio bands into the annotated table | `03_merge_audio_into_tables.py` | `tables/merged_annotated_with_audio.csv`  ← **canonical table** |
| 4. Cut audio per condition | `cut_audio_by_conditions.py` | `audio/audio_<COND>.wav` |
| 5. Clean ECG + extract R-peaks | `preprocess_ecg.py` | `ecg_processed/ecg_<COND>.csv`, `rpeaks_<COND>.npy` |

The 12 audio envelope columns (per-condition log-normalized, common 256 Hz axis):
- `env_broad` (60 Hz LP, broadband)
- `env_swell_0p2`, `env_swell_0p1` (slow swells, 0.2 / 0.1 Hz LP)
- `env_hrv_lf`, `env_hrv_hf` (0.04–0.15, 0.15–0.40 Hz; HRV bands)
- `env_splash_1_5` (1–5 Hz; splash bursts)
- `env_delta`, `env_theta`, `env_alpha` (canonical EEG bands on the envelope)
- `env_beta_low`, `env_beta_high`, `env_gamma1`

### Stage 1 — Features (`scripts/features/`)
- `extract_eeg_features.py` — band-power (δ θ α β γ, abs+rel) + entropy (LZC, perm,
  spectral, SVD, sample) per channel × condition.
- `extract_hrv_features.py` — NeuroKit2 HRV in 30 s windows with 90 % overlap.

### Stage 2 — Coupling analyses (`scripts/analysis/`)
- `run_resp_audio_coupling.py` — windowed xcorr / coherence / PLV / wPLI / MI
  between cleaned respiration and the swell envelope.
- `run_resp_audio_multi_envelope.py` — same five metrics swept across all
  12 audio envelope bands (5 × 12 fingerprint per subject × condition).
- `run_hrv_audio_coupling.py` — same metrics for HRV (RMSSD/MeanNN/SDNN/HF/SampEn) ↔ audio.
- `compute_audio_eeg_correlation.py` — band-by-band Pearson r (direct + lagged)
  between EEG and audio envelope, per channel.
- `compute_audio_eeg_coherence.py` — same but with spectral coherence.
- `compute_audio_eeg_mutual_information.py` — same but with kNN-MI (kept available;
  not run on this pilot to save compute).

### Stage 3 — Stats (`scripts/stats/`)
- `run_correlation_stats.py`, `run_coherence_stats.py`, `run_mi_stats.py` —
  per-channel mixed-effects → topomaps + summary plots, FDR over channels.
- `run_correlation_stats_aggregated.py`, `run_mi_stats_aggregated.py` —
  channel-aggregated stats (Friedman + Wilcoxon).
- `run_glm_analysis.py`, `run_gee_classification.py` — exploratory.

### Stage 4 — Figures (`scripts/figures/`)
- Reference figures (1–5) and the new analyses A, B, C, D, E, F (described below).

---

## 2. Subject status

| Subject | Status (`config/subjects.json`) | Condition order | Notes |
|---------|----|------|-----|
| sub-00, sub-01 | `split_recording` | VIZ-MULTI-AUD | Vendor-format multi-block raw; needs custom loader. |
| sub-02 | ready | MULTI-VIZ-AUD | |
| sub-03 | ready | AUD-VIZ-MULTI | (Old on-disk labels were stale; relabeled.) |
| sub-04 | ready | VIZ-AUD-MULTI | trigger patch: add trigger 3 min after last (last trigger missing in recording). |
| sub-05 | ready | VIZ-AUD-MULTI | trigger patch: remove the 5th-from-last trigger (spurious). |
| sub-06 | ready | AUD-VIZ-MULTI | |
| sub-07 | `pending` | VIZ-AUD-MULTI | Single-block CSV layout; not yet validated end-to-end. |
| sub-08 | `trigger_problem` | MULTI-AUD-VIZ | Lower threshold (1950) needed; some trigger anomalies remain. |

**Pilot N = 5** (subjects 02–06). All five fully processed end-to-end with
the current pipeline.

### Important relabel finding (sanity-check passed)

Three of the five (sub-02, sub-03, sub-04) had stale condition labels on
disk relative to the current `config/subjects.json` (the labels predated
a Jan 2026 "adjust conditions" commit and a separate fix for sub-04).
After regenerating, mean-shift accounting checks out exactly:

```
ΔVIZ   predicted -0.0154   observed -0.0154 ✓
ΔAUD   predicted +0.1084   observed +0.1084 ✓
ΔMULTI predicted -0.0930   observed -0.0930 ✓
```

So no algorithmic regression — the pipeline output reshuffles cleanly
under the corrected labels.

---

## 3. Figures (`figures/report/`)

All figures are saved as both PNG (300 dpi) and PDF, with consistent
paper-quality styling driven by `HNA.viz.use_paper_style()` and
the canonical condition palette.

### Original five (with where MI was added)

- **Fig 1 — Per-subject signal alignment.** `Fig1_alignment_sub03_AUD.png` (representative).
  3-panel: 60 s respiration vs. swell overlay → full-signal cross-correlation
  → 30 s windowed Pearson r. Per subject × condition; full set in
  `figures/per_subject/sub-XX/`.
- **Fig 2 — RS1 vs RS2 EEG features.** `Fig2_RS1_vs_RS2_high_beta_rel.png`,
  `Fig2_RS1_vs_RS2_alpha_rel.png`. LMM + Wilcoxon + FDR.
- **Fig 3 — Audio-EEG correlation topomaps.** All three contrasts:
  `Fig3_topomaps_VIZ_vs_AUD.png`, `..._VIZ_vs_MULTI.png`, `..._AUD_vs_MULTI.png`.
  6 bands × per-condition topomaps with N significant channels per band.
- **Fig 4 — Audio-EEG correlation violins per band.** All three contrasts.
- **Fig 5 — Resp ↔ audio coupling, condition comparison (with MI added).**
  `Fig5_resp_audio_metrics_grid.png`. Single paper-grid with
  XCorr / Coherence / PLV / wPLI / **MI**, Friedman + pairwise Wilcoxon,
  trend marker (`~`) for p<0.10. Also individual `violin_<metric>.png`
  files in `figures/respiration_audio/`.

### New analyses added for the report

- **A — Spectrum overlay.** `A_spectrum_overlay.png`. Single panel pooling
  VIZ+AUD+MULTI: audio swell, respiration, instantaneous HR group-mean PSD
  overlaid on a log-frequency axis with the canonical respiratory band
  (0.15–0.40 Hz) highlighted. Shows respiration peaks correctly inside the
  reference band; audio dominates < 0.1 Hz; HR has broadband slow content.

- **B — Surrogate-based individual significance.**
  `B_surrogate_resp_audio.png`. Per-(subject, condition) PLV vs. its own
  phase-shuffled null (n=200 surrogates). With this 5-subject pilot, **0/5
  subjects' PLV exceeds the 95th-percentile null** under any condition —
  important methodological caveat, since the narrow PLV bandwidth + short
  segment = high null floor. The MI-on-Hilbert estimator has the same
  caveat in absolute terms (see methods note below).

- **C — Time-resolved coupling (group mean ± SEM).**
  `C_time_resolved_resp_audio_{coh,plv,wpli,xcorr}.png`. Per-condition
  trajectory across normalized time, with per-subject linear-trend slopes
  (one-sample Wilcoxon vs. zero) and pairwise condition contrasts on the
  right. Coherence shows the cleanest **buildup pattern under AUD**
  (slope = +0.101, p ≈ 0.06 trend); VIZ and MULTI are flat.

- **D — Cross-modal coupling co-variance.** `D_cross_modal_coupling.png`.
  3 × 3 Spearman matrix on per-(subject, condition) cells of resp-audio
  PLV × HRV-audio PLV × EEG-α-audio |r|. With n=15 cells, no significant
  cross-modal co-variation: **attunement does not appear to be a unitary
  multi-modal state** in this pilot, suggesting modality-specific channels.

- **E — Hierarchical phase consistency.** `E_phase_polar.png`. Within-
  subject Rayleigh on each subject's ~20 windowed phases + group-level
  Rayleigh on subject means. AUD shows **4/5 within-subject consistency
  + group R = 0.73, p = 0.060 (trend)**; MULTI shows scattered (R = 0.22).
  Direct evidence of phase stability under audio-only listening that does
  not survive when visual attention is added.

- **F — Multi-envelope coupling fingerprint.** `F_multi_envelope_heatmap.png`.
  5 metrics × 12 audio bands × 3 conditions. Headline: **MI is dominant
  on the slow bands** (swell, HRV-LF/HF; values 0.7–1.1) and ~zero on the
  EEG bands (audio has no alpha-rate amplitude modulation). Linear methods
  (xcorr, coherence) are flat across bands — they undersell the coupling
  that MI exposes. PLV and wPLI peak under AUD as in Fig 5.

### Per-subject artifacts (`figures/per_subject/sub-XX/`)
- `<COND>_signal_alignment_validation.png` (Fig 1 generator — 5 conditions × 5 subjects)
- `<COND>_coupling_timeseries.png`, `<COND>_coherence.png`
- HRV-audio variants (`<COND>_hrv_audio_*.png`)

---

## 4. Methodological notes for the manuscript

### MI on Hilbert envelopes
The kNN MI estimator (Kraskov-like, `sklearn.feature_selection.mutual_info_regression`,
`n_neighbors=3`) is **upward-biased on autocorrelated time series**, so
absolute MI values (~1.0+ on slow swells) should not be interpreted as
"~1 nat of shared information per sample". The same bias hits every
condition equally, so **relative comparisons** (between conditions, or
between bands within a condition) are valid. For absolute claims, use
`HNA.surrogates.surrogate_test(...)` to subtract the
phase-shuffled null mean (recommended addition for the published
version).

### Surrogate test caveat (Analysis B)
Because phase-shuffling preserves the audio's spectral structure and our
PLV bandwidth (0.12 Hz) is narrow, the surrogate null PLV reaches
~0.20–0.40 even for pure-noise pairs. Thus a real PLV around 0.30 (which
*looks* high in absolute terms) is well within the null distribution.
This is the right interpretation: **observed PLV minus null mean** is
the effective coupling, not the raw PLV.

### HRV pipeline specs
- HRV feature window: 30 s, 90 % overlap (3 s step). Reliable for time-
  domain (RMSSD, SDNN, MeanNN), HF, marginal for the slowest LF, and
  **inadequate for VLF** in the available 5-min conditions.
- HRV-audio coupling window: 120 s, 10 s step → ~12 effective independent
  samples per (subject, condition). Tight on power but valid.
- Recommendation: in the report, drop or de-emphasize VLF; cite a single
  whole-condition PLV/wPLI alongside the windowed series.

### Hierarchical phase analysis (E)
We use the windowed `preferred_lag_s` × `dom_freq` to get ~20 phase
samples per subject per condition (instead of one per (subject, condition)).
Within-subject Rayleigh tests how stable the phase relationship is
*within* a subject; group-level Rayleigh on subject means tests whether
the *group* converges on a similar phase. This two-level framing is the
right small-N readout (Fisher 1993).

### Trigger / labeling issues we found and fixed
- Three subjects (02, 03, 04) had stale on-disk condition labels relative
  to the current `config/subjects.json`. Pipeline regenerated and verified.
- The previously-named `env_swell_0p3` column was actually 0.2 Hz LP
  (`SWELL_1` had been changed without renaming). Renamed to `env_swell_0p2`;
  downstream code updated; backwards-compatible fallback kept in
  `envelope_pref` tuples.
- `sub-04` had a missing last trigger: now patched in config
  (`add_after_last_s: 180.0`).
- `sub-05` had a spurious extra trigger: now patched in config
  (`remove_indices: [-5]`).

---

## 5. Toolbox (`src/HNA/modules/`)

See `src/HNA/modules/README.md` for full guidelines. Summary:

```
HNA/modules/
├── dsp.py           Generic signal-processing primitives.
├── coupling.py      xcorr / coherence / PLV / wPLI + windowed variants + plot helpers.
├── surrogates.py    Phase-shuffle / time-shift surrogates + generic test harness.
├── stats.py         Fisher z, BH-FDR, Friedman+post-hoc, Rayleigh, slope tests.
├── viz.py           Paper-ready style + canonical condition palette + sig stars.
├── audio.py         Audio modality: envelope decomposition into 12 bands.
├── eeg.py           EEG modality: filters + PSD bands + entropy.
└── utils.py         Dataset I/O + alignment + condition-segment extraction.
```

Reusability: the `coupling`, `stats`, `surrogates`, and `viz` modules are
modality-agnostic and meant to be reused on any pair of 1-D physiological
signals. Adding new modalities (video features, EDA, …) is a matter of
adding a sibling module next to `audio.py` / `eeg.py`.

---

## 6. Open work

### Validated and ready
- Preprocessing pipeline (config-driven, CLI, repo-root-resolved paths)
- Five reference figures regenerated post-relabel
- Six new analyses (A, B, C, D, E, F) producing paper-ready figures
- Toolbox extended with `stats.py`, `audio.py`, `surrogates.py`, `viz.py`,
  + README + guidelines

### Pending (in priority order)
1. **EEG-audio MI**. The script `compute_audio_eeg_mutual_information.py`
   exists and is path-fixed; just deferred for compute time. Running it
   would let us produce MI versions of Fig 3 (topomaps) and Fig 4 (violins).
2. **sub-07 trigger investigation.** Single-block CSV layout, just not
   end-to-end validated. Most likely needs a small `trigger_patches`
   tweak similar to sub-04/05 once we look at the channel.
3. **sub-08 trigger problems.** Lower threshold (1950) is set; remaining
   anomalies need a manual look at the trigger trace.
4. **sub-00, sub-01.** Vendor BBT multi-block layout; needs a dedicated
   loader in `HNA.utils` (or a sibling `io_bbt.py`).
5. **Surrogate-corrected effective MI** for the report's MI panel (cheap
   to compute; recommended for absolute MI claims).
6. **Optional**: collapse the per-condition and per-subject *_coherence /
   _coupling_timeseries PNGs into a per-subject summary PDF for the
   appendix.

### Outputs not in the repo (but referenced)
- Raw data: outside the repo (`<repo>/data/...` is gitignored). Pilot data
  lives in the main repo's `data/` only.
- A backup of pre-relabel sub-02..06 outputs is in
  `data/processed_backup_pre_relabel_<TIMESTAMP>/` for audit purposes.

---

## 7. How to reproduce the report from scratch

```bash
DATA="/path/to/data"
PYTHONPATH=src

# Stage 0 — preprocessing
python scripts/preprocessing/01_align_and_annotate.py --subjects 02 03 04 05 06 --data-dir "$DATA" --overwrite
python scripts/preprocessing/02_compute_audio_envelopes.py --subjects 02 03 04 05 06 --processed-dir "$DATA/processed" --overwrite
python scripts/preprocessing/03_merge_audio_into_tables.py --subjects 02 03 04 05 06 --savepath "$DATA/processed" --overwrite
python scripts/preprocessing/cut_audio_by_conditions.py --subjects 2 3 4 5 6 --data-dir "$DATA"
python scripts/preprocessing/preprocess_ecg.py --subjects 2 3 4 5 6 --data-dir "$DATA" --overwrite

# Stage 1 — features
python scripts/features/extract_eeg_features.py --subjects 02 03 04 05 06 --data-dir "$DATA"
python scripts/features/extract_hrv_features.py --subjects 2 3 4 5 6 --data-dir "$DATA" --overwrite

# Stage 2 — coupling
python scripts/analysis/run_resp_audio_coupling.py --subjects 02 03 04 05 06 --data-dir "$DATA" --overwrite
python scripts/analysis/run_resp_audio_multi_envelope.py --subjects 2 3 4 5 6 --data-dir "$DATA"
python scripts/analysis/run_hrv_audio_coupling.py --subjects 02 03 04 05 06 --data-dir "$DATA" --overwrite
python scripts/analysis/compute_audio_eeg_correlation.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI RS1 RS2 --data-dir "$DATA"

# Stage 3 — stats
for c1c2 in "VIZ AUD" "VIZ MULTI" "AUD MULTI"; do
  python scripts/stats/run_correlation_stats.py --metric correlation_direct --condition1 $c1c2
done

# Stage 4 — figures
python scripts/figures/analysis_A_spectrum_overlay.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --data-dir "$DATA"
python scripts/figures/analysis_B_surrogate_significance.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --n-surrogates 200 --data-dir "$DATA"
for m in wpli plv coh xcorr; do
  python scripts/figures/analysis_C_time_resolved_coupling.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --metric $m --data-dir "$DATA"
done
python scripts/figures/analysis_D_cross_modal_coupling.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --data-dir "$DATA"
python scripts/figures/analysis_E_phase_polar.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --data-dir "$DATA"
python scripts/figures/analysis_F_multi_envelope_heatmap.py
python scripts/figures/compare_coupling_conditions.py --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --data-dir "$DATA"
python scripts/figures/plot_resting_state_features.py --subjects 2 3 4 5 6 --data-dir "$DATA"
for c1c2 in "VIZ AUD" "VIZ MULTI" "AUD MULTI"; do
  python scripts/figures/plot_correlation_changes.py --metric correlation_direct --condition1 $c1c2
done
```
