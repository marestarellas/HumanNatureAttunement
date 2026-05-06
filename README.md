# HumanNatureAttunement

Exploring the neurophysiology of human attunement to nature through
multimodal coupling between sensory input (audio, video) and physiological
output (EEG, ECG/HRV, respiration).

The repository contains the **HNA Python toolbox** for multimodal coupling
analysis, the data-processing scripts of the *Multisensory Synchronization
as Mechanism of Nature Connectedness* project, and a preliminary results
report from the pilot dataset (n=5, sea-coast field recordings on the
coasts of Chile and Mallorca, Spain).

## Repository layout

```
HumanNatureAttunement/
├── src/HNA/modules/        ← the HNA toolbox (importable Python package)
│   ├── dsp.py              generic signal-processing primitives
│   ├── coupling.py         xcorr / coherence / PLV / wPLI / windowed MI + plot helpers
│   ├── surrogates.py       phase-shuffle / time-shift surrogates + generic test harness
│   ├── stats.py            Fisher z, BH-FDR, Friedman + post-hoc, circular Rayleigh, slope tests
│   ├── viz.py              paper-ready matplotlib style + canonical condition palette
│   ├── audio.py            MODALITY: audio envelope decomposition into 12 bands
│   ├── eeg.py              MODALITY: EEG band-power + entropy
│   ├── utils.py            data loading, alignment, condition extraction
│   └── README.md           toolbox guideline & module map
│
├── scripts/                ← thin CLI wrappers around the toolbox
│   ├── preprocessing/      01_align_and_annotate, 02_compute_audio_envelopes,
│   │                       03_merge_audio_into_tables, cut_audio_by_conditions, preprocess_ecg
│   ├── features/           extract_eeg_features, extract_hrv_features
│   ├── analysis/           run_resp_audio_coupling, run_hrv_audio_coupling,
│   │                       run_*_multi_envelope, compute_audio_eeg_correlation*
│   ├── stats/              run_correlation_stats, run_coherence_stats, run_mi_stats,
│   │                       run_glm_analysis, …
│   └── figures/            analysis_A_spectrum_overlay, analysis_B_surrogate_significance,
│                           analysis_C_time_resolved_coupling, analysis_D_cross_modal_coupling,
│                           analysis_E_phase_polar, analysis_F_*, analysis_RS1_vs_RS2_grid,
│                           analysis_features_grid, analysis_nature_vs_rest, …
│
├── config/
│   └── subjects.json       per-subject metadata: condition order, audio sync,
│                           trigger threshold, trigger patches, status
│
├── notebooks/              live notebooks (preprocessing, EEG features, …) +
│                           legacy/ (old prototypes kept for reference)
│
├── data/                   gitignored — see project_goal.md for layout
│   └── processed/sub-XX/
│       ├── tables/         merged_annotated_with_audio.csv, hrv_features_<COND>.csv,
│       │                   coupling_<COND>.json, …
│       ├── audio/          per-condition WAVs + envelope CSV
│       └── ecg_processed/  cleaned ECG + R-peaks per condition
│
├── reports/preliminary_results/
│   └── report.tex          LaTeX preliminary-results report (compiles with tectonic)
│
├── results/                intermediate stats CSVs
└── figures/                figure outputs
    └── report/             curated paper-ready set (tracked); rest is regenerable & gitignored
```

## Pipeline at a glance

```
raw EEG / ECG / respiration / audio
        │
        ▼
scripts/preprocessing/01_align_and_annotate.py
        │   per-subject trigger alignment, condition annotation
        ▼
scripts/preprocessing/02_compute_audio_envelopes.py
        │   12 band-organized audio envelope columns at 256 Hz
        ▼
scripts/preprocessing/03_merge_audio_into_tables.py
        │   → data/processed/sub-XX/tables/merged_annotated_with_audio.csv
        ▼
scripts/preprocessing/cut_audio_by_conditions.py + preprocess_ecg.py
        │   per-condition WAVs, cleaned ECG + R-peaks
        ▼
scripts/features/extract_eeg_features.py + extract_hrv_features.py
        │   band-power, entropy, HRV (NeuroKit2)
        ▼
scripts/analysis/{run_resp_audio_coupling, run_hrv_audio_coupling, …}
        │   xcorr / coherence / PLV / wPLI / MI per (subject, condition)
        ▼
scripts/stats/* + scripts/figures/analysis_*
        │   LMMs, Friedman + Wilcoxon post-hoc, paper-ready figures
        ▼
reports/preliminary_results/report.tex
```

The pipeline is config-driven: per-subject specifics (condition order,
audio sync time, trigger threshold, manual trigger patches) live in
`config/subjects.json` and the scripts default to repo-root-relative paths
(`<repo>/data`, `<repo>/results`, `<repo>/figures`) with explicit
`--data-dir` / `--results-dir` / `--figures-dir` CLI overrides.

## Running the full pilot pipeline

```bash
DATA="/path/to/data"
PYTHONPATH=src

# Stage 0 — preprocessing
python scripts/preprocessing/01_align_and_annotate.py     --subjects 02 03 04 05 06 --data-dir "$DATA" --overwrite
python scripts/preprocessing/02_compute_audio_envelopes.py --subjects 02 03 04 05 06 --processed-dir "$DATA/processed" --overwrite
python scripts/preprocessing/03_merge_audio_into_tables.py --subjects 02 03 04 05 06 --savepath "$DATA/processed" --overwrite
python scripts/preprocessing/cut_audio_by_conditions.py    --subjects 2 3 4 5 6 --data-dir "$DATA"
python scripts/preprocessing/preprocess_ecg.py             --subjects 2 3 4 5 6 --data-dir "$DATA" --overwrite

# Stage 1 — features
python scripts/features/extract_eeg_features.py            --subjects 02 03 04 05 06 --data-dir "$DATA"
python scripts/features/extract_hrv_features.py            --subjects 2 3 4 5 6 --data-dir "$DATA" --overwrite

# Stage 2 — coupling
python scripts/analysis/run_resp_audio_coupling.py         --subjects 02 03 04 05 06 --data-dir "$DATA" --overwrite
python scripts/analysis/run_hrv_audio_coupling.py          --subjects 02 03 04 05 06 --data-dir "$DATA" --overwrite
python scripts/analysis/run_resp_audio_multi_envelope.py   --subjects 2 3 4 5 6 --data-dir "$DATA"
python scripts/analysis/compute_audio_eeg_correlation.py   --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI RS1 RS2 --data-dir "$DATA"

# Stage 3 — figures (see scripts/figures/ for the full list)
python scripts/figures/analysis_A_spectrum_overlay.py      --subjects 2 3 4 5 6 --conditions VIZ AUD MULTI --data-dir "$DATA"
python scripts/figures/analysis_nature_vs_rest.py          --subjects 2 3 4 5 6 --modalities resp hrv_meannn hrv_meannn_swell_0p1 --data-dir "$DATA"
```

## Building the preliminary report

```bash
cd reports/preliminary_results
tectonic -X compile report.tex
# → report.pdf  (≈11 MB; gitignored, regenerated locally)
```

`tectonic` auto-downloads the necessary LaTeX packages on first run. Install
with `conda install -c conda-forge tectonic` or `cargo install tectonic`.

## Documentation

- [`src/HNA/modules/README.md`](src/HNA/modules/README.md) — toolbox guide:
  module map, coupling-method table, surrogate testing recipe, plotting
  style, guidelines for adding new modalities.
- [`REPORT.md`](REPORT.md) — narrative project status + reproduction instructions.
- [`reports/preliminary_results/report.tex`](reports/preliminary_results/report.tex) —
  the formal preliminary-results report (LaTeX source).

## Authors

Antoine Bellemare-Pepin, Mar Estarellas, Karim Jerbi, Michael Lifshitz.

## License

MIT.
