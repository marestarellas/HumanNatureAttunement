# HNA toolbox (`HNA`)

A library for **multimodal human-nature attunement** analyses. Built around
the question: *how do physiological rhythms (respiration, HRV, EEG) couple
to the temporal structure of natural environmental signals (audio, video, …)?*

The toolbox is organized so that **coupling methods are decoupled from
modalities**: any pair of 1-D signals can be coupled with any of the
implemented metrics, and per-modality helpers exist for the common
preprocessing steps (envelope extraction, band-power, R-peak detection,
respiration cleaning, …).

## Module map

```
HNA/
├── dsp.py             Generic signal-processing primitives (filters, envelope,
│                      NaN handling, resample, z-score, butter SOS).
├── io/                Optional pipeline-boundary helpers.
│   └── trace.py         ``Trace`` dataclass: 1-D ndarray + fs + name +
│                        modality + t0 + units, with ``resample_to`` /
│                        ``crop`` / ``align_to`` / ``from_dataframe``.
│                        Purely additive — coupling / features / stats /
│                        viz still accept plain ``(ndarray, fs)``.
├── coupling/          Four-family coupling subpackage. The package's
│   │                  ``__init__`` re-exports the full public surface so
│   │                  ``from HNA.coupling import …`` works
│   │                  uniformly across families.
│   ├── linear.py        Time-domain Pearson alignment with lag
│   │                    (``windowed_xcorr``).
│   ├── oscillatory.py   Frequency-/phase-domain coupling:
│   │                    • Welch coherence + windowed band-avg coherence
│   │                    • PLV / windowed PLV (phase consistency)
│   │                    • wPLI / windowed wPLI (debiased phase consistency)
│   │                    • PAC: Tort MI, Canolty MVL, windowed PAC,
│   │                      comodulogram (cross-frequency).
│   ├── information.py   Probabilistic and directional dependence:
│   │                    • windowed_mi (raw MI, biased)
│   │                    • effective_mi / windowed_effective_mi (bias-corrected)
│   │                    • Granger: granger_bivariate / granger_score /
│   │                      windowed_granger (linear-Gaussian directional)
│   │                    • transfer_entropy (lazy ``copent`` import).
│   ├── complexity.py    Scaling-structure coupling — exponent_matching,
│   │                    fluctuation_matching, mse_matching,
│   │                    windowed_exponent, complexity_coupling.
│   └── _plots.py        Shared plot helpers (plot_signal_alignment_validation,
│                        plot_coupling_over_time, plot_coherence_results).
│
├── surrogates.py      Phase-shuffle / time-shift surrogate generators +
│                      generic ``surrogate_test(metric_fn, x, y)`` harness.
├── stats.py           Linear (Fisher z), multiple-comparisons (BH-FDR),
│                      repeated-measures (Friedman + Wilcoxon post-hoc),
│                      circular (Rayleigh, mean, R), slope tests, AND
│                      cluster-based permutation testing (Maris/Oostenveld).
│
├── features/          Modality-agnostic per-signal feature extraction.
│   ├── psd.py           Welch PSD + band-integrated absolute / relative power.
│   ├── entropy.py       LZC, permutation, spectral, SVD, sample entropy.
│   ├── fractal.py       Higuchi / Katz / Petrosian FD, DFA α, Hurst R/S.
│   ├── aperiodic.py     FOOOF aperiodic offset / exponent / knee + R²
│   │                    (raw PSD or Welch-then-FOOOF in one shot).
│   ├── windowed.py      Generic ``windowed_channel_features`` iterator —
│   │                    slides any feature_fn over an EEG-like wide DataFrame.
│   └── __init__.py      Convenience aggregators (``all_entropies``,
│                        ``all_fractals``) plus legacy-named DataFrame
│                        wrappers (``compute_psd_features``, etc.) for
│                        backward compatibility.
│
├── modalities/        Per-modality cleaners only.
│   ├── audio.py         decompose_envelope into 12 band-organized columns
│   │                    (broad, swell, HRV-LF/HF, splash, δ, θ, α, β-low,
│   │                    β-high, γ1).
│   ├── eeg.py           filter_eeg (multi-channel Butterworth bandpass).
│   ├── respiration.py   clean_respiration (0.05–1 Hz BP + z-score).
│   └── ecg.py           preprocess_ecg_segment (NeuroKit2 cleaning +
│                        R-peak detection), compute_rolling_hrv_features,
│                        HRV ↔ audio resampling helpers.
│
├── viz/               Plotting subpackage.
│   ├── __init__.py      re-exports the style + palette + sig_stars +
│   │                    fmt_p + save_figure + add_significance_bar.
│   ├── _style.py        canonical condition palette + paper rcParams.
│   ├── forest.py        per-subject delta forest plots.
│   ├── polar.py         circular histogram + mean-vector arrow.
│   ├── spectrum.py      multi-condition Welch overlay with IQR shading.
│   └── topomap.py       EEG band topomap (lazy MNE import).
│
└── utils.py           I/O loaders + alignment + condition-segment extraction
                       (the data-layout-aware bits, currently dataset-specific).
```

The toolbox is structured so that:

- **Cleaners are modality-specific** (``modalities/``) — they assume things
  about the signal (NeuroKit2 for ECG, 0.05–1 Hz for respiration, etc.).
- **Features are modality-agnostic** (``features/``) — Welch PSD, fractal
  exponents, entropy, FOOOF aperiodic; all work on any 1-D signal.
- **Coupling families are modality-agnostic too** (``coupling/``) — any
  pair of 1-D signals can be coupled with any of the four families
  (linear / oscillatory / information / complexity).

## Coupling methods at a glance

The four families and the methods they contain:

| Family          | Method                | What it measures                                | When to use                                                                |
|-----------------|-----------------------|-------------------------------------------------|----------------------------------------------------------------------------|
| **Linear**      | Cross-correlation     | Linear amplitude alignment, with lag            | Quick directional check; locating preferred lag.                          |
| **Oscillatory** | Welch coherence       | Linear (amp+phase) coupling per freq            | Identifying which band carries the coupling.                              |
|                 | PLV                   | Phase consistency at one frequency              | Standard "entrainment" measure when you know a dominant frequency.        |
|                 | wPLI                  | Phase consistency, less zero-lag bias           | Robust alternative to PLV; same interpretation.                           |
|                 | PAC (Tort / Canolty)  | Phase of slow signal × amplitude of fast        | Cross-frequency coupling within or across modalities.                     |
| **Information** | Mutual information    | Any (linear or non-linear) dependency           | When linear methods undersell coupling (e.g., on Hilbert envelopes).      |
|                 | Effective MI          | MI minus its surrogate-bias estimate            | When you want absolute MI values comparable across signal pairs.          |
|                 | Granger               | Predictive directional coupling (linear-Gauss.) | Asks "does x drive y?" once both signals are stationary.                  |
|                 | Transfer entropy      | Non-linear directional coupling                 | When the linear-Gaussian Granger assumption is too strong.                |
| **Complexity**  | Exponent matching     | DFA α / Hurst / FD agreement                    | Tests whether scaling structure (not amplitude) matches between signals.  |
|                 | Fluctuation matching  | DFA F(s) curve correlation                      | Stronger than scalar α: matches across multiple scales.                   |
|                 | Complexity coupling   | Coupling on α(t) traces                         | Time-resolved version — composes with any scalar coupling metric.         |

All windowed estimators share a consistent ``(times_s, value)`` return
shape so any of them can be plugged into the same surrogate-test +
cluster-permutation pipeline.

## Surrogate testing

Use `surrogate_test(metric_fn, x, y)` to convert any of the above metrics
into a per-pair p-value and z-score relative to a phase-shuffled null.
This is the recommended way to claim coupling for an individual subject
in a small-N design.

## Statistics

For repeated-measures designs (subject × condition):
1. Per-subject metric → ``friedman_with_posthoc`` for an omnibus + paired
   Wilcoxon comparisons.
2. Apply ``fdr_bh`` for multi-channel / multi-band comparisons.
3. Visualize with ``viz.sig_stars`` (returns ``***``, ``**``, ``*``, ``~``,
   or ``ns``).

For circular data (preferred phases):
- Within-subject: ``rayleigh_test`` on each subject's windowed phases.
- Group-level: ``rayleigh_test`` on subject-level circular means.

## Plotting style

Always start figure scripts with::

    from HNA.viz import use_paper_style, CONDITION_COLORS, save_figure
    use_paper_style()

`save_figure(fig, "out_name")` writes both PNG (300 dpi) and PDF.

## Guidelines for adding a new analysis

1. **Reuse, don't reimplement.** The coupling/, surrogates/, stats/ helpers
   are designed to be combined. If you find yourself writing a new MI or
   FDR routine, extend the existing one instead.
2. **Modality preprocessing belongs in a modality module** (`audio.py`,
   `eeg.py`, …), so the script can stay short and any new dataset can
   reuse the same preprocessing.
3. **Scripts in `scripts/` are thin CLI wrappers**, not the source of
   truth. The "what we do" lives in the toolbox; the "for which subjects /
   files" lives in the script.
4. **Outputs go in three places**:
   - Per-subject artifacts → `data/processed/sub-XX/...`
   - Group/intermediate results → `<repo>/results/<topic>/`
   - Figures → `<repo>/figures/<topic>/`  (with `<repo>/figures/report/`
     reserved for publication-ready ones).

## Datasets

The **dataset-specific** part of the codebase lives in
`config/subjects.json` (per-subject metadata: condition order, audio sync
times, trigger patches, status) and the I/O helpers in `utils.py`. To add
a new dataset, add a sibling config file and an ``io_<dataset>.py`` module
(or extend ``utils.py``) — the rest of the toolbox should not need to change.
