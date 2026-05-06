# HNA toolbox (`HNA.modules`)

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
HNA/modules/
├── dsp.py           Generic signal-processing primitives (filters, envelope,
│                    NaN handling, resample, z-score, butter SOS).
├── coupling.py      Coupling metrics on any 2 1-D signals:
│                      • windowed cross-correlation  (time-domain amplitude)
│                      • Welch coherence + windowed band-avg coherence  (spectral)
│                      • PLV  /  windowed PLV       (phase consistency)
│                      • wPLI /  windowed wPLI      (debiased phase consistency)
│                    Plus shared plot helpers (plot_signal_alignment_validation,
│                    plot_coupling_over_time, plot_coherence_results).
├── surrogates.py    Phase-shuffle and time-shift surrogate generators plus
│                    a generic harness ``surrogate_test(metric_fn, x, y)`` that
│                    returns observed value, null distribution, p, and z-score.
├── stats.py         Linear (Fisher z), multiple-comparisons (BH-FDR), repeated-
│                    measures (Friedman + Wilcoxon post-hoc), circular
│                    (Rayleigh, mean, R), and slope-test helpers.
├── viz.py           Paper-ready matplotlib style + canonical condition palette
│                    + significance helpers (sig_stars, fmt_p, save_figure).
├── audio.py         MODALITY: audio envelope decomposition into a band-
│                    organized set of columns (broad, swell, HRV-LF/HF,
│                    splash, delta, theta, alpha, low/high beta, gamma1).
├── eeg.py           MODALITY: EEG bandpass + PSD-band features + entropy.
└── utils.py         I/O loaders + alignment + condition-segment extraction
                     (the data-layout-aware bits, currently dataset-specific).
```

When the project grows to include more modalities (ECG/HRV cleaning,
respiration cleaning, video features, etc.), the recommended pattern is
to add another modality module next to `audio.py` / `eeg.py`. The
**coupling**, **stats**, **surrogates**, and **viz** modules should stay
modality-agnostic.

## Coupling methods at a glance

| Method               | What it measures                       | When to use                                                                 |
|----------------------|----------------------------------------|------------------------------------------------------------------------------|
| Cross-correlation    | Linear amplitude alignment, with lag   | Quick directional check; locating preferred lag.                             |
| Welch coherence      | Linear (amp+phase) coupling per freq   | Identifying which band carries the coupling.                                 |
| PLV                  | Phase consistency at one frequency     | Standard "entrainment" measure when you know a dominant frequency.           |
| wPLI                 | Phase consistency, less zero-lag bias  | Robust alternative to PLV; same interpretation.                              |
| Mutual information   | Any (linear or non-linear) dependency  | When linear methods undersell coupling (e.g., on Hilbert envelopes).         |

All five are computed on the same windowing scheme (default 120 s window,
10 s step) so they can be compared directly.

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

    from HNA.modules.viz import use_paper_style, CONDITION_COLORS, save_figure
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
