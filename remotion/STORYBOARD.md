# Animation storyboard — quantifying sea-wave dynamics from video

A Remotion-rendered explainer (~70 seconds, 1920×1080, 30 fps) that walks
through each method implemented in `HNA.modules.video`. The architecture is:

  1. A Python script (`scripts/render_animation_assets.py`) **pre-renders** all
     visual assets from a chosen input video into `remotion/public/` and
     `remotion/src/data/`.
  2. Remotion (`npx remotion preview` or `npx remotion render`) composes the
     pre-rendered assets into the final animation. Scenes are pure React/TSX,
     so Remotion never needs Python or OpenCV at render time.

## Pre-rendered asset list (produced by the Python step)

| asset                          | scene(s)            | format     |
|--------------------------------|---------------------|------------|
| `public/source.mp4`            | 1, 2, 3             | re-encoded H.264 of the input clip, square-cropped |
| `public/flow_overlay.mp4`      | 3                   | flow vectors (HSV color wheel) on each frame       |
| `public/frame_diff.mp4`        | 2                   | absolute frame difference, contrast-stretched      |
| `public/edges.mp4`             | 4                   | Canny edge map per frame                           |
| `public/timestack.png`         | 6                   | x-t image of the chosen pixel column               |
| `public/mode_1.png … mode_k.png` | 5                 | normalized SVD/DMD spatial modes                   |
| `src/data/signals.json`        | all                 | every 1-D signal at frame rate + fps, scalar summaries |
| `src/data/summary.json`        | 5, 6, 7, 8          | timestack peak Hz/period, modal freqs/energies, complexity dict |

## Scene plan

### Scene 0 — Intro (0:00–0:03 — 3 s)
Title card: **"Quantifying sea-wave dynamics from video"**
Subtitle: *"Six families of methods, each producing a 1-D signal that couples
with EEG / HRV / EMG."*
- Background: muted, slow zoom on a still frame from the source clip.

### Scene 1 — Global temporal envelope (0:03–0:11 — 8 s)
Left half: source video plays.
Right half: three rolling line plots stacked — `luminance`, `blue_mean`,
`frame_diff` — each scrolling at frame rate, time cursor synced with the video.
Caption: *"Per-frame luminance, blue-channel mean, and frame difference give
the 'video envelope' — directly analogous to the audio envelope used in
audio-EEG entrainment."*

### Scene 2 — Frame difference (0:11–0:16 — 5 s)
Side-by-side: source video | `frame_diff.mp4` (the temporal-derivative video).
The `frame_diff` time series scrolls underneath both.
Caption: *"|I(t) − I(t−1)| highlights motion energy."*

### Scene 3 — Optical flow (0:16–0:24 — 8 s)
Source video on the left, `flow_overlay.mp4` on the right (Farneback flow as
HSV: hue = direction, saturation/value = magnitude).
Below: rolling `flow_mag_mean` (orange) and `flow_curl_abs_mean` (cyan).
Caption: *"Dense Farneback flow → mean magnitude (bulk wave motion) and mean
|curl| (rotational/turbulent content). Both peak at the swell period."*

### Scene 4 — Spatial complexity (0:24–0:32 — 8 s)
Left: source. Right: `edges.mp4`.
Below the edge clip: three values updating per frame — `edge_density`,
`spatial_psd_slope` (radial 2D-FFT slope), `fractal_dim` (box-counting).
A small inset shows the radial PSD with the linear fit drawn live.
Caption: *"Per-frame spatial-FFT slope and box-counting fractal dimension
quantify foam / turbulence. Conceptually parallel to the EEG aperiodic 1/f
exponent."*

### Scene 5 — Spatio-temporal modal decomposition (0:32–0:40 — 8 s)
Top: a 1×k strip of the top-4 SVD/DMD spatial modes (mode_1.png … mode_4.png)
with mode index labeled.
Below: their temporal coefficients (`modal_1`…`modal_4`) scrolling, color-coded
to the modes above. Annotation calls out the top mode's frequency from
`summary.json` (e.g. *"mode 1 ≈ 0.23 Hz → 4.3 s period"*).
Caption: *"SVD/DMD decomposition of the (pixels × time) matrix — each mode is
an oscillatory pattern with its own frequency and energy."*

### Scene 6 — Timestack (oceanographic) (0:40–0:48 — 8 s)
Animation: a vertical red line marks the chosen column on the source video; as
the video plays, the timestack image grows column-by-column on the right.
At ~6 s, a 1D-FFT panel slides up showing the PSD with the peak frequency
labeled (`summary.timestack.dominant_freq_hz`).
Caption: *"Timestack: one pixel column over time → 1-D FFT → dominant wave
period (Argus-style coastal monitoring; Holman & Stanley)."*

### Scene 7 — Time-resolved complexity (0:48–0:58 — 10 s)
Top: `flow_mag_mean` scrolling. A translucent yellow rectangle slides along it
representing the analysis window (`win_sec=4 s`).
Below: as the window slides, four values pulse — `perm_entropy`,
`hjorth_complexity`, `higuchi_fd`, `spectral_entropy` — each tracing its own
rolling line so complexity itself becomes a signal.
Caption: *"Sliding-window permutation entropy, Hjorth, Higuchi FD and spectral
entropy — complexity as a time series, ready for cross-correlation with
brain / heart / muscle signals."*

### Scene 8 — Outro: coupling promise (0:58–1:08 — 10 s)
Stacked-overlay panel: the source video scaled small in the corner; six
1-D signals stacked vertically (`luminance`, `flow_mag_mean`,
`fractal_dim`, `modal_1`, `wc_flow_mag_mean__perm_entropy`, plus a cartoon
EEG/HRV/EMG trace) with bidirectional arrows between the two groups.
Caption: *"Every signal above is a drop-in input for `HNA.modules.coupling` —
windowed cross-correlation, coherence, PLV, wPLI, mutual information."*

## Project structure

```
remotion/
  STORYBOARD.md            (this file)
  package.json
  tsconfig.json
  remotion.config.ts
  src/
    index.ts
    Root.tsx               (registers compositions)
    Showcase.tsx           (master sequence chaining all scenes)
    scenes/
      Intro.tsx            (Scene 0)
      GlobalEnvelope.tsx   (Scene 1)   ← implemented in this scaffold
      FrameDiff.tsx        (Scene 2)   ← TODO
      OpticalFlow.tsx      (Scene 3)   ← implemented in this scaffold
      SpatialComplexity.tsx (Scene 4)  ← TODO
      Modal.tsx            (Scene 5)   ← TODO
      Timestack.tsx        (Scene 6)   ← TODO
      ComplexityWindowed.tsx (Scene 7) ← implemented in this scaffold
      Outro.tsx            (Scene 8)
    components/
      ScrollingLine.tsx    (rolling line plot synced to currentFrame)
      MethodTitle.tsx      (title + subtitle overlay)
      Caption.tsx          (lower-third caption)
      ModeTile.tsx         (single SVD-mode tile with label)
    data/
      signals.json         (produced by the Python script)
      summary.json         (produced by the Python script)
  public/
    source.mp4             (produced by the Python script)
    flow_overlay.mp4       (produced by the Python script)
    frame_diff.mp4         (produced by the Python script)
    edges.mp4              (produced by the Python script)
    timestack.png          (produced by the Python script)
    mode_1.png … mode_4.png (produced by the Python script)
```

## Run order

```bash
# 1. Pre-render assets from a source video
python scripts/render_animation_assets.py --video video_examples/waves.mp4 --out remotion

# 2. Install Remotion deps (one-time)
cd remotion
npm install

# 3. Preview interactively
npx remotion preview

# 4. Render the final mp4
npx remotion render Showcase out/showcase.mp4
```

## Why Remotion

- React/TSX scenes give us programmatic SVG/Canvas overlays synced *exactly*
  to the rolling `currentFrame` — much cleaner than matplotlib animation for
  the kind of explainer described above.
- All physics-bearing computation already lives in `HNA.modules.video`; the
  pre-render step is the only Python touch point. Remotion just composes.
- Output is deterministic and re-renderable without any video editing GUI.

If Node/Remotion isn't desired in this repo, the same storyboard renders
cleanly from matplotlib's `FuncAnimation` — every scene's visual state at
frame `t` is a pure function of the pre-rendered signals + the source video
+ the current frame index, so the same data structures port directly.
