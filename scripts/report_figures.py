"""
Generate figures used by the LaTeX report (`report/main.tex`).

Produces under `report/figures/`:

  overlays_montage.png   single still frame of source / frame_diff / edges /
                         flow_arrows / patch_heatmap, side-by-side.
  signals_<name>.png     diagnostic signal plot for each example video (re-uses
                         the existing per-video PNGs if available).
  modal_modes.png        the four SVD spatial modes laid out 2x2.
  modal_freqs.png        bar chart of dominant modal frequencies for the
                         three example videos.
  complexity_grid.png    heatmap of complexity measures across signals (one
                         representative video).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.video import quantify_video  # noqa: E402

REPORT = ROOT / "report"
FIGS = REPORT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

PUBLIC = ROOT / "remotion" / "public"


def _grab_frame(path: Path, t_s: float = 1.5) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_s * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def overlays_montage() -> None:
    panels = [
        ("source.mp4",         "source"),
        ("frame_diff.mp4",     r"$|I(t)-I(t-1)|$"),
        ("edges.mp4",          "Canny edges"),
        ("flow_arrows.mp4",    "optical flow (Farneback)"),
        ("patch_heatmap.mp4",  "patch entropy heatmap"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4.4))
    for ax, (fname, label) in zip(axes, panels):
        p = PUBLIC / fname
        if not p.exists():
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"missing\n{fname}", ha="center", va="center")
            continue
        img = _grab_frame(p, t_s=2.0)
        ax.imshow(img); ax.set_title(label, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-method overlays from `scripts/render_animation_assets.py`",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS / "overlays_montage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def modal_modes() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for k, ax in enumerate(axes, start=1):
        p = PUBLIC / f"mode_{k}.png"
        if p.exists():
            ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        ax.set_title(f"mode {k}", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Top-4 SVD/POD spatial modes of the wave video", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS / "modal_modes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def modal_freqs_bar() -> None:
    """Compare top-4 DMD modal frequencies across the three example videos."""
    examples = ["waves", "rapid_waves", "generated_waves"]
    rows = []
    for n in examples:
        path = ROOT.parent.parent.parent / "video_examples" / f"{n}.mp4"
        if not path.exists():
            continue
        feats = quantify_video(str(path), target_long=128, modal_k=4,
                               include_complexity=False,
                               include_windowed_complexity=False,
                               include_timestack=False)
        if feats.modal is not None:
            rows.append((n, feats.modal.frequencies_hz, feats.modal.energies))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(4)
    width = 0.25
    for i, (name, freqs, energies) in enumerate(rows):
        ax.bar(x + (i - 1) * width, freqs, width, label=name)
    ax.set_xticks(x); ax.set_xticklabels([f"mode {k}" for k in range(1, 5)])
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Dominant DMD mode frequencies (top-4 by energy)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "modal_freqs.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def multiscale_flow_panel() -> None:
    """Plot multi-scale optical flow signals on waves.mp4."""
    from HNA.modules.video import extract_optical_flow_multiscale
    path = ROOT.parent.parent.parent / "video_examples" / "waves.mp4"
    if not path.exists():
        return
    sigs = extract_optical_flow_multiscale(str(path),
                                           target_long=192, max_frames=192,
                                           temporal_strides=(1, 3, 9),
                                           spatial_lowpass_sigma_px=6.0)
    fps = 24.0
    n = len(next(iter(sigs.values())))
    t = np.arange(n) / fps

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.0), sharex=True)

    ax = axes[0]
    for dt, color in zip((1, 3, 9), ("#0F5C8A", "#2D8FBE", "#7ABDDC")):
        v = sigs[f"flow_dt{dt}_full_mag_mean"]
        ax.plot(t, v, lw=1.5, color=color, label=f"Δt = {dt} frames")
    ax.set_ylabel("mean |flow|")
    ax.set_title("Temporal-stride decomposition (full flow field)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for scale, color in zip(("large", "small"),
                            ("#0F5C8A", "#C2410C")):
        v = sigs[f"flow_dt3_{scale}_mag_mean"]
        ax.plot(t, v, lw=1.5, color=color,
                label=f"{scale}-scale (sigma=6 px {'low-pass' if scale=='large' else 'residual'})")
    ax.set_ylabel("mean |flow|")
    ax.set_xlabel("time (s)")
    ax.set_title("Spatial decomposition of the flow field, Δt = 3")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-scale optical flow on waves.mp4", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "multiscale_flow.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def lacunarity_panel() -> None:
    """Plot fractal_dim_grad and lacunarity time series for the three clips."""
    from HNA.modules.video import extract_spatial_complexity
    examples = ["waves", "rapid_waves", "generated_waves"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.0), sharex=False)
    rows_fd = []
    rows_lac = []
    for name in examples:
        path = ROOT.parent.parent.parent / "video_examples" / f"{name}.mp4"
        if not path.exists():
            continue
        s = extract_spatial_complexity(str(path), target_long=192,
                                       max_frames=192)
        rows_fd.append((name, s["fractal_dim_grad"]))
        rows_lac.append((name, s["lacunarity_grad"]))

    ax = axes[0]
    for name, v in rows_fd:
        v = np.asarray(v); v = v[np.isfinite(v)]
        ax.plot(v, lw=1.4, label=f"{name}  μ={v.mean():.3f}")
    ax.set_ylabel("fractal_dim_grad")
    ax.set_title("Box-counting fractal dim on the binarised gradient image")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, v in rows_lac:
        v = np.asarray(v); v = v[np.isfinite(v)]
        ax.plot(v, lw=1.4, label=f"{name}  μ={v.mean():.3f}")
    ax.set_ylabel("lacunarity_grad")
    ax.set_xlabel("frame")
    ax.set_title("Lacunarity (Plotnick/Allgood) on the same gradient binarisation")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    fig.suptitle("Per-frame fractality + lacunarity time series", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "lacunarity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def complexity_grid() -> None:
    """Heatmap of complexity measures across signals for one example video."""
    path = ROOT.parent.parent.parent / "video_examples" / "waves.mp4"
    if not path.exists():
        return
    feats = quantify_video(str(path), target_long=160,
                           include_windowed_complexity=False,
                           modal_k=2, max_frames=192)
    keys = ["luminance", "frame_diff", "flow_mag_mean", "flow_curl_abs_mean",
            "edge_density", "spatial_psd_slope", "fractal_dim",
            "patch_entropy"]
    measures = ["perm_entropy", "sample_entropy", "spectral_entropy",
                "svd_entropy", "higuchi_fd", "katz_fd",
                "hjorth_complexity", "dfa_alpha", "hurst",
                "lz_complexity", "psd_slope"]
    M = np.full((len(keys), len(measures)), np.nan)
    for i, k in enumerate(keys):
        if k not in feats.complexity:
            continue
        for j, m in enumerate(measures):
            v = feats.complexity[k].get(m, np.nan)
            M[i, j] = v if v == v else np.nan

    # Robust per-column z-score for visual comparison
    Z = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        mu = np.nanmean(col); sd = np.nanstd(col) + 1e-9
        Z[:, j] = (col - mu) / sd

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    ax.set_xticks(range(len(measures))); ax.set_xticklabels(measures, rotation=40, ha="right")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(Z[i, j]) < 1.4 else "white")
    fig.colorbar(im, ax=ax, label="column z-score")
    ax.set_title("Complexity measures × signals  (example: waves.mp4)")
    fig.tight_layout()
    fig.savefig(FIGS / "complexity_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def pixel_spectrum_panel() -> None:
    """
    Visualise the per-pixel temporal FFT on the natural-waves clip:
    (a) peak frequency map, (b) low-band power, (c) mid-band power,
    (d) spectral entropy map, plus the mean-across-pixels PSD curve with
    the dominant-frequency dotted line.
    """
    from HNA.modules.video import extract_pixel_spectrum
    path = ROOT.parent.parent.parent / "video_examples" / "waves.mp4"
    if not path.exists():
        return
    ps = extract_pixel_spectrum(str(path), fps=24.0,
                                target_long=96, max_frames=192)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))

    im0 = axes[0].imshow(ps.peak_freq_map, cmap="turbo",
                         vmin=0, vmax=min(2.0, ps.freqs[-1]))
    axes[0].set_title("peak frequency [Hz]"); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.log1p(ps.band_power_maps["low"]),
                         cmap="magma")
    axes[1].set_title("log(1 + low-band power)\n0.05–0.25 Hz")
    axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(np.log1p(ps.band_power_maps["mid"]),
                         cmap="magma")
    axes[2].set_title("log(1 + mid-band power)\n0.25–0.5 Hz")
    axes[2].axis("off"); plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(ps.spectral_entropy_map, cmap="cividis",
                         vmin=0, vmax=1)
    axes[3].set_title("spectral entropy\n(per-pixel, normalized)")
    axes[3].axis("off"); plt.colorbar(im3, ax=axes[3], fraction=0.046)

    ax = axes[4]
    f = ps.freqs[1:]; P = ps.mean_spectrum[1:]
    ax.semilogy(f, P, color="#0F5C8A", lw=2)
    pf = float(f[np.argmax(P)])
    ax.axvline(pf, color="#C2410C", ls="--",
               label=f"peak {pf:.3f} Hz")
    ax.set_xlim(0, min(2.0, f[-1]))
    ax.set_xlabel("Hz")
    ax.set_ylabel("mean power across pixels")
    ax.set_title("mean spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-pixel temporal FFT — sync_index={ps.sync_index:.3f}, "
                 f"coherence={ps.coherence_index:.3f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGS / "pixel_spectrum.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def spatial_fft_signals_panel() -> None:
    """Show the three new per-frame 2D-FFT signals across the example clips."""
    from HNA.modules.video import extract_spatial_fft_signals
    examples = ["waves", "rapid_waves", "generated_waves"]
    sigs = {}
    for n in examples:
        p = ROOT.parent.parent.parent / "video_examples" / f"{n}.mp4"
        if p.exists():
            sigs[n] = extract_spatial_fft_signals(str(p), target_long=160,
                                                  max_frames=192)
    if not sigs:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True)
    keys = ["spatial_fft_peak_k", "spatial_fft_anisotropy", "spatial_fft_orientation"]
    titles = ["peak radial wavenumber (normalized)",
              "log(horizontal-stripe / vertical-stripe power)",
              "dominant orientation (radians, [0, π))"]
    for ax, k, t in zip(axes, keys, titles):
        for n, d in sigs.items():
            ax.plot(d[k], lw=1.4, label=n, alpha=0.85)
        ax.set_ylabel(k.replace("spatial_fft_", ""))
        ax.set_title(t, fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(FIGS / "spatial_fft_signals.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Rendering report figures into", FIGS)
    overlays_montage()
    modal_modes()
    modal_freqs_bar()
    complexity_grid()
    pixel_spectrum_panel()
    spatial_fft_signals_panel()
    multiscale_flow_panel()
    lacunarity_panel()
    print("Done.")
