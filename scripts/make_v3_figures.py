"""
Render the figures used by `report/main_v3.tex`.

The script is self-contained: it runs the analysis pipeline on the
four example clips, holds the resulting feature dictionaries in
memory, and from that single source produces every figure of the
report.

Inputs
------
Four mp4 clips named `example-01.mp4` ... `example-04.mp4`, by
default under `video_examples/`. Override with `--video-dir`.

Outputs (under `report/figures/`)
---------------------------------
v3_clips_overview.png      reference frame from each clip + 1-line tag
v3_raw_motion.png          luminance + flow_mag + frame_diff per clip
v3_multiscale_flow.png     temporal-stride flow per clip (4 panels)
v3_dmd_top_modes.png       top-2 DMD spatial modes per clip with f-Hz tag
v3_pixel_spectrum.png      per-pixel peak-frequency map per clip + mean spectra
v3_complexity.png          per-frame complexity metrics per clip
v3_fingerprint.png         z-scored heatmap re-ordered by feature family

Cost: ~ 1 -- 2 minutes per clip on CPU at the defaults below.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modalities.video import (  # noqa: E402
    quantify_video, extract_optical_flow_multiscale,
    extract_pixel_spectrum, extract_modal, _stack_video,
)

OUT_DIR = ROOT / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_NAMES = ["example-01", "example-02", "example-03", "example-04"]
CLIP_COLORS = {
    "example-01": "#1f77b4",
    "example-02": "#d97706",
    "example-03": "#2e7d32",
    "example-04": "#7e3ff2",
}
CLIP_TAG = {
    "example-01": "turbulent shallow water",
    "example-02": "slow deep swell",
    "example-03": "organised striped swell",
    "example-04": "fast wind chop",
}
FAMILY_COLORS = {
    "raw": "#9CC4E4",
    "oscillatory": "#F2C879",
    "complexity": "#C8A2C8",
}


# ---------------------------------------------------------------------------
# Data pipeline (in-memory; one pass per clip)
# ---------------------------------------------------------------------------

@dataclass
class ClipBundle:
    name: str
    fs: float
    duration_s: float
    signals: Dict[str, np.ndarray]
    flow_multiscale: Dict[str, np.ndarray]
    pixel_spectrum: object
    modal: object
    summary: Dict[str, float]


def build_bundles(video_dir: Path,
                  cap_seconds: float = 12.0) -> Dict[str, ClipBundle]:
    """Run quantify_video + multiscale flow + pixel spectrum on each clip
    and return a per-clip bundle. Limits each clip to `cap_seconds`."""
    bundles: Dict[str, ClipBundle] = {}
    for name in CLIP_NAMES:
        path = video_dir / f"{name}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"missing clip: {path}")
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        max_frames = int(round(cap_seconds * fps))
        print(f"[{name}] quantify_video (cap={cap_seconds:g}s)")
        feats = quantify_video(str(path),
                               target_long=128,
                               max_frames=max_frames,
                               include_flow=True,
                               include_spatial=True,
                               include_spatial_fft=True,
                               include_modal=True,
                               include_timestack=True,
                               include_complexity=False,
                               modal_target_long=96,
                               modal_k=4)
        print(f"[{name}] extract_optical_flow_multiscale")
        flow = extract_optical_flow_multiscale(str(path),
                                               target_long=128,
                                               max_frames=max_frames)
        print(f"[{name}] extract_pixel_spectrum")
        pix = extract_pixel_spectrum(str(path), fps=feats.fs,
                                     target_long=64,
                                     max_frames=max_frames)

        signals = {k: np.asarray(v) for k, v in feats.signals.items()}
        signals.update({k: np.asarray(v) for k, v in flow.items()})

        # cross-clip summary scalars
        summary = {
            "timestack_peak_hz": (feats.timestack.dominant_freq_hz
                                  if feats.timestack else float("nan")),
            "pixel_sync_index": float(pix.sync_index),
            "pixel_coherence_index": float(pix.coherence_index),
            "pixel_peak_freq_median": float(np.median(pix.peak_freq_map)),
        }
        # add common per-clip means
        for k in ("flow_mag_mean", "flow_curl_abs_mean",
                  "fractal_dim_grad", "lacunarity_grad",
                  "patch_entropy", "spatial_psd_slope",
                  "edge_density", "spatial_fft_anisotropy",
                  "spatial_fft_peak_k"):
            if k in signals:
                summary[f"{k}__mean"] = float(np.nanmean(signals[k]))
        # multi-scale flow summaries
        for dt in (1, 3, 9):
            key = f"flow_dt{dt}_full_mag_mean"
            if key in flow:
                summary[f"flow_dt{dt}_mag__mean"] = float(np.nanmean(flow[key]))
        # luminance complexity summaries (DFA + 1/f slope)
        from HNA.modalities.video import complexity_summary
        if "luminance" in signals:
            cs = complexity_summary(signals["luminance"], fs=feats.fs)
            summary["luminance_dfa"] = cs.get("dfa_alpha", float("nan"))
            summary["luminance_psd_slope"] = cs.get("psd_slope", float("nan"))

        bundles[name] = ClipBundle(
            name=name, fs=float(feats.fs),
            duration_s=float(feats.meta.duration_s),
            signals=signals, flow_multiscale=flow,
            pixel_spectrum=pix, modal=feats.modal,
            summary=summary,
        )
    return bundles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_frame(video_dir: Path, name: str,
                    t_s: float = 1.5) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_dir / f"{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_s * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame from {name}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    if s > 1e-9:
        x = x / s
    return x


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_clips_overview(video_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, name in zip(axes, CLIP_NAMES):
        img = reference_frame(video_dir, name, t_s=1.5)
        h, w = img.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2; x0 = (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        ax.imshow(img); ax.set_axis_off()
        ax.set_title(name, color=CLIP_COLORS[name],
                     fontsize=12, fontweight="bold", pad=4)
        ax.text(0.5, -0.06, CLIP_TAG[name], transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, color="#374151")
    fig.suptitle("Four sea states used as running examples",
                 fontsize=12, fontweight="bold", color="#0F5C8A", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v3_clips_overview.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_clips_overview.png")


def fig_raw_motion(bundles: Dict[str, ClipBundle]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13, 7.0))
    for ri, name in enumerate(CLIP_NAMES):
        b = bundles[name]
        t = np.arange(len(b.signals["luminance"])) / b.fs
        c = CLIP_COLORS[name]
        traces = [
            ("luminance",     b.signals["luminance"],
             "raw  -  luminance"),
            ("flow_mag_mean", b.signals.get("flow_mag_mean",
                                            np.zeros_like(t)),
             "raw  -  optical-flow magnitude"),
            ("frame_diff",    b.signals.get("frame_diff",
                                            np.zeros_like(t)),
             "raw  -  frame difference"),
        ]
        for ci, (key, sig, title) in enumerate(traces):
            ax = axes[ri, ci]
            ax.plot(t, _normalise(sig), color=c, linewidth=1.0,
                    alpha=0.95)
            ax.axhline(0, color="#cccccc", linewidth=0.5)
            ax.set_xlim(t.min(), t.max())
            ax.tick_params(axis="both", labelsize=8)
            if ri == 0:
                ax.set_title(title, fontsize=10, color="#0F5C8A",
                             fontweight="bold", pad=4)
            if ci == 0:
                ax.set_ylabel(name, color=c, fontsize=10,
                              fontweight="bold", rotation=0,
                              labelpad=42, va="center")
            if ri == 3:
                ax.set_xlabel("time (s)", fontsize=9)
            ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Raw envelopes per clip "
        "(z-scored along time; same axis scale across clips)",
        fontsize=11.5, color="#0F5C8A", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "v3_raw_motion.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_raw_motion.png")


def fig_multiscale_flow(bundles: Dict[str, ClipBundle]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.6))
    for ax, name in zip(axes.flat, CLIP_NAMES):
        b = bundles[name]
        t = np.arange(len(b.signals["luminance"])) / b.fs
        c = CLIP_COLORS[name]
        for dt, alpha, lw in ((1, 0.4, 1.0),
                              (3, 0.65, 1.0),
                              (9, 1.0, 1.4)):
            key = f"flow_dt{dt}_full_mag_mean"
            if key not in b.flow_multiscale:
                continue
            ax.plot(t[:len(b.flow_multiscale[key])],
                    b.flow_multiscale[key],
                    color=c, alpha=alpha, linewidth=lw,
                    label=fr"$\Delta t={dt}$")
        ax.set_xlim(t.min(), t.max())
        ax.set_title(f"{name} -- {CLIP_TAG[name]}",
                     color=c, fontsize=10.5, fontweight="bold", pad=2)
        ax.set_ylabel("mean |flow|", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9)
    fig.suptitle(
        "Multi-scale optical flow at three temporal strides",
        fontsize=11.5, color="#0F5C8A", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "v3_multiscale_flow.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_multiscale_flow.png")


def _compute_top_modes(video_dir: Path, name: str,
                       k_top: int = 2) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                Tuple[int, int]]:
    """Re-fit DMD just to grab the spatial mode magnitudes."""
    path = str(video_dir / f"{name}.mp4")
    fps_meta = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or 30.0
    res = extract_modal(path, fps=fps_meta, target_long=96,
                        max_frames=300, k=4)
    X, shape = _stack_video(path, target_long=96, stride=1,
                            max_frames=300)
    try:
        from pydmd import DMD as PyDMD
    except ImportError:
        # SVD fallback
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        modes = (U[:, :k_top] * S[:k_top]).T
        modes = np.abs(modes).reshape(k_top, *shape)
        return modes, res.frequencies_hz[:k_top], shape

    dmd = PyDMD(svd_rank=8).fit(X)
    eigs = np.asarray(dmd.eigs)
    log_lam = np.log(eigs + 1e-30) * fps_meta
    freqs_full = np.abs(log_lam.imag) / (2 * np.pi)
    modes_full = np.asarray(dmd.modes)
    dyn_full = np.asarray(dmd.dynamics)
    amps = (np.linalg.norm(modes_full, axis=0)
            * np.abs(dyn_full).mean(axis=1))

    keys = np.round(freqs_full, 4)
    seen: Dict[float, int] = {}
    for i, key in enumerate(keys):
        cur = seen.get(key)
        if cur is None or amps[i] > amps[cur]:
            seen[key] = i
    keep = np.array(sorted(seen.values()), dtype=int)
    freqs_dedup = freqs_full[keep]
    amps_dedup = amps[keep]
    order = np.argsort(-amps_dedup)[:k_top]

    modes_kept = modes_full[:, keep][:, order]
    freqs_kept = freqs_dedup[order]

    out = np.zeros((k_top, shape[0], shape[1]), dtype=np.float32)
    for i in range(min(k_top, modes_kept.shape[1])):
        out[i] = np.abs(modes_kept[:, i]).reshape(shape).astype(np.float32)
    if modes_kept.shape[1] < k_top:
        for i in range(modes_kept.shape[1], k_top):
            out[i] = np.nan
        freqs_padded = np.full(k_top, np.nan)
        freqs_padded[:modes_kept.shape[1]] = freqs_kept
        return out, freqs_padded, shape
    return out, freqs_kept, shape


def fig_dmd_top_modes(video_dir: Path) -> None:
    K = 2
    fig, axes = plt.subplots(K, 4, figsize=(13, 5.6))
    for ci, name in enumerate(CLIP_NAMES):
        modes, freqs, _ = _compute_top_modes(video_dir, name, k_top=K)
        for ri in range(K):
            ax = axes[ri, ci]
            mode = modes[ri]
            if np.all(np.isnan(mode)):
                ax.text(0.5, 0.5, "no mode", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11,
                        color="#888888")
                ax.set_axis_off(); continue
            mode = mode - np.nanmin(mode)
            mode = mode / (np.nanmax(mode) + 1e-12)
            ax.imshow(mode, cmap="magma", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(CLIP_COLORS[name])
                spine.set_linewidth(2)
            f = freqs[ri]
            label = (f"$f_{{{ri+1}}}$ = {f:.3f} Hz"
                     if np.isfinite(f) else "no f")
            ax.set_title(label, fontsize=10, color="#0F5C8A", pad=2)
            if ri == 0:
                ax.text(0.5, 1.16, f"{name}",
                        transform=ax.transAxes, ha="center",
                        fontsize=11, fontweight="bold",
                        color=CLIP_COLORS[name])
            if ci == 0:
                ax.set_ylabel(f"top-{ri+1} mode", fontsize=10,
                              labelpad=6)
    fig.suptitle(
        "Top-2 DMD spatial modes per clip "
        "(magnitude $|\\varphi_k|$ on the down-sampled grid)",
        fontsize=11.5, color="#0F5C8A", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_DIR / "v3_dmd_top_modes.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_dmd_top_modes.png")


def fig_pixel_spectrum(bundles: Dict[str, ClipBundle]) -> None:
    fig = plt.figure(figsize=(13, 6.4))
    gs = mpl.gridspec.GridSpec(2, 4, figure=fig,
                               height_ratios=[1.05, 0.95],
                               hspace=0.36, wspace=0.28)
    vmax = 2.0
    im = None
    for ci, name in enumerate(CLIP_NAMES):
        b = bundles[name]
        d = b.pixel_spectrum
        ax_map = fig.add_subplot(gs[0, ci])
        im = ax_map.imshow(d.peak_freq_map, cmap="turbo",
                           vmin=0, vmax=vmax, aspect="equal")
        ax_map.set_xticks([]); ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_edgecolor(CLIP_COLORS[name])
            spine.set_linewidth(2)
        ax_map.set_title(name, fontsize=11, color=CLIP_COLORS[name],
                         fontweight="bold", pad=4)
        ax_map.text(0.5, -0.06, "peak-frequency map",
                    transform=ax_map.transAxes, ha="center", va="top",
                    fontsize=8.5, color="#374151")

        ax_spec = fig.add_subplot(gs[1, ci])
        f = d.freqs; P = d.mean_spectrum
        ax_spec.semilogy(f, P + 1e-9, color=CLIP_COLORS[name],
                         linewidth=1.2)
        ax_spec.set_xlim(0, 2.0)
        ax_spec.set_xlabel("frequency (Hz)", fontsize=9)
        if ci == 0:
            ax_spec.set_ylabel("mean P(f)", fontsize=9)
        ax_spec.tick_params(axis="both", labelsize=8)
        ax_spec.grid(True, alpha=0.25)
        ts = b.summary.get("timestack_peak_hz", float("nan"))
        if np.isfinite(ts):
            ax_spec.axvline(ts, color="#666666", linestyle="--",
                            linewidth=0.9)
            ax_spec.text(ts, ax_spec.get_ylim()[1] * 0.5,
                         f" timestack {ts:.2f} Hz",
                         fontsize=7.5, color="#666666",
                         rotation=90, va="center")

    cbar_ax = fig.add_axes([0.92, 0.55, 0.012, 0.36])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("peak f (Hz)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Per-pixel temporal FFT  -  every pixel reports its dominant frequency",
        fontsize=11.5, color="#0F5C8A", fontweight="bold", y=0.995)
    fig.savefig(OUT_DIR / "v3_pixel_spectrum.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_pixel_spectrum.png")


CPLX_METRICS = [
    ("edge_density__mean",      "edge density"),
    ("fractal_dim_grad__mean",  "fractal D (grad)"),
    ("lacunarity_grad__mean",   "lacunarity (grad)"),
    ("patch_entropy__mean",     "patch entropy"),
    ("spatial_psd_slope__mean", "spatial 1/f slope"),
    ("luminance_dfa",           "luminance DFA"),
]


def fig_complexity(bundles: Dict[str, ClipBundle]) -> None:
    n_metrics = len(CPLX_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(13, 3.2))
    x = np.arange(len(CLIP_NAMES))
    for ax, (key, label) in zip(axes, CPLX_METRICS):
        vals = [bundles[n].summary.get(key, np.nan) for n in CLIP_NAMES]
        bar_colors = [CLIP_COLORS[n] for n in CLIP_NAMES]
        ax.bar(x, vals, color=bar_colors, edgecolor="#222", linewidth=0.6)
        ax.set_title(label, fontsize=10, color="#0F5C8A", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("example-", "ex-") for n in CLIP_NAMES],
                           rotation=35, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        "Per-frame complexity battery, mean over clip",
        fontsize=11.5, color="#0F5C8A", fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "v3_complexity.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_complexity.png")


FINGERPRINT_ROWS: List[Tuple[str, str, str]] = [
    ("flow_mag_mean__mean",          "flow magnitude",            "raw"),
    ("flow_curl_abs_mean__mean",     "|curl|  (turbulence)",      "raw"),
    ("flow_dt9_mag__mean",           "Delta-t=9 flow (slow)",     "raw"),
    ("timestack_peak_hz",            "timestack peak (Hz)",       "oscillatory"),
    ("pixel_sync_index",             "pixel-spectrum sync",       "oscillatory"),
    ("pixel_coherence_index",        "pixel-spectrum coherence",  "oscillatory"),
    ("spatial_fft_anisotropy__mean", "2D-FFT anisotropy",         "oscillatory"),
    ("spatial_fft_peak_k__mean",     "2D-FFT peak wavenumber",    "oscillatory"),
    ("fractal_dim_grad__mean",       "fractal D (grad)",          "complexity"),
    ("lacunarity_grad__mean",        "lacunarity (grad)",         "complexity"),
    ("patch_entropy__mean",          "patch entropy",             "complexity"),
    ("edge_density__mean",           "edge density",              "complexity"),
    ("spatial_psd_slope__mean",      "spatial 1/f slope",         "complexity"),
    ("luminance_dfa",                "luminance DFA",             "complexity"),
    ("luminance_psd_slope",          "luminance 1/f slope",       "complexity"),
]


def fig_fingerprint(bundles: Dict[str, ClipBundle]) -> None:
    rows: List[List[float]] = []
    labels: List[str] = []
    families: List[str] = []
    for key, label, fam in FINGERPRINT_ROWS:
        vals = [bundles[n].summary.get(key, np.nan) for n in CLIP_NAMES]
        rows.append([float(v) if v is not None else np.nan for v in vals])
        labels.append(label); families.append(fam)
    M = np.array(rows, dtype=float)

    Z = np.zeros_like(M)
    for i, row in enumerate(M):
        finite = row[np.isfinite(row)]
        if finite.size:
            mu = finite.mean(); sd = finite.std() + 1e-9
            Z[i] = (row - mu) / sd
        else:
            Z[i] = np.nan

    fam_groups: List[Tuple[str, int, int]] = []
    cur_fam = families[0]; start = 0
    for i, f in enumerate(families):
        if f != cur_fam:
            fam_groups.append((cur_fam, start, i))
            cur_fam = f; start = i
    fam_groups.append((cur_fam, start, len(families)))

    sep = np.full((1, Z.shape[1]), np.nan)
    blocks: List[np.ndarray] = []
    block_labels: List[str] = []
    block_families: List[str] = []
    for k, (fam, s_, e_) in enumerate(fam_groups):
        if k:
            blocks.append(sep); block_labels.append(""); block_families.append("sep")
        blocks.append(Z[s_:e_])
        block_labels.extend(labels[s_:e_])
        block_families.extend([fam] * (e_ - s_))
    Zg = np.vstack(blocks)
    n_rows, n_cols = Zg.shape

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    cmap = mpl.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(np.ma.masked_invalid(Zg), cmap=cmap,
                   vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{n}\n({CLIP_TAG[n]})" for n in CLIP_NAMES],
                       rotation=0, ha="center", fontsize=9.5)
    for j, n in enumerate(CLIP_NAMES):
        ax.get_xticklabels()[j].set_color(CLIP_COLORS[n])
        ax.get_xticklabels()[j].set_fontweight("bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(block_labels, fontsize=9)

    for j in range(n_cols):
        for i in range(n_rows):
            v = Zg[i, j]
            if np.isfinite(v):
                src_i = sum(1 for fam in block_families[:i] if fam != "sep")
                raw_v = M[src_i, j]
                if np.isfinite(raw_v):
                    txt = (f"{raw_v:.2f}" if abs(raw_v) >= 0.01
                           else f"{raw_v:.2e}")
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8.0, color="#202020")

    for i, fam in enumerate(block_families):
        if fam == "sep":
            continue
        rect = mpl.patches.Rectangle(
            (-0.55, i - 0.5), 0.18, 1.0,
            facecolor=FAMILY_COLORS[fam], edgecolor="none",
            transform=ax.transData, clip_on=False)
        ax.add_patch(rect)

    for fam, color in FAMILY_COLORS.items():
        idxs = [i for i, f in enumerate(block_families) if f == fam]
        if not idxs:
            continue
        mid = (min(idxs) + max(idxs)) / 2.0
        ax.text(-1.6, mid, fam, ha="right", va="center",
                fontsize=11, color=color, fontweight="bold",
                rotation=90)

    ax.tick_params(axis="x", which="both", length=0)
    ax.set_title("Cross-clip fingerprint, organised by feature family",
                 fontsize=12, color="#0F5C8A", fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
    cbar.set_label("row z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "v3_fingerprint.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("  v3_fingerprint.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", type=Path,
                   default=ROOT / "video_examples",
                   help="directory containing example-01..04.mp4")
    p.add_argument("--cap-seconds", type=float, default=12.0,
                   help="cap each clip to this many seconds (default 12)")
    args = p.parse_args()

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    print(f"writing v3 figures to {OUT_DIR} ...")
    print(f"reading clips from {args.video_dir}")

    bundles = build_bundles(args.video_dir, cap_seconds=args.cap_seconds)
    fig_clips_overview(args.video_dir)
    fig_raw_motion(bundles)
    fig_multiscale_flow(bundles)
    fig_dmd_top_modes(args.video_dir)
    fig_pixel_spectrum(bundles)
    fig_complexity(bundles)
    fig_fingerprint(bundles)
    print("done.")


if __name__ == "__main__":
    main()
