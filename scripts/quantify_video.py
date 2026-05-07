"""
Demo: extract oscillation & complexity features from one or more videos and
save them to disk. The 1-D signals are saved as a CSV (one row per frame, one
column per feature) and a small JSON of summary scalars.

Example
-------
    python scripts/quantify_video.py \
        --videos video_examples/waves.mp4 video_examples/rapid_waves.mp4 \
        --out results/video_features --plot

The CSV produced for each video can be fed directly into the coupling tools
(`HNA.modules.coupling`) -- pass `fs = video_fps` (saved in the JSON sidecar).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.video import quantify_video  # noqa: E402


def _to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    return o


def run_one(path: str, out_dir: Path, target_long: int, stride: int,
            max_seconds: float | None, modal_k: int, do_plot: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(path).stem

    max_frames = None
    if max_seconds is not None:
        # we need fps to convert; quantify_video also accepts max_frames
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        max_frames = int(round(max_seconds * fps))

    print(f"\n=== {path} ===")
    feats = quantify_video(path,
                           target_long=target_long,
                           stride=stride,
                           max_frames=max_frames,
                           modal_k=modal_k)
    print(f"  fps={feats.meta.fps:.2f}  duration={feats.meta.duration_s:.2f}s "
          f"frames={feats.meta.n_frames}  fs(used)={feats.fs:.2f} Hz")
    for n in feats.notes:
        print(f"  note: {n}")

    df = feats.as_dataframe()
    csv_path = out_dir / f"{name}_signals.csv"
    df.to_csv(csv_path, index=False)
    print(f"  signals -> {csv_path}  shape={df.shape}")

    summary = {
        "path": path,
        "fps": feats.meta.fps,
        "fs_used": feats.fs,
        "n_frames_used": int(len(df)),
        "duration_s": feats.meta.duration_s,
        "complexity": feats.complexity,
        "timestack": None,
        "modal": None,
    }
    if feats.timestack is not None:
        summary["timestack"] = {
            "dominant_freq_hz": feats.timestack.dominant_freq_hz,
            "dominant_period_s": feats.timestack.dominant_period_s,
            "column_index": feats.timestack.column_index,
        }
    if feats.modal is not None:
        summary["modal"] = {
            "method": feats.modal.method,
            "frequencies_hz": feats.modal.frequencies_hz,
            "growth_rates": feats.modal.growth_rates,
            "energies": feats.modal.energies,
        }
    json_path = out_dir / f"{name}_summary.json"
    json_path.write_text(json.dumps(_to_jsonable(summary), indent=2))
    print(f"  summary -> {json_path}")

    if do_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not available -> skipping plots")
            return
        keys_show = [
            "luminance", "frame_diff",
            "flow_mag_mean", "flow_curl_abs_mean",
            "edge_density", "spatial_psd_slope",
        ]
        keys_show = [k for k in keys_show if k in feats.signals]
        t = np.arange(len(df)) / feats.fs
        fig, axes = plt.subplots(len(keys_show), 1, figsize=(10, 1.8 * len(keys_show)),
                                 sharex=True)
        if len(keys_show) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys_show):
            ax.plot(t, feats.signals[k], lw=1.0)
            ax.set_ylabel(k, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{name} — video-derived signals (fs={feats.fs:.1f} Hz)")
        fig.tight_layout()
        png_path = out_dir / f"{name}_signals.png"
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"  plot    -> {png_path}")

        if feats.timestack is not None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].imshow(feats.timestack.timestack_image.T, aspect="auto",
                           cmap="gray", origin="lower",
                           extent=[0, len(df) / feats.fs, 0,
                                   feats.timestack.timestack_image.shape[1]])
            axes[0].set_title("timestack (column × time)")
            axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("row")
            axes[1].semilogy(feats.timestack.psd_freqs, feats.timestack.psd_power)
            axes[1].axvline(feats.timestack.dominant_freq_hz, color="r", ls="--",
                            label=f"peak {feats.timestack.dominant_freq_hz:.3f} Hz "
                                  f"({feats.timestack.dominant_period_s:.2f}s)")
            axes[1].set_xlim(0, min(2.0, feats.fs / 2))
            axes[1].set_xlabel("Hz"); axes[1].set_ylabel("PSD")
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
            fig.suptitle(f"{name} — timestack analysis")
            fig.tight_layout()
            ts_path = out_dir / f"{name}_timestack.png"
            fig.savefig(ts_path, dpi=130)
            plt.close(fig)
            print(f"  timestack plot -> {ts_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Quantify oscillation & complexity from videos.")
    p.add_argument("--videos", nargs="+", required=True, help="Paths to video files.")
    p.add_argument("--out", default="results/video_features", help="Output directory.")
    p.add_argument("--target-long", type=int, default=192,
                   help="Resize the long axis of frames to this many pixels (default 192).")
    p.add_argument("--stride", type=int, default=1,
                   help="Decode every Nth frame (default 1).")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Optional cap on processed duration (seconds).")
    p.add_argument("--modal-k", type=int, default=4)
    p.add_argument("--plot", action="store_true", help="Save diagnostic PNGs.")
    args = p.parse_args()

    out_dir = Path(args.out)
    for v in args.videos:
        if not os.path.exists(v):
            print(f"!! missing: {v}", file=sys.stderr)
            continue
        run_one(v, out_dir,
                target_long=args.target_long,
                stride=args.stride,
                max_seconds=args.max_seconds,
                modal_k=args.modal_k,
                do_plot=args.plot)


if __name__ == "__main__":
    main()
