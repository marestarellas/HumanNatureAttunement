"""
Pre-render visual assets for the Remotion showcase animation
(see remotion/STORYBOARD.md).

Usage
-----
    python scripts/render_animation_assets.py \
        --video video_examples/waves.mp4 \
        --out remotion \
        --max-seconds 8

The script writes to:
  <out>/public/             source.mp4, flow_overlay.mp4, frame_diff.mp4,
                            edges.mp4, timestack.png, mode_*.png
  <out>/src/data/           signals.json, summary.json

All visual assets are square-cropped and resized to 540×540 (sized for a
1920×1080 / two-panel layout). Length is capped by `--max-seconds`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from HNA.modules.video import quantify_video, iter_frames  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    y = (h - s) // 2; x = (w - s) // 2
    return frame[y:y + s, x:x + s]


def _flow_to_hsv(u: np.ndarray, v: np.ndarray, max_mag: float) -> np.ndarray:
    mag = np.hypot(u, v)
    ang = np.arctan2(v, u)
    h = ((ang + np.pi) / (2 * np.pi) * 180.0).astype(np.uint8)
    s = np.full_like(h, 255)
    val = np.clip(mag / max(max_mag, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    hsv = np.stack([h, s, val], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _draw_flow_arrows(frame_bgr: np.ndarray, u: np.ndarray, v: np.ndarray,
                      grid: int = 22, max_mag: float = 1.0,
                      darken: float = 0.55) -> np.ndarray:
    """
    Draw discrete flow arrows on a regular grid. Arrow color encodes direction
    (HSV → BGR), length encodes magnitude. Far more readable than the HSV
    flood-fill on a low-contrast water surface.
    """
    out = (frame_bgr.astype(np.float32) * darken).clip(0, 255).astype(np.uint8)
    H, W = u.shape
    # Grid step in flow-coords; we draw onto the (target_size x target_size) frame
    Th, Tw = frame_bgr.shape[:2]
    sx = Tw / W; sy = Th / H
    step_h = max(1, H // grid); step_w = max(1, W // grid)
    arrow_scale = 0.55 * min(Tw / grid, Th / grid) / max(max_mag, 1e-6)
    for j in range(step_h // 2, H, step_h):
        for i in range(step_w // 2, W, step_w):
            uu = float(u[j, i]); vv = float(v[j, i])
            mag = (uu * uu + vv * vv) ** 0.5
            if mag < 0.05:
                continue
            x0 = int(i * sx); y0 = int(j * sy)
            dx = int(uu * arrow_scale); dy = int(vv * arrow_scale)
            ang = np.arctan2(vv, uu)
            hue = int(((ang + np.pi) / (2 * np.pi)) * 180) % 180
            sat = 255
            val = int(min(255, 60 + (mag / max(max_mag, 1e-6)) * 195))
            color = cv2.cvtColor(
                np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(out, (x0, y0), (x0 + dx, y0 + dy),
                            color, thickness=2, tipLength=0.35, line_type=cv2.LINE_AA)
    return out


def _patch_entropy_map(gray: np.ndarray, patch: int = 24, bins: int = 32) -> np.ndarray:
    """Per-patch Shannon entropy → low-resolution map of shape (h//patch, w//patch)."""
    h, w = gray.shape
    H = (h // patch) * patch; W = (w // patch) * patch
    g = gray[:H, :W].reshape(H // patch, patch, W // patch, patch).swapaxes(1, 2)
    g = g.reshape(g.shape[0], g.shape[1], -1)
    out = np.zeros(g.shape[:2], dtype=np.float32)
    edges = np.linspace(0, 256, bins + 1)
    for j in range(g.shape[0]):
        for i in range(g.shape[1]):
            hist, _ = np.histogram(g[j, i], bins=edges, density=True)
            hist = hist[hist > 0]
            out[j, i] = float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0
    return out


def _writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_assets(video_path: str, out_dir: Path, target_size: int = 540,
                  max_seconds: float | None = None,
                  modal_k: int = 4) -> None:
    public = out_dir / "public"
    data_dir = out_dir / "src" / "data"
    public.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Quantify the whole clip first (cheap on small frames).
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    max_frames = int(round(max_seconds * fps)) if max_seconds else None
    print(f"[1/5] Quantifying {video_path}  fps={fps:.2f}  frames={n_total} "
          f"max_frames={max_frames}")
    feats = quantify_video(video_path,
                           target_long=192,
                           max_frames=max_frames,
                           include_windowed_complexity=True,
                           windowed_complexity_signals=("luminance", "frame_diff",
                                                        "flow_mag_mean",
                                                        "patch_entropy"),
                           windowed_complexity_win_sec=2.0,
                           windowed_complexity_step_sec=0.25,
                           modal_k=modal_k,
                           modal_target_long=96)
    n_used = len(next(iter(feats.signals.values())))

    # 2) Re-render the source clip cropped to a square at target_size and
    #    simultaneously every per-frame overlay used by the animation.
    print("[2/5] Re-encoding source.mp4 + overlays "
          "(frame_diff, motion_trail, edges, flow_overlay, flow_arrows, "
          "patch_heatmap, column_sweep)")
    sz = (target_size, target_size)
    src_w = _writer(public / "source.mp4", fps, sz)
    diff_w = _writer(public / "frame_diff.mp4", fps, sz)
    trail_w = _writer(public / "motion_trail.mp4", fps, sz)
    edge_w = _writer(public / "edges.mp4", fps, sz)
    flow_w = _writer(public / "flow_overlay.mp4", fps, sz)
    arr_w = _writer(public / "flow_arrows.mp4", fps, sz)
    heat_w = _writer(public / "patch_heatmap.mp4", fps, sz)
    sweep_w = _writer(public / "column_sweep.mp4", fps, sz)

    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    flow_max = 1.0
    motion_acc = None     # exponentially-decayed motion accumulator
    sweep_canvas = None   # builds up the timestack column-by-column
    column_index = target_size // 2
    written = 0
    sweep_total = max_frames or n_total or 1

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and written >= max_frames):
            break
        sq = _square_crop(frame)
        sq = cv2.resize(sq, sz, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)

        src_w.write(sq)

        # patch-entropy heatmap overlay (uses tuned patch_size)
        ent_map = _patch_entropy_map(gray, patch=24, bins=32)
        ent_norm = cv2.normalize(ent_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ent_big = cv2.resize(ent_norm, sz, interpolation=cv2.INTER_NEAREST)
        ent_bgr = cv2.applyColorMap(ent_big, cv2.COLORMAP_MAGMA)
        # draw thin grid lines so the patch structure is visible
        h_step = sz[1] // ent_map.shape[0]
        w_step = sz[0] // ent_map.shape[1]
        for y in range(0, sz[1], h_step):
            cv2.line(ent_bgr, (0, y), (sz[0], y), (20, 20, 20), 1)
        for x in range(0, sz[0], w_step):
            cv2.line(ent_bgr, (x, 0), (x, sz[1]), (20, 20, 20), 1)
        heat_blend = cv2.addWeighted(sq, 0.45, ent_bgr, 0.55, 0)
        heat_w.write(heat_blend)

        # column-sweep canvas: accumulate the chosen pixel column into a panel
        if sweep_canvas is None:
            sweep_canvas = np.zeros_like(sq)
        col_x = int((written / max(1, sweep_total - 1)) * (sz[0] - 1))
        col_x = max(0, min(sz[0] - 1, col_x))
        sweep_canvas[:, col_x:col_x + 1, :] = sq[:, column_index:column_index + 1, :]
        # draw the live sampling line on a copy of the source
        sweep_left = sq.copy()
        cv2.line(sweep_left, (column_index, 0), (column_index, sz[1] - 1),
                 (60, 80, 255), 2, cv2.LINE_AA)
        # draw progress cursor on the canvas
        sweep_right = sweep_canvas.copy()
        cv2.line(sweep_right, (col_x, 0), (col_x, sz[1] - 1),
                 (60, 220, 255), 1, cv2.LINE_AA)
        # The mp4 stores both halves stacked (left=source w/ red line,
        # right=accumulating timestack); Remotion can clip to either half.
        # Here we just composite right-half-only at full size (the scene
        # already shows source.mp4 on the left).
        sweep_w.write(sweep_right)

        if prev_gray is None:
            zero = np.zeros_like(sq)
            diff_w.write(zero); trail_w.write(zero)
            edge_w.write(cv2.cvtColor(cv2.Canny(gray, 35, 110), cv2.COLOR_GRAY2BGR))
            flow_w.write(zero); arr_w.write(sq.copy())
        else:
            d = cv2.absdiff(gray, prev_gray)
            d_n = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_bgr = cv2.applyColorMap(d_n, cv2.COLORMAP_INFERNO)
            diff_w.write(d_bgr)

            # exponential motion-energy accumulator → trail effect
            if motion_acc is None:
                motion_acc = d.astype(np.float32)
            motion_acc = 0.85 * motion_acc + 1.0 * d.astype(np.float32)
            mt_n = cv2.normalize(motion_acc, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
            mt_bgr = cv2.applyColorMap(mt_n, cv2.COLORMAP_INFERNO)
            mt_blend = cv2.addWeighted(sq, 0.35, mt_bgr, 0.75, 0)
            trail_w.write(mt_blend)

            edges = cv2.Canny(gray, 35, 110)
            edge_w.write(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

            # tuned Farneback for smoother flow on water
            small_prev = cv2.resize(prev_gray, (192, 192), interpolation=cv2.INTER_AREA)
            small_cur = cv2.resize(gray, (192, 192), interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_cur, None,
                0.5, 4, 21, 5, 7, 1.5, 0)
            u = flow[..., 0]; v = flow[..., 1]
            this_max = float(np.percentile(np.hypot(u, v), 95))
            flow_max = 0.9 * flow_max + 0.1 * this_max
            hsv_bgr = _flow_to_hsv(u, v, max_mag=max(flow_max, 0.5))
            hsv_bgr = cv2.resize(hsv_bgr, sz, interpolation=cv2.INTER_LINEAR)
            flow_w.write(cv2.addWeighted(sq, 0.35, hsv_bgr, 0.65, 0))

            # discrete arrows version (much more readable than HSV)
            arr = _draw_flow_arrows(sq, u, v, grid=22,
                                    max_mag=max(flow_max, 0.5))
            arr_w.write(arr)

        prev_gray = gray
        written += 1
    cap.release()
    src_w.release(); diff_w.release(); trail_w.release()
    edge_w.release(); flow_w.release(); arr_w.release()
    heat_w.release(); sweep_w.release()

    # 3) Timestack PNG (column over time, column = midline)
    print("[3/5] Timestack PNG")
    ts = feats.timestack
    if ts is not None and ts.timestack_image.size > 0:
        ts_img = ts.timestack_image  # (n_frames, h)
        ts_norm = cv2.normalize(ts_img.T, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ts_bgr = cv2.applyColorMap(ts_norm, cv2.COLORMAP_VIRIDIS)
        ts_bgr = cv2.resize(ts_bgr, (1080, 540), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(public / "timestack.png"), ts_bgr)

    # 4) Modal mode PNGs (top-k DMD spatial modes)
    print("[4/5] DMD modal mode PNGs")
    if feats.modal is not None:
        # Recompute DMD spatial modes here so the PNGs match the
        # frequencies/energies stored in feats.modal exactly.
        from HNA.modules.video import _stack_video  # type: ignore
        X, shape = _stack_video(video_path, target_long=96,
                                stride=1, max_frames=max_frames)
        try:
            from pydmd import DMD as _PyDMD
            dmd = _PyDMD(svd_rank=int(2 * modal_k)).fit(X)
            modes = np.asarray(dmd.modes)              # (P, k_eff) complex
            eigs = np.asarray(dmd.eigs)
            log_lam = np.log(eigs + 1e-30) * float(feats.fs)
            freqs = np.abs(log_lam.imag) / (2 * np.pi)
            dyn = np.asarray(dmd.dynamics)
            amps = np.linalg.norm(modes, axis=0) * np.abs(dyn).mean(axis=1)
            # de-duplicate conjugate pairs (highest amp wins)
            keys = np.round(freqs, 4)
            seen: Dict[float, int] = {}
            for i, key in enumerate(keys):
                cur = seen.get(key)
                if cur is None or amps[i] > amps[cur]:
                    seen[key] = i
            keep = np.array(sorted(seen.values()), dtype=int)
            order = np.argsort(-amps[keep])[:modal_k]
            keep = keep[order]
            modes = modes[:, keep]
            for i in range(modes.shape[1]):
                phi = modes[:, i].reshape(shape)
                # |phi| -- captures spatial extent of the mode
                m = np.abs(phi)
                m = (m - m.min()) / (max(m.max(), 1e-9) - m.min())
                m8 = (m * 255).astype(np.uint8)
                mode_bgr = cv2.applyColorMap(m8, cv2.COLORMAP_TURBO)
                mode_bgr = cv2.resize(mode_bgr, (target_size, target_size),
                                      interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(public / f"mode_{i+1}.png"), mode_bgr)
        except Exception as e:
            print(f"  PyDMD failed ({e}); falling back to SVD mode PNGs")
            U, _, _ = np.linalg.svd(X, full_matrices=False)
            for i in range(min(modal_k, U.shape[1])):
                mode = U[:, i].reshape(shape)
                mode = mode - mode.min(); mode = mode / max(mode.max(), 1e-9)
                mode8 = (mode * 255).astype(np.uint8)
                mode_bgr = cv2.applyColorMap(mode8, cv2.COLORMAP_TURBO)
                mode_bgr = cv2.resize(mode_bgr, (target_size, target_size),
                                      interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(public / f"mode_{i+1}.png"), mode_bgr)

    # 5) Signals JSON + summary JSON
    print("[5/5] signals.json + summary.json")
    signals_dump: Dict[str, list] = {k: [float(x) for x in v]
                                     for k, v in feats.signals.items()}
    payload = {
        "fps": float(feats.meta.fps),
        "fs_used": float(feats.fs),
        "n_frames": int(n_used),
        "duration_s": float(n_used / feats.fs),
        "video_target_size": int(target_size),
        "signals": signals_dump,
    }
    (data_dir / "signals.json").write_text(json.dumps(payload))

    summary = {
        "fps": float(feats.meta.fps),
        "duration_s": float(n_used / feats.fs),
        "complexity": {k: {kk: (None if vv != vv else float(vv))
                            for kk, vv in v.items()}
                       for k, v in feats.complexity.items()},
        "timestack": (None if ts is None else {
            "dominant_freq_hz": float(ts.dominant_freq_hz),
            "dominant_period_s": float(ts.dominant_period_s),
            "column_index": int(ts.column_index),
        }),
        "modal": (None if feats.modal is None else {
            "method": feats.modal.method,
            "frequencies_hz": feats.modal.frequencies_hz.tolist(),
            "energies": feats.modal.energies.tolist(),
        }),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAssets written under {out_dir}/public/ and {out_dir}/src/data/")
    print(f"  duration: {n_used / feats.fs:.2f}s ({n_used} frames at {feats.fs:.2f} fps)")
    if ts is not None:
        print(f"  timestack peak: {ts.dominant_freq_hz:.3f} Hz "
              f"({ts.dominant_period_s:.2f}s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="remotion")
    p.add_argument("--target-size", type=int, default=540)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--modal-k", type=int, default=4)
    args = p.parse_args()
    render_assets(args.video, Path(args.out),
                  target_size=args.target_size,
                  max_seconds=args.max_seconds,
                  modal_k=args.modal_k)


if __name__ == "__main__":
    main()
