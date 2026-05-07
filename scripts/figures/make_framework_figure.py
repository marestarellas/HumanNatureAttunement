"""
Render the v3-report framework figure: a 3x3 grid where rows are
spatial-scale tiers (whole-image / per-patch / per-pixel) and columns
are feature families (raw / oscillatory / complexity). Each cell lists
the corresponding extractors of ``HNA.modalities.video``.

Output: ``reports/video_v3/figures/framework.png``
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "video_v3" / "figures" / "framework.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- content of each cell -------------------------------------------------
# Each entry is (header, bullets, source-file footer).
TABLE = [
    # row = whole-image
    [
        ("Envelope-like signals",
         ["luminance, R/G/B mean",
          "spatial std, frame_diff",
          "flow_mag_mean / curl / div",
          "flow_dir_entropy"],
         "whole_image.py"),
        ("Spatial pattern descriptors",
         ["spatial_fft_peak_k",
          "spatial_fft_anisotropy",
          "spatial_fft_orientation",
          "(timestack & DMD modal_k)"],
         "spatial_fft.py + timestack.py + modal.py"),
        ("Frame-level complexity",
         ["edge_density, fractal_dim",
          "lacunarity, GLCM",
          "patch_entropy (mean,p95,...)",
          "spatial_psd_slope"],
         "whole_image.py"),
    ],
    # row = per-patch
    [
        ("Per-patch motion",
         ["patch frame_diff map",
          "(per-patch flow magnitude)"],
         "per_patch.py"),
        ("",
         ["—  spatial pattern at the",
          "    patch tier is captured",
          "    by the whole-image",
          "    spatial-FFT scalars"],
         ""),
        ("Per-patch texture maps",
         ["patch_entropy grid",
          "edge_density grid",
          "fractal_dim_grad grid",
          "lacunarity_grad grid",
          "GLCM contrast / homog grids"],
         "per_patch.py"),
    ],
    # row = per-pixel
    [
        ("",
         ["—  per-pixel raw value is",
          "    just the input intensity"],
         ""),
        ("Per-pixel temporal FFT",
         ["peak_freq_map (Hz)",
          "band_power_low/mid/high",
          "spectral_entropy_map",
          "+ sliding-window variant",
          "DMD modes (full-res)"],
         "per_pixel.py + modal.py"),
        ("Per-pixel temporal complexity",
         ["higuchi_fd_map",
          "perm_entropy_map",
          "(dfa_alpha_map; opt-in)"],
         "per_pixel.py"),
    ],
]

ROW_LABELS = ["whole-image\nscalar / frame", "per-patch\nlow-res map",
              "per-pixel\nfull-res map"]
COL_LABELS = ["raw\n(envelopes & motion)",
              "oscillatory\n(rhythms & periods)",
              "complexity\n(entropy, fractal, scaling)"]

# Colour per feature family, deliberately muted.
COL_COLORS = ["#9CC4E4", "#F2C879", "#C8A2C8"]
COL_FILL = ["#E7F1FB", "#FBF1DC", "#F2E5F2"]


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(13.0, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.set_aspect("auto")

    # Title
    ax.text(6.0, 7.7, "A two-axis framework for video features",
            ha="center", va="bottom", fontsize=15, fontweight="bold",
            color="#0F5C8A")
    ax.text(6.0, 7.42,
            "rows = spatial scale of analysis,  "
            "columns = feature family.  "
            "Every 1-D signal in a cell is a coupling-ready time series.",
            ha="center", va="bottom", fontsize=9.5, color="#374151")

    # column headers
    for ci, (lab, col) in enumerate(zip(COL_LABELS, COL_COLORS)):
        x0 = 2.5 + ci * 3.0
        ax.text(x0 + 1.5, 7.05, lab, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=col)

    # row headers
    for ri, lab in enumerate(ROW_LABELS):
        y0 = 5.7 - ri * 1.95
        ax.text(1.0, y0, lab, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#0F5C8A")

    # cells
    for ri in range(3):
        for ci in range(3):
            x0 = 2.5 + ci * 3.0
            y0 = 4.8 - ri * 1.95
            w, h = 2.85, 1.78
            patch = FancyBboxPatch((x0, y0), w, h,
                                   boxstyle="round,pad=0.04,rounding_size=0.10",
                                   linewidth=1.0,
                                   edgecolor=COL_COLORS[ci],
                                   facecolor=COL_FILL[ci])
            ax.add_patch(patch)
            header, bullets, footer = TABLE[ri][ci]
            if header:
                ax.text(x0 + 0.10, y0 + h - 0.20, header,
                        ha="left", va="top",
                        fontsize=9.5, fontweight="bold",
                        color="#0F5C8A")
            line_y = y0 + h - 0.50
            for b in bullets:
                # bullet point
                marker = "•" if header else " "
                ax.text(x0 + 0.10, line_y,
                        f"{marker} {b}" if header else b,
                        ha="left", va="top", fontsize=8.5,
                        color="#1F2937")
                line_y -= 0.20
            if footer:
                ax.text(x0 + w - 0.10, y0 + 0.08, footer,
                        ha="right", va="bottom",
                        fontsize=7.0, style="italic",
                        color="#6E7781")

    # Legend strip at the bottom
    ax.text(6.0, 0.45,
            "Every cell that lists a 1-D signal feeds the coupling toolbox  "
            "(linear / oscillatory / information statistics).  "
            "Sliding-window variants of the spectral & modal cells exist "
            "for non-stationary clips.",
            ha="center", va="center", fontsize=8.5, color="#374151")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
