"""
Demonstration figures for the wave-synthesis report (`report/wave_synthesis.tex`).

Generates four PNGs under `report/figures/`:

  pm_spectrum.png         The Pierson-Moskowitz / JONSWAP ocean spectra
                          along with a 2-D directional Phillips spectrum.
  fft_ocean_snapshots.png Three Tessendorf-style FFT ocean heightfields
                          at U10 = 4, 7, 12 m/s, rendered as shaded
                          surface plots.
  fp_vs_U10.png           Pierson-Moskowitz peak-frequency / period vs.
                          wind-speed cheat sheet, with a target band
                          (0.1 -- 0.5 Hz) shaded.
  pipeline_tiers.png      Schematic comparison of the recommended pipelines
                          (parameters in / mp4 out).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

GRAVITY = 9.81


# -------------------------------------------------------------------------
# Spectra -----------------------------------------------------------------

def pm_spectrum(f: np.ndarray, U10: float) -> np.ndarray:
    """Pierson-Moskowitz omnidirectional frequency spectrum (m^2/Hz)."""
    alpha = 8.1e-3
    beta = 0.74
    omega = 2 * np.pi * f
    omega_p = GRAVITY / U10
    out = np.zeros_like(omega)
    mask = omega > 0
    out[mask] = (alpha * GRAVITY ** 2 / omega[mask] ** 5) * \
                np.exp(-beta * (omega_p / omega[mask]) ** 4)
    return out


def jonswap_spectrum(f: np.ndarray, U10: float, fetch_km: float = 100.0,
                     gamma: float = 3.3) -> np.ndarray:
    """JONSWAP spectrum (Hasselmann 1973)."""
    F = fetch_km * 1000.0
    omega = 2 * np.pi * f
    omega_p = 22.0 * (GRAVITY ** 2 / (U10 * F)) ** (1.0 / 3.0)
    alpha = 0.076 * (U10 ** 2 / (F * GRAVITY)) ** 0.22
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r = np.exp(-((omega - omega_p) ** 2) / (2 * (sigma * omega_p) ** 2))
    pm = np.zeros_like(omega)
    mask = omega > 0
    pm[mask] = alpha * GRAVITY ** 2 / omega[mask] ** 5 * \
               np.exp(-1.25 * (omega_p / omega[mask]) ** 4)
    return pm * (gamma ** r)


def phillips_2d(N: int, L: float, U10: float, wind_dir: float = 0.0,
                damping: float = 0.001) -> np.ndarray:
    """Tessendorf's directional Phillips spectrum on an N x N grid."""
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(Kx ** 2 + Ky ** 2)
    K[K == 0] = 1e-9
    A = 0.0008
    Lc = U10 ** 2 / GRAVITY
    cosang = (Kx * np.cos(wind_dir) + Ky * np.sin(wind_dir)) / K
    P = A * np.exp(-1.0 / (K * Lc) ** 2) / K ** 4 * cosang ** 2
    P[cosang < 0] *= 0.07          # suppress against-wind components
    P *= np.exp(-(K * damping * Lc) ** 2)
    P[np.isnan(P)] = 0.0
    return P


def fft_ocean_height(N: int, L: float, U10: float, wind_dir: float, t: float,
                     seed: int = 1) -> np.ndarray:
    """One snapshot of a Tessendorf-style FFT ocean (height only)."""
    rng = np.random.default_rng(seed)
    P = phillips_2d(N, L, U10, wind_dir)
    H0 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) \
         * np.sqrt(np.maximum(P, 0) / 2.0)

    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(Kx ** 2 + Ky ** 2)
    omega = np.sqrt(GRAVITY * np.maximum(K, 1e-9))
    H = H0 * np.exp(1j * omega * t) + np.conj(np.flip(H0)) * np.exp(-1j * omega * t)
    h = np.fft.ifft2(H).real
    # Normalize so gradients are well-conditioned for shading.
    h = h / max(np.std(h), 1e-9) * 0.6
    return h


def shade_heightfield(h: np.ndarray) -> np.ndarray:
    """Cheap Lambertian + sky-reflection shading -> RGB image in [0,1]."""
    gx, gy = np.gradient(h)
    n = np.stack([-gx, -gy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9

    light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
    diffuse = np.clip((n * light).sum(axis=-1), 0.0, 1.0)
    half = (light + np.array([0.0, 0.0, 1.0])); half /= np.linalg.norm(half)
    specular = np.clip((n * half).sum(axis=-1), 0.0, 1.0) ** 64

    sky_horizon = np.array([0.78, 0.86, 0.94])
    sky_zenith  = np.array([0.36, 0.55, 0.78])
    sky = (n[..., 2:3] * sky_zenith + (1 - n[..., 2:3]) * sky_horizon)

    deep_water  = np.array([0.06, 0.18, 0.28])
    shallow     = np.array([0.18, 0.36, 0.48])
    base = (1 - diffuse[..., None]) * deep_water + diffuse[..., None] * shallow

    img = 0.45 * sky + 0.55 * base + 1.4 * specular[..., None] * np.array([1.0, 0.97, 0.85])
    return np.clip(img, 0, 1)


# -------------------------------------------------------------------------
# Figures -----------------------------------------------------------------

def fig_spectra() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    f = np.linspace(0.001, 0.6, 800)
    for U in (4, 7, 12):
        axes[0].plot(f, pm_spectrum(f, U), label=f"PM, U10={U} m/s",
                      lw=1.8)
    for U in (7,):
        axes[0].plot(f, jonswap_spectrum(f, U, fetch_km=80, gamma=3.3),
                      "--", lw=1.5,
                      label=f"JONSWAP, U10={U}, fetch=80km, γ=3.3")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel(r"$S(f)$  $(m^2/\mathrm{Hz})$")
    axes[0].set_title("Omnidirectional sea-surface spectra")
    axes[0].set_xlim(0, 0.6)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].axvspan(0.1, 0.5, color="0.85", zorder=0)

    P = phillips_2d(256, 256.0, 7.0, wind_dir=0.0)
    P_shift = np.fft.fftshift(P)
    extent = (-np.pi, np.pi, -np.pi, np.pi)  # k_x, k_y in arbitrary units
    axes[1].imshow(np.log1p(P_shift), origin="lower",
                    extent=extent, cmap="viridis", aspect="equal")
    axes[1].set_title("Phillips 2-D directional spectrum (wind →)")
    axes[1].set_xlabel(r"$k_x$"); axes[1].set_ylabel(r"$k_y$")
    axes[1].add_patch(FancyArrowPatch((-2.6, 0), (-1.0, 0),
                                      arrowstyle='-|>', mutation_scale=18,
                                      color="white"))
    fig.tight_layout()
    fig.savefig(FIGS / "pm_spectrum.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_fft_ocean_snapshots() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, U in zip(axes, (4, 7, 12)):
        h = fft_ocean_height(N=512, L=400.0, U10=float(U),
                              wind_dir=np.deg2rad(25), t=2.0, seed=2025)
        img = shade_heightfield(h)
        ax.imshow(img)
        ax.set_title(f"FFT ocean — $U_{{10}}$ = {U} m/s "
                      f"($f_p \\approx$ {0.13 * GRAVITY / U:.3f} Hz)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Tessendorf-style FFT ocean (NumPy, ~120 lines)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "fft_ocean_snapshots.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_fp_vs_U10() -> None:
    fig, ax1 = plt.subplots(figsize=(8.8, 4.4))
    U = np.linspace(2, 18, 200)
    fp = 0.13 * GRAVITY / U
    Tp = 1.0 / fp
    ax1.plot(U, fp, color="#0F5C8A", lw=2.2, label=r"PM peak freq. $f_p$")
    ax1.set_xlabel("Wind speed at 10 m height, $U_{10}$  [m/s]")
    ax1.set_ylabel("Peak frequency $f_p$  [Hz]", color="#0F5C8A")
    ax1.set_xlim(2, 18); ax1.set_ylim(0, 0.7)
    ax1.grid(True, alpha=0.3)
    ax1.axhspan(0.1, 0.5, color="0.9", zorder=0)
    ax1.text(2.3, 0.46, "target band\n(0.1 -- 0.5 Hz)", fontsize=10,
             color="#444")
    ax2 = ax1.twinx()
    ax2.plot(U, Tp, color="#C2410C", lw=1.8, ls="--",
             label=r"Period $T_p = 1/f_p$")
    ax2.set_ylabel("Period $T_p$  [s]", color="#C2410C")
    ax2.set_ylim(0, 16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95))
    ax1.set_title("Wind-speed → wave-period map (Pierson-Moskowitz)")
    fig.tight_layout()
    fig.savefig(FIGS / "fp_vs_U10.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline_tiers() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    tiers = [
        ("Tier 1: NumPy/CuPy FFT ocean",
         "spectrum + IFFT + simple shading",
         "seconds–minutes",
         "stylised → mid",
         3.6),
        ("Tier 2: Taichi or moderngl shader",
         "GPU FFT ocean + raymarched water",
         "real-time",
         "mid → high",
         2.2),
        ("Tier 3: Blender bpy + Cycles (CUDA/OptiX)",
         "Ocean modifier (Mastin-Watterberg-Mareda) + path-traced render",
         "30–120 min / 2-min clip @ 1080p",
         "near photorealistic",
         0.8),
    ]
    for (title, desc, cost, realism, y) in tiers:
        ax.add_patch(plt.Rectangle((0.5, y), 9, 1.4,
                                   facecolor="#EAF2F8",
                                   edgecolor="#0F5C8A", lw=1.4))
        ax.text(0.7, y + 1.0, title, fontsize=12, fontweight="bold",
                color="#0F5C8A")
        ax.text(0.7, y + 0.55, desc, fontsize=10)
        ax.text(0.7, y + 0.18, f"cost: {cost}    |    realism: {realism}",
                fontsize=9, color="#444")
        ax.add_patch(FancyArrowPatch((9.7, y + 0.7), (10.1, y + 0.7),
                                     arrowstyle='-|>', mutation_scale=14,
                                     color="#C2410C"))
        ax.text(10.15, y + 0.7, "mp4", fontsize=10, color="#C2410C",
                va="center")
    ax.text(5, 5.5, "wave-synthesis pipelines spanning the realism / cost trade-off",
            ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "pipeline_tiers.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Rendering wave-synthesis report figures into", FIGS)
    fig_spectra()
    fig_fft_ocean_snapshots()
    fig_fp_vs_U10()
    fig_pipeline_tiers()
    print("Done.")
