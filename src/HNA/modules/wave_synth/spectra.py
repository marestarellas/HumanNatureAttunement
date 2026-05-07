"""
Sea-surface wave spectra and dispersion relations.

References
----------
Pierson, W. J. & Moskowitz, L. (1964). A proposed spectral form for fully
  developed wind seas based on the similarity theory of Kitaigorodskii.
  J. Geophys. Res. 69 (24), 5181-5190.
Hasselmann, K. et al. (1973). Measurements of wind-wave growth and swell
  decay during the JONSWAP project.
Tessendorf, J. (2001). Simulating Ocean Water. SIGGRAPH course notes.
"""
from __future__ import annotations

import numpy as np

GRAVITY: float = 9.81  # m / s^2

# ---------------------------------------------------------------------------
# Wind <-> peak-frequency conversions (PM, fully developed sea)
# ---------------------------------------------------------------------------

def fp_to_U10(fp_hz: float) -> float:
    """Pierson-Moskowitz inverse: U10 (m/s) for desired peak frequency f_p (Hz)."""
    if fp_hz <= 0:
        raise ValueError("f_p must be positive.")
    return 0.13 * GRAVITY / float(fp_hz)


def U10_to_fp(U10: float) -> float:
    """Pierson-Moskowitz peak frequency (Hz) for wind speed U10 (m/s)."""
    if U10 <= 0:
        raise ValueError("U10 must be positive.")
    return 0.13 * GRAVITY / float(U10)


# ---------------------------------------------------------------------------
# Frequency-domain (1-D) spectra
# ---------------------------------------------------------------------------

def pierson_moskowitz(f_hz: np.ndarray, U10: float) -> np.ndarray:
    """Pierson-Moskowitz omnidirectional spectrum S(f), units m^2 / Hz."""
    omega = 2 * np.pi * np.asarray(f_hz, float)
    omega_p = GRAVITY / U10
    alpha = 8.1e-3
    out = np.zeros_like(omega)
    m = omega > 0
    out[m] = (alpha * GRAVITY ** 2 / omega[m] ** 5) * \
             np.exp(-1.25 * (omega_p / omega[m]) ** 4)
    return out


def jonswap(f_hz: np.ndarray, U10: float, fetch_km: float = 100.0,
            gamma: float = 3.3) -> np.ndarray:
    """JONSWAP spectrum (Hasselmann 1973). Adds a peak-enhancement factor `gamma`."""
    F = float(fetch_km) * 1000.0
    omega = 2 * np.pi * np.asarray(f_hz, float)
    omega_p = 22.0 * (GRAVITY ** 2 / (U10 * F)) ** (1.0 / 3.0)
    alpha = 0.076 * (U10 ** 2 / (F * GRAVITY)) ** 0.22
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r = np.exp(-((omega - omega_p) ** 2) / (2 * (sigma * omega_p) ** 2))
    pm = np.zeros_like(omega)
    m = omega > 0
    pm[m] = alpha * GRAVITY ** 2 / omega[m] ** 5 * \
            np.exp(-1.25 * (omega_p / omega[m]) ** 4)
    return pm * (gamma ** r)


# ---------------------------------------------------------------------------
# 2-D directional Phillips spectrum (Tessendorf 2001)
# ---------------------------------------------------------------------------

def phillips_2d(N: int, L: float, U10: float, wind_dir_rad: float = 0.0,
                damping_frac: float = 0.001,
                spreading: str = "cosine_squared",
                A: float = 8e-4) -> np.ndarray:
    """
    2-D directional Phillips spectrum on an N x N wavenumber grid spanning
    [-pi*N/L, +pi*N/L] in each direction. Returns a real-valued array of
    shape (N, N) with FFT-shifted indexing (k = 0 at index 0).

    Parameters
    ----------
    N : int          grid size
    L : float        physical patch size in meters; sets dk = 2*pi/L
    U10 : float      wind speed at 10 m, m/s
    wind_dir_rad : float
        Wind direction (radians; 0 = +x).
    damping_frac : float
        Suppression of wavelengths < damping_frac * L_c (Tessendorf trick to
        kill the high-k sawtooth).
    spreading : {'cosine_squared','mitsuyasu','donelan'}
        Directional spreading model. Affects only the angular part.
    A : float
        Phillips amplitude scale (small constant).
    """
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(Kx ** 2 + Ky ** 2)
    K_safe = np.where(K == 0, 1e-9, K)

    Lc = U10 ** 2 / GRAVITY
    cosang = (Kx * np.cos(wind_dir_rad) + Ky * np.sin(wind_dir_rad)) / K_safe

    base = A * np.exp(-1.0 / (K_safe * Lc) ** 2) / K_safe ** 4

    if spreading == "cosine_squared":
        ang = cosang ** 2
    elif spreading == "donelan":
        # Donelan-Banner narrower spreading: cos^4
        ang = cosang ** 4
    elif spreading == "mitsuyasu":
        # Mitsuyasu: |cos|^(2s) with s growing with k; here s=4 (rough)
        ang = np.abs(cosang) ** 8
    else:
        raise ValueError(f"Unknown spreading {spreading!r}")
    P = base * ang

    # Suppress against-wind half-plane and very high-k components.
    P[cosang < 0] *= 0.07
    P *= np.exp(-(K_safe * damping_frac * Lc) ** 2)

    P[K == 0] = 0.0
    return P


# ---------------------------------------------------------------------------
# Dispersion relation
# ---------------------------------------------------------------------------

def dispersion_omega(K: np.ndarray, depth_m: float = 200.0) -> np.ndarray:
    """
    Linear water-wave dispersion: omega(k) = sqrt(g k tanh(k h)).
    Returns array of same shape as K, with omega[K==0] = 0.
    """
    K = np.asarray(K, float)
    Ksafe = np.where(K == 0, 1e-9, K)
    return np.sqrt(GRAVITY * Ksafe * np.tanh(Ksafe * depth_m)) * (K > 0)
