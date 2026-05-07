"""
Tessendorf-style FFT ocean: deterministic, parameter-controlled,
periodic-tile heightfield + choppy displacement + foam.

Internally stores ONE H0 array per wave component (a "wave bank"), so the
final heightfield can be built per frame as a weighted superposition with
arbitrary, time-varying per-component weights -- enabling fade-in/out,
amplitude envelopes, and other non-stationary stimuli.

References: Tessendorf 2001 (Simulating Ocean Water).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .spectra import GRAVITY, phillips_2d, dispersion_omega


@dataclass
class OceanState:
    h: np.ndarray         # heightfield, (N, N)
    dx: np.ndarray        # horizontal displacement x, (N, N)
    dy: np.ndarray        # horizontal displacement y, (N, N)
    foam: np.ndarray      # foam intensity in [0, 1], (N, N)
    grad_x: np.ndarray    # dh/dx, (N, N)
    grad_y: np.ndarray    # dh/dy, (N, N)


class FFTOcean:
    """
    Build the spectrum bank at construction time; sum it on demand at
    `state_at(t, weights=...)`.

    Each wave component is normalised so weight=1 produces a heightfield
    whose std equals natural_Hs / 4, where natural_Hs = 0.21 U10^2 / g.
    A weight of 0 turns the component off; a weight of 2 doubles its
    amplitude. With multiple components present, the combined Hs is
    sqrt(sum_k (weight_k * natural_Hs_k)^2) (independent variances).

    Parameters
    ----------
    components : list of (U10, wind_dir_rad, weight, spreading), optional
        Each entry is one wave system. If None, a single component is
        constructed from (U10, wind_dir_rad).
    """

    def __init__(self,
                 N: int = 256,
                 L: float = 200.0,
                 U10: float = 6.0,
                 wind_dir_rad: float = 0.0,
                 depth_m: float = 200.0,
                 choppiness: float = 1.0,
                 seed: int = 42,
                 components: Optional[List[Tuple[float, float, float, str]]] = None):
        self.N = int(N)
        self.L = float(L)
        self.U10 = float(U10)
        self.wind_dir_rad = float(wind_dir_rad)
        self.depth_m = float(depth_m)
        self.choppiness = float(choppiness)
        self.seed = int(seed)

        if not components:
            components = [(self.U10, self.wind_dir_rad, 1.0, "cosine_squared")]
        self.components = list(components)

        kx = 2 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        ky = 2 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
        K = np.sqrt(Kx ** 2 + Ky ** 2)
        Ksafe = np.where(K == 0, 1e-9, K)
        self.Kx = Kx
        self.Ky = Ky
        self.K = K
        self.Ksafe = Ksafe
        self.omega = dispersion_omega(K, depth_m=self.depth_m)
        self.Khat_x = self.Kx / self.Ksafe
        self.Khat_y = self.Ky / self.Ksafe

        rng = np.random.default_rng(self.seed)

        # Build per-component H0 arrays, each normalised to natural Hs.
        self._H0_per: List[np.ndarray] = []
        self._H0neg_per: List[np.ndarray] = []
        self._natural_Hs_per: List[float] = []
        self._default_weights: List[float] = []
        for (U_c, dir_c, w_c, spread_c) in self.components:
            P_c = phillips_2d(N=self.N, L=self.L, U10=float(U_c),
                              wind_dir_rad=float(dir_c),
                              spreading=str(spread_c))
            xi_r = rng.standard_normal((self.N, self.N))
            xi_i = rng.standard_normal((self.N, self.N))
            H0 = (xi_r + 1j * xi_i) * np.sqrt(np.maximum(P_c, 0) / 2.0)
            H0_neg = np.conj(np.flip(np.flip(H0, axis=0), axis=1))
            # Normalise so this component alone delivers std(h) = natural_Hs/4.
            Hs_c = 0.21 * (float(U_c) ** 2) / GRAVITY
            target_std = max(Hs_c / 4.0, 1e-3)
            h_test = np.fft.ifft2(H0 + H0_neg).real
            cur = float(h_test.std())
            if cur > 1e-15:
                H0 = H0 * (target_std / cur)
                H0_neg = H0_neg * (target_std / cur)
            self._H0_per.append(H0)
            self._H0neg_per.append(H0_neg)
            self._natural_Hs_per.append(Hs_c)
            self._default_weights.append(float(w_c))

    # -----------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        return len(self._H0_per)

    @property
    def default_weights(self) -> List[float]:
        return list(self._default_weights)

    def natural_Hs(self) -> List[float]:
        """Per-component PM-derived significant wave heights (meters)."""
        return list(self._natural_Hs_per)

    # -----------------------------------------------------------------------

    def state_at(self, t: float,
                 weights: Optional[Sequence[float]] = None) -> OceanState:
        """
        Compute heightfield + displacement + gradients + foam at time `t`,
        with optional per-component overriding weights.

        weights : list of float of length `n_components`, optional
            If None, uses the construction-time default weights.
        """
        if weights is None:
            weights = self._default_weights
        if len(weights) != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} weights, got {len(weights)}.")

        e_pos = np.exp(1j * self.omega * float(t))
        e_neg = np.conj(e_pos)
        H = np.zeros_like(self._H0_per[0])
        for w, H0, H0n in zip(weights, self._H0_per, self._H0neg_per):
            if w == 0.0:
                continue
            H = H + float(w) * (H0 * e_pos + H0n * e_neg)

        h = np.fft.ifft2(H).real

        Dx = np.fft.ifft2(-1j * self.Khat_x * H).real
        Dy = np.fft.ifft2(-1j * self.Khat_y * H).real
        lam = self.choppiness
        dx_disp = lam * Dx
        dy_disp = lam * Dy

        grad_x = np.fft.ifft2(1j * self.Kx * H).real
        grad_y = np.fft.ifft2(1j * self.Ky * H).real

        # Jacobian-derived foam
        dDx_dx = np.fft.ifft2(1j * self.Kx * (-1j * self.Khat_x * H)).real
        dDx_dy = np.fft.ifft2(1j * self.Ky * (-1j * self.Khat_x * H)).real
        dDy_dx = np.fft.ifft2(1j * self.Kx * (-1j * self.Khat_y * H)).real
        dDy_dy = np.fft.ifft2(1j * self.Ky * (-1j * self.Khat_y * H)).real
        J = (1.0 + lam * dDx_dx) * (1.0 + lam * dDy_dy) \
            - (lam ** 2) * dDx_dy * dDy_dx
        foam = np.clip(1.0 - J, 0.0, 1.0)

        return OceanState(h=h, dx=dx_disp, dy=dy_disp, foam=foam,
                          grad_x=grad_x, grad_y=grad_y)

    # -----------------------------------------------------------------------

    def sample(self, state: OceanState,
               wx: np.ndarray, wy: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear sampling on the periodic heightfield/foam at world (wx, wy)."""
        N = self.N; L = self.L
        gx = (wx % L) / L * N
        gy = (wy % L) / L * N
        i0 = np.floor(gx).astype(np.int32) % N
        j0 = np.floor(gy).astype(np.int32) % N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        fx = gx - np.floor(gx)
        fy = gy - np.floor(gy)
        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy

        def _bi(field):
            return (w00 * field[j0, i0] + w10 * field[j0, i1]
                    + w01 * field[j1, i0] + w11 * field[j1, i1])

        h = _bi(state.h)
        gxh = _bi(state.grad_x)
        gyh = _bi(state.grad_y)
        foam = _bi(state.foam)
        return h, gxh, gyh, foam
