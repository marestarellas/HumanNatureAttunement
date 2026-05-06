"""
Surrogate-data hypothesis tests for coupling metrics.

A surrogate breaks the temporal relationship between two signals while
preserving each signal's marginal properties (e.g. power spectrum). The
distribution of a coupling metric over many surrogates gives a per-pair
null distribution against which the observed value can be tested.

This module provides:
- ``phase_shuffle(x, rng=None)``: phase-randomize one signal in the Fourier
  domain. Preserves the magnitude spectrum exactly; randomizes phases.
- ``time_shift(x, rng=None, min_shift=None)``: circular time shift. Cheap
  but only valid for short-stationarity assumptions.
- ``surrogate_test(metric_fn, x, y, n=500, method='phase_shuffle', ...)``:
  generic harness. Returns the observed metric, the null distribution,
  a one-tailed p-value, and a z-score relative to the null.

Notes
-----
Phase shuffling is the standard for stationary signals with structured
power spectra (audio envelope, respiration, slow HR). Time-shift surrogates
are only OK when the signal is reasonably stationary across the window
(short conditions are fine; long resting-state with drifts may violate this).

These tests are *one-sided*: we test whether the observed metric is more
extreme than the null, with the direction set by ``higher_is_better``.
"""

from __future__ import annotations

from typing import Callable, Tuple, Literal

import numpy as np


# ----------------------------- surrogate generators -----------------------------
def phase_shuffle(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Phase-randomize a 1-D signal via the Fourier transform.

    Preserves the power spectrum exactly. Randomizes the phase of every
    non-DC, non-Nyquist bin while keeping the spectrum Hermitian so the
    output is real-valued.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    X = np.fft.rfft(x)
    n_freq = len(X)

    # Keep DC (and Nyquist if n even) untouched.
    has_nyquist = (n % 2 == 0)
    new_phase = np.zeros(n_freq)
    rand_count = n_freq - 1 - (1 if has_nyquist else 0)
    if rand_count > 0:
        new_phase[1:1 + rand_count] = rng.uniform(-np.pi, np.pi, rand_count)
    # Keep Nyquist phase 0 (real) for the conjugate symmetry to be consistent.

    Y = np.abs(X) * np.exp(1j * new_phase)
    Y[0] = X[0]                  # DC
    if has_nyquist:
        Y[-1] = X[-1].real       # Nyquist
    y = np.fft.irfft(Y, n=n)
    return y


def time_shift(x: np.ndarray, rng: np.random.Generator | None = None,
               min_shift: int | None = None) -> np.ndarray:
    """Circular time shift of a 1-D signal by a random offset.

    ``min_shift`` (if given) prevents trivial near-zero shifts.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    if min_shift is None:
        min_shift = max(1, n // 20)
    shift = int(rng.integers(min_shift, n - min_shift))
    return np.roll(x, shift)


# ----------------------------- generic surrogate harness -----------------------------
def surrogate_test(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n: int = 500,
    method: Literal["phase_shuffle", "time_shift"] = "phase_shuffle",
    surrogate_target: Literal["x", "y"] = "y",
    higher_is_better: bool = True,
    rng_seed: int | None = 0,
) -> Tuple[float, np.ndarray, float, float]:
    """Run a surrogate-data hypothesis test for a coupling metric.

    Parameters
    ----------
    metric_fn : callable(x, y) -> float
        Coupling metric to test (e.g. PLV, Pearson r, coherence band-avg).
    x, y : np.ndarray
        Two 1-D signals of equal length.
    n : int
        Number of surrogates.
    method : str
        How to generate surrogates. "phase_shuffle" preserves spectrum.
    surrogate_target : "x" or "y"
        Which signal to surrogate-permute. Default surrogates ``y``.
    higher_is_better : bool
        If True, p = P(null >= observed). If False, p = P(null <= observed).
    rng_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    observed : float
    null : np.ndarray of shape (n,)
    p : float
        Empirical one-sided p-value with +1/+1 add-one correction.
    z : float
        (observed - mean(null)) / std(null). NaN if std is zero.
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    surrogate_fn = phase_shuffle if method == "phase_shuffle" else time_shift

    observed = float(metric_fn(x, y))
    null = np.empty(n, dtype=float)
    for i in range(n):
        if surrogate_target == "y":
            y_surr = surrogate_fn(y, rng=rng)
            null[i] = metric_fn(x, y_surr)
        else:
            x_surr = surrogate_fn(x, rng=rng)
            null[i] = metric_fn(x_surr, y)

    if higher_is_better:
        p = (np.sum(null >= observed) + 1) / (n + 1)
    else:
        p = (np.sum(null <= observed) + 1) / (n + 1)

    null_mean = float(np.nanmean(null))
    null_std = float(np.nanstd(null))
    z = (observed - null_mean) / null_std if null_std > 0 else float("nan")

    return observed, null, float(p), float(z)
