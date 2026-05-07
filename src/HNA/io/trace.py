"""Trace --- a 1-D signal + its metadata, in one place.

The :class:`Trace` dataclass is *purely additive* sugar for analysis
scripts. The toolbox core (coupling / features / stats / viz) keeps
accepting plain ``(ndarray, fs)`` --- so existing pipelines run
unchanged. Use ``Trace`` only when you want metadata (fs, modality
name, t0 offset, units) to travel together with the array.

Example
-------
>>> import pandas as pd
>>> df = pd.read_csv("merged_annotated_with_audio.csv")
>>> audio = Trace.from_dataframe(df, "env_swell_0p2", fs=256, modality="audio")
>>> hrv   = Trace.from_dataframe(df, "HRV_MeanNN",   fs=4,   modality="hrv")
>>> hrv2  = hrv.align_to(audio)            # resample + crop to audio's grid
>>> from HNA.coupling import dcca_rho
>>> res = dcca_rho(audio.data, hrv2.data)  # plain ndarrays --- core unchanged

Design rules
------------
- A Trace **never** mutates in place. Every transformation returns a
  new ``Trace`` with new ``data`` and updated metadata.
- Methods are limited to *alignment glue* (resample, crop, align_to).
  Feature extraction and coupling stay outside the class.
- ``Trace.data`` and ``Trace.fs`` are always plain ``np.ndarray`` /
  ``float``, so you can drop them into any existing function via
  ``func(t.data, t.fs)``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
# Trace dataclass
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trace:
    """A 1-D signal with metadata.

    Attributes
    ----------
    data : np.ndarray, shape (N,)
        The 1-D signal. Cast to ``float`` on construction.
    fs : float
        Sampling rate in Hz.
    name : str
        Human-readable label (column name, channel id, derived feature
        name, ...). Used by viz helpers for legends/titles.
    modality : str
        Free-form modality tag --- ``"audio"``, ``"eeg"``, ``"hrv"``,
        ``"resp"``, ``"video"``, etc. Useful for grouping in figures
        and for downstream sanity checks.
    t0 : float, default 0.0
        Time of ``data[0]`` in seconds, relative to whatever common
        clock the calling pipeline uses (typically the trigger-aligned
        recording start).
    units : str, default ""
        Free-form units string for documentation / figure axes.
    """

    data: np.ndarray
    fs: float
    name: str
    modality: str
    t0: float = 0.0
    units: str = ""

    def __post_init__(self):
        # Frozen dataclasses can't simply assign in __post_init__ ---
        # use object.__setattr__ to coerce types without breaking
        # immutability semantics.
        arr = np.asarray(self.data, dtype=float).ravel()
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "fs", float(self.fs))
        object.__setattr__(self, "t0", float(self.t0))

    # -- convenience properties --------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.data.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs if self.fs > 0 else float("nan")

    @property
    def t_end(self) -> float:
        return self.t0 + self.duration_s

    @property
    def times_s(self) -> np.ndarray:
        """Sample-time vector ``t0 + arange(N) / fs``."""
        return self.t0 + np.arange(self.n_samples) / self.fs

    # -- transformations ---------------------------------------------
    def resample_to(self, fs_new: float, *, kind: str = "linear") -> "Trace":
        """Return a new Trace at the requested sample rate.

        ``kind="linear"`` (default) uses :func:`numpy.interp` on the
        original sample-time grid; ``kind="poly"`` uses
        :func:`scipy.signal.resample_poly` (anti-aliased, slower, only
        valid for stationary regular grids).
        """
        if fs_new <= 0:
            raise ValueError("fs_new must be positive")
        if abs(fs_new - self.fs) < 1e-12:
            return self

        if kind == "linear":
            t_old = self.times_s
            n_new = max(2, int(round(self.duration_s * fs_new)))
            t_new = self.t0 + np.arange(n_new) / fs_new
            data_new = np.interp(t_new, t_old, self.data)
        elif kind == "poly":
            from scipy.signal import resample_poly
            from fractions import Fraction
            ratio = Fraction(fs_new / self.fs).limit_denominator(1000)
            data_new = resample_poly(
                self.data, ratio.numerator, ratio.denominator)
        else:
            raise ValueError(f"unknown kind={kind!r}; use 'linear' or 'poly'")

        return replace(self, data=data_new, fs=fs_new)

    def crop(self, t_start: float, t_end: float) -> "Trace":
        """Return a new Trace covering ``[t_start, t_end)`` (seconds).

        Times are interpreted on the same clock as ``self.t0``. The new
        Trace's ``t0`` is updated to the actual start time of the kept
        samples (so subsequent ``times_s`` is still correct).
        """
        if t_end <= t_start:
            raise ValueError("t_end must be > t_start")
        i0 = max(0, int(np.ceil((t_start - self.t0) * self.fs)))
        i1 = min(self.n_samples,
                  int(np.floor((t_end - self.t0) * self.fs)))
        if i1 <= i0:
            return replace(self, data=np.empty(0, dtype=float),
                            t0=t_start)
        new_t0 = self.t0 + i0 / self.fs
        return replace(self, data=self.data[i0:i1].copy(), t0=new_t0)

    def align_to(self, other: "Trace", *, kind: str = "linear") -> "Trace":
        """Resample + crop so this Trace shares ``other``'s grid.

        Returns a new Trace whose ``fs`` equals ``other.fs`` and whose
        samples cover the temporal overlap with ``other``. Useful for
        feeding two cross-modal signals into a coupling estimator that
        requires a common sampling rate.
        """
        # Cropped to the temporal intersection
        t_lo = max(self.t0, other.t0)
        t_hi = min(self.t_end, other.t_end)
        if t_hi <= t_lo:
            raise ValueError(
                f"no temporal overlap between {self.name!r} "
                f"({self.t0:.3f}-{self.t_end:.3f} s) and "
                f"{other.name!r} ({other.t0:.3f}-{other.t_end:.3f} s)"
            )
        cropped = self.crop(t_lo, t_hi)
        if abs(cropped.fs - other.fs) < 1e-12:
            return cropped
        # Resample to the OTHER's exact sample grid (so indices match
        # bin-for-bin, not just fs)
        n_new = int(np.floor((t_hi - t_lo) * other.fs))
        t_new = np.arange(n_new) / other.fs + t_lo
        data_new = np.interp(t_new, cropped.times_s, cropped.data)
        return replace(cropped, data=data_new, fs=other.fs, t0=t_lo)

    def with_data(self, new_data: np.ndarray) -> "Trace":
        """Return a new Trace with the same metadata but a different array.

        Useful after applying a filter / preprocessing step that
        preserves fs and time base.
        """
        new_data = np.asarray(new_data, dtype=float).ravel()
        if new_data.size != self.n_samples:
            raise ValueError(
                f"with_data requires the same length ({self.n_samples}); "
                f"got {new_data.size}. Use resample_to / crop for length "
                f"changes."
            )
        return replace(self, data=new_data)

    # -- factories ---------------------------------------------------
    @classmethod
    def from_array(
        cls,
        arr,
        fs: float,
        name: str,
        modality: str,
        t0: float = 0.0,
        units: str = "",
    ) -> "Trace":
        """Build a Trace from a 1-D array-like."""
        return cls(np.asarray(arr, dtype=float).ravel(), fs=fs,
                    name=name, modality=modality, t0=t0, units=units)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        column: str,
        fs: float,
        modality: str,
        t0: float = 0.0,
        units: str = "",
        *,
        name: Optional[str] = None,
    ) -> "Trace":
        """Pull a column out of a wide DataFrame (e.g.\ a merged-CSV row).

        ``name`` defaults to ``column``. The DataFrame is assumed to
        have a uniform sample rate (which is the standard schema of
        ``merged_annotated_with_audio.csv``).
        """
        if column not in df.columns:
            raise KeyError(f"column {column!r} not in DataFrame")
        return cls.from_array(
            df[column].to_numpy(),
            fs=fs, name=name or column, modality=modality,
            t0=t0, units=units,
        )

    # -- repr --------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Trace(name={self.name!r}, modality={self.modality!r}, "
            f"fs={self.fs:g} Hz, n={self.n_samples}, "
            f"t=[{self.t0:.3f}, {self.t_end:.3f}] s"
            + (f", units={self.units!r}" if self.units else "")
            + ")"
        )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def align_traces(*traces: "Trace", to: Optional["Trace"] = None,
                  kind: str = "linear") -> tuple:
    """Align several Traces onto a common time + fs grid.

    By default, the *first* Trace's grid is the reference. Pass
    ``to=`` to override (any Trace, including one not in ``traces``).
    Returns a tuple of new Traces in the same order, all sharing
    ``to.fs`` and the temporal intersection of all inputs.
    """
    if not traces:
        raise ValueError("align_traces() needs at least one Trace")
    ref = to if to is not None else traces[0]
    # Find common temporal window
    t_lo = max(ref.t0, *(t.t0 for t in traces))
    t_hi = min(ref.t_end, *(t.t_end for t in traces))
    if t_hi <= t_lo:
        raise ValueError("traces have no shared temporal overlap")
    # Build a fake reference Trace covering the intersection so each
    # trace.align_to() lands on the same grid
    n = int(np.floor((t_hi - t_lo) * ref.fs))
    common = Trace(np.zeros(n), fs=ref.fs, name="__ref__",
                    modality="__ref__", t0=t_lo)
    return tuple(t.align_to(common, kind=kind) for t in traces)
