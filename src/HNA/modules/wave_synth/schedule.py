"""
Tiny scheduling primitives used to make any numeric parameter time-varying.

A `Schedule` is a callable: given a clock time `t` (seconds), it returns
the parameter's value at that time. Any synthesise() argument that is
documented as schedule-able may be a plain scalar, a Schedule, or any
other callable accepting one float and returning one float.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, List, Sequence, Union

import numpy as np


@dataclass
class Schedule:
    """
    Piecewise interpolation over (t, y) keyframes.

    Parameters
    ----------
    t : list of float
        Strictly increasing keyframe times, in seconds.
    y : list of float
        Values at the keyframe times. Same length as `t`.
    kind : {'linear', 'step', 'smoothstep'}
        Interpolation between adjacent keyframes.
    extrapolate : {'clip', 'wrap'}
        Behavior outside the keyframe range. 'clip' returns the boundary
        value; 'wrap' loops the schedule modulo its total span.
    """
    t: List[float]
    y: List[float]
    kind: str = "linear"
    extrapolate: str = "clip"

    def __post_init__(self):
        self._t = np.asarray(self.t, dtype=np.float64)
        self._y = np.asarray(self.y, dtype=np.float64)
        if self._t.shape != self._y.shape or self._t.ndim != 1:
            raise ValueError("Schedule.t and .y must be 1-D arrays of equal length.")
        if self._t.size < 1:
            raise ValueError("Schedule needs at least one keyframe.")
        if self._t.size > 1 and not np.all(np.diff(self._t) > 0):
            raise ValueError("Schedule.t must be strictly increasing.")

    def __call__(self, time: float) -> float:
        ts = self._t; ys = self._y
        if ts.size == 1:
            return float(ys[0])
        if self.extrapolate == "wrap":
            span = ts[-1] - ts[0]
            time = ts[0] + ((float(time) - ts[0]) % span)
        else:
            time = float(np.clip(time, ts[0], ts[-1]))

        if self.kind == "linear":
            return float(np.interp(time, ts, ys))
        if self.kind == "step":
            idx = int(np.searchsorted(ts, time, side="right") - 1)
            return float(ys[max(0, min(idx, ys.size - 1))])
        if self.kind == "smoothstep":
            i = int(np.searchsorted(ts, time, side="right") - 1)
            i = max(0, min(i, ts.size - 2))
            u = (time - ts[i]) / max(ts[i + 1] - ts[i], 1e-12)
            u = u * u * (3.0 - 2.0 * u)
            return float((1.0 - u) * ys[i] + u * ys[i + 1])
        raise ValueError(f"Schedule.kind={self.kind!r} not recognised.")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def evaluate(value: Any, time: float) -> Any:
    """Resolve `value` at time `t`. Callables and Schedules are evaluated;
    everything else passes through unchanged."""
    if callable(value):
        return value(time)
    return value


def evaluate_dataclass(d, time: float):
    """Return a copy of dataclass `d` with every callable/Schedule field
    evaluated at `time`. Tuple/list fields are not recursed into."""
    if d is None:
        return None
    if not is_dataclass(d):
        return d
    new_kwargs = {}
    for f in fields(d):
        v = getattr(d, f.name)
        # Schedules are dataclasses too -- they're identifiable by being callable.
        if callable(v):
            try:
                new_kwargs[f.name] = v(time)
            except TypeError:
                # Not a (time,) callable -- leave as-is.
                pass
    if new_kwargs:
        return replace(d, **new_kwargs)
    return d


def schedule_from_keyframes(*pairs, kind: str = "linear",
                            extrapolate: str = "clip") -> Schedule:
    """Convenience: `schedule_from_keyframes((0, 0.0), (30, 1.5), (60, 1.5))`."""
    ts = [float(p[0]) for p in pairs]
    ys = [float(p[1]) for p in pairs]
    return Schedule(t=ts, y=ys, kind=kind, extrapolate=extrapolate)
