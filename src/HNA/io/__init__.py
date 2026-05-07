"""HNA.io --- pipeline-boundary I/O helpers.

Currently exports the optional :class:`Trace` dataclass for carrying a
1-D signal together with its metadata (fs, modality, name, t0, units)
through analysis scripts. The toolbox core (coupling / features /
stats / viz) does **not** depend on Trace --- it is purely additive
sugar at the script boundary.
"""
from __future__ import annotations

from .trace import Trace, align_traces

__all__ = ["Trace", "align_traces"]
