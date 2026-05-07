"""HNA paper-ready plotting helpers.

This package replaces the former single-file ``viz.py``. The original
public surface (style, palette, sig_stars / fmt_p / save_figure /
add_significance_bar) is preserved at the top level — every existing
``from HNA.viz import …`` import keeps working unchanged.

New higher-level helpers live in the sibling modules:

- :mod:`HNA.viz.forest`    — per-subject delta forest plots
- :mod:`HNA.viz.polar`     — circular histogram + mean-vector arrow
- :mod:`HNA.viz.spectrum`  — multi-condition Welch spectrum overlay
- :mod:`HNA.viz.topomap`   — band-topomap helper (lazy MNE import)

Each is independent and can be imported on demand.
"""

# Re-export the original public surface from the moved file.
from ._style import (  # noqa: F401
    CONDITION_COLORS,
    MODALITY_COLORS,
    CONDITION_ORDER,
    PAPER_RC,
    use_paper_style,
    sig_stars,
    fmt_p,
    add_significance_bar,
    save_figure,
    _figsize,
)

# New higher-level helpers (lazy loadable; imports here so that
# ``from HNA.viz import forest_plot`` etc. work directly).
from .forest import forest_plot  # noqa: F401
from .polar import polar_phase_plot  # noqa: F401
from .spectrum import spectrum_overlay  # noqa: F401
# topomap is intentionally NOT auto-imported because it requires `mne`
# (heavy dependency). Use ``from HNA.viz.topomap import band_topomap``.

__all__ = [
    # Style / palette
    "CONDITION_COLORS", "MODALITY_COLORS", "CONDITION_ORDER", "PAPER_RC",
    "use_paper_style",
    # Significance
    "sig_stars", "fmt_p", "add_significance_bar",
    # Save
    "save_figure",
    # New helpers
    "forest_plot",
    "polar_phase_plot",
    "spectrum_overlay",
]
