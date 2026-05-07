"""HNA coupling subpackage — four-family layout.

Each submodule corresponds to one coupling family; the public surface is
re-exported here so that the legacy flat ``coupling.py`` import paths
keep working::

    from HNA.coupling import windowed_xcorr            # linear
    from HNA.coupling import band_coherence_windowed   # oscillatory (spectral)
    from HNA.coupling import plv_phase_sync            # oscillatory (phase)
    from HNA.coupling import tort_modulation_index     # oscillatory (cross-frequency)
    from HNA.coupling import windowed_mi               # information
    from HNA.coupling import effective_mi              # information (bias-corrected)
    from HNA.coupling import granger_bivariate         # information (directional)
    from HNA.coupling import exponent_matching         # complexity

Submodules
----------
- :mod:`.linear`        — xcorr (time-domain Pearson alignment).
- :mod:`.oscillatory`   — coherence + PLV / wPLI + PAC (frequency / phase /
                           cross-frequency).
- :mod:`.information`   — MI, effective MI, Granger, TE (probabilistic and
                           directional dependence).
- :mod:`.complexity`    — complexity matching (scaling-structure coupling).
- :mod:`._plots`        — shared plot helpers across families.

Underscore helpers (``_butter_bandpass``, ``_dominant_freq``,
``_wpli_from_phase``, ``_coh_choose_nperseg``) are also re-exported so
external notebooks that imported them from the old flat module still work.
"""

# ---- Linear (time-domain xcorr) ----
from .linear import (  # noqa: F401
    windowed_xcorr,
    XCorrResult,
)

# ---- Oscillatory (spectral / phase / cross-frequency) ----
from .oscillatory import (  # noqa: F401
    # Spectral
    band_coherence,
    band_coherence_windowed,
    CoherenceResult,
    # Phase synchronisation
    plv_phase_sync,
    windowed_plv,
    PLVResult,
    wpli_phase_sync,
    windowed_wpli,
    WPLIResult,
    # Cross-frequency (PAC)
    tort_modulation_index,
    canolty_mvl,
    windowed_pac,
    comodulogram,
    TortMIResult,
    CanoltyMVLResult,
    # Internal helpers — kept for legacy notebooks
    _butter_bandpass,
    _dominant_freq,
    _wpli_from_phase,
    _coh_choose_nperseg,
)

# ---- Information (MI + effective MI + Granger + TE + PID + Phi-ID) ----
from .information import (  # noqa: F401
    windowed_mi,
    effective_mi,
    windowed_effective_mi,
    granger_bivariate,
    granger_score,
    windowed_granger,
    GrangerResult,
    transfer_entropy,
    transfer_entropy_binned,
    pid_2source,
    phi_id,
)

# ---- Complexity (linear coupling on complexity features) ----
from .complexity import (  # noqa: F401
    exponent_matching,
    exponent_correlation,
    fluctuation_curve,
    fluctuation_matching,
    mse_curve,
    mse_matching,
    windowed_exponent,
    complexity_coupling,
)

# ---- Cross-complexity (genuinely scale-aware bivariate) ----
from .cross_complexity import (  # noqa: F401
    dcca,
    dcca_rho,
    cross_sample_entropy,
    multiscale_cross_entropy,
)

# ---- Shared plot helpers ----
from ._plots import (  # noqa: F401
    plot_signal_alignment_validation,
    plot_coupling_over_time,
    plot_coherence_results,
)

# Re-export NaN interpolation from dsp under its legacy in-package name
# (the old flat coupling.py exposed ``_nan_interp`` and at least one
# notebook imports it).
from ..dsp import interpolate_nan as _nan_interp  # noqa: F401


__all__ = [
    # Linear
    "windowed_xcorr", "XCorrResult",
    # Oscillatory: spectral
    "band_coherence", "band_coherence_windowed", "CoherenceResult",
    # Oscillatory: phase
    "plv_phase_sync", "windowed_plv", "PLVResult",
    "wpli_phase_sync", "windowed_wpli", "WPLIResult",
    # Oscillatory: cross-frequency
    "tort_modulation_index", "canolty_mvl", "windowed_pac", "comodulogram",
    "TortMIResult", "CanoltyMVLResult",
    # Information
    "windowed_mi", "effective_mi", "windowed_effective_mi",
    "granger_bivariate", "granger_score", "windowed_granger", "GrangerResult",
    "transfer_entropy",
    # Complexity
    "exponent_matching", "exponent_correlation",
    "fluctuation_curve", "fluctuation_matching",
    "mse_curve", "mse_matching",
    "windowed_exponent", "complexity_coupling",
    # Plot helpers
    "plot_signal_alignment_validation",
    "plot_coupling_over_time",
    "plot_coherence_results",
]
