"""Per-modality preprocessing for the HNA toolbox.

Each submodule encapsulates the cleaning / feature-extraction steps that are
*specific to one signal type* (audio, EEG, respiration, ECG/HRV, ...). The
goal is that any analysis script can do, e.g.::

    from HNA.modalities.respiration import clean_respiration
    from HNA.modalities.ecg import preprocess_ecg_segment
    from HNA.modalities.audio import decompose_envelope
    from HNA.modalities.eeg import filter_eeg, compute_psd_features

without ever re-implementing a Butterworth bandpass or a NeuroKit2 wrapper.

The coupling / surrogate / stats / viz modules in :mod:`HNA` stay
modality-agnostic and can be combined freely with any of the helpers below.
"""

from . import audio, eeg, respiration, ecg

__all__ = ["audio", "eeg", "respiration", "ecg"]
