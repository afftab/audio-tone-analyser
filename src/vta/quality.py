"""Deterministic audio_quality classification from DSP features.

Technical quality only -- independent of emotional tone and of
background-noise presence, per the brief's explicit warning against
conflating the two. Thresholds are a documented heuristic (3 labeled clips,
all "clear", cannot support fitting) -- see the build plan limitations.
"""

from vta.dsp_features import DSPFeatures

# Clipping: fraction of samples pinned at (tanh-saturated) full scale.
# 0.03, not 0.02: moderate soft-clipping spreads 1-4.4%, so 0.02 sat
# inside it and over-called "slight" as "severe".
SEVERE_CLIPPING = 0.03
SLIGHT_CLIPPING = 0.001

# SNR: signal-to-noise-floor estimate in dB.
SEVERE_SNR_DB = 8.0
SLIGHT_SNR_DB = 18.0

# Occupied bandwidth: frequency below which 99% of spectral energy sits.
# 1400, not 2000: clear speech spans 1490-2900 Hz, so 2000 false-fired on
# clean controls.
SEVERE_BANDWIDTH_HZ = 1200.0
SLIGHT_BANDWIDTH_HZ = 1400.0


def classify_audio_quality(feats: DSPFeatures) -> str:
    severe = (
        feats.clipping_ratio > SEVERE_CLIPPING
        or feats.snr_db < SEVERE_SNR_DB
        or feats.occupied_bandwidth_hz < SEVERE_BANDWIDTH_HZ
    )
    if severe:
        return "severely_impaired"

    slight = (
        feats.clipping_ratio > SLIGHT_CLIPPING
        or feats.snr_db < SLIGHT_SNR_DB
        or feats.occupied_bandwidth_hz < SLIGHT_BANDWIDTH_HZ
    )
    if slight:
        return "slightly_impaired"

    return "clear"
