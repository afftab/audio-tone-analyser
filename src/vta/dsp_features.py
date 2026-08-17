"""Deterministic acoustic feature extraction.

These features resolve `audio_quality` and `background_noise_severity`
without any learned model, and feed the emotional_intensity cross-check.
Windowed (per-second) energy/pitch series are kept, not just scalars, per
PLAN.md §10 item 3 (a single mean over a long call flattens mid-call escalation).
"""

from dataclasses import dataclass, field

import librosa
import numpy as np

from vta.audio_io import NormalizedAudio

FRAME_LENGTH = 2048
HOP_LENGTH = 512
WINDOW_S = 1.0  # size of the coarse time-series bucket for arc detection


@dataclass
class DSPFeatures:
    duration_s: float

    # Technical quality
    clipping_ratio: float  # fraction of samples at/near full scale
    snr_db: float  # signal-energy vs noise-floor estimate
    occupied_bandwidth_hz: float  # frequency below which 99% of spectral energy sits
    telephony_band_ratio: float  # energy fraction inside 300-3400 Hz band

    # Prosody (voiced frames only)
    pitch_mean_hz: float
    pitch_std_hz: float
    voiced_ratio: float

    # Energy dynamics
    energy_mean_db: float
    energy_std_db: float
    energy_dynamic_range_db: float
    energy_contour_db: list[float] = field(default_factory=list)  # per-WINDOW_S bucket


# Removed: onset_rate_per_s, pitch_range_hz and spectral_flatness_* --
# superseded by ASR word timestamps, pitch_std_hz and the PANNs tagger.


def _clipping_ratio(y: np.ndarray, threshold: float = 0.99) -> float:
    return float(np.mean(np.abs(y) >= threshold))


def _snr_db(rms_frames: np.ndarray) -> float:
    """Signal energy = high-percentile frame RMS, noise floor = low-percentile."""
    if len(rms_frames) == 0:
        return 0.0
    signal = np.percentile(rms_frames, 95)
    noise = max(np.percentile(rms_frames, 10), 1e-8)
    return float(20 * np.log10(signal / noise))


def _power_spectrum(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """One whole-signal power spectrum, shared by the two band measures
    (which used to each recompute this identical rfft)."""
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    return spec, freqs


def _occupied_bandwidth(
    spec: np.ndarray, freqs: np.ndarray, energy_fraction: float = 0.99
) -> float:
    cumulative = np.cumsum(spec)
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0
    idx = np.searchsorted(cumulative, energy_fraction * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _telephony_band_ratio(
    spec: np.ndarray, freqs: np.ndarray, lo: float = 300.0, hi: float = 3400.0
) -> float:
    total = spec.sum()
    if total <= 0:
        return 0.0
    band_mask = (freqs >= lo) & (freqs <= hi)
    return float(spec[band_mask].sum() / total)


def _energy_contour(rms_db: np.ndarray, frame_times: np.ndarray, duration_s: float) -> list[float]:
    """Mean frame energy per WINDOW_S bucket (one grouped reduction, not a
    per-bucket mask)."""
    n_buckets = max(1, int(np.ceil(duration_s / WINDOW_S)))
    if len(rms_db) == 0:
        return [0.0] * n_buckets
    idx = np.clip((frame_times / WINDOW_S).astype(int), 0, n_buckets - 1)
    sums = np.bincount(idx, weights=rms_db, minlength=n_buckets)
    counts = np.bincount(idx, minlength=n_buckets)
    # Buckets with no frames fall back to the clip minimum, as before.
    filler = float(np.min(rms_db))
    means = np.divide(sums, counts, out=np.full(n_buckets, filler), where=counts > 0)
    return [float(v) for v in means[:n_buckets]]


def compute_dsp_features(audio: NormalizedAudio) -> DSPFeatures:
    y, sr = audio.samples, audio.sr
    if len(y) == 0:
        raise ValueError("empty audio signal")

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)

    # fmax 400 Hz: speech f0 tops out near 300-350 Hz. librosa's music
    # default (2093 Hz) searched harmonics with no fundamentals in them.
    f0, voiced_flag, _ = librosa.pyin(
        y,
        sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=400.0,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else 0.0

    # Per-second energy contour for arc detection
    frame_times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=HOP_LENGTH)
    contour = _energy_contour(rms_db, frame_times, audio.duration_s)

    spec, freqs = _power_spectrum(y, sr)

    return DSPFeatures(
        duration_s=audio.duration_s,
        clipping_ratio=_clipping_ratio(y),
        snr_db=_snr_db(rms),
        occupied_bandwidth_hz=_occupied_bandwidth(spec, freqs),
        telephony_band_ratio=_telephony_band_ratio(spec, freqs),
        pitch_mean_hz=float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0,
        pitch_std_hz=float(np.std(voiced_f0)) if len(voiced_f0) else 0.0,
        voiced_ratio=voiced_ratio,
        energy_mean_db=float(np.mean(rms_db)),
        energy_std_db=float(np.std(rms_db)),
        energy_dynamic_range_db=float(np.ptp(rms_db)),
        energy_contour_db=contour,
    )
