"""DSP, ASR-derived and LLM-payload behaviour.

Pure-function tests only -- nothing here loads a model or calls the API.
"""

import numpy as np
import pytest

from vta.asr import PAUSE_GAP_S, Word, _speaking_rate_wpm
from vta.dsp_features import (
    WINDOW_S,
    _energy_contour,
    _occupied_bandwidth,
    _power_spectrum,
    _telephony_band_ratio,
)


# --- speaking rate ---

def _words(spans):
    return [Word(text=f"w{i}", start_s=a, end_s=b) for i, (a, b) in enumerate(spans)]


def test_speaking_rate_excludes_pauses():
    """Two words 10s apart are not "two words per 10 seconds" of speech."""
    words = _words([(0.0, 0.3), (10.0, 10.3)])
    # Speech time is 0.3s + 0.3s worth of span with the 9.7s gap removed.
    assert _speaking_rate_wpm(words) == pytest.approx(2 / 0.6 * 60, rel=0.01)


def test_speaking_rate_keeps_sub_threshold_gaps():
    """Ordinary inter-word gaps are speech, not pause."""
    gap = PAUSE_GAP_S / 2
    words = _words([(0.0, 0.5), (0.5 + gap, 1.0 + gap)])
    span = 1.0 + gap
    assert _speaking_rate_wpm(words) == pytest.approx(2 / span * 60, rel=0.01)


def test_speaking_rate_is_invariant_to_pause_length():
    """Same words, longer dead air -> same rate. The old formula collapsed."""
    short_pause = _words([(0.0, 0.4), (0.5, 0.9), (10.0, 10.4)])
    long_pause = _words([(0.0, 0.4), (0.5, 0.9), (120.0, 120.4)])

    assert _speaking_rate_wpm(short_pause) == pytest.approx(_speaking_rate_wpm(long_pause))

    # What the old whole-span formula did with the same two inputs.
    def whole_span(w):
        return len(w) / (w[-1].end_s - w[0].start_s) * 60

    assert whole_span(short_pause) / whole_span(long_pause) > 10


@pytest.mark.parametrize("words", [[], _words([(0.0, 0.5)])])
def test_speaking_rate_degenerate_inputs(words):
    assert _speaking_rate_wpm(words) == 0.0


# --- energy contour ---

def test_energy_contour_matches_the_reference_loop():
    """The vectorized bucketing must agree with the mask-per-bucket original."""
    rng = np.random.default_rng(0)
    rms_db = rng.normal(-30, 5, size=900)
    frame_times = np.linspace(0.0, 29.9, 900)
    duration_s = 30.0

    n = max(1, int(np.ceil(duration_s / WINDOW_S)))
    expected = []
    for b in range(n):
        mask = (frame_times >= b * WINDOW_S) & (frame_times < (b + 1) * WINDOW_S)
        expected.append(float(np.mean(rms_db[mask])) if np.any(mask) else float(np.min(rms_db)))

    assert _energy_contour(rms_db, frame_times, duration_s) == pytest.approx(expected)


def test_energy_contour_fills_empty_buckets_with_the_minimum():
    rms_db = np.array([-20.0, -25.0])
    frame_times = np.array([0.1, 0.2])  # everything in bucket 0
    out = _energy_contour(rms_db, frame_times, duration_s=3.0)
    assert len(out) == 3
    assert out[0] == pytest.approx(-22.5)
    assert out[1] == out[2] == pytest.approx(-25.0)


# --- spectrum ---

def test_shared_spectrum_matches_independent_computation():
    """Sharing one FFT between the two band measures must not change them."""
    sr = 16_000
    t = np.arange(sr) / sr
    y = (np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)

    spec, freqs = _power_spectrum(y, sr)
    independent = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    assert spec == pytest.approx(independent)

    # 440 Hz dominates, so nearly all energy sits inside the telephony band.
    assert _telephony_band_ratio(spec, freqs) > 0.9
    assert 400 < _occupied_bandwidth(spec, freqs) < 7000


# --- LLM payload ---

def test_long_contour_is_downsampled_and_disclosed():
    from vta.tone_llm import MAX_CONTOUR_POINTS, _prosody_block, ProsodySummary

    long_call = ProsodySummary(
        pitch_mean_hz=200.0, pitch_std_hz=30.0, energy_mean_db=-30.0,
        energy_std_db=5.0, energy_dynamic_range_db=40.0, voiced_ratio=0.5,
        speaking_rate_wpm=180.0,
        energy_contour_db=[float(i) for i in range(1800)],  # a 30-minute call
    )
    block = _prosody_block(long_call)
    series = block["energy_contour_db_per_second"]
    assert len(series) <= MAX_CONTOUR_POINTS
    assert "energy_contour_note" in block, "a resampled series must say so"
    # Block-averaging keeps the arc: the series still rises monotonically.
    assert series == sorted(series)


def test_short_contour_is_passed_through_untouched():
    from vta.tone_llm import _prosody_block, ProsodySummary

    short = ProsodySummary(
        pitch_mean_hz=200.0, pitch_std_hz=30.0, energy_mean_db=-30.0,
        energy_std_db=5.0, energy_dynamic_range_db=40.0, voiced_ratio=0.5,
        speaking_rate_wpm=180.0, energy_contour_db=[1.0, 2.0, 3.0],
    )
    block = _prosody_block(short)
    assert block["energy_contour_db_per_second"] == [1.0, 2.0, 3.0]
    assert "energy_contour_note" not in block
