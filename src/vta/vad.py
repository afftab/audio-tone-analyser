"""Silero VAD: voice-activity segmentation -> long_silence_present.

MIT-licensed, ~2M params, CPU-only. Loaded once per process via torch.hub
cache (downloaded at build time, not on first request).
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from vta.audio_io import NormalizedAudio

SILERO_SR = 16_000  # Silero's supported rate; matches our normalization target
LONG_SILENCE_THRESHOLD_S = 12.0  # gap length past which dead air is "unusually long"
# Heuristic, not fitted: the 3 clips' longest real gap is 8.9s, all labeled
# false (see PLAN.md limitations).


@dataclass
class VadResult:
    speech_segments: list[tuple[float, float]]  # (start_s, end_s)
    silence_gaps_s: list[float]  # duration of every gap between speech segments (incl. leading/trailing)
    longest_gap_s: float
    long_silence_present: bool
    speech_ratio: float


@lru_cache(maxsize=1)
def _load_model():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def run_vad(audio: NormalizedAudio, threshold_s: float = LONG_SILENCE_THRESHOLD_S) -> VadResult:
    if audio.sr != SILERO_SR:
        raise ValueError(f"expected {SILERO_SR} Hz audio, got {audio.sr}")

    model, get_speech_timestamps = _load_model()
    wav = torch.from_numpy(audio.samples.astype(np.float32))

    segments = get_speech_timestamps(
        wav, model, sampling_rate=SILERO_SR, return_seconds=True
    )
    speech_segments = [(s["start"], s["end"]) for s in segments]

    gaps = []
    cursor = 0.0
    for start, end in speech_segments:
        if start > cursor:
            gaps.append(start - cursor)
        cursor = max(cursor, end)
    if audio.duration_s > cursor:
        gaps.append(audio.duration_s - cursor)

    longest_gap = max(gaps) if gaps else 0.0
    speech_time = sum(e - s for s, e in speech_segments)
    speech_ratio = speech_time / audio.duration_s if audio.duration_s > 0 else 0.0

    return VadResult(
        speech_segments=speech_segments,
        silence_gaps_s=gaps,
        longest_gap_s=longest_gap,
        long_silence_present=longest_gap >= threshold_s,
        speech_ratio=speech_ratio,
    )
