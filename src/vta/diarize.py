"""Speaker diarization -> speaker-labelled transcript.

The brief asks for the customer's tone, but an undiarized transcript gives
the LLM no way to tell customer from agent. The agent holds 53%/61%/59% of
the speech across the three provided calls, and on call_003 delivers four
long policy explanations while the customer's sentiment is a handful of
turns. Flattened into one string that signal is diluted by a party scripted
to stay neutral -- a plausible driver of the neutral bias on long calls.

Which speaker is the customer is left to the LLM, which reads it off the
greeting more reliably than any heuristic here would.
"""

import os
from dataclasses import dataclass

from vta.asr import Word
from vta.audio_io import NormalizedAudio
from vta.config import HF_TOKEN

# Measured but not adopted -- off by default, VTA_DIARIZE=1 to enable.
# It changed no prediction on the 3 calls while tripling latency (170s vs
# 57s), and can't be validated above n=3 (IEMOCAP is single-speaker).
DIARIZE_ENABLED = os.environ.get("VTA_DIARIZE", "0") != "0"

# Support calls are two-party; fixing this removes a degree of freedom
# clustering gets wrong on short clips.
NUM_SPEAKERS = 2


@dataclass
class Turn:
    start_s: float
    end_s: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[Turn]
    labelled_transcript: str
    speech_share: dict[str, float]  # speaker -> fraction of total speech time


_PIPELINE = None


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from pyannote.audio import Pipeline

        # pyannote.audio 4.x renamed `use_auth_token` -> `token`.
        _PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=HF_TOKEN
        )
    return _PIPELINE


def _diarize_turns(audio: NormalizedAudio) -> list[Turn]:
    import torch

    pipe = _load_pipeline()
    waveform = torch.from_numpy(audio.samples).float().unsqueeze(0)
    out = pipe(
        {"waveform": waveform, "sample_rate": audio.sr}, num_speakers=NUM_SPEAKERS
    )
    # 4.x wraps the result; the Annotation with .itertracks is on
    # .speaker_diarization.
    ann = getattr(out, "speaker_diarization", out)
    return [
        Turn(start_s=seg.start, end_s=seg.end, speaker=label)
        for seg, _, label in ann.itertracks(yield_label=True)
    ]


def _assign(word: Word, turns: list[Turn]) -> str:
    """Speaker for a word: midpoint containment, nearest-turn fallback.

    Containment alone drops words straddling a diarization gap, which are
    often the turn-boundary words carrying sentiment.
    """
    mid = (word.start_s + word.end_s) / 2
    for t in turns:
        if t.start_s <= mid <= t.end_s:
            return t.speaker
    return min(
        turns,
        key=lambda t: 0.0 if t.start_s <= mid <= t.end_s
        else min(abs(mid - t.start_s), abs(mid - t.end_s)),
    ).speaker


def build_labelled_transcript(words: list[Word], turns: list[Turn]) -> str:
    """Group consecutive words by speaker into 'SPEAKER_xx: ...' lines."""
    if not words or not turns:
        return ""
    lines: list[str] = []
    current: str | None = None
    buf: list[str] = []
    for w in words:
        spk = _assign(w, turns)
        if spk != current and buf:
            lines.append(f"{current}: {' '.join(buf)}")
            buf = []
        current = spk
        buf.append(w.text)
    if buf:
        lines.append(f"{current}: {' '.join(buf)}")
    return "\n".join(lines)


def speaker_audio(audio: NormalizedAudio, turns: list[Turn], speaker: str):
    """A speaker's segments concatenated into one NormalizedAudio.

    Fine for the summary statistics taken here; not for anything
    continuity-sensitive.
    """
    import numpy as np

    from vta.audio_io import NormalizedAudio

    sr = audio.sr
    chunks = [
        audio.samples[int(t.start_s * sr): int(t.end_s * sr)]
        for t in turns
        if t.speaker == speaker
    ]
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return None
    samples = np.concatenate(chunks).astype(np.float32)
    return NormalizedAudio(
        samples=samples,
        sr=sr,
        duration_s=len(samples) / sr,
        source_path=audio.source_path,
    )


def diarize(audio: NormalizedAudio, words: list[Word]) -> DiarizationResult | None:
    """None when disabled, when the ASR gave no word timestamps, or on failure.

    Callers fall back to the flat transcript; this enhances one field group
    rather than being a pipeline dependency.
    """
    if not DIARIZE_ENABLED or not words:
        return None
    try:
        turns = _diarize_turns(audio)
    except Exception:
        return None
    if not turns:
        return None

    share: dict[str, float] = {}
    for t in turns:
        share[t.speaker] = share.get(t.speaker, 0.0) + (t.end_s - t.start_s)
    total = sum(share.values()) or 1.0

    return DiarizationResult(
        turns=turns,
        labelled_transcript=build_labelled_transcript(words, turns),
        speech_share={k: v / total for k, v in share.items()},
    )
