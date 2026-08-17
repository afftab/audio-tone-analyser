"""pyannote/segmentation-3.0 -> speaker_overlap_present.

Stereo channels in the production calls are dual-mono (verified in
audio_io.py's docstring), so overlap cannot be read off channel separation
and must come from the acoustic overlap-detection model itself.

Uses segmentation-3.0 directly rather than the full speaker-diarization-3.1
pipeline (segmentation + embedding + clustering). Profiling showed
diarization was 58-75% of total per-clip pipeline latency; overlap only
needs "how many local speaker slots are active per frame," which comes
entirely from segmentation, upstream of the embedding/clustering steps that
make diarization slow. Verified this gives near-identical overlap durations
(0.35s/0.89s/2.3s vs. the full pipeline's 0.35s/0.89s/2.3s on the 3 provided
calls) at 34-61x the speed (28.8s/27.2s/274.4s -> 0.76s/0.80s/4.48s).

segmentation-3.0 is permutation_invariant (local speaker-slot identities
aren't consistent across sliding-window chunks -- "slot 1" in one chunk
isn't necessarily the same physical speaker as "slot 1" in the next), so
pyannote.audio's Inference class skips its usual cross-chunk aggregation by
default: naively averaging raw per-chunk outputs would blend unrelated
speaker identities. "How many slots are active right now" doesn't have that
problem -- it's permutation-invariant by construction -- so it's computed
per chunk via a pre_aggregation_hook and only *that* scalar is aggregated
across overlapping windows.
"""

from dataclasses import dataclass
from functools import lru_cache

import os

import numpy as np

from vta.audio_io import NormalizedAudio
from vta.config import HF_TOKEN

MIN_OVERLAP_S = 0.5  # total overlapped duration below this is not "enough to affect understanding"
STEP_S = float(os.environ.get("VTA_OVERLAP_STEP", "2.0"))  # aggregation hop; segmentation-3.0's window is a fixed 10s
ACTIVE_SPEAKER_THRESHOLD = 1.5  # aggregated active-slot count classed as "2+ speakers"


@dataclass
class OverlapResult:
    total_overlap_s: float
    speaker_overlap_present: bool


def _count_active_speakers(outputs: np.ndarray) -> np.ndarray:
    """Permutation-invariant per-frame 'how many of the local speaker slots
    are active' estimate -- safe to aggregate across chunks even though the
    per-slot identities themselves are not (see module docstring)."""
    return outputs.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=1)
def _load_inference():
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
    return Inference(model, step=STEP_S, pre_aggregation_hook=_count_active_speakers)


def detect_overlap(audio: NormalizedAudio, threshold_s: float = MIN_OVERLAP_S) -> OverlapResult:
    import torch

    inference = _load_inference()
    waveform = torch.from_numpy(audio.samples).float().unsqueeze(0)
    out = inference({"waveform": waveform, "sample_rate": audio.sr})

    active_speaker_count = out.data[:, 0]
    frame_duration = out.sliding_window.step
    overlapped_frames = active_speaker_count >= ACTIVE_SPEAKER_THRESHOLD
    total_overlap_s = float(np.sum(overlapped_frames) * frame_duration)

    return OverlapResult(
        total_overlap_s=total_overlap_s,
        speaker_overlap_present=total_overlap_s >= threshold_s,
    )
