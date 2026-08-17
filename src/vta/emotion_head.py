"""emotion2vec_plus_large: acoustic emotion posteriors + arousal proxy.

Feeds the LLM tone head (as corroborating acoustic evidence, not a
replacement for the semantic transcript read) and the emotional_intensity
cross-check. Resolved independently of transcript content -- this only ever
sees the waveform.
"""

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from vta.audio_io import NormalizedAudio

# HF Hub rather than funasr's default modelscope loader, whose downloader
# has no resume support.
HF_REPO_ID = os.environ.get("VTA_EMOTION2VEC", "emotion2vec/emotion2vec_plus_large")


class Emotion2vecUnavailable(RuntimeError):
    """Raised without a network attempt when VTA_DISABLE_EMOTION2VEC is set.

    Set it when the 1.94GB download isn't cached yet -- pipeline.py treats
    this signal as optional and falls back.
    """

# How "activated" each class is, independent of valence. Corroborates
# emotional_intensity only; valence stays with the LLM's transcript read.
AROUSAL_WEIGHTS = {
    "angry": 1.0,
    "surprised": 0.9,
    "fearful": 0.85,
    "happy": 0.6,
    "disgusted": 0.6,
    "sad": 0.35,
    "neutral": 0.1,
    # Excluded: uncertainty mass, not activation.
    "other": 0.0,
    "unknown": 0.0,
}


@dataclass
class EmotionResult:
    posteriors: dict[str, float]
    arousal: float  # weighted sum of posteriors x AROUSAL_WEIGHTS, in [0, 1]


# lru_cache memoizes returns but not exceptions; one attempt per process
# instead, so a failed load isn't retried per clip.
_LOAD_FAILURE: Exception | None = None


@lru_cache(maxsize=1)
def _load_model_uncached():
    if os.environ.get("VTA_DISABLE_EMOTION2VEC"):
        raise Emotion2vecUnavailable("VTA_DISABLE_EMOTION2VEC is set")

    from funasr import AutoModel
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id=HF_REPO_ID)
    return AutoModel(model=local_path, disable_update=True, log_level="ERROR")


def _load_model():
    global _LOAD_FAILURE
    if _LOAD_FAILURE is not None:
        raise _LOAD_FAILURE
    try:
        return _load_model_uncached()
    except Exception as e:
        _LOAD_FAILURE = e
        raise


def reset_model_cache() -> None:
    """Clear the cached model and any sticky load failure."""
    global _LOAD_FAILURE
    _LOAD_FAILURE = None
    _load_model_uncached.cache_clear()


# Raw labels are "<chinese>/<english>"; take the last segment so the English
# side matches AROUSAL_WEIGHTS ("<unk>" maps to "unknown").
def _clean_label(raw: str) -> str:
    raw = raw.strip()
    if raw == "<unk>":
        return "unknown"
    if "/" in raw:
        return raw.split("/")[-1].strip().lower()
    return raw.strip().lower()


# Chunked so cost is linear in duration: full self-attention over every 20ms
# frame is quadratic in clip length. Posteriors are duration-weight-averaged.
EMOTION2VEC_CHUNK_S = 20.0


def _run_on_samples(model, samples: np.ndarray, sr: int) -> dict[str, float]:
    """Run emotion2vec on one contiguous audio array -> {label: prob}."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), samples, sr)
        result = model.generate(str(tmp_path), granularity="utterance", extract_embedding=False)
    finally:
        tmp_path.unlink(missing_ok=True)
    entry = result[0]
    return {
        _clean_label(label): float(score)
        for label, score in zip(entry["labels"], entry["scores"])
    }


def _chunk(samples: np.ndarray, sr: int, chunk_s: float):
    """Yield (chunk, duration_weight) for non-overlapping chunks. Weights sum to 1."""
    chunk_n = int(chunk_s * sr)
    total = len(samples)
    if total <= chunk_n:
        yield samples, 1.0
        return
    start = 0
    while start < total:
        chunk = samples[start : start + chunk_n]
        yield chunk, len(chunk) / total
        start += chunk_n


def classify_emotion(audio: NormalizedAudio) -> EmotionResult:
    model = _load_model()

    # Aggregate at the posterior level rather than pooling raw frame
    # features -- standard for chunked SER, and needs no funasr internals.
    chunk_posteriors: list[dict[str, float]] = []
    weights: list[float] = []
    for chunk, weight in _chunk(audio.samples, audio.sr, EMOTION2VEC_CHUNK_S):
        chunk_posteriors.append(_run_on_samples(model, chunk, audio.sr))
        weights.append(weight)

    all_labels = set().union(*chunk_posteriors)
    total_w = sum(weights)
    posteriors = {
        label: sum(cp.get(label, 0.0) * w for cp, w in zip(chunk_posteriors, weights)) / total_w
        for label in all_labels
    }

    # Normalize by confidently-classified mass, not the raw total, so an
    # uncertain clip reflects the split among classes the model committed to.
    confident_mass = sum(v for k, v in posteriors.items() if k not in ("unknown", "other"))
    raw = sum(posteriors.get(k, 0.0) * w for k, w in AROUSAL_WEIGHTS.items())
    arousal = raw / confident_mass if confident_mass > 1e-6 else 0.0
    arousal = max(0.0, min(1.0, arousal))

    return EmotionResult(posteriors=posteriors, arousal=arousal)
