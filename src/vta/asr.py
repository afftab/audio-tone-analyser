"""nvidia/parakeet-tdt-0.6b-v2 via NeMo: transcript + word timestamps.

The transcript feeds the LLM tone head (semantic tone/intensity boundary,
e.g. frustrated vs upset, is not recoverable from prosody alone). Word
timestamps give a real speaking-rate measurement, superseding the DSP
onset-rate proxy.

A Moonshine backend was evaluated and removed: with no word timestamps it
returned speaking_rate_wpm=0.0, which then reached the LLM as if measured,
and left `words` empty, silently disabling diarization. See
TECHNICAL_MEMO.md for the comparison.
"""

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from vta.audio_io import NormalizedAudio

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
ASR_SR = 16_000

# Gaps longer than this are pauses, not speech, and leave the speaking-rate
# denominator: longer than inter-word gaps, shorter than a turn pause.
PAUSE_GAP_S = 0.4


@dataclass
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass
class AsrResult:
    transcript: str
    words: list[Word]
    # Words per minute of *speech*, with pauses removed -- see _speaking_rate.
    speaking_rate_wpm: float


@lru_cache(maxsize=1)
def _load_model():
    import nemo.collections.asr as nemo_asr

    return nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)


def _speaking_rate_wpm(words: list[Word]) -> float:
    """Words per minute over speech time, excluding pauses.

    Dividing by the whole word-span instead counts silence as speech
    (call_001: 68.9 vs 202.4 wpm -- 20s of pause in a 30s clip). The gap
    threshold is not knife-edge: sweeping PAUSE_GAP_S 0.3-0.6s moves the
    result by under 7 wpm.
    """
    if len(words) < 2:
        return 0.0
    span = words[-1].end_s - words[0].start_s
    pause_s = sum(
        max(0.0, nxt.start_s - cur.end_s)
        for cur, nxt in zip(words, words[1:])
        if nxt.start_s - cur.end_s > PAUSE_GAP_S
    )
    speech_s = span - pause_s
    if speech_s <= 0:
        return 0.0
    return (len(words) / speech_s) * 60.0


def transcribe(audio: NormalizedAudio) -> AsrResult:
    if audio.sr != ASR_SR:
        raise ValueError(f"expected {ASR_SR} Hz audio, got {audio.sr}")

    model = _load_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), audio.samples, audio.sr)
        outputs = model.transcribe([str(tmp_path)], timestamps=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    hyp = outputs[0]
    transcript = hyp.text or ""

    word_ts = (hyp.timestamp or {}).get("word", []) if hasattr(hyp, "timestamp") else []
    words = [
        Word(text=w["word"], start_s=w["start"], end_s=w["end"]) for w in word_ts
    ]
    words.sort(key=lambda w: w.start_s)

    return AsrResult(
        transcript=transcript,
        words=words,
        speaking_rate_wpm=_speaking_rate_wpm(words),
    )
