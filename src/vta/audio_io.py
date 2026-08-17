"""ffmpeg-based normalization: any input container/codec -> 16 kHz mono WAV.

Stereo channels in the production calls are dual-mono (verified: channel
correlation 1.0000, identical per-channel RMS) so downmixing loses no
agent/customer separation that wasn't already absent.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000

# A hung ffmpeg decode stalls the one worker thread and every clip after it.
# Generous, but finite.
FFMPEG_TIMEOUT_S = float(os.environ.get("VTA_FFMPEG_TIMEOUT_S", "300"))


class AudioNormalizationError(RuntimeError):
    pass


@dataclass
class NormalizedAudio:
    samples: np.ndarray  # float32, mono, TARGET_SR
    sr: int
    duration_s: float
    source_path: Path


def _ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioNormalizationError("ffmpeg not found on PATH")
    return path


def normalize_to_wav(input_path: Path, out_path: Path | None = None) -> Path:
    """Decode any supported audio file to 16kHz mono PCM16 WAV via ffmpeg."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise AudioNormalizationError(f"input file not found: {input_path}")

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = Path(tmp.name)
        tmp.close()

    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-sample_fmt",
        "s16",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        out_path.unlink(missing_ok=True)
        raise AudioNormalizationError(
            f"ffmpeg timed out after {FFMPEG_TIMEOUT_S:.0f}s decoding {input_path.name}"
        ) from None
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise AudioNormalizationError(
            f"ffmpeg failed for {input_path.name}: {proc.stderr.strip()[-2000:]}"
        )
    return out_path


def load_normalized(input_path: Path) -> NormalizedAudio:
    """Normalize then load into memory as float32 in [-1, 1]."""
    wav_path = normalize_to_wav(input_path)
    try:
        # Always 1-D: ffmpeg was given `-ac 1`.
        samples, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        duration_s = len(samples) / sr
        return NormalizedAudio(
            samples=samples, sr=sr, duration_s=duration_s, source_path=input_path
        )
    finally:
        wav_path.unlink(missing_ok=True)
