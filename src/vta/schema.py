"""Required output schema for a single analyzed clip (per PROJECT_BRIEF.md §2)."""

from typing import Literal

from pydantic import BaseModel, Field

EmotionalTone = Literal["neutral", "satisfied", "frustrated", "upset", "distressed"]
EmotionalIntensity = Literal["low", "medium", "high"]
NoiseSeverity = Literal["none", "low", "medium", "high"]
AudioQuality = Literal["clear", "slightly_impaired", "severely_impaired"]


class ClipResult(BaseModel):
    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str = ""
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)
