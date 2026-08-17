"""End-to-end per-clip pipeline: audio file -> ClipResult.

Combines the deterministic 7-field acoustic stack with the LLM tone head,
per PLAN.md's architecture split (technical fields never touch the LLM;
tone/intensity never derived from noise/quality heuristics).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from vta.asr import transcribe
from vta.audio_io import load_normalized
from vta.diarize import diarize, speaker_audio
from vta.dsp_features import compute_dsp_features
from vta.emotion_head import classify_emotion
from vta.events_panns import detect_background_noise
from vta.overlap_pyannote import detect_overlap
from vta.quality import classify_audio_quality
from vta.schema import ClipResult
from vta.tone_llm import ProsodySummary, TokenUsage, classify_tone
from vta.vad import run_vad

# The old emotion2vec->schema tone override was removed (see analyze_clip);
# posteriors still reach the LLM as context but no longer set the label.


@dataclass
class PipelineDiagnostics:
    duration_s: float
    processing_s: float
    transcript: str
    llm_reasoning: str
    llm_token_usage: TokenUsage
    stage_timings_s: dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    result: ClipResult
    diagnostics: PipelineDiagnostics


def _timed(fn, *args, **kwargs):
    # perf_counter, not time(): wall-clock can step backwards under an NTP
    # adjustment and these are durations, not timestamps.
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def analyze_clip(path: Path) -> AnalysisResult:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    audio, timings["load_normalize"] = _timed(load_normalized, path)
    feats, timings["dsp_features"] = _timed(compute_dsp_features, audio)
    vad_result, timings["vad"] = _timed(run_vad, audio)
    asr_result, timings["asr"] = _timed(transcribe, audio)
    noise_result, timings["panns_noise"] = _timed(
        detect_background_noise, audio, feats.telephony_band_ratio,
        feats.occupied_bandwidth_hz,
    )
    overlap_result, timings["pyannote_overlap"] = _timed(detect_overlap, audio)
    try:
        emotion_result, timings["emotion2vec"] = _timed(classify_emotion, audio)
    except Exception:
        # Corroborating signal, not one of the 9 required fields: a failure
        # here must not fail the whole clip.
        emotion_result = None
        timings["emotion2vec"] = 0.0
    audio_quality = classify_audio_quality(feats)

    prosody = ProsodySummary(
        pitch_mean_hz=feats.pitch_mean_hz,
        pitch_std_hz=feats.pitch_std_hz,
        energy_mean_db=feats.energy_mean_db,
        energy_std_db=feats.energy_std_db,
        energy_dynamic_range_db=feats.energy_dynamic_range_db,
        voiced_ratio=feats.voiced_ratio,
        speaking_rate_wpm=asr_result.speaking_rate_wpm,
        energy_contour_db=feats.energy_contour_db,
        arousal=emotion_result.arousal if emotion_result else None,
        emotion_posteriors=emotion_result.posteriors if emotion_result else None,
    )
    # Speaker-labelled transcript so the LLM can isolate the customer; which
    # speaker that is stays the LLM's call (tone_llm.SYSTEM_PROMPT). Falls
    # back to the flat transcript when diarization is off.
    diarization, timings["diarize"] = _timed(diarize, audio, asr_result.words)
    tone_transcript = (
        diarization.labelled_transcript
        if diarization and diarization.labelled_transcript
        else asr_result.transcript
    )

    # Per-speaker prosody: whole-clip features average in the agent, who
    # holds ~half the speech. Both speakers are sent, keyed to the labels.
    per_speaker_prosody: dict[str, ProsodySummary] = {}
    t_ps = time.perf_counter()
    if diarization:
        for spk in sorted(diarization.speech_share):
            spk_audio = speaker_audio(audio, diarization.turns, spk)
            if spk_audio is None or spk_audio.duration_s < 0.5:
                continue
            try:
                spk_feats = compute_dsp_features(spk_audio)
            except Exception:
                continue
            spk_arousal = spk_posteriors = None
            try:
                spk_emotion = classify_emotion(spk_audio)
                spk_arousal, spk_posteriors = spk_emotion.arousal, spk_emotion.posteriors
            except Exception:
                pass
            per_speaker_prosody[spk] = ProsodySummary(
                pitch_mean_hz=spk_feats.pitch_mean_hz,
                pitch_std_hz=spk_feats.pitch_std_hz,
                energy_mean_db=spk_feats.energy_mean_db,
                energy_std_db=spk_feats.energy_std_db,
                energy_dynamic_range_db=spk_feats.energy_dynamic_range_db,
                voiced_ratio=spk_feats.voiced_ratio,
                # Speaking rate is measured over the whole clip by the ASR; a
                # per-speaker figure would need words grouped by speaker.
                speaking_rate_wpm=asr_result.speaking_rate_wpm,
                energy_contour_db=spk_feats.energy_contour_db,
                arousal=spk_arousal,
                emotion_posteriors=spk_posteriors,
            )
    timings["per_speaker_prosody"] = time.perf_counter() - t_ps

    (tone_judgment, token_usage), timings["llm_tone"] = _timed(
        classify_tone, tone_transcript, prosody, per_speaker_prosody or None
    )

    # Taken as final: both acoustic post-corrections tried were removed
    # (see TECHNICAL_MEMO.md §2c).
    tone = tone_judgment.step2_emotional_tone
    intensity = tone_judgment.step5_emotional_intensity

    # emotion2vec still votes on confidence, on the neutral/non-neutral split
    # only -- no taxonomy mapping, so it can't reintroduce those errors. The
    # two values are a documented heuristic, not a fit (PLAN.md §6).
    confidence = 0.82
    if emotion_result is not None:
        scored = {k: v for k, v in emotion_result.posteriors.items()
                  if k not in ("unknown", "other")}
        if scored:
            acoustic_neutral = max(scored, key=scored.get) == "neutral"
            llm_neutral = tone == "neutral"
            confidence = 0.82 if acoustic_neutral == llm_neutral else 0.65

    clip_result = ClipResult(
        emotional_tone=tone,
        emotional_intensity=intensity,
        background_noise_present=noise_result.background_noise_present,
        background_noise_type=noise_result.background_noise_type,
        background_noise_severity=noise_result.background_noise_severity,
        audio_quality=audio_quality,
        speaker_overlap_present=overlap_result.speaker_overlap_present,
        long_silence_present=vad_result.long_silence_present,
        confidence=confidence,
    )

    diagnostics = PipelineDiagnostics(
        duration_s=audio.duration_s,
        processing_s=time.perf_counter() - t_start,
        transcript=tone_transcript,
        llm_reasoning=" | ".join(
            [
                f"[lexical] {tone_judgment.step1_lexical_evidence}",
                f"[acoustic] {tone_judgment.step3_acoustic_evidence}",
                f"[intensity] {tone_judgment.step4_intensity_rationale}",
            ]
        ),
        llm_token_usage=token_usage,
        stage_timings_s=timings,
    )

    return AnalysisResult(result=clip_result, diagnostics=diagnostics)
