"""End-to-end per-clip pipeline: audio file -> ClipResult.

Combines the deterministic 7-field acoustic stack with the LLM tone head,
per the build plan's architecture split (technical fields never touch the LLM;
tone/intensity never derived from noise/quality heuristics).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from vta.asr import transcribe
from vta.audio_io import load_normalized
from vta.diarize import diarize, speaker_audio
from vta.dsp_features import compute_dsp_features
from vta.emotion_head import EmotionResult, classify_emotion
from vta.events_panns import NoiseResult, detect_background_noise
from vta.overlap_pyannote import OverlapResult, detect_overlap
from vta.quality import classify_audio_quality
from vta.schema import ClipResult
from vta.tone_llm import ProsodySummary, TokenUsage, ToneJudgment, classify_tone
from vta.vad import VadResult, run_vad

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


@dataclass
class LocalAnalysis:
    """Everything analyze_clip computes before the LLM tone call. Split out
    so a batch of clips can run this part sequentially (it's CPU-bound and
    the local models are already internally multithreaded -- running many
    at once would oversubscribe the same cores, not speed anything up) while
    the LLM call, which is pure network I/O, runs concurrently across the
    batch instead. See jobs._run_job and TECHNICAL_MEMO.md §5."""

    t_start: float
    local_elapsed_s: float
    duration_s: float
    tone_transcript: str
    prosody: ProsodySummary
    per_speaker_prosody: dict[str, ProsodySummary] | None
    emotion_result: EmotionResult | None
    noise_result: NoiseResult
    overlap_result: OverlapResult
    vad_result: VadResult
    audio_quality: str
    timings: dict[str, float]


def _timed(fn, *args, **kwargs):
    # perf_counter, not time(): wall-clock can step backwards under an NTP
    # adjustment and these are durations, not timestamps.
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def analyze_clip(path: Path) -> AnalysisResult:
    """Single-clip convenience wrapper: local stages then the LLM call, in
    one call. Batch processing uses analyze_clip_local + classify_tone +
    finish_clip_with_tone directly instead, to run the LLM calls
    concurrently across a batch (see jobs._run_job)."""
    local = analyze_clip_local(path)
    (tone_judgment, token_usage), llm_s = _timed(
        classify_tone, local.tone_transcript, local.prosody, local.per_speaker_prosody or None
    )
    return finish_clip_with_tone(local, tone_judgment, token_usage, llm_s)


def analyze_clip_local(path: Path) -> LocalAnalysis:
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

    return LocalAnalysis(
        t_start=t_start,
        local_elapsed_s=time.perf_counter() - t_start,
        duration_s=audio.duration_s,
        tone_transcript=tone_transcript,
        prosody=prosody,
        per_speaker_prosody=per_speaker_prosody or None,
        emotion_result=emotion_result,
        noise_result=noise_result,
        overlap_result=overlap_result,
        vad_result=vad_result,
        audio_quality=audio_quality,
        timings=timings,
    )


def finish_clip_with_tone(
    local: LocalAnalysis,
    tone_judgment: ToneJudgment,
    token_usage: TokenUsage,
    llm_timing_s: float,
) -> AnalysisResult:
    """Combines LocalAnalysis with the (possibly concurrently-run) LLM tone
    result. Split from analyze_clip_local so a batch can run the LLM calls
    for many clips at once -- see jobs._run_job."""
    timings = dict(local.timings)
    timings["llm_tone"] = llm_timing_s

    # Taken as final: both acoustic post-corrections tried were removed
    # (see TECHNICAL_MEMO.md §2c).
    tone = tone_judgment.step2_emotional_tone
    intensity = tone_judgment.step5_emotional_intensity

    # emotion2vec still votes on confidence, on the neutral/non-neutral split
    # only -- no taxonomy mapping, so it can't reintroduce those errors. The
    # two values are a documented heuristic, not a fit (the build plan §6).
    confidence = 0.82
    if local.emotion_result is not None:
        scored = {k: v for k, v in local.emotion_result.posteriors.items()
                  if k not in ("unknown", "other")}
        if scored:
            acoustic_neutral = max(scored, key=scored.get) == "neutral"
            llm_neutral = tone == "neutral"
            confidence = 0.82 if acoustic_neutral == llm_neutral else 0.65

    clip_result = ClipResult(
        emotional_tone=tone,
        emotional_intensity=intensity,
        background_noise_present=local.noise_result.background_noise_present,
        background_noise_type=local.noise_result.background_noise_type,
        background_noise_severity=local.noise_result.background_noise_severity,
        audio_quality=local.audio_quality,
        speaker_overlap_present=local.overlap_result.speaker_overlap_present,
        long_silence_present=local.vad_result.long_silence_present,
        confidence=confidence,
    )

    diagnostics = PipelineDiagnostics(
        duration_s=local.duration_s,
        # Sum of actual work (local phase + this LLM call), not wall-clock
        # since local.t_start -- in a batch, the LLM calls for several clips
        # run concurrently, so wall-clock-since-start would double-count
        # queueing time as if it were this clip's own processing time.
        processing_s=local.local_elapsed_s + llm_timing_s,
        transcript=local.tone_transcript,
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
