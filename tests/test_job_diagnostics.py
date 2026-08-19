"""_run_job used to discard analyze_clip's diagnostics -- the dashboard had
no way to show a stage-timing breakdown, a cost decomposition, or why a clip
got the label it did. These are regression tests for that plumbing.

_run_job now calls analyze_clip_local (Phase 1, sequential) and classify_tone
(Phase 2, concurrent across the batch) separately rather than one combined
analyze_clip call -- see pipeline.LocalAnalysis and TECHNICAL_MEMO.md §5.
Tests mock those two seams instead of analyze_clip directly."""

import time

import pytest

from vta.events_panns import NoiseResult
from vta.overlap_pyannote import OverlapResult
from vta.pipeline import LocalAnalysis
from vta.schema import ClipResult
from vta.tone_llm import ProsodySummary, TokenUsage, ToneJudgment
from vta.vad import VadResult

RESULT = ClipResult(
    emotional_tone="frustrated", emotional_intensity="medium",
    background_noise_present=False, background_noise_type="",
    background_noise_severity="none", audio_quality="clear",
    speaker_overlap_present=False, long_silence_present=False, confidence=0.82,
)

TONE_JUDGMENT = ToneJudgment(
    step1_lexical_evidence="evidence text",
    step2_emotional_tone="frustrated",
    step3_acoustic_evidence="arousal text",
    step4_intensity_rationale="intensity text",
    step5_emotional_intensity="medium",
)
TOKEN_USAGE = TokenUsage(
    prompt_tokens=1842, cached_tokens=0, completion_tokens=510, reasoning_tokens=190,
)


def _fake_local_analysis() -> LocalAnalysis:
    return LocalAnalysis(
        t_start=time.perf_counter(),
        local_elapsed_s=15.7,
        duration_s=30.9,
        tone_transcript="Come on. Hi, I'm Erica...",
        prosody=ProsodySummary(
            pitch_mean_hz=200.0, pitch_std_hz=20.0, energy_mean_db=-40.0,
            energy_std_db=10.0, energy_dynamic_range_db=30.0, voiced_ratio=0.6,
            speaking_rate_wpm=150.0, energy_contour_db=[-40.0],
        ),
        per_speaker_prosody=None,
        emotion_result=None,
        noise_result=NoiseResult(
            background_noise_present=False, background_noise_type="",
            background_noise_severity="none", top_labels=[], noise_prob_mass=0.0,
        ),
        overlap_result=OverlapResult(total_overlap_s=0.0, speaker_overlap_present=False),
        vad_result=VadResult(
            speech_segments=[(0.0, 30.9)], silence_gaps_s=[], longest_gap_s=0.0,
            long_silence_present=False, speech_ratio=1.0,
        ),
        audio_quality="clear",
        timings={"asr": 9.3, "emotion2vec": 5.2},
    )


def test_run_job_carries_diagnostics_into_the_file_outcome(jobs_module, tmp_path, monkeypatch):
    jobs = jobs_module
    job_dir = jobs.JOBS_DIR / "j1"
    job_dir.mkdir()
    (job_dir / "call_001.ogg").write_bytes(b"fake audio")

    job = jobs.Job(id="j1", created_at=0.0)
    job.files["call_001.ogg"] = jobs.FileOutcome(name="call_001.ogg", status="pending")
    jobs._JOBS["j1"] = job

    monkeypatch.setattr(jobs, "analyze_clip_local", lambda path: _fake_local_analysis())
    monkeypatch.setattr(jobs, "classify_tone", lambda *a, **k: (TONE_JUDGMENT, TOKEN_USAGE))

    jobs._run_job("j1", job_dir)

    outcome = job.files["call_001.ogg"]
    assert outcome.status == "done"
    assert outcome.result == RESULT.model_dump()
    assert outcome.stage_timings_s["asr"] == 9.3
    assert outcome.stage_timings_s["emotion2vec"] == 5.2
    assert "llm_tone" in outcome.stage_timings_s  # added by finish_clip_with_tone (Phase 2)
    assert outcome.transcript.startswith("Come on")
    assert "[lexical]" in outcome.llm_reasoning
    assert outcome.token_usage == {
        "prompt_tokens": 1842, "cached_tokens": 0,
        "completion_tokens": 510, "reasoning_tokens": 190,
    }
    # cost_breakdown's components must reconcile with the single cost_usd
    # figure everything else in the dashboard (and the CSV) reports.
    assert outcome.cost_breakdown["fresh_tokens"] == 1842
    total = (
        outcome.cost_breakdown["fresh_input_usd"] + outcome.cost_breakdown["cached_input_usd"]
        + outcome.cost_breakdown["reasoning_usd"] + outcome.cost_breakdown["visible_output_usd"]
    )
    assert outcome.cost_usd == pytest.approx(total)


def test_run_job_failure_leaves_diagnostics_unset(jobs_module, monkeypatch):
    """A clip that errors must not report stale or partial diagnostics."""
    jobs = jobs_module
    job_dir = jobs.JOBS_DIR / "j2"
    job_dir.mkdir()
    (job_dir / "bad.ogg").write_bytes(b"corrupt")

    job = jobs.Job(id="j2", created_at=0.0)
    job.files["bad.ogg"] = jobs.FileOutcome(name="bad.ogg", status="pending")
    jobs._JOBS["j2"] = job

    def boom(path):
        raise RuntimeError("decode failed")
    monkeypatch.setattr(jobs, "analyze_clip_local", boom)

    jobs._run_job("j2", job_dir)

    outcome = job.files["bad.ogg"]
    assert outcome.status == "error"
    assert outcome.stage_timings_s is None
    assert outcome.transcript is None
    assert outcome.cost_breakdown is None


def test_run_job_runs_llm_calls_concurrently_across_the_batch(jobs_module, monkeypatch):
    """The whole point of the Phase 1 / Phase 2 split: N clips' LLM calls
    should overlap in wall-clock time, not run back to back. A sleeping fake
    classify_tone makes this measurable instead of assumed."""
    jobs = jobs_module
    job_dir = jobs.JOBS_DIR / "j3"
    job_dir.mkdir()

    n = 5
    sleep_s = 0.2
    job = jobs.Job(id="j3", created_at=0.0)
    for i in range(n):
        name = f"call_{i}.ogg"
        (job_dir / name).write_bytes(b"fake audio")
        job.files[name] = jobs.FileOutcome(name=name, status="pending")
    jobs._JOBS["j3"] = job

    monkeypatch.setattr(jobs, "analyze_clip_local", lambda path: _fake_local_analysis())

    def slow_classify_tone(*a, **k):
        time.sleep(sleep_s)
        return TONE_JUDGMENT, TOKEN_USAGE

    monkeypatch.setattr(jobs, "classify_tone", slow_classify_tone)

    t0 = time.perf_counter()
    jobs._run_job("j3", job_dir)
    elapsed = time.perf_counter() - t0

    assert all(f.status == "done" for f in job.files.values())
    # Sequential would take n * sleep_s (1.0s here); concurrent should take
    # roughly one sleep_s plus overhead. The midpoint is a generous margin
    # against CI slowness without being able to pass if it ran sequentially.
    assert elapsed < (n * sleep_s) / 2, (
        f"took {elapsed:.2f}s -- looks sequential, not concurrent (n={n}, sleep_s={sleep_s})"
    )
