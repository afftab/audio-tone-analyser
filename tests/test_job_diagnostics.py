"""_run_job used to discard analyze_clip's diagnostics -- the dashboard had
no way to show a stage-timing breakdown, a cost decomposition, or why a clip
got the label it did. These are regression tests for that plumbing."""

import pytest

from vta.pipeline import AnalysisResult, PipelineDiagnostics
from vta.schema import ClipResult
from vta.tone_llm import TokenUsage

RESULT = ClipResult(
    emotional_tone="frustrated", emotional_intensity="medium",
    background_noise_present=False, background_noise_type="",
    background_noise_severity="none", audio_quality="clear",
    speaker_overlap_present=False, long_silence_present=False, confidence=0.82,
)


@pytest.fixture
def jobs_module(tmp_path, monkeypatch):
    from vta import jobs
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "_JOBS", {})
    jobs.JOBS_DIR.mkdir(parents=True)
    return jobs


def test_run_job_carries_diagnostics_into_the_file_outcome(jobs_module, tmp_path, monkeypatch):
    jobs = jobs_module
    job_dir = jobs.JOBS_DIR / "j1"
    job_dir.mkdir()
    (job_dir / "call_001.ogg").write_bytes(b"fake audio")

    job = jobs.Job(id="j1", created_at=0.0)
    job.files["call_001.ogg"] = jobs.FileOutcome(name="call_001.ogg", status="pending")
    jobs._JOBS["j1"] = job

    analysis = AnalysisResult(
        result=RESULT,
        diagnostics=PipelineDiagnostics(
            duration_s=30.9, processing_s=22.9, transcript="Come on. Hi, I'm Erica...",
            llm_reasoning="[lexical] evidence text | [acoustic] arousal text",
            llm_token_usage=TokenUsage(prompt_tokens=1842, cached_tokens=0,
                                        completion_tokens=510, reasoning_tokens=190),
            stage_timings_s={"asr": 9.3, "llm_tone": 7.2, "emotion2vec": 5.2},
        ),
    )
    monkeypatch.setattr(jobs, "analyze_clip", lambda path: analysis)

    jobs._run_job("j1", job_dir)

    outcome = job.files["call_001.ogg"]
    assert outcome.status == "done"
    assert outcome.result == RESULT.model_dump()
    assert outcome.stage_timings_s == {"asr": 9.3, "llm_tone": 7.2, "emotion2vec": 5.2}
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
    monkeypatch.setattr(jobs, "analyze_clip", boom)

    jobs._run_job("j2", job_dir)

    outcome = job.files["bad.ogg"]
    assert outcome.status == "error"
    assert outcome.stage_timings_s is None
    assert outcome.transcript is None
    assert outcome.cost_breakdown is None
