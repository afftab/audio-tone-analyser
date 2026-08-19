"""The spend cap is the only thing standing between a public login and an
unbounded OpenAI bill, so its two hard properties get tested directly:

  - it survives a restart (the ledger is on disk, not in process memory)
  - it survives concurrency (LLM_CONCURRENCY workers cannot each pass the
    same "is there room?" check and collectively overshoot)

Plus the batch-level behaviour: running out of budget marks clips `skipped`
and makes no API call, rather than failing them as pipeline errors.
"""

import threading

import pytest

from vta.budget import BudgetExhausted, SpendLedger


@pytest.fixture
def ledger(tmp_path):
    """A cap of 1 cent with 0.2-cent reservations: 5 calls of headroom."""
    return SpendLedger(
        path=tmp_path / "ledger.json", cap_usd=0.01, reserve_per_call_usd=0.002
    )


def test_spend_accumulates_and_reports_remaining(ledger):
    assert ledger.state().spent_usd == 0.0
    assert ledger.state().remaining_usd == pytest.approx(0.01)

    for _ in range(3):
        assert ledger.reserve() is True
        ledger.settle(0.0005)

    state = ledger.state()
    assert state.calls == 3
    assert state.spent_usd == pytest.approx(0.0015)
    assert state.remaining_usd == pytest.approx(0.0085)
    assert state.exhausted is False


def test_reserve_refuses_once_the_cap_would_be_crossed(ledger):
    # Settle right up to the edge: 0.009 spent leaves 0.001, under one
    # 0.002 reservation.
    assert ledger.reserve() is True
    ledger.settle(0.009)

    assert ledger.state().exhausted is True
    assert ledger.reserve() is False
    # Refusing must not itself consume headroom.
    assert ledger.state().reserved_usd == 0.0


def test_a_failed_call_gives_its_reservation_back(ledger):
    assert ledger.reserve() is True
    assert ledger.state().reserved_usd == pytest.approx(0.002)

    ledger.release()

    state = ledger.state()
    assert state.reserved_usd == 0.0
    assert state.spent_usd == 0.0
    assert state.calls == 0


def test_spend_survives_a_restart(tmp_path):
    path = tmp_path / "ledger.json"
    first = SpendLedger(path=path, cap_usd=0.01, reserve_per_call_usd=0.002)
    first.reserve()
    first.settle(0.004)

    # A new process reads the same file rather than starting from zero.
    second = SpendLedger(path=path, cap_usd=0.01, reserve_per_call_usd=0.002)
    assert second.state().spent_usd == pytest.approx(0.004)
    assert second.state().calls == 1


def test_an_unreadable_ledger_reads_as_zero_rather_than_crashing(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{ this is not json")

    ledger = SpendLedger(path=path, cap_usd=0.01)
    assert ledger.state().spent_usd == 0.0


def test_concurrent_reservations_never_exceed_the_cap(tmp_path):
    """The reason reserve/settle exists instead of a plain remaining() check.

    Twenty threads race for five reservations' worth of headroom. Exactly
    five may win: any more means two threads read the same headroom and both
    decided it was theirs.
    """
    ledger = SpendLedger(
        path=tmp_path / "ledger.json", cap_usd=0.01, reserve_per_call_usd=0.002
    )
    granted = []
    granted_lock = threading.Lock()
    start = threading.Event()

    def worker():
        start.wait()
        if ledger.reserve():
            with granted_lock:
                granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    start.set()  # release them together, to actually contend
    for t in threads:
        t.join()

    assert len(granted) == 5
    assert ledger.state().reserved_usd == pytest.approx(0.01)


def test_exhausted_budget_skips_clips_without_calling_the_api(
    jobs_module, monkeypatch
):
    """A budget stop is not a pipeline failure, and must not bill anything."""
    from test_job_diagnostics import TOKEN_USAGE, TONE_JUDGMENT, _fake_local_analysis

    jobs = jobs_module
    # Drain the ledger the fixture installed.
    jobs.LEDGER.reserve()
    jobs.LEDGER.settle(jobs.LEDGER.cap_usd)
    assert jobs.LEDGER.state().exhausted is True

    job_dir = jobs.JOBS_DIR / "j1"
    job_dir.mkdir()
    job = jobs.Job(id="j1", created_at=0.0)
    for name in ("a.ogg", "b.ogg"):
        (job_dir / name).write_bytes(b"fake audio")
        job.files[name] = jobs.FileOutcome(name=name, status="pending")
    jobs._JOBS["j1"] = job

    calls = []

    def _tone(*args, **kwargs):
        calls.append(1)
        return TONE_JUDGMENT, TOKEN_USAGE

    monkeypatch.setattr(jobs, "analyze_clip_local", lambda path: _fake_local_analysis())
    monkeypatch.setattr(jobs, "classify_tone", _tone)

    jobs._run_job("j1", job_dir)

    assert calls == [], "the paid call must not run once the cap is reached"
    for outcome in job.files.values():
        assert outcome.status == "skipped"
        assert outcome.result is None
        assert "Spend cap reached" in outcome.error

    summary = job.summary()
    assert summary["counts"]["skipped"] == 2
    assert summary["counts"]["error"] == 0
    assert summary["budget"]["exhausted"] is True


def test_a_batch_within_budget_still_runs_and_records_spend(jobs_module, monkeypatch):
    from test_job_diagnostics import TOKEN_USAGE, TONE_JUDGMENT, _fake_local_analysis

    jobs = jobs_module
    job_dir = jobs.JOBS_DIR / "j2"
    job_dir.mkdir()
    (job_dir / "a.ogg").write_bytes(b"fake audio")
    job = jobs.Job(id="j2", created_at=0.0)
    job.files["a.ogg"] = jobs.FileOutcome(name="a.ogg", status="pending")
    jobs._JOBS["j2"] = job

    monkeypatch.setattr(jobs, "analyze_clip_local", lambda path: _fake_local_analysis())
    monkeypatch.setattr(jobs, "classify_tone", lambda *a, **k: (TONE_JUDGMENT, TOKEN_USAGE))

    jobs._run_job("j2", job_dir)

    outcome = job.files["a.ogg"]
    assert outcome.status == "done"
    # The ledger charged exactly what the dashboard reported for the clip.
    assert jobs.LEDGER.state().spent_usd == pytest.approx(outcome.cost_usd)
    assert jobs.LEDGER.state().calls == 1
