"""Fixtures shared by the job-level test modules."""

import pytest


@pytest.fixture
def jobs_module(tmp_path, monkeypatch):
    """The jobs module with every piece of real global state redirected.

    Three things would otherwise leak between tests or into the deployment:
    the on-disk job directory, the in-memory job registry, and the spend
    ledger. The ledger matters most -- _run_job settles every tone call
    against it, so an unpatched suite would write to the deployment's real
    ledger and bill itself against the live cap.
    """
    from vta import jobs
    from vta.budget import SpendLedger

    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "_JOBS", {})
    monkeypatch.setattr(
        jobs, "LEDGER", SpendLedger(path=tmp_path / "ledger.json", cap_usd=1.0)
    )
    jobs.JOBS_DIR.mkdir(parents=True)
    return jobs
