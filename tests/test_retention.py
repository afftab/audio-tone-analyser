"""Retention of uploaded call audio (brief §5 treats it as confidential)."""

import time

import pytest


@pytest.fixture
def jobs_module(tmp_path, monkeypatch):
    from vta import jobs

    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "_JOBS", {})
    jobs.JOBS_DIR.mkdir(parents=True)
    return jobs


def _make_job(jobs, job_id: str, *, age_s: float, status: str = "done"):
    job = jobs.Job(id=job_id, created_at=time.time() - age_s, status=status)
    jobs._JOBS[job_id] = job
    d = jobs.JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "call.ogg").write_bytes(b"audio")
    return job


def test_expired_jobs_are_purged_from_memory_and_disk(jobs_module):
    jobs = jobs_module
    _make_job(jobs, "old", age_s=jobs.RETENTION_S + 60)
    _make_job(jobs, "fresh", age_s=10)

    purged = jobs.purge_expired_jobs()

    assert purged == ["old"]
    assert "old" not in jobs._JOBS
    assert not (jobs.JOBS_DIR / "old").exists(), "audio must be deleted, not just forgotten"
    assert "fresh" in jobs._JOBS
    assert (jobs.JOBS_DIR / "fresh" / "call.ogg").exists()


def test_in_flight_jobs_are_never_purged(jobs_module):
    """A worker thread is mid-read of that audio."""
    jobs = jobs_module
    _make_job(jobs, "running", age_s=jobs.RETENTION_S * 10, status="processing")

    assert jobs.purge_expired_jobs() == []
    assert (jobs.JOBS_DIR / "running").exists()


def test_job_count_is_capped(jobs_module, monkeypatch):
    jobs = jobs_module
    monkeypatch.setattr(jobs, "MAX_RETAINED_JOBS", 3)
    for i in range(6):
        _make_job(jobs, f"j{i}", age_s=100 - i)  # j0 oldest

    jobs.purge_expired_jobs()

    assert sorted(jobs._JOBS) == ["j3", "j4", "j5"]
    for gone in ("j0", "j1", "j2"):
        assert not (jobs.JOBS_DIR / gone).exists()


def test_orphaned_directories_are_reaped(jobs_module):
    """Left by a previous process: unreachable through the UI, still audio."""
    jobs = jobs_module
    orphan = jobs.JOBS_DIR / "orphan"
    orphan.mkdir()
    (orphan / "call.ogg").write_bytes(b"audio")
    import os

    old = time.time() - (jobs.RETENTION_S + 3600)
    os.utime(orphan, (old, old))

    assert "orphan" in jobs.purge_expired_jobs()
    assert not orphan.exists()


def test_uploads_are_moved_not_copied(jobs_module, tmp_path):
    """Copying doubled peak disk for every batch and left the source behind."""
    jobs = jobs_module
    src = tmp_path / "incoming" / "call.ogg"
    src.parent.mkdir()
    src.write_bytes(b"audio")

    jobs._extract_batch([src], [], jobs.JOBS_DIR / "j")

    assert (jobs.JOBS_DIR / "j" / "call.ogg").read_bytes() == b"audio"
    assert not src.exists()
