"""Guards for the things that are wrong on a public deployment.

These are regression tests for defects found in the sweep, not coverage for
its own sake: each one fails against the code as it was.
"""

import importlib
import io
import zipfile

import pytest


# --- Placeholder credentials must not reach a deployment ---

def _reload_config(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import vta.config

    return importlib.reload(vta.config)


def test_dev_environment_allows_placeholders(monkeypatch):
    cfg = _reload_config(
        monkeypatch, VTA_ENV="dev", SESSION_SECRET=None,
        DASHBOARD_PASSWORD=None, DASHBOARD_USERNAME=None,
    )
    cfg.require_production_secrets()  # must not raise on a laptop


@pytest.mark.parametrize("missing", ["SESSION_SECRET", "DASHBOARD_PASSWORD"])
def test_production_rejects_placeholder_secrets(monkeypatch, missing):
    env = {
        "VTA_ENV": "production",
        "SESSION_SECRET": "a-real-secret",
        "DASHBOARD_PASSWORD": "a-real-password",
        "DASHBOARD_USERNAME": "a-real-user",
    }
    env[missing] = None  # fall back to the published placeholder
    cfg = _reload_config(monkeypatch, **env)
    with pytest.raises(cfg.InsecureConfigError) as exc:
        cfg.require_production_secrets()
    assert missing in str(exc.value)


def test_production_accepts_real_secrets(monkeypatch):
    cfg = _reload_config(
        monkeypatch, VTA_ENV="production", SESSION_SECRET="s3cret",
        DASHBOARD_PASSWORD="hunter2", DASHBOARD_USERNAME="ops",
    )
    cfg.require_production_secrets()


# --- ZIP handling ---

def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_zip_members_cannot_escape_the_job_dir(tmp_path):
    """Traversal members must land inside job_dir, flattened to a basename."""
    from vta import jobs

    zpath = tmp_path / "evil.zip"
    zpath.write_bytes(_zip_bytes({"../../escaped.wav": b"x", "/abs/rooted.wav": b"y"}))
    job_dir = tmp_path / "job"

    jobs._extract_batch([], [zpath], job_dir)

    written = sorted(p.name for p in job_dir.iterdir())
    assert written == ["escaped.wav", "rooted.wav"]
    assert not (tmp_path.parent / "escaped.wav").exists()


def test_zip_bomb_is_refused(tmp_path, monkeypatch):
    """A highly compressible member must not be expanded without limit."""
    from vta import jobs

    monkeypatch.setattr(jobs, "MAX_EXTRACTED_BYTES", 1024 * 1024)
    zpath = tmp_path / "bomb.zip"
    zpath.write_bytes(_zip_bytes({"big.wav": b"\0" * (8 * 1024 * 1024)}))

    with pytest.raises(jobs.UploadTooLarge):
        jobs._extract_batch([], [zpath], tmp_path / "job")


def test_bad_zip_is_reported_not_raised(tmp_path):
    from vta import jobs

    zpath = tmp_path / "broken.zip"
    zpath.write_bytes(b"not a zip at all")
    errors = jobs._extract_batch([], [zpath], tmp_path / "job")
    assert any("not a valid ZIP" in e for e in errors)
