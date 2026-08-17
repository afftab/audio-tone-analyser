"""In-memory batch job manager: upload -> per-file processing -> results.

Single-process, in-memory state is a deliberate simplicity choice for an
evaluation dashboard (brief §7) -- not a claim this scales past one worker.

Retention: uploaded audio is production-call material, confidential per brief
§5, so batches are not kept indefinitely -- both the job record and the
on-disk copy are dropped after RETENTION_S, and the batch count is capped.
This is the minimum that stops an evaluation deployment accumulating call
recordings, not a real retention policy.
"""

import csv
import io
import json
import os
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vta.pipeline import analyze_clip
from vta.pricing import ClipCost, batch_totals, clip_cost_breakdown

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"}

JOBS_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "jobs"

# --- Limits ---
# Total upload bytes; the Space tier has 16GB RAM (several GB in models).
MAX_UPLOAD_BYTES = int(os.environ.get("VTA_MAX_UPLOAD_MB", "512")) * 1024 * 1024
# Bytes written while expanding a ZIP. A few-KB archive can expand to
# gigabytes: compression ratio matters here, not upload size.
MAX_EXTRACTED_BYTES = int(os.environ.get("VTA_MAX_EXTRACTED_MB", "1024")) * 1024 * 1024
MAX_FILES_PER_BATCH = int(os.environ.get("VTA_MAX_FILES", "500"))

# --- Retention ---
RETENTION_S = float(os.environ.get("VTA_JOB_RETENTION_S", str(6 * 3600)))
MAX_RETAINED_JOBS = int(os.environ.get("VTA_MAX_RETAINED_JOBS", "20"))


class UploadTooLarge(Exception):
    """Upload or ZIP expansion exceeded the configured byte budget."""


@dataclass
class FileOutcome:
    name: str
    status: str  # "pending" | "processing" | "done" | "error"
    result: dict | None = None
    error: str | None = None
    expected: dict | None = None  # from CSV result_json column, if provided
    processing_s: float | None = None
    audio_s: float | None = None
    cost_usd: float | None = None
    # Diagnostics from analyze_clip, surfaced for the dashboard.
    stage_timings_s: dict[str, float] | None = None
    transcript: str | None = None
    llm_reasoning: str | None = None
    token_usage: dict | None = None
    cost_breakdown: dict | None = None


@dataclass
class Job:
    id: str
    created_at: float
    status: str = "validating"  # validating | processing | done | error
    validation_errors: list[str] = field(default_factory=list)
    files: dict[str, FileOutcome] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def summary(self) -> dict:
        with self._lock:
            counts = {"pending": 0, "processing": 0, "done": 0, "error": 0}
            for f in self.files.values():
                counts[f.status] += 1
            costs = [
                ClipCost(usd=f.cost_usd, audio_s=f.audio_s)
                for f in self.files.values()
                if f.cost_usd is not None and f.audio_s is not None
            ]
            return {
                "id": self.id,
                "status": self.status,
                "validation_errors": self.validation_errors,
                "counts": counts,
                "total": len(self.files),
                "cost": batch_totals(costs),
                "files": [asdict(f) for f in self.files.values()],
            }


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _purge_job_dir(job_id: str) -> None:
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)


def purge_expired_jobs(now: float | None = None) -> list[str]:
    """Drop batches past RETENTION_S, and the oldest beyond MAX_RETAINED_JOBS.

    Called on upload rather than from a timer -- state only accumulates when
    the process is in use. A batch still processing is never purged; its
    worker is mid-read of the audio. Returns the purged ids.
    """
    now = time.time() if now is None else now
    with _JOBS_LOCK:
        finished = [j for j in _JOBS.values() if j.status in ("done", "error")]
        expired = {j.id for j in finished if now - j.created_at > RETENTION_S}
        # Oldest-first, keeping the newest MAX_RETAINED_JOBS.
        by_age = sorted(finished, key=lambda j: j.created_at)
        overflow = max(0, len(by_age) - MAX_RETAINED_JOBS)
        expired.update(j.id for j in by_age[:overflow])
        for job_id in expired:
            _JOBS.pop(job_id, None)

    for job_id in expired:
        _purge_job_dir(job_id)

    # Directories with no in-memory job are left over from a previous
    # process: unreachable through the UI, so pure retained audio.
    if JOBS_DIR.exists():
        with _JOBS_LOCK:
            live = set(_JOBS)
        for d in JOBS_DIR.iterdir():
            if not d.is_dir() or d.name in live:
                continue
            try:
                age = now - d.stat().st_mtime
            except OSError:
                continue
            if age > RETENTION_S:
                shutil.rmtree(d, ignore_errors=True)
                expired.add(d.name)

    return sorted(expired)


def _extract_batch(upload_paths: list[Path], zip_paths: list[Path], job_dir: Path) -> list[str]:
    """Materialize uploaded files (and any zip contents) into job_dir. Returns validation errors."""
    errors = []
    job_dir.mkdir(parents=True, exist_ok=True)
    budget = MAX_EXTRACTED_BYTES

    for zpath in zip_paths:
        try:
            with zipfile.ZipFile(zpath) as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                if len(members) > MAX_FILES_PER_BATCH:
                    errors.append(
                        f"{zpath.name}: {len(members)} entries exceeds the "
                        f"{MAX_FILES_PER_BATCH}-file limit, skipped"
                    )
                    continue
                for member in members:
                    # Flattening to the basename is also what makes this
                    # traversal-safe: Path("../../x").name is "x".
                    target = job_dir / Path(member.filename).name
                    # Budget the actual bytes, not the header's declared
                    # file_size, which an attacker controls.
                    with zf.open(member) as src, open(target, "wb") as dst:
                        while chunk := src.read(1024 * 1024):
                            budget -= len(chunk)
                            if budget < 0:
                                dst.close()
                                target.unlink(missing_ok=True)
                                raise UploadTooLarge(
                                    f"{zpath.name}: expands past the "
                                    f"{MAX_EXTRACTED_BYTES // (1024 * 1024)} MB limit"
                                )
                            dst.write(chunk)
        except zipfile.BadZipFile:
            errors.append(f"{zpath.name}: not a valid ZIP archive")

    for upath in upload_paths:
        target = job_dir / upath.name
        if str(upath) != str(target):
            # move, not copy -- the caller deletes its temp dir right after,
            # and copying doubled peak disk per batch.
            shutil.move(str(upath), str(target))

    return errors


def create_job(upload_paths: list[Path], zip_paths: list[Path], csv_path: Path | None) -> Job:
    purge_expired_jobs()

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job = Job(id=job_id, created_at=time.time())

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    try:
        errors = _extract_batch(upload_paths, zip_paths, job_dir)
    except UploadTooLarge:
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)
        _purge_job_dir(job_id)
        raise
    job.validation_errors.extend(errors)

    manifest_rows: dict[str, dict] = {}
    if csv_path is not None:
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "name" not in reader.fieldnames:
                    job.validation_errors.append("CSV manifest missing required 'name' column")
                else:
                    for row in reader:
                        name = row.get("name", "").strip()
                        if not name:
                            continue
                        raw_json = (row.get("result_json") or "").strip()
                        expected = None
                        if raw_json:
                            try:
                                expected = json.loads(raw_json)
                            except json.JSONDecodeError:
                                job.validation_errors.append(
                                    f"{name}: result_json is not valid JSON, ignoring"
                                )
                        manifest_rows[name] = expected
        except OSError as e:
            job.validation_errors.append(f"Could not read CSV manifest: {e}")

    on_disk = [p for p in job_dir.iterdir() if p.is_file()]
    audio_files = {
        p.name: p for p in on_disk if p.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    if manifest_rows:
        for name in manifest_rows:
            if name not in audio_files:
                job.validation_errors.append(f"{name}: listed in manifest but file not found in batch")
        for name in audio_files:
            if name not in manifest_rows:
                job.validation_errors.append(f"{name}: uploaded but not listed in manifest")

    for name, path in audio_files.items():
        job.files[name] = FileOutcome(
            name=name, status="pending", expected=manifest_rows.get(name)
        )

    unsupported = [
        p.name
        for p in on_disk
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS and p.suffix.lower() != ".csv"
    ]
    for name in unsupported:
        job.validation_errors.append(f"{name}: unsupported file type, skipped")

    job.status = "processing"
    thread = threading.Thread(target=_run_job, args=(job.id, job_dir), daemon=True)
    thread.start()

    return job


def _run_job(job_id: str, job_dir: Path) -> None:
    job = get_job(job_id)
    if job is None:
        return

    for name, outcome in list(job.files.items()):
        # Lock the field writes only, never analyze_clip: summary() takes
        # the same lock to serve the poller, and a clip takes tens of seconds.
        with job._lock:
            outcome.status = "processing"
        try:
            analysis = analyze_clip(job_dir / name)
            breakdown = clip_cost_breakdown(analysis.diagnostics.llm_token_usage)
            with job._lock:
                outcome.result = analysis.result.model_dump()
                outcome.processing_s = analysis.diagnostics.processing_s
                outcome.audio_s = analysis.diagnostics.duration_s
                outcome.cost_usd = breakdown.total_usd
                outcome.stage_timings_s = analysis.diagnostics.stage_timings_s
                outcome.transcript = analysis.diagnostics.transcript
                outcome.llm_reasoning = analysis.diagnostics.llm_reasoning
                outcome.token_usage = asdict(analysis.diagnostics.llm_token_usage)
                outcome.cost_breakdown = asdict(breakdown)
                outcome.status = "done"
        except Exception as e:  # noqa: BLE001 -- one bad file must not fail the batch
            with job._lock:
                outcome.status = "error"
                outcome.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"

    with job._lock:
        job.status = "done"


def results_csv(job: Job) -> str:
    buf = io.StringIO()
    fieldnames = [
        "name",
        "status",
        "error",
        "emotional_tone",
        "emotional_intensity",
        "background_noise_present",
        "background_noise_type",
        "background_noise_severity",
        "audio_quality",
        "speaker_overlap_present",
        "long_silence_present",
        "confidence",
        # Trailing diagnostic columns (not part of the required output schema).
        "audio_s",
        "processing_s",
        "cost_usd",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    with job._lock:
        outcomes = list(job.files.values())
    for outcome in outcomes:
        row = {"name": outcome.name, "status": outcome.status, "error": outcome.error or ""}
        if outcome.result:
            row.update(outcome.result)
        if outcome.audio_s is not None:
            row["audio_s"] = f"{outcome.audio_s:.1f}"
        if outcome.processing_s is not None:
            row["processing_s"] = f"{outcome.processing_s:.1f}"
        if outcome.cost_usd is not None:
            row["cost_usd"] = f"{outcome.cost_usd:.6f}"
        writer.writerow(row)
    return buf.getvalue()


def results_json(job: Job) -> str:
    with job._lock:
        costs = [
            ClipCost(usd=f.cost_usd, audio_s=f.audio_s)
            for f in job.files.values()
            if f.cost_usd is not None and f.audio_s is not None
        ]
        files = {name: asdict(outcome) for name, outcome in job.files.items()}
    return json.dumps({"cost": batch_totals(costs), "files": files}, indent=2)
