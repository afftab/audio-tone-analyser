"""FastAPI dashboard: login, batch upload, progress, results, export (brief §7)."""

import hashlib
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from vta.config import (
    DASHBOARD_PASSWORD,
    DASHBOARD_USERNAME,
    IS_DEV,
    SESSION_SECRET,
    require_production_secrets,
)
from vta.jobs import (
    MAX_UPLOAD_BYTES,
    UploadTooLarge,
    create_job,
    get_job,
    results_csv,
    results_json,
)

BASE_DIR = Path(__file__).resolve().parent

# Fail fast rather than serve with forgeable session cookies.
require_production_secrets()

app = FastAPI(title="Voice Tone & Background Noise Analyzer")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    # Spaces terminates TLS in front of us; only relax this for local http.
    https_only=not IS_DEV,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _static_version(rel_path: str) -> str:
    """Content hash for a `?v=` cache-buster on static asset links.

    StaticFiles sets no Cache-Control, so browsers may serve a stale app.css
    after an edit/restart. A content-hashed URL forces a real fetch; hashed
    per request so it stays correct without a restart.
    """
    try:
        return hashlib.sha256((BASE_DIR / "static" / rel_path).read_bytes()).hexdigest()[:10]
    except OSError:
        return "0"


templates.env.globals["static_version"] = _static_version


# Coarse per-IP brute-force limiter. In-process only: resets on restart,
# spans no replicas. Enough to stall an unattended guessing loop.
LOGIN_MAX_ATTEMPTS = 8
LOGIN_WINDOW_S = 300.0
_login_attempts: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Spaces puts a proxy in front, so honour the first XFF hop when present.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _login_blocked(ip: str) -> bool:
    now = time.monotonic()
    with _login_lock:
        recent = [t for t in _login_attempts.get(ip, []) if now - t < LOGIN_WINDOW_S]
        _login_attempts[ip] = recent
        return len(recent) >= LOGIN_MAX_ATTEMPTS


def _record_failure(ip: str) -> None:
    with _login_lock:
        _login_attempts.setdefault(ip, []).append(time.monotonic())


def _clear_failures(ip: str) -> None:
    with _login_lock:
        _login_attempts.pop(ip, None)


def _is_authed(request: Request) -> bool:
    return bool(request.session.get("authed"))


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = _client_ip(request)
    if _login_blocked(ip):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Too many failed attempts. Try again in a few minutes."},
            status_code=429,
        )

    # compare_digest, not `==`: constant-time, and both fields always
    # compared so timing doesn't reveal which one was wrong.
    user_ok = secrets.compare_digest(username, DASHBOARD_USERNAME)
    pass_ok = secrets.compare_digest(password, DASHBOARD_PASSWORD)
    if user_ok and pass_ok:
        _clear_failures(ip)
        # New session on privilege change: a cookie fixed before login
        # must not survive it.
        request.session.clear()
        request.session["authed"] = True
        return RedirectResponse("/", status_code=303)

    _record_failure(ip)
    return templates.TemplateResponse(
        request, "login.html", {"error": "Invalid username or password"}, status_code=401
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not _is_authed(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/findings", response_class=HTMLResponse)
def findings(request: Request):
    if not _is_authed(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "findings.html", {})


@app.post("/upload")
async def upload(request: Request, files: list[UploadFile]):
    if not _is_authed(request):
        return RedirectResponse("/login", status_code=303)

    tmp_dir = Path(tempfile.mkdtemp(prefix="vta_upload_"))
    audio_paths: list[Path] = []
    zip_paths: list[Path] = []
    csv_path: Path | None = None

    try:
        budget = MAX_UPLOAD_BYTES
        for uf in files:
            if not uf.filename:
                continue
            dest = tmp_dir / Path(uf.filename).name
            # Stream: uf.read() would load the whole upload into RSS.
            written = 0
            with open(dest, "wb") as out:
                while chunk := await uf.read(1024 * 1024):
                    written += len(chunk)
                    if written > budget:
                        raise UploadTooLarge(
                            f"upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB batch limit"
                        )
                    out.write(chunk)
            budget -= written

            suffix = dest.suffix.lower()
            if suffix == ".zip":
                zip_paths.append(dest)
            elif suffix == ".csv":
                csv_path = dest
            else:
                audio_paths.append(dest)

        job = create_job(audio_paths, zip_paths, csv_path)
    except UploadTooLarge as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return templates.TemplateResponse(
            request, "index.html", {"error": str(e)}, status_code=413
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    else:
        # create_job has moved what it needs; leaving this would strand
        # confidential call audio in the system temp dir.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    if not _is_authed(request):
        return RedirectResponse("/login", status_code=303)
    job = get_job(job_id)
    if job is None:
        return HTMLResponse("Job not found", status_code=404)
    return templates.TemplateResponse(request, "job.html", {"job_id": job_id})


@app.get("/jobs/{job_id}/status.json")
def job_status(request: Request, job_id: str):
    # Real status codes: a 200 with an error body used to crash the client
    # poller into data.counts.
    if not _is_authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    job = get_job(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(job.summary())


@app.get("/jobs/{job_id}/download.csv")
def job_download_csv(request: Request, job_id: str):
    if not _is_authed(request):
        return RedirectResponse("/login", status_code=303)
    job = get_job(job_id)
    if job is None:
        return PlainTextResponse("not found", status_code=404)
    return PlainTextResponse(
        results_csv(job),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_results.csv"'},
    )


@app.get("/jobs/{job_id}/download.json")
def job_download_json(request: Request, job_id: str):
    if not _is_authed(request):
        return RedirectResponse("/login", status_code=303)
    job = get_job(job_id)
    if job is None:
        return PlainTextResponse("not found", status_code=404)
    return PlainTextResponse(
        results_json(job),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_results.json"'},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
