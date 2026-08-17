---
title: Voice Tone Analyzer
emoji: 🎧
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Emotional tone and background-noise analysis for call audio
---

# Voice Tone & Background Noise Analyzer

Classifies emotional tone and detects background noise / technical issues in
call-center audio, per the AutoAce AI technical trial brief. See `PLAN.md`
for the full design rationale and `TECHNICAL_MEMO.md` for the writeup of
approaches tested, validation results, cost/latency analysis, and known
limitations.

## Architecture

Nine required fields are resolved by seven independent local models/DSP
routines plus one external LLM call (only transcript text + numeric
features leave local infrastructure -- audio never does):

| Field | Source | Cost |
|---|---|---|
| `audio_quality` | DSP (clipping, SNR, bandwidth) | $0 |
| `background_noise_present/type/severity` | PANNs (CNN14) + DSP cross-check | $0 |
| `speaker_overlap_present` | pyannote/segmentation-3.0 (direct, not the full diarization pipeline -- see TECHNICAL_MEMO.md §1/§4) | $0 |
| `long_silence_present` | Silero VAD | $0 |
| `emotional_tone`, `emotional_intensity` | GPT-5.6 Luna (transcript + prosody features, 5-field evidence-ordered structured output) | ~$0.0003-0.0015/audio-min |
| (transcript, speaking rate) | nvidia/parakeet-tdt-0.6b-v2 (feeds the LLM head) | $0 |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH.

```bash
cd voice-tone-analyzer
cp .env.example .env   # fill in HF_TOKEN and OPENAI_API_KEY
uv sync
```

`HF_TOKEN` needs access to the gated `pyannote/segmentation-3.0` and
`pyannote/speaker-diarization-3.1` models (accept their terms on
huggingface.co, then generate a token with read access).

## Run the dashboard locally

```bash
uv run uvicorn app.main:app --port 8000
```

Visit `http://localhost:8000`, log in with `DASHBOARD_USERNAME`/`DASHBOARD_PASSWORD`
from `.env` (defaults: `autoace` / `changeme` -- **change these before
deploying**), upload a ZIP of the evaluation batch (or the audio files +
`labels.csv` directly), and download results as CSV/JSON once processing
completes.

### emotion2vec+

The `emotion2vec_plus_large` model (1.94GB) provides a corroborating
arousal signal for `emotional_intensity`, fetched via the Hugging Face Hub
mirror (`emotion2vec/emotion2vec_plus_large`) rather than funasr's default
modelscope loader -- the latter has no resume support and can fail
unpredictably on a slow connection; the HF mirror does real resumable range
requests. If it isn't predownloaded (see Docker build below) and a request
needs it before the download finishes, the pipeline degrades gracefully --
the tone/intensity head just runs without that cross-check. Set
`VTA_DISABLE_EMOTION2VEC=1` to skip it outright (e.g. slow/offline build
environments).

## Batch processing from the command line

```bash
uv run python scripts/generate_predictions.py   # the 3 provided calls -> predictions.json/csv
uv run python scripts/validate_synthetic.py     # synthetic validation -> validation_results.json
```

## Deployment (Docker)

```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t vta .
docker run -p 8000:8000 --env-file .env vta
```

The build predownloads parakeet, Silero VAD, pyannote/segmentation-3.0, and
emotion2vec+ so the deployed container never pays a cold-start model
download on first request (per the brief's latency requirement). PANNs
weights are also fetched at build time via direct `curl` (its own
downloader shells out to `wget`, unavailable in the slim image).

Any container host works (Fly.io, Render, HF Spaces, a VPS...) -- the app is
a standard FastAPI/uvicorn service with no host-specific code. Set
`SESSION_SECRET`, `DASHBOARD_USERNAME`, `DASHBOARD_PASSWORD`, `HF_TOKEN`, and
`OPENAI_API_KEY` as environment variables/secrets on whichever host is used.

The Dockerfile sets `VTA_ENV=production`, which makes the app **refuse to
start** if `SESSION_SECRET`, `DASHBOARD_USERNAME` or `DASHBOARD_PASSWORD` are
still at their development placeholders. Those placeholders are in
`src/vta/config.py` and therefore public; `SESSION_SECRET` is the session
cookie's signing key, so a known value lets anyone forge a logged-in session.
Generate one with:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

## Data handling

Uploaded audio is written to a per-job directory and processed in-process;
only derived transcript text and numeric acoustic features are sent to the
OpenAI API for the tone/intensity call. No audio is uploaded to any other
third-party service.

Audio does not persist indefinitely. A batch's job record and its on-disk
audio are both deleted after `VTA_JOB_RETENTION_S` (default 6 hours), and at
most `VTA_MAX_RETAINED_JOBS` (default 20) completed batches are kept;
directories orphaned by a previous process are reaped on the same schedule.
Uploads are capped at `VTA_MAX_UPLOAD_MB` (512) with ZIP expansion capped
separately at `VTA_MAX_EXTRACTED_MB` (1024).

Validation artifacts (`validation_results*.json`, sweep logs) are gitignored:
they embed verbatim IEMOCAP reference transcripts, which that corpus's
licence does not permit redistributing. `.github/workflows/sync-to-hub.yml`
re-checks this before pushing to a public Space and fails the deploy if any
such file, or any audio, is tracked.
