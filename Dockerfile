FROM python:3.11-slim AS base

# --- Root-only setup: system packages and the uv binary ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Hugging Face Spaces runs containers as UID 1000. ---
# /root is mode 700, so caches and weights must live under that user's HOME
# or the app cannot read them at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Anything but "dev" makes vta.config refuse to start on the public
    # placeholder credentials; set the real ones as Space secrets.
    VTA_ENV=production
WORKDIR $HOME/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY --chown=user . .
RUN uv sync --frozen

# --- Predownload model weights at build time, not on first request ---
# All caches resolve under $HOME so the runtime user owns them.
ENV VTA_DISABLE_EMOTION2VEC=""

RUN mkdir -p $HOME/panns_data && \
    curl -sL -o $HOME/panns_data/class_labels_indices.csv \
      "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" && \
    curl -sL -o "$HOME/panns_data/Cnn14_mAP=0.431.pth" \
      "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

RUN uv run python -c "\
from nemo.collections import asr as nemo_asr; \
nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')"

RUN uv run python -c "\
import torch; \
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)"

# pyannote/segmentation-3.0 is gated: build-time secrets must be mounted at
# /run/secrets, not passed as build-args. Needs an HF_TOKEN Space secret.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" uv run python -c "\
from pyannote.audio import Model; \
import os; \
Model.from_pretrained('pyannote/segmentation-3.0', token=os.environ['HF_TOKEN'])"

# emotion2vec+ (1.94GB) is best-effort: the pipeline degrades gracefully, so
# a failure here must not fail the build.
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='emotion2vec/emotion2vec_plus_large')" || \
    echo "WARNING: emotion2vec+ prefetch failed; will download on first use"

# Spaces routes to the port declared as `app_port` in README.md frontmatter.
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
