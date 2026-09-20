# syntax=docker/dockerfile:1.7
#
# Two bases, one Dockerfile.
#   CPU  (default, works on arm64 Macs):  docker build backend
#   CUDA (Proxmox GPU box):               docker build --build-arg BASE=cuda backend
#
# Model weights are deliberately NOT baked in. They are multi-GB and would make
# the image unusable; they land in a named volume on the HF cache instead.

FROM python:3.11-slim-bookworm AS cpu-base
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda-base
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

ARG BASE=cpu
FROM ${BASE}-base AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_XET_HIGH_PERFORMANCE=1 \
    DATA_DIR=/data

# ffmpeg: torchaudio and several providers shell out to it for decoding.
# libsndfile1: soundfile backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv

# EXTRAS selects which model stacks get installed. Keep it minimal to keep the
# image small; "all" pulls torch + diffusers + transformers (several GB).
#   EXTRAS=""            API only, stub providers  (~250 MB)
#   EXTRAS="tts"         + Chatterbox
#   EXTRAS="music,sfx"   + MusicGen and Stable Audio
#   EXTRAS="all"         everything
ARG EXTRAS=""

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-strategy unsafe-best-match \
        --extra-index-url "${TORCH_INDEX_URL}" \
        $( [ -n "${EXTRAS}" ] && echo ".[${EXTRAS}]" || echo "." ) \


COPY app ./app

RUN mkdir -p /data /models && \
    useradd --create-home --uid 10001 app && \
    chown -R app:app /srv /data /models
USER app

EXPOSE 8000

# start_period is 15 minutes on purpose: a first boot on a fresh volume may pull
# ~10 GB of weights. Checks that fail during the grace period do not count
# against `retries`, and one success ends it early, so this is free when fast.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --start-interval=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
