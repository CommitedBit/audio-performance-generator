# syntax=docker/dockerfile:1.7
#
# Two bases, one Dockerfile.
#   CPU  (default, works on arm64 Macs):  docker build backend
#   CUDA (the Proxmox GPU box):           docker build --build-arg BASE=cuda backend
#
# Python is provided by uv in both cases rather than by the distro, so the
# interpreter version is a build arg and neither base needs a PPA.
# Model weights are deliberately NOT baked in -- they are multi-GB and land in
# a named volume on the HF cache instead.

# These two are used in FROM lines, so they MUST be declared before the first
# FROM -- an ARG after it is stage-scoped and resolves to empty here.
ARG BASE=cpu
# CUDA_TAG must be >= 12.8 for Blackwell / RTX 50-series (sm_120). CUDA 12.4 and
# earlier carry no sm_120 kernels and fail at runtime with
# "no kernel image is available for execution on the device".
ARG CUDA_TAG=13.0.2-cudnn-runtime-ubuntu24.04

FROM debian:bookworm-slim AS cpu-base
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM nvidia/cuda:${CUDA_TAG} AS cuda-base
# Must be a Blackwell-capable index. cu128 is the FIRST one carrying sm_120 and
# it starts at torch 2.7.0; cu126 and older are Blackwell-dead.
ARG TORCH_INDEX=cu130
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_INDEX}

FROM ${BASE}-base AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_XET_HIGH_PERFORMANCE=1 \
    DATA_DIR=/data \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:$PATH

# ffmpeg: torchaudio and several providers shell out to it for decoding.
# libsndfile1: the soundfile backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Chatterbox is the reason this is a knob. chatterbox-tts 0.1.7 hard-pins
# torch==2.6.0 on Python < 3.14 -- and torch 2.6.0 has no wheel on ANY
# sm_120-capable index. On Python >= 3.14 the same package declares
# torch>=2.9.0, which does resolve to a Blackwell build.
ARG PYTHON_VERSION=3.12
RUN uv python install ${PYTHON_VERSION} \
    && uv venv /opt/venv --python ${PYTHON_VERSION}

WORKDIR /srv

# EXTRAS selects which model stacks get installed.
#   EXTRAS=""            API only, stub providers  (~805 MB, measured)
#   EXTRAS="tts"         + Chatterbox
#   EXTRAS="music,sfx"   + ACE-Step and Stable Audio 3
#   EXTRAS="all"         everything
ARG EXTRAS=""

# TORCH_PIN is applied as a dependency OVERRIDE, not merely installed first.
# Installing first is not enough: the model packages declare exact torch pins
# (chatterbox-tts torch==2.6.0, ACE-Step torch==2.10.0+cu128, stable-audio-3
# torch==2.7.1) and the resolver will happily downgrade to satisfy them --
# including to versions with no sm_120 kernels at all. An override file is what
# makes the pin stick. Space-separate multiple specs.
ARG TORCH_PIN=""

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${TORCH_PIN}" ]; then \
        printf '%s\n' ${TORCH_PIN} > /tmp/overrides.txt ; \
        uv pip install --index-strategy unsafe-best-match \
            --extra-index-url "${TORCH_INDEX_URL}" ${TORCH_PIN} ; \
    else \
        : > /tmp/overrides.txt ; \
    fi \
    && uv pip install --index-strategy unsafe-best-match \
        --extra-index-url "${TORCH_INDEX_URL}" \
        --overrides /tmp/overrides.txt \
        $( [ -n "${EXTRAS}" ] && echo ".[${EXTRAS}]" || echo "." ) \
    && python -c "import torch; print('torch', torch.__version__)" 2>/dev/null || true

COPY app ./app

RUN mkdir -p /data /models \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /srv /data /models /opt/venv
USER app

EXPOSE 8000

# start_period is 15 minutes on purpose: a first boot on a fresh volume may pull
# ~10 GB of weights. Checks that fail during the grace period do not count
# against `retries`, and one success ends it early, so this is free when fast.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --start-interval=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
