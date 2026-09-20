# Audio Performance Generator

A local-first audio clip generator and timeline editor. Generate voice, sound
effects and music from text, drop each result on a multi-track timeline, then
trim and arrange it.

Generation runs on **your own hardware** through a model server in Docker.
A cloud provider (ElevenLabs) remains available as an optional fallback, but
nothing requires it and no API key ever reaches the browser.

```
┌──────────────┐      /api      ┌──────────────────┐
│   frontend   │ ─────────────► │     backend      │
│ React + Vite │ ◄───────────── │ FastAPI + models │
│    nginx     │                │   CUDA / CPU     │
└──────────────┘                └────────┬─────────┘
                                         │
                            ┌────────────┴────────────┐
                            │  models:  HF weights    │
                            │  data:    audio, voices │
                            └─────────────────────────┘
```

## Quick start

Nothing to download, no GPU needed — this boots with placeholder providers that
emit tones, so you can see the whole pipeline work before committing to
multi-gigabyte weights.

```bash
cp .env.example .env
docker compose up --build
```

Then open <http://localhost:8080>. The **Server** tab lists every provider and
says exactly why each one is or is not available.

## Running with real models on a GPU

On the Proxmox box, with an NVIDIA GPU passed through and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed in the guest:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

The CUDA overlay switches the image to a CUDA base, installs the full model
stack (`EXTRAS=all`), sets `DEVICE=cuda`, and reserves the GPU.

Set `DEV_STUB=0` in `.env` once real providers are working so the placeholders
stop being offered.

### Proxmox: use a VM, not an LXC container

**Put the GPU in a Linux VM with VFIO passthrough.** LXC GPU sharing looks
simpler and is genuinely lighter, but it is the path where people lose days:
the guest's userspace NVIDIA driver must match the host's *exactly*, and
Proxmox VE 9 shipped an LXC 6.0.4 regression that broke
nvidia-container-toolkit in unprivileged containers outright until lxc-pve
6.0.5. With a VM, every one of those failure modes simply does not exist.

The one real risk with passthrough is IOMMU on consumer hardware, and it costs
30 seconds to check before committing. On the Proxmox node:

```bash
dmesg | grep -e DMAR -e IOMMU          # want: DMAR: IOMMU enabled
dmesg | grep remapping                 # want: Enabled IRQ remapping
```

If interrupt remapping is absent you need
`echo "options vfio_iommu_type1 allow_unsafe_interrupts=1" > /etc/modprobe.d/iommu_unsafe_interrupts.conf`.
Then blacklist the host drivers (`nouveau`, `nvidia*`), reboot, and add the card
to the VM as a PCI device.

Inside the VM, install Docker and nvidia-container-toolkit normally. Confirm the
GPU is visible to Docker *before* touching compose:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Note the tradeoff: the VM owns the card exclusively while it runs, so nothing
else on the Proxmox host can use it concurrently.

### Models

Defaults, and why:

| Track | Model | Weights licence | Peak VRAM |
| --- | --- | --- | --- |
| voice | **Chatterbox** (Resemble AI) | MIT — code *and* weights | ~3.2 GB |
| music | **ACE-Step 1.5** | MIT | 6–8 GB tier |
| sfx | **Stable Audio 3 small-sfx** | Stability Community (free under $1M rev) | ~2.4 GB |

Chatterbox is not the highest-scoring open TTS model — Kokoro-82M and Maya1
both rank above it and are Apache-2.0, though neither does zero-shot voice
cloning. What makes Chatterbox the right default here is that it is the only
expressive cloning model in the top tier whose weights are MIT: every
clearly better-sounding one (Breeze TTS 2, Fish S2 Pro, Higgs TTS 3, Voxtral
TTS, F5-TTS, XTTS-v2) ships research or non-commercial weights. Breeze's
licence restricts generated **outputs**, not just the weights.

MusicGen is still registered but is no longer a sensible default: CC-BY-NC
weights and outclassed by ACE-Step, which is MIT.

**Sizing the SFX/music models is a VRAM decision, not a taste one.** Stability
split the small tier because at 459M, mixing SFX and music data "degrades
musical coherence", while medium handles both in one checkpoint:

- **8 GB** — two small specialists, `small-sfx` + `small-music`, ~2.4 GB each.
  This is the default.
- **12 GB+** — set `SA3_MODEL=stabilityai/stable-audio-3-medium` and one
  ~6.5 GB checkpoint serves both tracks, with 380s max length instead of 120s.

> **Two traps with Stable Audio 3 medium.** Flash Attention 2 is **mandatory**
> and its absence fails *silently* — output collapses to static rather than
> raising. It needs compute capability ≥ 8.0 (RTX 3090 / 4090 class; Turing and
> older cannot run medium at all). The provider refuses to load rather than
> generate garbage, and reports why. The small checkpoints carry no such
> requirement.

### First run downloads weights

Weights are deliberately **not** baked into the image — that would put it at
15–25 GB and couple every code change to a multi-gigabyte rebuild. They
download on first use into the `models` named volume, which survives container
recreation. Expect ~10 GB and a slow first request per model.

The healthcheck allows a **15 minute** start period for exactly this. Failed
checks during that window do not count against `retries`, and one success ends
it early, so it costs nothing when startup is fast.

Stable Audio 3 is a **gated** HuggingFace repo: accept its licence on the model
page, then put a read token in `HF_TOKEN`. It is read at runtime, never baked
into the image.

## Configuration

Everything is environment-driven; see [.env.example](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `auto` | `auto` picks cuda → mps → cpu |
| `PROVIDERS` | `auto` | Comma-separated ids, or everything that imports |
| `DEV_STUB` | `1` | Offer the tone-generating placeholders |
| `MAX_CONCURRENT_JOBS` | `1` | Generation is serialised; one GPU, one model |
| `MODEL_IDLE_TIMEOUT` | `600` | Unload an idle model to free VRAM; `0` disables |
| `ELEVENLABS_API_KEY` | — | Optional cloud fallback; unset means not offered |
| `HF_TOKEN` | — | Required for gated repos (Stable Audio 3) |
| `SA3_MODEL` | — | Pin one checkpoint for both music and sfx (12 GB+) |
| `ACESTEP_BACKEND` | `pt` | `pt` avoids vLLM, whose pinned CUDA kernels are fragile |

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Liveness, device, GPU VRAM, provider readiness |
| `GET` | `/v1/models` | Providers, licences, voices, tunable params, defaults |
| `POST` | `/v1/audio/speech` | TTS; waits inline and returns the finished clip |
| `POST` | `/v1/audio/music` | Enqueues; returns `202` and a job id |
| `POST` | `/v1/audio/sfx` | Enqueues; returns `202` and a job id |
| `GET` | `/v1/jobs/{id}` | Job status; `DELETE` cancels one still queued |
| `GET` | `/v1/audio/{id}` | The generated bytes |
| `GET`/`POST`/`DELETE` | `/v1/voices` | Voice-clone reference samples |
| `GET` | `/v1/storage` | Disk usage |

Voice generation takes seconds, so the request simply waits. Music can take
minutes — far too long to hold a connection open through a proxy — so it
returns a job to poll. The frontend polls with backoff from 500ms to 3s.

Why no cancel mid-generation: once work is inside torch it cannot be
interrupted, so `DELETE /v1/jobs/{id}` only succeeds while a job is still
queued and returns `409` otherwise, rather than pretending.

## Development without Docker

```bash
# backend
cd backend
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
DEV_STUB=1 .venv/bin/python -m uvicorn app.main:app --reload --port 8000

# frontend
npm --prefix frontend install
npm --prefix frontend run dev
```

Vite proxies `/api` to `localhost:8000`, matching what nginx does in the
container, so `VITE_API_BASE` stays `/api` in both and CORS never arises.

Install a model stack with an extra: `.[tts]`, `.[music]`, `.[sfx]`, or `.[all]`.

## Adding a model

Subclass `Provider` in `backend/app/providers/`, implement `_load()` and
`generate()`, and register it in `backend/app/registry.py`. Declare its
`license` accurately — it is shown in the UI.

The frontend needs no change: it renders whatever `/v1/models` reports.

## Licensing

Model **weights** frequently carry different terms from the **code** that runs
them, and the difference is easy to miss. MusicGen is the classic trap: MIT
code, CC-BY-NC weights. F5-TTS is the same shape — MIT code, CC-BY-NC weights.
Each provider declares its own licence and the Server tab displays it.

The defaults above are all commercially usable: Chatterbox and ACE-Step are
MIT, Stable Audio 3 is free below a $1M annual revenue threshold. Swap in
anything else and check its weights terms first — and note that at least one
model in this space (Breeze TTS 2) restricts the **generated audio**, not just
the weights.

Voice cloning: only clone a voice you have permission to use. Several
jurisdictions now regulate this directly.

## Status

Working: local generation for all three track types behind a provider
abstraction, job queue, timeline placement, drag and trim, playback.

Not yet built: mixdown/export (the timeline cannot be bounced to a file),
project persistence (clips live in memory and are lost on reload, though their
audio survives on the server), and undo.
