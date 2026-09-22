# Audio Performance Generator

A local-first audio clip generator and timeline editor. Generate voice, sound
effects and music from text, drop each result on a multi-track timeline, then
trim and arrange it.

Generation runs on **your own GPU**. A cloud provider (ElevenLabs) stays
available as an optional fallback, but nothing requires it and no API key ever
reaches the browser.

## Where things run

| | Runs what | Why |
| --- | --- | --- |
| **Proxmox VM, RTX 5090** | The full stack — `compose.gpu.yml` | The models need the GPU |
| **Your Mac** | Vite dev server, and `docker-compose.yml` for smoke tests | arm64, no CUDA |

`docker-compose.yml` on the Mac runs **stub providers that emit tones**, not
real models. It exists to exercise the API, the job queue and the UI without a
GPU. It is not a smaller version of the real thing.

## Deploying to the GPU box

On the VM, once `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
works:

```bash
git clone <this repo> && cd audio-performance-generator
cp .env.example .env          # set HF_TOKEN
docker compose -f compose.gpu.yml up -d --build
```

Five containers come up: `gateway`, `tts`, `music`, `sfx`, `frontend`.
First boot downloads tens of GB of weights into the `models` volume — the
healthcheck allows 15 minutes for it.

### Why four backend containers instead of one

The three model stacks declare **mutually incompatible dependencies**. Straight
from their own metadata:

| package | python | torch | transformers |
| --- | --- | --- | --- |
| `chatterbox-tts` 0.1.7 | `<3.14` | `==2.6.0` | `==5.2.0` |
| `ACE-Step-1.5` | `>=3.11,<3.13` | `==2.10.0+cu128` | — |
| `stable-audio-3` | `>=3.10` | `==2.7.1` | `>=5.8.0` |

The torch pins are exact and mutually exclusive, and the transformers
constraints contradict outright. One Python environment cannot hold all three,
so each model gets its own image. The **gateway** merges them into a single
`/v1/models` and one set of endpoints, so the frontend sees one API and is
unaware of the split.

All three pins happen to resolve on the `cu128` index at cp312, which is why
every service shares CUDA 12.8.1 and Python 3.12 and differs only in torch.

### Blackwell (RTX 50-series) is sm_120

This bites in two places, both silent until runtime:

- **CUDA ≥ 12.8 and a torch from `cu128` or later are mandatory.** Earlier
  builds carry no sm_120 kernels and fail with *"no kernel image is available
  for execution on the device"*. `torch==2.6.0` — which `chatterbox-tts` pins —
  has **zero** wheels on any sm_120-capable index. The `TORCH_PIN` build arg
  overrides it as a resolver override, not merely an earlier install, because
  the resolver will otherwise downgrade right back.
- **`stable-audio-3`'s own pyproject points torch at the `cu126` index**, which
  is Blackwell-dead. Same override mechanism fixes it, keeping the `2.7.1` pin
  but taking the `cu128` build.

### Proxmox: use a VM, not an LXC container

LXC GPU sharing looks lighter but is where people lose days: the guest's
userspace NVIDIA driver must match the host's exactly, and Proxmox VE 9 shipped
an LXC 6.0.4 regression that broke nvidia-container-toolkit in unprivileged
containers until lxc-pve 6.0.5. With VFIO passthrough none of that exists.

Check IOMMU before committing — it costs 30 seconds:

```bash
dmesg | grep -e DMAR -e IOMMU     # want: DMAR: IOMMU enabled
dmesg | grep remapping            # want: Enabled IRQ remapping
```

Then blacklist `nouveau`/`nvidia*` on the host, reboot, and add the card to the
VM as a PCI device. The VM owns the card exclusively while it runs.

## Developing against the remote box

Run the UI on your Mac against the real models:

```bash
npm --prefix frontend install
VITE_API_TARGET=http://<gpu-vm-ip>:8000 npm --prefix frontend run dev
```

Vite proxies `/api` there, matching what nginx does in the container, so
`VITE_API_BASE` stays `/api` in both and CORS never arises.

For backend work, `docker context` lets you drive the remote daemon without
leaving your Mac:

```bash
docker context create gpubox --docker "host=ssh://user@gpu-vm"
docker --context gpubox compose -f compose.gpu.yml up -d --build
docker --context gpubox compose logs -f music
```

> **The API has no authentication.** It is fine on a trusted LAN. Do not expose
> port 8000 to the internet — put it behind Tailscale/WireGuard or a reverse
> proxy with auth. Anyone who can reach it can generate audio on your GPU and
> read every clip in the data volume.

## Models

| Track | Model | Weights licence | Peak VRAM |
| --- | --- | --- | --- |
| voice | **Chatterbox** (Resemble AI) | MIT — code *and* weights | ~3.2 GB |
| music | **ACE-Step 1.5** | MIT | 6–8 GB tier |
| sfx | **Stable Audio 3 small-sfx** | Stability Community (free under $1M rev) | ~2.4 GB |

Chatterbox is not the top-scoring open TTS model — Kokoro-82M and Maya1 both
rank above it and are Apache-2.0 — but neither does zero-shot voice cloning.
Chatterbox is the only expressive cloning model in the top tier whose weights
are MIT; every clearly better-sounding one (Breeze TTS 2, Fish S2 Pro, Higgs
TTS 3, Voxtral, F5-TTS, XTTS-v2) ships research or non-commercial weights, and
Breeze's licence restricts generated **outputs**, not just the weights.

MusicGen is still registered but is no longer a sensible default: CC-BY-NC
weights and outclassed by ACE-Step, which is MIT.

### Stable Audio 3: start with the small checkpoints

Sizing is a VRAM decision, not a taste one — Stability split the small tier
because at 459M, mixing SFX and music data "degrades musical coherence", while
medium handles both from one checkpoint.

At 32 GB you have room for medium, **but do not set it yet**:

> Medium **requires Flash Attention 2, and without it output silently collapses
> to static** rather than raising. Its published compute-capability list stops
> at 9.0 — **sm_120 is not on it**, and no confirmed report of medium running
> on an RTX 50-series card was found. The small checkpoints carry no FA2
> requirement, so they are the default. Set
> `SA3_MODEL=stabilityai/stable-audio-3-medium` only after confirming it
> produces real audio on your card, and listen to the first clip.

The provider refuses to load medium when `flash_attn` is absent rather than
generating garbage.

## Configuration

See [.env.example](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_TAG` | `12.8.1-cudnn-runtime-ubuntu24.04` | Must be ≥ 12.8 for sm_120 |
| `TORCH_INDEX` | `cu128` | First Blackwell-capable index |
| `MAX_CONCURRENT_JOBS` | `1` | Generation is serialised per service |
| `MODEL_IDLE_TIMEOUT` | `1800` | Unload idle models to free VRAM; `0` disables |
| `SA3_MODEL` | — | Leave blank for the small checkpoints |
| `ACESTEP_BACKEND` | `pt` | `pt` avoids nano-vllm, reported broken on sm_120 |
| `HF_TOKEN` | — | Required for gated repos (Stable Audio 3) |
| `ELEVENLABS_API_KEY` | — | Optional cloud fallback; blank means not offered |
| `VITE_API_TARGET` | `http://localhost:8000` | Where the Mac dev server proxies `/api` |

## API

The gateway presents one API over all model services.

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Per-upstream up/down, provider readiness |
| `GET` | `/v1/models` | Providers, licences, voices, params, which service serves each |
| `POST` | `/v1/audio/speech` | TTS; waits inline and returns the finished clip |
| `POST` | `/v1/audio/music` | Enqueues; returns `202` and a job id |
| `POST` | `/v1/audio/sfx` | Enqueues; returns `202` and a job id |
| `GET` | `/v1/jobs/{id}` | Job status; `DELETE` cancels one still queued |
| `GET` | `/v1/audio/{id}` | The generated bytes |
| `GET`/`POST`/`DELETE` | `/v1/voices` | Voice-clone reference samples |

Job ids are prefixed with their service (`music:abc123`) so polling routes
without server-side state and survives a gateway restart. Audio needs no
routing — every service shares one data volume and the gateway reads it
directly.

A queued job can be cancelled; a running one cannot. Once work is inside torch
it is uninterruptible, so `DELETE` returns `409` rather than pretending.

## Adding a model

Subclass `Provider` in `backend/app/providers/`, implement `_load()` and
`generate()`, register it in `backend/app/registry.py`. Declare its `license`
accurately — it is shown in the UI.

If it needs a torch version that conflicts with an existing service, give it
its own service in `compose.gpu.yml` with its own `TORCH_PIN`. The frontend
needs no change either way: it renders whatever `/v1/models` reports.

## Licensing

Model **weights** frequently carry different terms from the **code** that runs
them. MusicGen is the classic trap — MIT code, CC-BY-NC weights — and F5-TTS is
the same shape. Each provider declares its own licence and the Server tab shows
it.

The defaults are all commercially usable: Chatterbox and ACE-Step are MIT,
Stable Audio 3 is free below a $1M annual revenue threshold. Check any
replacement's weights terms, and note that at least one model in this space
(Breeze TTS 2) restricts the **generated audio**, not just the weights.

Voice cloning: only clone a voice you have permission to use.

## Status

Working: local generation for all three track types behind a provider
abstraction, a gateway over per-model services, job queue, timeline placement,
drag and trim, playback.

Not built: mixdown/export (the timeline cannot be bounced to a file), project
persistence (clips are in memory and lost on reload, though their audio
survives on the server), undo, and API authentication.
