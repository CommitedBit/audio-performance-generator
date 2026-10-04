# Audio Performance Generator

[![ci](https://github.com/CommitedBit/audio-performance-generator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CommitedBit/audio-performance-generator/actions/workflows/ci.yml)

> **New here?**
> - [HANDOFF.md](HANDOFF.md): where the project stands and what comes next.
> - [AGENTS.md](AGENTS.md): setup and the rules for anyone changing the
>   code.
> - [docs/codex-kickoff.md](docs/codex-kickoff.md): the ready-made first
>   message, if you're handing the project to a coding agent.

A local-first audio clip generator and timeline editor. Generate voice, sound
effects and music from text, drop each result on a multi-track timeline, then
trim and arrange it.

Generation runs on **your own GPU**. ElevenLabs stays available as an optional
cloud provider. It is used only when you pick it, never as an automatic
fallback unless you opt in with `ALLOW_CLOUD_DEFAULT=1`, and its API key never
reaches the browser.

## Where things run

| | Runs what | Why |
| --- | --- | --- |
| **Proxmox VM, RTX 5090** | The full stack — `compose.gpu.yml` | The models need the GPU |
| **Your Mac** | Vite dev server; `docker-compose.yml` for smoke tests | arm64, no CUDA |

`docker-compose.yml` runs **tone-generating stubs, not real models**. It exists
to exercise the API, the job queue and the UI without a GPU. It is not a
smaller version of the real thing.

## Deploying to the GPU box

Build **on** the GPU box and drive it from your Mac. Do not build locally:
nothing CUDA can run on an arm64 Mac, and compiling CUDA extensions under QEMU
is a multi-day proposition.

```bash
docker context create gpu --docker "host=ssh://user@gpu-vm"
export DOCKER_CONTEXT=gpu
docker compose -f compose.gpu.yml up -d --build
```

Only the build context crosses SSH — never the 15–20 GB image. Day to day:

```bash
docker compose -f compose.gpu.yml logs -f models
docker compose -f compose.gpu.yml exec models nvidia-smi
```

Prerequisite: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
must work inside the VM first.

### First run: `scripts/smoke_gpu.py`

Use this instead of the `up` command above for the first start, and again after
any change to models, pins or the Dockerfile:

```bash
cp .env.example .env              # set HF_TOKEN, and API_KEY if exposing
python3 scripts/smoke_gpu.py      # add --split for the split topology
```

It works on the VM or from the Mac with `DOCKER_CONTEXT=gpu`, because every
request goes through `docker compose exec`. The steps:

1. **Preflight.** Checks that the GPU is visible inside containers, that there
   is enough free disk, and that `HF_TOKEN` is set. It also reminds you to
   accept the gated Stable Audio 3 licence.
2. **Build and start**, then wait for `/health/ready`.
3. **Generate.** One seeded clip each for voice (Chatterbox), music (ACE-Step)
   and sfx (Stable Audio 3), each with the model named explicitly. This runs
   three times:
   - **first:** downloads the weights, so allow up to an hour on a fresh volume
   - **warm:** the models are already loaded
   - **cold:** after restarting the model services, the weights load from disk
4. **Check every clip.** Each must come from the right model, be the right
   length, and not be silent. The script also records each model's time and
   peak VRAM.

Everything lands in `smoke-results/<timestamp>/` (or the new directory
`--out DIR` names): the clips, `report.txt`, and
`results.json`, which holds the timings and VRAM peaks used for model sizing.
On failure the compose logs are saved there too. Listen to the clips:
automated checks can tell sound from silence, but not good from bad.

`--quick` skips the restart round. `--dev` runs the same flow against the local
placeholder stack, to test the script itself.

`--idle-check` is opt-in. It checks that idle unloading gives the VRAM back.
First it restarts the model services, so the baseline comes from processes that
have never loaded a model. After the last round it waits until no model is
loaded, then requires two things:

- **Each model service's own PyTorch allocations** (from its `/health`) are
  back within 0.25 GiB of the baseline. This is what catches a model left
  referenced: even the smallest, Stable Audio 3 small (459M parameters), holds
  about 0.85 GiB at half precision.
- **The card's VRAM** (from `nvidia-smi`) is back within 1 GiB per model
  service. Each process keeps the kernels and library workspaces it loaded, so
  this check alone would miss a small model left resident. It catches memory
  held outside PyTorch.

`report.txt` and `results.json` record the baseline, peak and after-unload VRAM
and both residuals. Neither allowance has been measured on the 5090. The
default `MODEL_IDLE_TIMEOUT` (1800 s) is too long to wait for, and the check
refuses anything over 300 s, so give the run a short one; compose passes it
through:

```bash
MODEL_IDLE_TIMEOUT=60 python3 scripts/smoke_gpu.py --idle-check
```

A timeout that short also unloads models between rounds, so take timings and
peaks from a run without it, and add `--no-build --quick` when the check is a
second pass. The stack keeps the short timeout until it is next started without
it (`docker compose -f compose.gpu.yml up -d`). With `--dev` the stubs load and
unload but hold no VRAM: the unload is checked, and VRAM is reported as not
measured. The check has not yet run on the GPU.

### Disk

Weights are roughly **31 GB** (ACE-Step ~10 GB, Stable Audio 3 medium ~10.5 GB,
Chatterbox 3–14 GB). With images, build cache and generated audio, give the VM
**500 GB** on an NVMe-backed virtual disk. Use a real virtual disk, not
virtiofs.

Weights live in the `models` **named volume** — daemon-side, and symlink-capable
for HuggingFace's blobs/snapshots layout. Generated audio and voice references
bind-mount to `DATA_PATH` (default `/srv/story/data`) so you can find them
outside Docker. You don't need to create or `chown` it: a one-shot `data-init`
container gives it to the app's uid (10001) before anything else starts, and
the app containers themselves never run as root.

### Proxmox: use a VM, not an LXC container

LXC GPU sharing looks lighter but is where people lose days: the guest's
userspace driver must match the host's exactly, and Proxmox VE 9 shipped an LXC
6.0.4 regression that broke nvidia-container-toolkit in unprivileged containers
until lxc-pve 6.0.5. VFIO passthrough avoids all of it.

```bash
dmesg | grep -e DMAR -e IOMMU     # want: DMAR: IOMMU enabled
dmesg | grep remapping            # want: Enabled IRQ remapping
```

Then blacklist `nouveau`/`nvidia*` on the host, reboot, add the card to the VM
as a PCI device. The VM owns the card exclusively while it runs.

## Why the dependency handling looks unusual

The three model packages declare **disjoint transformers ranges**:

| package | transformers | torch (declared) |
| --- | --- | --- |
| `ACE-Step-1.5` | `>=4.51.0,<4.58.0` | `==2.10.0+cu128` (linux) |
| `chatterbox-tts` | `==5.2.0` | `==2.6.0` |
| `stable-audio-3` | `>=5.8.0` | `==2.7.1` |

No resolver can satisfy those together, so all three install with `--no-deps`
against the hand-written pin set in
[backend/requirements-models.txt](backend/requirements-models.txt).

The **torch pins are not code constraints**. ACE-Step declares
`torch==2.7.1+cu128` on Windows and `torch==2.10.0+cu128` on Linux from the same
codebase, and none of the three gates on `torch.__version__` anywhere in its
source. One torch (2.7.1+cu128, which ships sm_120) serves all three, so this is
**one model container**, not three.

> This combination is not blessed by any of the three packages. It is derived
> from their actual import surfaces and needs validating on first run — see
> below. If it misbehaves, `compose.gpu.split.yml` gives ACE-Step its own
> container. No application code changes: the gateway makes topology a
> compose-file decision.

### Validating the unified environment

Before building, check that every GPU image's dependency set still resolves.
This needs no GPU and takes seconds:

```bash
python3 scripts/check_deps.py
```

It reads the build args from the compose files and resolves each set the way
the Dockerfile installs it (torch pins as overrides, Python 3.12, x86_64
manylinux), then again wheel-only. The two results must match. If they don't,
some package would have to be compiled on the GPU box. CI runs this on every
pull request.

After the stack boots, `scripts/smoke_gpu.py` (above) generates and checks one
clip per track type. To see why a model is unavailable:

```bash
docker compose -f compose.gpu.yml exec gateway curl -s 127.0.0.1:8000/health/ready
```

A wrong transformers version usually surfaces as an ImportError at model load,
which shows up in the provider's `unavailable_reason` and in the readiness
problems.

The generation path also runs an automatic sanity check on every clip and fails
the job rather than storing silence — see *Silent failures* below.

### Blackwell (RTX 50-series) is sm_120

CUDA ≥ 12.8 and a torch from `cu128` or later are mandatory; earlier builds
carry no sm_120 kernels and fail with *"no kernel image is available for
execution on the device"*. Two specific traps:

- `torch==2.6.0`, which `chatterbox-tts` pins, has **zero** wheels on any
  sm_120-capable index.
- `stable-audio-3`'s pyproject points torch at the **cu126** index, which is
  also Blackwell-dead.

`TORCH_PIN` overrides both — as a *resolver override*, not merely an earlier
install, because the resolver would otherwise downgrade straight back.

## Models

| Track | Model | Weights licence | Peak VRAM |
| --- | --- | --- | --- |
| voice | **Chatterbox** (Resemble AI) | MIT — code *and* weights | ~3.2 GB |
| music | **ACE-Step 1.5** | MIT | 6–8 GB tier |
| sfx | **Stable Audio 3** | Stability Community (free under $1M rev) | 2.4 GB small / 6.5 GB medium |

Chatterbox is not the top-scoring open TTS model — Kokoro-82M and Maya1 rank
above it and are Apache-2.0 — but neither does zero-shot voice cloning.
Chatterbox is the only expressive cloning model in the top tier whose weights
are MIT; every clearly better-sounding one ships research or non-commercial
weights, and Breeze TTS 2's licence restricts generated **outputs**, not just
the weights.

MusicGen is still registered but is not a sensible default: CC-BY-NC weights,
outclassed by ACE-Step, which is MIT.

**Flash Attention is not required.** Stable Audio 3 wraps its `flash_attn`
import in try/except and falls through a four-tier cascade its own source
comments call *math-equivalent* (flex_attention → chunked-halo SDPA → full
masked SDPA). The "output collapses to static" warning still in its README
describes a bug fixed in May 2026. Installing flash-attn is an optional
speed-up; on sm_120 it would pin the image to one prebuilt wheel, so it is left
out.

At 32 GB you have room for `SA3_MODEL=stabilityai/stable-audio-3-medium`, which
serves music and sfx from one checkpoint with 380 s max length instead of 120 s.

## Security

The API ships **unauthenticated and bound to loopback**. Both matter:

- Compose publishes on `127.0.0.1` by default. Docker's published ports
  **bypass ufw**, so `0.0.0.0` would expose the GPU to your whole LAN.
- Set `API_KEY` before exposing it anywhere, then enter the same value in the
  UI on the **Server** page. The browser stores it and sends it with every
  request, and the gateway enforces it; `/health` stays open so container
  probes keep working.
- nginx deliberately does **not** attach the key for you. A proxy that
  injects the key authenticates every anonymous caller along with you, so
  `API_KEY` would protect nothing on an exposed port.

To reach it from another machine, prefer Tailscale or an SSH tunnel over
publishing a port:

```bash
tailscale serve --bg --https=443 http://127.0.0.1:8000
```

Anyone who can reach an unauthenticated instance can monopolise the GPU and
read or upload voice-clone reference samples.

## Silent failures

Several failure modes here produce a clean exit code and unusable audio: an
amplitude-collapsed decode, a deprecated TensorRT engine, a model on the wrong
device. Every generated clip is checked for an RMS floor and for being a
constant signal, and the job **fails** rather than storing silence and
reporting success.

The same rule applies to the stack as a whole:

- **No placeholder tones in production.** The tone generators exist only with
  `DEV_STUB=1`. They used to be added automatically whenever no real model was
  available, which meant a box whose models had all failed served test tones
  and reported healthy.
- **No silent cloud fallback.** ElevenLabs is never the automatic default unless
  `ALLOW_CLOUD_DEFAULT=1`. A Chatterbox that failed to load would otherwise send
  your script to a paid API.
- **Health is reported per capability**, and only a local model counts as `ok`.
  `/health/ready` checks named models (`REQUIRED_PROVIDERS`), because a
  capability can look fine while running on a fallback, such as MusicGen
  standing in for a broken ACE-Step.

## Configuration

See [.env.example](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_TAG` | `12.8.1-cudnn-runtime-ubuntu24.04` | Must be ≥ 12.8 for sm_120 |
| `TORCH_INDEX` | `cu128` | First Blackwell-capable index |
| `API_KEY` | — | Blank disables auth; set before exposing |
| `BIND_ADDR` | `127.0.0.1` | Do not change without reading *Security* |
| `DATA_PATH` | `/srv/story/data` | Host path for generated audio |
| `MAX_CONCURRENT_JOBS` | `1` | Generation is serialised |
| `MODEL_IDLE_TIMEOUT` | `1800` | Unload idle models to free VRAM; `0` disables |
| `SA3_MODEL` | — | Blank = small checkpoints |
| `ACESTEP_BACKEND` | `pt` | `pt` avoids nano-vllm, broken on sm_120 |
| `HF_TOKEN` | — | Required for gated repos (Stable Audio 3) |
| `REQUIRED_PROVIDERS` | `chatterbox,acestep,stable-audio-3-sfx` | What `/health/ready` waits for; `none` = any provider per capability |
| `ALLOW_CLOUD_DEFAULT` | `0` | `1` lets ElevenLabs stand in automatically for a failed local model |
| `PROVIDER_RETRY_SECONDS` | `300` | How long a failed model load is reported before the next retry |
| `MODELS_CACHE_SECONDS` | `3` | Gateway caches discovery this long |
| `DEV_STUB` | `0` | `1` adds placeholder tone generators (dev only) |
| `DEVICE` | `auto` | `auto` picks cuda, then mps, then cpu |
| `PROVIDERS` | `auto` | Comma-separated provider ids a service registers; `auto` = all |
| `ELEVENLABS_API_KEY` | — | Optional cloud provider; blank = not advertised |
| `FORWARD_TIMEOUT` | `600` | Gateway → model service, seconds; nginx and Vite allow 660 |
| `DISCOVERY_TIMEOUT` | `10` | Gateway's per-upstream `/v1/models` timeout, seconds |
| `UPSTREAMS` | set by compose | Gateway: `key=url,...` of the model services |
| `GPU_LOCK_FILE` | set by the split compose | Cross-container GPU slot prefix; blank = off |
| `ACESTEP_MODEL` | `acestep-v15-base` | ACE-Step DiT checkpoint |
| `ACESTEP_LM_MODEL` | `acestep-5Hz-lm-0.6B` | ACE-Step planning LM; `none` disables it |
| `ACESTEP_PROJECT_ROOT` | `$HF_HOME/acestep` | Where ACE-Step keeps its checkpoints |
| `MUSICGEN_MODEL` | `facebook/musicgen-medium` | Legacy provider |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8080` | Browser origins allowed to call the API |
| `LOG_LEVEL` | `INFO` | |
| `DATA_DIR` | `/data` | In-container data path; compose mounts `DATA_PATH` there |
| `VITE_API_TARGET` | `http://localhost:8000` | Where the Mac dev server proxies `/api` |

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Liveness, always `200`: per-capability `ok`/`cloud`/`stub`/`down`, upstreams, required providers. No API key |
| `GET` | `/health/ready` | `200` only when every `REQUIRED_PROVIDERS` entry can generate, else `503` with the reasons. No API key. Never use it as a container healthcheck |
| `GET` | `/v1/models` | Providers, licences, voices, params, defaults |
| `POST` | `/v1/audio/speech` | TTS; waits inline, returns the finished clip |
| `POST` | `/v1/audio/music` | Enqueues; returns `202` and a job id |
| `POST` | `/v1/audio/sfx` | Enqueues; returns `202` and a job id |
| `GET` | `/v1/jobs/{id}` | Job status; `DELETE` cancels one still queued |
| `GET` | `/v1/audio/{id}` | The generated bytes |
| `GET`/`POST`/`DELETE` | `/v1/voices` | Voice-clone reference samples |

Job ids are prefixed with their service (`models:abc123`) so polling routes
with no server-side state and survives a gateway restart. Audio needs no
routing — every service shares one data volume and the gateway reads it
directly.

A queued job can be cancelled; a running one cannot. Once work is inside torch
it is uninterruptible, so `DELETE` returns `409` rather than pretending.

## Developing against the remote box

The frontend needs Node 22.22.2+ on the 22 line, or 24.15+ / 26+ (the floor is
in `frontend/package.json` `engines`).

```bash
npm --prefix frontend install
VITE_API_TARGET=http://<gpu-vm-ip>:8000 npm --prefix frontend run dev
```

Vite proxies `/api` there, matching what nginx does in the container, so
`VITE_API_BASE` stays `/api` in both and CORS never arises.

## Tests

Set up once (Python 3.10+; Node 22.22.2+, 24.15+ or 26+; see AGENTS.md):

```bash
scripts/bootstrap.sh
```

Then run every check before you push:

```bash
scripts/check.sh
```

Add `--full` when dependencies, pins, Docker or the stack changed. It adds
dependency resolution, the contract snapshot check and the dev stack end to end
(restart round included), so it needs network access and Docker.

No GPU, torch or weights needed: providers are faked, the gateway talks to fake
upstreams, and the GPU-slot tests use real subprocesses. The frontend tests
(vitest) stub `fetch` and fake Web Audio, so they need neither a server nor a
browser. CI (`.github/workflows/ci.yml`) runs these on every pull request,
plus the frontend build, dependency resolution, the dev stack end to end
(`scripts/smoke_gpu.py --dev`, restart round included) and a
`docker compose config` of every topology.

## Adding a model

Subclass `Provider` in `backend/app/providers/`, implement `_load()` and
`generate()`, register it in `backend/app/registry.py`, and add tests beside
the existing ones in `backend/tests/`. Declare its `license`
accurately — it is shown in the UI. Add its runtime deps to
`requirements-models.txt` and its install spec to the `MODELS` build arg.

If it needs a torch that genuinely conflicts, give it its own service the way
`compose.gpu.split.yml` does. The frontend needs no change either way.

## Licensing

This repository's code is [MIT-licensed](LICENSE). That covers the code only:
the models it downloads keep their own terms, as described below.

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

The latest release is the newest versioned section of
[CHANGELOG.md](CHANGELOG.md); anything under `[Unreleased]` is on `main` but
not yet released. 0.3.0 is the unified final version of this work; the 0.3.z releases after it
correct and add documentation and tooling. [HANDOFF.md](HANDOFF.md) has the
current state and what comes next.

**Implemented and tested without a GPU:**
- Generation for all three track types behind one provider interface.
- A gateway that makes topology a compose decision.
- Per-capability health and readiness.
- The job queue, silent-output detection and optional API-key auth.
- Timeline placement, trim, playback with Stop, and gain and fades.
- CI on every pull request.

**Not yet proven on the hardware:** nothing has run on the RTX 5090. The model
loads, VRAM use and audio quality are unverified until
`scripts/smoke_gpu.py` passes on the VM.

**Not built yet:**
- Mixdown and export.
- Project persistence. Clips live in memory and are lost on reload, though
  their audio survives on the server.
- Undo.
- The planned LLM director (see [docs/plan.md](docs/plan.md)).

## Contributing

Branches, commits, PRs, versioning and releases are described in
[docs/git-workflow.md](docs/git-workflow.md). Coding agents start at
[AGENTS.md](AGENTS.md).
