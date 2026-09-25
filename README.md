# Audio Performance Generator

A local-first audio clip generator and timeline editor. Generate voice, sound
effects and music from text, drop each result on a multi-track timeline, then
trim and arrange it.

Generation runs on **your own GPU**. ElevenLabs stays available as an optional
fallback, but nothing requires it and no API key ever reaches the browser.

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

After first boot, generate one clip per track type and listen to it:

```bash
curl -s localhost:8000/v1/models | jq '.providers[] | {id, available, unavailable_reason}'
```

Every provider should report `available: true`. A wrong transformers version
usually surfaces as an ImportError at model load, which shows up as the
provider's `unavailable_reason`.

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
| `VITE_API_TARGET` | `http://localhost:8000` | Where the Mac dev server proxies `/api` |

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Per-upstream up/down; no API key required |
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

```bash
npm --prefix frontend install
VITE_API_TARGET=http://<gpu-vm-ip>:8000 npm --prefix frontend run dev
```

Vite proxies `/api` there, matching what nginx does in the container, so
`VITE_API_BASE` stays `/api` in both and CORS never arises.

## Adding a model

Subclass `Provider` in `backend/app/providers/`, implement `_load()` and
`generate()`, register it in `backend/app/registry.py`. Declare its `license`
accurately — it is shown in the UI. Add its runtime deps to
`requirements-models.txt` and its install spec to the `MODELS` build arg.

If it needs a torch that genuinely conflicts, give it its own service the way
`compose.gpu.split.yml` does. The frontend needs no change either way.

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
abstraction, a gateway that makes topology a compose decision, job queue,
silent-output detection, optional API-key auth, timeline placement, drag and
trim, playback.

Not built: mixdown/export, project persistence (clips are in memory and lost on
reload, though their audio survives on the server), and undo.
