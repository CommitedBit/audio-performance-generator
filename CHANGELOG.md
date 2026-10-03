# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) — while below 1.0, a minor bump
marks new features or changed behaviour and a patch bump marks fixes. How to
cut a release: [docs/git-workflow.md](docs/git-workflow.md#releases).

## [Unreleased]

### Fixed
- **MP3 audio from ElevenLabs reports its real length.** Only WAV was measured
  server-side, so ElevenLabs speech reported 0 s and sfx the length it was
  asked for. MP3 is now measured from its frames (stdlib, no new dependency).
  An mp3 with no audio frames, such as an error page, fails the job instead of
  storing nothing.
- **A running job no longer jumps to the top of `GET /v1/jobs`.** When the
  oldest job was still running, the job list's cleanup moved it to the end of
  the list, so it listed as the newest, and kept one job more than its limit.
  Cleanup now drops the oldest *finished* jobs and leaves everything else in
  place.

## [0.1.0] - 2026-10-03

The first stable foundation. Merges PRs #4–#16.

> **The GPU path is unverified.** Nothing in this release has run on the RTX
> 5090. Every check here works without a GPU: model-library calls are matched
> against their pinned sources, and the API is exercised against placeholder
> providers. The release is proven on the hardware only when
> `scripts/smoke_gpu.py` passes on the VM.

### Added
- **Local generation for all three track types**, behind one provider
  interface:
  - Chatterbox (voice)
  - ACE-Step 1.5 (music)
  - Stable Audio 3 (sfx and music)
  - MusicGen (legacy)
  - ElevenLabs (optional cloud provider)
- **A gateway** that presents any number of model services as one API, so
  splitting the models across containers is purely a compose decision
  (`compose.gpu.yml`, `compose.gpu.split.yml`). #4
- **A Blackwell (sm_120) toolchain:** CUDA 12.8, cu128 wheels, and per-image
  dependency sets installed `--no-deps`, with torch pins applied as overrides.
  #4, #8
- **Per-capability health** (`ok` / `cloud` / `stub` / `down`), and
  `GET /health/ready`, which waits for `REQUIRED_PROVIDERS`. #7
- **`scripts/smoke_gpu.py`**, the GPU acceptance run. It records load times
  and VRAM peaks. `--dev` runs the same flow against the placeholder stack.
  #9
- **`scripts/check_deps.py`**: every GPU image's dependency set must resolve
  wheel-only, exactly as the Dockerfile installs it. #8
- **Contract tests:** the providers' calls into ACE-Step, Stable Audio 3,
  Chatterbox and diffusers are checked against snapshots of the pinned
  sources (`scripts/snapshot_contracts.py`). #10
- **Clips gain new fields:** `sourceDuration`, `label`, `gain`, `fadeIn`,
  `fadeOut` and `audioId`. Missing cached audio is re-fetched from the
  server. #12
- **A test suite and CI:** backend pytest (150 tests), frontend vitest (91),
  dependency resolution, the dev stack end to end, and compose validation,
  run on every pull request. #5, #8, #9, #11
- **Project documentation:**
  - `CLAUDE.md` and `AGENTS.md`, for coding agents
  - `HANDOFF.md`, the current state
  - `docs/git-workflow.md`, the git and release process
  - the MIT `LICENSE`, the PR and issue templates, Dependabot, and the
    release workflow

  #14, #15, #16
- **The release version is shown** in `/health` and on the Settings page.
  #16

### Changed
- **No automatic cloud fallback.** ElevenLabs is used only when chosen
  explicitly, unless `ALLOW_CLOUD_DEFAULT=1`. #7
- **Placeholder tone providers exist only with `DEV_STUB=1`.** Before, they
  were added whenever no real model was available, and the service reported
  healthy. #7
- **Timeout defaults agree** across code, compose, `.env.example`, nginx and
  Vite. A test now fails if they drift apart. #14

### Fixed
- **The idle sweep froze the whole API** while any model was loading. #6
- **Stable Audio 3 checkpoints stalled each other's loads and unloads.** #6
- **Stable Audio 3's diffusers fallback crashed on every generation:** it
  passed `audio_end_in_s=` where diffusers 0.40 expects `duration=`. #10
- **`scripts/check_deps.py` blamed the pins for a network outage.** It
  reported an unreachable package index as "the image would build a package
  from source". It now retries, and then says the index was unreachable. #16
- **The API client:**
  - one failed poll failed a whole generation
  - an abort was reported as "server unreachable"
  - abandoned server jobs were never cancelled
  - ids failed over plain HTTP

  #11
- **Playback and the timeline:**
  - Play could be silent (autoplay policy)
  - clips drifted out of sync by their decode time
  - Stop did nothing
  - left-trim moved the audio
  - trims could not be undone
  - an evicted IndexedDB cache lost a clip's audio for good

  #12

### Security
- **All 24 `npm audit` advisories cleared,** including react-router and vite.
  #13
- **The API key is enforced by the gateway** and never injected by nginx.
  Ports bind to loopback by default. #4
