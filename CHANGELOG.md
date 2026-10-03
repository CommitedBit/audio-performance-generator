# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) — while below 1.0, a minor bump
marks new features or changed behaviour and a patch bump marks fixes. How to
cut a release: [docs/git-workflow.md](docs/git-workflow.md#releases).

## [Unreleased]

### Changed
- **The frontend build moves to Vite 8** with `@vitejs/plugin-react` 6, which
  are upgraded together, as neither installs without the other. Vite 8
  bundles with rolldown.
- **The frontend tests move to jsdom 30.** One test fixture changed: it built
  a fetch `Response` from a jsdom `Blob`, which jsdom 30 no longer supports.
  The app code is unaffected.
- **TypeScript moves to 5.9 and stays on 5.x.** TypeScript 7 is blocked:
  `typescript-eslint`, even the newest release, accepts only TypeScript below
  6.1. Dependabot now skips TypeScript majors, with that reason in its
  config. Supersedes #31, #32, #33 and #35; #34 (globals 17) merged as is.
- **The frontend needs Node 22.22.2+** (or 24.15+ / 26+), declared in
  `frontend/package.json` `engines` and `.nvmrc`, and checked by
  `scripts/bootstrap.sh`. On older Node, npm silently skips rolldown's
  native binding, and every frontend command then fails. CI and the Docker
  image already use the current Node 22.

## [0.2.0] - 2026-10-03

### Changed
- **The frontend lint toolchain moves to ESLint 10** (`eslint` 10,
  `@eslint/js` 10, `eslint-plugin-react-hooks` 7) in one step, replacing three
  Dependabot PRs that each failed alone. react-hooks 7's new
  `set-state-in-effect` rule flagged four effects that copied derived values
  into state, costing an extra render with stale values:
  - the generation panel's provider and voice are now derived during render
  - its length adjusts when the provider changes, during render, so typing is
    still never clobbered
  - `useModels` raises `loading` in `refresh()`

  The behaviour is unchanged. New tests pin it, and they pass on the old code
  too. Dependabot now groups ESLint packages so they arrive together. #29
- **vitest 5** (from Dependabot #25). It is folded in here because both
  upgrades rewrite `package-lock.json`. All 103 frontend tests run under it.
  #29
- **GitHub Actions moved to their current major versions,** clearing CI's
  Node 20 deprecation warnings, and **eight frontend dependencies** received
  minor and patch updates (Dependabot). #21, #22

### Added
- **`scripts/bootstrap.sh`: one command from a fresh checkout to a working
  environment** (`backend/.venv` with dev extras, then `npm ci`). AGENTS.md
  says which checks need network, Docker or the GPU VM.
  `scripts/check.sh` skips the Docker checks with a note when Docker is
  absent, as in most agent sandboxes, rather than failing; CI still runs
  them. #30
- **`scripts/smoke_gpu.py --idle-check` (opt-in) checks that idle unloading
  frees VRAM.** It restarts the model services so the baseline comes from
  processes that have never loaded a model, waits for the idle sweep to unload
  every model, and then requires each process's PyTorch allocations back
  within 0.25 GiB of the baseline and the card's VRAM within 1 GiB per model
  service. Baseline, peak and after-unload VRAM and both residuals go into
  `report.txt` and `results.json`. It needs a short timeout:
  `MODEL_IDLE_TIMEOUT=60 python3 scripts/smoke_gpu.py --idle-check`; above
  300 s it refuses to start. It has not run on the GPU, so neither allowance
  is measured. With `--dev` it checks the unload only and reports VRAM as not
  measured, for which `docker-compose.yml` now passes `MODEL_IDLE_TIMEOUT`
  through to the stubs. #27
- **A model service's `/health` reports its own PyTorch allocations**
  (`gpu.torch_allocated_gb`, `gpu.torch_reserved_gb`) beside the card's free
  and total VRAM. `--idle-check` reads them: after an unload, `nvidia-smi`
  still shows the kernels a process keeps, and Stable Audio 3 small left
  resident would fit inside that. #27

### Fixed
- **`scripts/smoke_gpu.py` read the gateway's cached view of a model service
  it had just replaced.** For 3 s after `up` recreated a service (for example
  a run with a different `MODEL_IDLE_TIMEOUT`), the gateway still reported
  the old process: ready, with its models loaded. A run chained straight
  after another then failed every submit with a 502. The script now waits the
  cache out after `up` and after every restart. #27
- **CI's e2e job and `scripts/check.sh --full` skipped the smoke script's
  restart round** (`--quick`), while HANDOFF.md counted a restart as covered.
  Both now run it. #27
- **docs/plan.md and CI named a shell-script smoke runner that never
  existed;** they now name `scripts/smoke_gpu.py`. docs/plan.md's M2 also
  promised an idle-unload VRAM check the script did not have; it now points
  at `--idle-check`. #27

## [0.1.1] - 2026-10-03

### Fixed
- **A running job reports its phase:**
  - loading `<model>`, which happens on a first load and can take minutes
  - generating
  - saving

  The generation panel shows the phase and the elapsed time. Before, a job
  read "generating" from before its model even started loading, and its
  progress jumped from 0 to 1 at the end. #19
- **MP3 audio from ElevenLabs reports its real length.** Only WAV was measured
  server-side, so ElevenLabs speech reported 0 s and sfx the length it was
  asked for. MP3 is now measured from its frames (stdlib, no new dependency).
  An mp3 with no audio frames, such as an error page, fails the job instead of
  storing nothing. #18
- **A running job no longer jumps to the top of `GET /v1/jobs`.** When the
  oldest job was still running, the job list's cleanup moved it to the end of
  the list, so it listed as the newest, and kept one job more than its limit.
  Cleanup now drops the oldest *finished* jobs and leaves everything else in
  place. #17

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
