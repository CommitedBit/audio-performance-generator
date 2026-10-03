> **Status (2026-10-03):** M0–M2 tooling and all M1 PRs are done (PRs #4–#14; see HANDOFF.md). The GPU run (M2) and M3–M5 remain. Step-by-step details below are as originally approved; HANDOFF.md is the current state.

# Plan: make the repo safe for agents, then add a local-LLM director

## Context

PR #4 (`fix/make-it-run`) turned a broken app into a local-generation stack
(Chatterbox / ACE-Step 1.5 / Stable Audio 3 behind a gateway) and absorbed
~16 rounds of automated review. Three problems block further agentic work:

- **No permanent verification.** The 28 review harnesses lived in `/tmp` and
  are gone. No tests, no CI — nothing protects those fixes from regression.
- **The GPU path has never run.** Verified only as "imports and calls the real
  APIs correctly"; no audio has come out of the 5090.
- **Failures are silent.** Tone stubs replace missing models
  (`registry.py:67-70`) and `/health` still says ok. ElevenLabs silently
  becomes the voice default if Chatterbox fails (`registry.py:30`), sending
  story text to a paid API. MusicGen silently stands in for a broken ACE-Step
  (`musicgen.py:37`).

User decisions: do **both** (agent-safe codebase + in-app director); local LLM
via **Ollama, mostly in system RAM**; GPU VM gets **64-128 GB** of 256 GB.

That shapes the plan:
- The 5090's 32 GB goes to audio, so bigger checkpoints become possible, but
  not all at once.
- The VM's RAM is shared between Ollama and any parked audio models.
- Ollama must be hidden from the GPU.
- CPU LLM inference is slow, so director work is async and resumable, and
  repair round-trips must be rare.

## M0 — user steps first

```bash
sudo xcodebuild -license accept
git fetch && git reset origin/fix/make-it-run && git status
```

This fixes the local index, which has been stale since Sep 21.
- **Why not stash and pull:** files added since then are untracked, so a pull
  would refuse to overwrite them.
- **What reset does:** a mixed reset moves only HEAD and the index; no files
  change.
- **What to expect:** `git status` should come back clean. If it shows
  anything, inspect it before going further.

Then two decisions for you (recommended):
1. **Merge PR #4.** It is strictly better than `main`, and later PRs then
   target `main` cleanly.
2. **Turn off Codex Auto-fix.** Otherwise new work keeps landing on a branch
   that review is still rewriting.

## M1a — foundation (4 PRs, before the GPU run)

**PR A — test harness + CI skeleton**

The conftest has three jobs:
- Set `DATA_DIR` (to a temp dir) and `DEV_STUB` *before* any app import:
  `Settings()` creates its directories on import (`config.py:39-44`).
- Reset the `lru_cache`d `get_settings`, the registry singleton, and
  `main.queue` for every test: `JobQueue` semaphores bind to the first event
  loop that uses them.
- Install a fake `torch` in `sys.modules`.

Tests pinning current behaviour, rebuilt from the lost harnesses:
- **API:** `with TestClient(app)`, so the lifespan runs. Health, models, the
  three generate routes, jobs, and WAV duration measured from the stored file.
- **Validation:** `_validated_seconds` and `SpeechBody` text resolution.
- **Gateway:** routing, `service:` id prefixing, 404 vs 503 + Retry-After,
  `preference_rank` defaults, `/v1/jobs` aggregation.
- **Concurrency:**
  - load lock: a single `_load` under a race, and the cooldown checked inside
    the lock
  - `unload_if_idle` vs `in_use`
  - `rng_scope` exclusivity and writer preference
  - remote lane vs local semaphore
  - the SA3 shared pipeline and its call lock
  - `gpu_slot` across subprocesses
- **ElevenLabs auth gate:** fake `httpx`.
- **Auth middleware** and the `/health` exemption.

Also in this PR:
- ruff.
- `.github/workflows/ci.yml` running backend tests, frontend lint/build, and
  `docker compose config -q` for dev, gpu, and gpu+split. CI runs on every
  `pull_request`, with no branch filter.

**PR B — stall fixes (test-first)**
- **Sweeper freeze.** `_sweep_idle_models` (`main.py:63-73`) runs on the event
  loop and blocks on `_load_lock`, which `load()` holds for a multi-minute
  `_load()`.
  - Fix: `unload_if_idle` uses `acquire(blocking=False)` (busy means skip),
    and the sweep runs in `asyncio.to_thread`.
- **SA3 cross-checkpoint stall.** `_load` holds the module-wide `_SHARED_LOCK`
  for the whole build (`stable_audio.py:235-244`), and `unload` waits on it
  (`:274`). With the default separate sfx and music checkpoints, sweeping one
  while the other builds blocks the sweep while it holds the first provider's
  `_load_lock`.
  - Fix: one build lock per checkpoint key, so `_SHARED_LOCK` guards only the
    bookkeeping.
- **Freeing VRAM.** Add `gc.collect()` to `unload()`, so an unload actually
  releases VRAM.
- **Testing.** Assert latency (the sweep returns promptly during a blocked
  fake load), not the return value.

**PR C — fail loudly, readiness, local-first**
- **Stubs:** only when `DEV_STUB=1`.
- **Health statuses:** `ok`, `stub` (dev only), `cloud`, `degraded`, `down`,
  per capability.
- **`REQUIRED_PROVIDERS` on the gateway** (default
  `chatterbox,acestep,stable-audio-3-sfx`).
  - `GET /health/ready` returns 503 unless all of them are available; add it to
    `OPEN_PATHS` (`auth.py:24`).
  - `/health` stays HTTP-200 liveness. Document that no HEALTHCHECK or
    `service_healthy` may point at `/ready`.
- **`ALLOW_CLOUD_DEFAULT=0`:** ElevenLabs is used only when explicitly selected,
  never as an automatic default.
- **Gateway caching:** a ~3 s TTL cache on `_gather_models()`
  (`gateway.py:94-119`), which `/health` also uses. Today one slow upstream's
  10 s timeout fails the gateway's own 10 s healthcheck.
- **Logging:** the gateway logs an ERROR when a required provider changes state
  (not per service, so split services don't raise false alarms).
- **Schemas:** wire the unused `HealthResponse` (`schemas.py:78`) as the
  `response_model`; update the frontend `HealthInfo` (`client.ts:293-302`) and
  SettingsPage.

**PR D — dependency CI + build defaults**
- **Wheel-only resolution** for each of the three requirement sets, mirroring
  the Dockerfile exactly: pyproject + the requirement set + the `TORCH_PIN`
  override, py3.12, x86_64 manylinux, cu128.
  - Allowlist pure-Python sdists if needed. Verify `antlr4-python3-runtime`,
    which `omegaconf` pulls in.
- **No CUDA image builds in CI** (hosted runners don't have the disk). Instead,
  check that a CPU-torch import works for each topology.
- **Dockerfile defaults:** align with compose: `CUDA_TAG` 13.0.2 → 12.8.1 and
  `TORCH_INDEX` cu130 → cu128 (`Dockerfile:18,26`). Pin the `uv` image
  (`:48`).

## M2 — first real GPU run (right after M1a)

`scripts/smoke_gpu.py` on the VM. It comes this early because the result may
force the split topology, which changes the dependency checks, the readiness
defaults, and CLAUDE.md.

- **Preflight:** `nvidia-smi` in a CUDA container, free disk (expect ~31 GB of
  first downloads), `HF_TOKEN`, a reminder to accept the gated SA3 licence.
- **Models load lazily** (`base.py:231`), so `/health` is green in seconds and
  says nothing about the models.
  - The script generates one seeded voice, music, and sfx clip, with explicit
    local providers.
  - It polls the jobs with long timeouts (speech returns 202 after 180 s), then
    requires `/health/ready` to return 200.
- **Each WAV is checked** for duration and RMS, and that it is not a stub tone.
- **Timings:** download, cold-from-disk, and warm loads timed separately. Peak
  VRAM per model recorded.
- **Idle unload (opt-in):** `--idle-check` confirms that an idle unload frees
  VRAM. It needs a short timeout, so it is not part of the default run:
  `MODEL_IDLE_TIMEOUT=60 python3 scripts/smoke_gpu.py --idle-check`.
- **On failure:** save the compose logs; print a pass/fail report.
- **README:** rewrite the deploy section around the script (the manual `chown`
  step is obsolete).

## M1b — remaining foundation (4 PRs)

**PR E — contract snapshots**
- `scripts/snapshot_contracts.py` parses the source tarballs at the pinned refs
  (no torch needed; needs network, so it is run by hand).
- It writes signatures and dataclass sources to `backend/tests/contracts/`;
  tests build fakes from them offline.
- Pinned refs: ACE-Step `ca1e85fe9430`, stable-audio-3 `779434a90819`,
  chatterbox-tts `0.1.7`. Chatterbox (`from_pretrained`, `generate`, `sr`) has
  never been checked.
- A pin-agreement test across `compose.gpu.yml`, `compose.gpu.split.yml` and
  `pyproject.toml`.

**PR F — frontend tests + client fixes**
- Add vitest.
- `waitForJob` retries transient poll failures with backoff
  (`client.ts:235-258`).
- Aborts stay `AbortError`, not `ApiError(0)` (`:139-145`), and an abort calls
  `cancelJob`.
- `crypto.randomUUID` fallback for plain-HTTP LAN access.

**PR G — the `Clip` change, done once**
- New fields: `sourceDuration`, `label`, `gain`, `fadeIn`, `fadeOut`, and
  server `audioId`, with IndexedDB kept as a cache.
- Playback (`Timeline.tsx:23-32`, `AudioEngine.ts`):
  - one `AudioContext`, resumed on the Play gesture
  - decode before computing `t0`
  - keep sources so Stop works
  - try/finally
  - gain and fades applied
- Left trim moves `start` with `offset`, and the handles can extend back out to
  `sourceDuration`.

**PR H — `CLAUDE.md` + hygiene**
- **CLAUDE.md:**
  - architecture map
  - invariants and why: the dependency topology, Blackwell pins, `_load_lock`,
    `rng_scope`, `gpu_slot`, the remote lane, the SA3 shared pipeline
  - verification protocol: reproduce before fixing, a failing test first, CI
    as the gate
  - harness traps: zsh does not word-split; `TestClient` needs `with`; never
    keep evidence in `/tmp`
- **Hygiene:**
  - `.dockerignore` files
  - untrack `.claude/launch.json`
  - drop the `pytorch-cu124` index
  - reconcile the `MODEL_IDLE_TIMEOUT` and `FORWARD_TIMEOUT` defaults
  - complete the README config table
  - fix the stale voice-sfx requirements comment
- The react-router bump (2 high advisories) goes in a separate PR.

## M3 — hardware tuning (LLM in RAM)

1. **Ollama** under an `llm` compose profile:
   - **Hidden from the GPU:** no GPU reservation, plus
     `NVIDIA_VISIBLE_DEVICES=void` and `CUDA_VISIBLE_DEVICES=` in case
     nvidia is the default runtime.
   - **Pinned** image tag.
   - **Limits:** `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`,
     `OLLAMA_KEEP_ALIVE`, and a `mem_limit` so it can't starve the audio stack.
   - Its own volume, internal network only.
   - **Model:** `DIRECTOR_MODEL`, an 8-14B quantized instruct model. A 70B on
     CPU takes hours for a long plan. `OLLAMA_URL` can point at an Ollama
     hosted elsewhere.
2. **VRAM budget** from M2's measurements. Start with everything resident:
   Chatterbox + SA3 medium (one checkpoint for both) + ACE-Step 2B with the
   1.7B LM. XL only if the numbers allow it.
3. **VRAM-aware loading** (only if XL is wanted): check `mem_get_info` before a
   load; evict LRU idle providers via `unload_if_idle`, never one `in_use`.
4. **RAM tier, conditional.**
   - First see whether the page cache already makes warm reloads fast.
   - If not: `park()`/`unpark()` hooks (unload by default), a
     `MODEL_RAM_BUDGET_GB` LRU, and only for objects verified to support
     `.to("cpu")` (MusicGen and diffusers yes; the others need checking
     against pinned source).
   - Hooks run under `_load_lock` and must not re-enter it.
   - Expose ACE-Step's offload flags (`acestep.py:109-110,128`) as env vars.

## M4 — the director (story → timeline)

**Its own service.** Same CPU backend image, entrypoint `app.director`,
listed in `UPSTREAMS` as `director`. The existing `director:` prefix routing
and `/v1/jobs` aggregation then work unchanged, the gateway stays stateless,
and redeploying the gateway doesn't kill a render.
- It returns an empty provider list from `/v1/models`.
- The gateway gets a pass-through for `/v1/director/*`.
- The director calls the gateway for generation, so routing and readiness live
  in one place. It reads `API_KEY` from env.

**State.** Jobs use the `Job.public()` shape, so `waitForJob` works unchanged.
An async runner can cancel running jobs. A per-plan manifest is written
atomically under `/data/director/<plan>/` (cue status, seed, `audio_id`,
duration), so renders survive restarts and can resume.

**LLM client.** One function, `chat_json(messages, schema)`, over Ollama's
native `/api/chat`:
- `format` set to a schema built per request, with an `enum` of valid voice
  ids
- `num_ctx` set per request (the default is small and silently truncates long
  stories)
- `think:false`
- streamed, to show progress

It stays swappable for an OpenAI-compatible backend.

**Plan format.** The LLM emits a simple *ordered script*: `characters` plus
items of type `line`, `sfx`, `music_start`/`music_stop`, `ambience`, or
`pause`. The server derives timing anchors from it:
- default: after the previous line
- `with` + a fraction, for overlap
- `from`/`until` spans, for music beds and ambience

Validation checks the schema and the semantics. Only structural errors go back
to the LLM for repair; out-of-range numbers are clamped with a warning.

**Casting.** Chatterbox has a single built-in voice (`chatterbox.py:57`).
Everything else is an uploaded clone.
- The plan editor has a casting step per character: an existing reference, an
  upload, or an explicitly chosen ElevenLabs voice (cloud, opt-in).
- It warns when characters share a voice.
- A preset-voice local TTS provider is a possible follow-up.

**Layout.** One track per character (capped near 8, then shared lanes),
shared sfx lanes, and two music tracks for crossfades. Overlap is checked
per track only.

**Render.**
- **Order:** voice lines first, then their measured durations resolve the
  anchors, then sfx and music.
- **Length clamping:** to the rendering provider's `ParamSpec` (ACE-Step
  10-600 s, SA3 1-120/380 s, ElevenLabs sfx 0.5-22 s).
  - Below the minimum: generate the minimum and trim.
  - Above the maximum: crossfaded segments, or a loop.
- **Seeds:** `hash(plan_seed, cue_id)`, so inserting a cue doesn't reseed the
  rest. Reproducibility is best-effort, on the same machine.
- **Retries:**
  - 502/503/504/404 with backoff
  - `SilentOutputError` once, with seed+1
- **Partial results:** reported as such; re-render only the failed cues.
- **Pacing:** 1-2 cues in flight, grouped by provider, so manual generation
  isn't starved.
- **Providers:** always explicit, never the cloud default.

**Frontend.**
- A `/director` page: story → script editor → casting → render progress →
  import.
- Store additions: `importTimeline`, `addTrack`, and `addClip` accepting
  `start` and `label` and returning the id.

**Evals.**
- **Golden stories** in `backend/tests/director/`: the plan validates, every
  character is cast, no overlaps within a track, length is within target.
- **Real-LLM evals** run behind an opt-in `@pytest.mark.llm`.
- **CI** covers the parser, validator, repair loop and layout using recorded
  LLM responses.

## M5 — finish the loop

- **Mixdown:** `POST /v1/mixdown` renders the timeline server-side from stored
  audio, applying gain and fades, to WAV/MP3. Story in, finished file out.
- **Server-side projects:** `/v1/projects`. Because clips already carry
  `audioId` (PR G), this just stores the document.
- **Undo/redo** over store actions.

## How the agentic work runs

Each PR runs as one Workflow: implement (failing test first) → adversarial
verification (reproduce before and after) → full suite → push, with CI as the
gate and the evidence in the PR description. `CLAUDE.md` carries the lessons
between sessions.

## Critical files

- **Backend:**
  - `backend/app/{registry,main,gateway,jobs,config,auth,schemas}.py`
  - `backend/app/providers/{base,stable_audio,chatterbox,acestep,musicgen}.py`
  - new: `backend/tests/`, `backend/app/director/`, `scripts/`
- **Frontend:**
  - `frontend/src/api/client.ts`, `frontend/src/store/useProjectStore.ts`,
    `frontend/src/types/timeline.ts`
  - `frontend/src/components/{Timeline,GenerationPanel}.tsx`,
    `frontend/src/engine/AudioEngine.ts`, `frontend/src/pages/SettingsPage.tsx`
  - new: `frontend/src/pages/DirectorPage.tsx`
- **Infra:**
  - `compose.gpu.yml`, `compose.gpu.split.yml`, `docker-compose.yml`,
    `backend/Dockerfile`, `.env.example`, `README.md`
  - new: `.github/workflows/ci.yml`, `.dockerignore`s, `CLAUDE.md`

## Verification

- **M1:** the suite passes locally and in CI. Each regression test is first
  shown to fail on the pre-fix version of its file. PR B's tests show the API
  responding during a blocked load. Snapshots regenerate byte-identically.
- **M2:** `smoke_gpu.py` exits 0 on the VM and `/health/ready` returns 200.
  You listen to the three clips. A run with `--idle-check` shows that idle
  unloading returns the VRAM.
- **M3:** with Ollama loaded, every audio model still loads, `nvidia-smi` shows
  no Ollama process, and reload timings are recorded.
- **M4:** unit tests for the script parser, anchors, layout, clamping and
  resume. The opt-in LLM evals pass against the chosen model. A story → import
  round trip works in the browser, and a killed render resumes.
- **M5:** a mixdown of a director timeline plays as one file; a project
  survives a reload.
