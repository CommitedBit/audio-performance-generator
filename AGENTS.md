# AGENTS.md

Instructions for coding agents (Codex and others). Read these first:

1. **[CLAUDE.md](CLAUDE.md)**: architecture, the invariants and why each exists,
   every check command, and known traps. It applies to any agent, not just
   Claude.
2. **[HANDOFF.md](HANDOFF.md)**: current state, what is and is not verified,
   review priorities, and the backlog of foundational work.

## Environment setup

```bash
scripts/bootstrap.sh     # backend/.venv with dev extras, frontend npm ci (needs network)
scripts/check.sh         # every offline check
```

Requirements:
- **Python 3.10+.**
- **Node 22.22.2+ on the 22 line, or 24.15+ / 26+.** The floor is in
  `frontend/package.json` `engines`; `.nvmrc` selects the Node 22 line.
  Below the floor, npm silently skips vite 8's native bundler binding, and
  the frontend cannot build. `bootstrap.sh` checks both versions.
- **`uv`** for `check_deps.py`.

| Check | Needs |
| --- | --- |
| backend pytest + ruff, frontend lint/build/test | nothing after bootstrap |
| `docker compose config` (all topologies) | Docker. Skipped with a note when it is absent |
| `scripts/snapshot_contracts.py --check` | network (`scripts/check.sh --full`). **CI does not run it**: run it whenever a pin or a provider's model-library call changes |
| `scripts/check_deps.py` | network, Docker and `uv` (`--full`). Skipped without Docker or `uv`; CI's `deps` job runs it |
| `scripts/smoke_gpu.py --dev` | Docker (`--full`). Skipped when absent; CI's `e2e` job always runs it |
| `scripts/smoke_gpu.py` on real models | the owner's GPU VM; agents cannot run it |

When a check cannot run where you are, say so in the PR. CI runs all of them
except the contract-snapshot check (`snapshot_contracts.py --check`, which
compares against upstream sources over the network) and the GPU run. CI's
`backend` job still runs `tests/test_contracts.py` against the committed
snapshots.

## Non-negotiables

- **Never force-push.** Never rewrite a branch someone else has pushed.
  Resolve conflicts with a merge, not a rebase.
- **One concern per PR, branched from `main`.** Follow
  [docs/git-workflow.md](docs/git-workflow.md): branch names, commits, merge
  commits, versioning and releases. Add a `CHANGELOG.md` `[Unreleased]` line
  for every behaviour change. Run `scripts/check.sh` before pushing.
- **Write the failing test first, then fix.** Prove every fix by re-breaking it
  in a scratch copy and watching a test fail. CI
  (`.github/workflows/ci.yml`) must pass before anything merges.
- **The GPU path has never run.** Nothing here proves the models load or
  produce good audio on the RTX 5090. Don't claim it does. Only
  `scripts/smoke_gpu.py` run on the GPU VM can show that.
- **Keep the invariants** listed in CLAUDE.md:
  - the per-image dependency sets
  - the cu128/Blackwell pins
  - the non-blocking idle sweep
  - stubs only with `DEV_STUB`
  - no automatic cloud fallback
  - `/health` (liveness) is separate from `/health/ready` (readiness)

  If one blocks you, say why in the PR rather than working around it.
- **Model-library calls must match the pinned sources.** After changing a pin
  or a provider's call into ACE-Step, Stable Audio 3, Chatterbox or diffusers,
  run `python3 scripts/snapshot_contracts.py` and keep
  `backend/tests/test_contracts.py` green.
- **Comments explain why,** usually by naming the failure they prevent. Match
  the surrounding style.
