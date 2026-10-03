# AGENTS.md

Instructions for coding agents (Codex and others). Read these first:

1. **[CLAUDE.md](CLAUDE.md)**: architecture, the invariants and why each exists,
   every check command, and known traps. It applies to any agent, not just
   Claude.
2. **[HANDOFF.md](HANDOFF.md)**: current state, what is and is not verified,
   review priorities, and the backlog of foundational work.

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
