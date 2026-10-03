# Git workflow, versions and maintenance

How changes get into `main`, how versions are cut, and how the repository is
kept healthy. These rules apply to people and coding agents alike; AGENTS.md
and CLAUDE.md point here.

## Branches

- `main` is the only long-lived branch. It is protected (see
  [Repository settings](#repository-settings)) and always releasable.
- Work happens on short-lived branches cut from `main`, named for the change:

  | Prefix | For |
  | --- | --- |
  | `feat/` | new behaviour |
  | `fix/` | a bug fix |
  | `test/` | tests only |
  | `ci/` | CI, scripts that only CI runs |
  | `chore/` | dependencies, tooling, housekeeping |
  | `docs/` | documentation only |
  | `refactor/` | behaviour-preserving restructuring |

- Merged branches are deleted automatically. A branch with no open PR and no
  commits for 30 days is stale: merge it, or delete it after checking that
  `git log main..<branch>` is empty or unwanted.

## Commits

- **Subject:** imperative, under 72 characters ("Fix eviction reordering the
  job list").
- **Body:** explains *why*: the failure, the evidence, what was not
  verified. The diff already says what changed.
- **Size:** one logical change per commit. Never commit generated output,
  `.env`, model weights or `smoke-results/`.

## Pull requests

- **One concern per PR.** Fill in the template: Why / What / Verification /
  Not verified / checklist.
- **Test first.** A fix needs a test that fails without it. Mutation-check it:
  re-break the fix in a scratch copy and watch the test fail.
- **CI is the gate.** All five jobs (`backend`, `frontend`, `deps`, `e2e`,
  `compose`) must pass. Run `scripts/check.sh` before pushing, and
  `scripts/check.sh --full` when dependencies, pins, Docker or the compose
  stack changed.
- **Stack PRs only when one genuinely builds on another.** Each targets the
  branch below it. Merge bottom-up, retargeting each to `main` before merging
  it.
- **Merge with "Create a merge commit".** Not squash: a squash would leave
  any stacked branch above carrying commits that `main` no longer has.
- **Never force-push** a branch someone else may have fetched. Bring in
  `main` with a merge, not a rebase.
- **The owner merges.** Agents open PRs and report CI; they do not merge to
  `main`.

## Versions

Semantic Versioning, pre-1.0:

- **0.y.0** for new features or any change in behaviour or API.
- **0.y.z** for fixes that change nothing else.
- **1.0.0** once the GPU path is proven and the API is meant to be stable.

The version lives in five places, which `backend/tests/test_version.py` keeps
in agreement:
- `backend/app/__init__.py`, the source; `/health` reports it
- `backend/pyproject.toml`
- `frontend/package.json`; the Settings page shows it next to the server's
- `frontend/package-lock.json`
- the newest release in `CHANGELOG.md`

Never edit them by hand; use `scripts/bump_version.py`.

Every PR that changes behaviour adds a line under `## [Unreleased]` in
`CHANGELOG.md`, under Added, Changed, Fixed or Security.

## Releases

1. **Bump the version.** On a `chore/release-X.Y.Z` branch, run
   `scripts/bump_version.py X.Y.Z`. It moves all five version fields and turns
   `[Unreleased]` into `[X.Y.Z] - <date>`. It refuses an empty `[Unreleased]`
   and a version that doesn't go up.
2. **Open the release PR.** The owner merges it once CI passes.
3. **Tag the merged commit on `main`:**
   ```bash
   gh api repos/CommitedBit/audio-performance-generator/git/refs -f ref=refs/tags/vX.Y.Z -f sha=<merge-commit-sha>
   ```
   or, with a local git identity:
   ```bash
   git tag -a vX.Y.Z -m vX.Y.Z <sha> && git push origin vX.Y.Z
   ```
4. **The release publishes itself.** `.github/workflows/release.yml` checks
   that the tag matches `app.__version__` and publishes a GitHub Release with
   that version's CHANGELOG section.
5. **Record GPU status for releases that touch models, pins or the
   Dockerfile.** Run `scripts/smoke_gpu.py` on the VM and note the result in
   the release notes. A release whose GPU path is untested says so.

Tags are never moved or deleted once pushed. If a release is wrong, fix it in
a new patch release.

**Hotfixes:** branch `fix/…` from `main`, then follow the normal PR flow and a
patch release. There are no release branches while the project is pre-1.0.

## Dependencies

- **Dependabot.** It opens weekly PRs for GitHub Actions and the frontend's npm
  packages, with minor and patch updates grouped. Treat each as a normal PR:
  CI must pass, and anything touching playback or the build needs the
  frontend tests.
- **npm audit.** `npm --prefix frontend audit` should report nothing. Prefer
  `npm audit fix` without `--force`, which stays within the declared ranges.
- **Python model pins** (ACE-Step, Stable Audio 3, Chatterbox, diffusers,
  torch) are updated by hand only. They are coupled to each other, to CUDA,
  and to the contract snapshots:
  1. Change the pin everywhere it appears: the compose `MODELS` args,
     `backend/pyproject.toml` extras, and `requirements-models*.txt`.
     `tests/test_contracts.py` fails if one place is missed.
  2. `python3 scripts/snapshot_contracts.py` regenerates the snapshots. Fix
     any provider call the new version changed.
  3. `python3 scripts/check_deps.py`: every image must still resolve
     wheel-only.
  4. `scripts/smoke_gpu.py` on the VM before releasing.
- **API dependencies** (`backend/pyproject.toml` `dependencies`: fastapi,
  uvicorn, pydantic, numpy, httpx) are bumped by hand with the backend tests
  as the gate.
- **The CUDA base image** (`CUDA_TAG`) changes only together with
  `TORCH_INDEX` and the torch pins. `check_deps.py` rejects anything older
  than CUDA 12.8 or cu128 (Blackwell).

## Repository settings

These are applied to `CommitedBit/audio-performance-generator`, and they are
part of the foundation. Change them deliberately and record the change here.

- **Branch protection on `main`:**
  - changes arrive by pull request, with no required approvals (a
    single-owner repo cannot approve its own PRs)
  - required status checks: `backend`, `frontend`, `deps`, `e2e`, `compose`
    (not strict, so an out-of-date branch may still merge after green CI)
  - no force-pushes, no deletion
- **Automatically delete head branches** after merge: on.
- **Merge methods:** merge commits are the convention. The repo still allows
  squash and rebase, but don't use them.

Read the current state back with:
```bash
gh api repos/CommitedBit/audio-performance-generator/branches/main/protection
```
```bash
gh api repos/CommitedBit/audio-performance-generator --jq '{delete_branch_on_merge, allow_merge_commit}'
```

## Local environment notes (the owner's Mac)

- **`/usr/bin/git` refuses to run** until the Xcode licence is accepted
  (`sudo xcodebuild -license accept`; only the owner can do this). Until then
  use `/Library/Developer/CommandLineTools/usr/bin/git`, or put
  `/Library/Developer/CommandLineTools/usr/bin` first on `PATH`, which also
  fixes `gh` commands that shell out to git.
- **Pushes authenticate through `gh`** (`gh auth git-credential` is the
  credential helper for github.com).
- **No global git identity is set.** Set `user.name` and `user.email` before
  committing locally, or pass them per command with
  `git -c user.name=… -c user.email=…`.
- **The GPU VM** is driven from the Mac with `DOCKER_CONTEXT=gpu`. See the
  README.
