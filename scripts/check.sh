#!/usr/bin/env bash
# Every check in CLAUDE.md, in one command. Run it before every push.
#
#   scripts/check.sh          offline checks: backend tests + ruff, frontend
#                             lint/build/test, compose config for all topologies
#   scripts/check.sh --full   also: dependency resolution, contract snapshots
#                             against the pinned sources (both need network) and
#                             the dev stack end to end, restart included (Docker)
#
# Written for bash 3.2 (the macOS default): no associative arrays, no mapfile.
# Runs every check even after a failure, then exits non-zero if any failed.
set -uo pipefail

cd "$(dirname "$0")/.."
FULL=0
case "${1:-}" in
  --full) FULL=1 ;;
  "") ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

# The backend venv if present (local), else whatever python3 has the deps (CI).
PY=python3
[ -x backend/.venv/bin/python ] && PY="$PWD/backend/.venv/bin/python"

FAILED=()
run() {
  local name=$1; shift
  printf '\n== %s\n' "$name"
  if "$@"; then
    printf 'ok    %s\n' "$name"
  else
    printf 'FAIL  %s\n' "$name"
    FAILED+=("$name")
    return 1
  fi
}

run "backend tests"   bash -c "cd backend && '$PY' -m pytest -q -p no:cacheprovider"
run "backend ruff"    bash -c "cd backend && '$PY' -m ruff check app tests && '$PY' -m ruff check --config pyproject.toml ../scripts"
run "frontend lint"   npm --prefix frontend run lint --silent
run "frontend build"  npm --prefix frontend run build --silent
run "frontend tests"  npm --prefix frontend test --silent
# Sandboxes (Codex's among them) often have no Docker. Skip, say so, and let
# CI's compose and e2e jobs cover it -- a missing tool is not a failing check.
HAVE_DOCKER=0
command -v docker >/dev/null 2>&1 && HAVE_DOCKER=1
if [ "$HAVE_DOCKER" = 1 ]; then
  run "compose: dev"    docker compose -f docker-compose.yml config -q
  run "compose: gpu"    docker compose -f compose.gpu.yml config -q
  run "compose: split"  docker compose -f compose.gpu.yml -f compose.gpu.split.yml config -q
else
  printf '\nskip  compose config: docker is not installed here (CI runs it)\n'
fi

if [ "$FULL" = 1 ]; then
  # check_deps.py needs docker compose (to read the build args) and uv (to resolve).
  if [ "$HAVE_DOCKER" = 1 ] && command -v uv >/dev/null 2>&1; then
    run "dependency resolution"  python3 scripts/check_deps.py
  else
    printf '\nskip  dependency resolution: needs docker and uv (CI deps runs it)\n'
  fi
  run "contract snapshots"     python3 scripts/snapshot_contracts.py --check
  if [ "$HAVE_DOCKER" = 1 ]; then
    # smoke-results/ may also hold the owner's GPU runs, the only record of
    # them (it is not in git). So the dev run writes to a directory of its own,
    # and only that one is removed, only when the run passes; a failed run's
    # results and logs stay for reading. smoke_gpu.py refuses an --out that
    # already exists.
    dev_run="smoke-results/dev-check-$(date +%Y%m%d-%H%M%S)-$$"
    # No --quick, as in CI: the restart round is part of what this covers.
    if run "dev stack end to end"   python3 scripts/smoke_gpu.py --dev --out "$dev_run"; then
      dev_ok=1
    else
      dev_ok=0
    fi
    # The smoke run leaves the dev stack up; take it down either way.
    docker compose -f docker-compose.yml down >/dev/null 2>&1
    if [ "$dev_ok" = 1 ]; then
      rm -rf "$dev_run"
      rmdir smoke-results 2>/dev/null || true
    else
      printf 'kept  the failed dev run: %s\n' "$dev_run"
    fi
  else
    printf '\nskip  dev stack end to end: docker is not installed here (CI e2e runs it)\n'
  fi
fi

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "all checks passed"
