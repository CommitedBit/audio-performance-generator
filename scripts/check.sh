#!/usr/bin/env bash
# Every check in CLAUDE.md, in one command. Run it before every push.
#
#   scripts/check.sh          offline checks: backend tests + ruff, frontend
#                             lint/build/test, compose config for all topologies
#   scripts/check.sh --full   also: dependency resolution, contract snapshots
#                             against the pinned sources (both need network) and
#                             the dev stack end to end (needs Docker)
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
  fi
}

run "backend tests"   bash -c "cd backend && '$PY' -m pytest -q -p no:cacheprovider"
run "backend ruff"    bash -c "cd backend && '$PY' -m ruff check app tests && '$PY' -m ruff check --config pyproject.toml ../scripts"
run "frontend lint"   npm --prefix frontend run lint --silent
run "frontend build"  npm --prefix frontend run build --silent
run "frontend tests"  npm --prefix frontend test --silent
run "compose: dev"    docker compose -f docker-compose.yml config -q
run "compose: gpu"    docker compose -f compose.gpu.yml config -q
run "compose: split"  docker compose -f compose.gpu.yml -f compose.gpu.split.yml config -q

if [ "$FULL" = 1 ]; then
  run "dependency resolution"  python3 scripts/check_deps.py
  run "contract snapshots"     python3 scripts/snapshot_contracts.py --check
  run "dev stack end to end"   python3 scripts/smoke_gpu.py --dev --quick
  # The smoke run leaves the dev stack up; take it down either way.
  docker compose -f docker-compose.yml down >/dev/null 2>&1
  rm -rf smoke-results
fi

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "all checks passed"
