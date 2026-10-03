#!/usr/bin/env bash
# One command from a fresh checkout to a working development environment:
#
#   scripts/bootstrap.sh
#
# Creates backend/.venv with the API and its dev extras (pytest, ruff), installs
# the frontend's locked dependencies, and points at scripts/check.sh. Safe to
# re-run. Needs Python 3.10+ (the images run 3.12) and Node 22 with npm; uses uv
# when it is installed. Needs network for the installs; after that, the offline
# checks need none (see AGENTS.md, "Environment setup").
#
# Written for bash 3.2 (the macOS default).
set -euo pipefail
cd "$(dirname "$0")/.."

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 -- $2" >&2; exit 1; }
}
need python3 "install Python 3.10+ (the images use 3.12)"
need npm "install Node 22 with npm (the frontend image uses node:22)"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "Python 3.10+ is required, found $(python3 --version)" >&2; exit 1; }
# The frontend toolchain's floor (frontend/package.json "engines"): jsdom 30
# needs 22.22.2+, vite 8 22.12+. Below it npm silently SKIPS rolldown's native
# binding (an optional dependency whose engines do not match), and every
# frontend command then fails with "Cannot find module @rolldown/binding-...".
node -e '
  const [a, b, c] = process.versions.node.split(".").map(Number);
  const ok = (a === 22 && (b > 22 || (b === 22 && c >= 2))) || (a === 24 && b >= 15) || a >= 26;
  process.exit(ok ? 0 : 1);' \
  || { echo "Node $(node --version) is too old for the frontend: need 22.22.2+ (or 24.15+ / 26+)." \
            "Install the current Node 22 (e.g. https://nodejs.org or nvm install 22; see .nvmrc)." >&2; exit 1; }

echo "== backend: backend/.venv"
if command -v uv >/dev/null 2>&1; then
  [ -x backend/.venv/bin/python ] || uv venv --quiet --python python3 backend/.venv
  VIRTUAL_ENV="$PWD/backend/.venv" uv pip install --quiet -e "./backend[dev]"
else
  [ -x backend/.venv/bin/python ] || python3 -m venv backend/.venv
  backend/.venv/bin/python -m pip install --quiet --upgrade pip
  backend/.venv/bin/python -m pip install --quiet -e "./backend[dev]"
fi

echo "== frontend: npm ci"
npm --prefix frontend ci --no-audit --no-fund

echo
echo "ready. next: scripts/check.sh   (add --full for the network and Docker checks)"
