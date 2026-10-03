#!/usr/bin/env python3
"""Move every version field to X.Y.Z and cut the CHANGELOG's [Unreleased] section.

The release version is written in five places, and they must agree
(backend/tests/test_version.py):
  backend/app/__init__.py      __version__ (what the API reports in /health)
  backend/pyproject.toml       [project] version
  frontend/package.json        version (what the UI shows)
  frontend/package-lock.json   both root version fields
  CHANGELOG.md                 the newest release heading

usage: scripts/bump_version.py X.Y.Z [--date YYYY-MM-DD] [--root PATH]

Refuses a version that is not greater than the current one, and an empty
[Unreleased] section -- a release with no recorded changes is a mistake.
See docs/git-workflow.md#releases for the rest of the release steps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def current_version(root: Path) -> str:
    m = re.search(r'^__version__ = "([^"]+)"', (root / "backend/app/__init__.py").read_text(), re.M)
    if not m:
        sys.exit("cannot find __version__ in backend/app/__init__.py")
    return m.group(1)


def _sub_once(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"version field not found in {path}")
    path.write_text(new)


def cut_changelog(path: Path, version: str, date: str) -> None:
    text = path.read_text()
    m = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[)", text, re.M | re.S)
    if not m:
        sys.exit("CHANGELOG.md has no [Unreleased] section followed by a release")
    if not m.group(1).strip():
        sys.exit("[Unreleased] is empty: record the changes before releasing")
    heading = f"## [Unreleased]\n\n## [{version}] - {date}\n\n"
    path.write_text(text[: m.start()] + heading + m.group(1).lstrip("\n") + text[m.end():])


def bump(root: Path, version: str, date: str) -> None:
    if not SEMVER.match(version):
        sys.exit(f"not a X.Y.Z version: {version}")
    old = current_version(root)
    if tuple(map(int, SEMVER.match(version).groups())) <= tuple(map(int, SEMVER.match(old).groups())):
        sys.exit(f"{version} is not greater than the current {old}")

    cut_changelog(root / "CHANGELOG.md", version, date)
    _sub_once(root / "backend/app/__init__.py", r'^__version__ = "[^"]+"', f'__version__ = "{version}"')
    _sub_once(root / "backend/pyproject.toml", r'^version = "[^"]+"', f'version = "{version}"')
    # The first "version" key in package.json is the top-level one.
    _sub_once(root / "frontend/package.json", r'^  "version": "[^"]+"', f'  "version": "{version}"')
    lock_path = root / "frontend/package-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["version"] = version
    lock["packages"][""]["version"] = version
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"{old} -> {version}: updated 5 files; review the diff, then open the release PR")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    bump(Path(a.root), a.version, a.date)


if __name__ == "__main__":
    main()
