"""One release version, written in five places, must agree everywhere.

Before this, the backend said 0.1.0 and the frontend 0.0.0, so nothing could
say which release a running box was on. scripts/bump_version.py moves them
together; this test catches a hand edit that moves only one.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app import __version__

ROOT = Path(__file__).resolve().parents[2]
VERSIONED = ["backend/app/__init__.py", "backend/pyproject.toml", "frontend/package.json",
             "frontend/package-lock.json", "CHANGELOG.md"]


def _versions(root: Path) -> dict[str, str]:
    init = re.search(r'^__version__ = "([^"]+)"', (root / VERSIONED[0]).read_text(), re.M).group(1)
    pyproject = re.search(r'^version = "([^"]+)"', (root / VERSIONED[1]).read_text(), re.M).group(1)
    package = json.loads((root / VERSIONED[2]).read_text())["version"]
    lock = json.loads((root / VERSIONED[3]).read_text())
    released = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", (root / VERSIONED[4]).read_text(), re.M)
    return {
        "app.__version__": init,
        "pyproject.toml": pyproject,
        "package.json": package,
        "package-lock.json": lock["version"],
        'package-lock.json packages[""]': lock["packages"][""]["version"],
        "CHANGELOG newest release": released[0] if released else "none",
    }


def test_every_version_field_agrees():
    versions = _versions(ROOT)
    assert set(versions.values()) == {__version__}, versions


def test_health_reports_the_version(api):
    assert api.get("/health").json()["version"] == __version__


# -- scripts/bump_version.py ------------------------------------------------------

def _copy_versioned(tmp_path: Path) -> Path:
    for rel in VERSIONED:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def _set_unreleased(changelog: Path, body: str) -> None:
    """Replace whatever [Unreleased] holds, so a test does not depend on it."""
    text = changelog.read_text()
    changelog.write_text(re.sub(r"(?ms)^## \[Unreleased\]\n.*?(?=^## \[)", f"## [Unreleased]\n{body}\n", text, count=1))


def _bump(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts/bump_version.py"), *args, "--root", str(root)],
                          capture_output=True, text=True)


def _next_minor() -> str:
    major, minor, _ = map(int, __version__.split("."))
    return f"{major}.{minor + 1}.0"


def test_bump_moves_every_field_and_cuts_the_changelog(tmp_path):
    root = _copy_versioned(tmp_path)
    changelog = root / "CHANGELOG.md"
    _set_unreleased(changelog, "\n### Fixed\n- a thing\n")
    new = _next_minor()

    r = _bump(root, new, "--date", "2030-01-02")
    assert r.returncode == 0, r.stderr
    assert set(_versions(root).values()) == {new}
    text = changelog.read_text()
    assert f"## [Unreleased]\n\n## [{new}] - 2030-01-02\n\n### Fixed\n- a thing\n" in text
    assert text.count("## [Unreleased]") == 1


@pytest.mark.parametrize("version, unreleased, error", [
    ("0.0.1", "- a thing\n", "not greater than"),
    ("1.2", "- a thing\n", "not a X.Y.Z version"),
    (None, "", "[Unreleased] is empty"),
])
def test_bump_refuses_mistakes(tmp_path, version, unreleased, error):
    root = _copy_versioned(tmp_path)
    changelog = root / "CHANGELOG.md"
    _set_unreleased(changelog, unreleased)
    before = {rel: (root / rel).read_text() for rel in VERSIONED}

    r = _bump(root, version or _next_minor())
    assert r.returncode != 0 and error in r.stderr
    assert {rel: (root / rel).read_text() for rel in VERSIONED} == before    # nothing half-written
