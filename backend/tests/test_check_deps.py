"""scripts/check_deps.py tells a network failure apart from a pin failure.

CI once failed with "the image would build a package from source" when
download.pytorch.org answered 503: the wheel-only resolve could not reach the
index, and the script read that as a resolution difference. These tests fake
uv, so they run offline.
"""
from __future__ import annotations

import importlib.util
import time
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_deps.py"
OUTAGE = ("error: Request failed after 3 retries\n  Caused by: Failed to fetch: "
          "`https://download.pytorch.org/whl/cu128/soxr/`\n  Caused by: HTTP status server error (503)")


@pytest.fixture
def check_deps(monkeypatch):
    spec = importlib.util.spec_from_file_location("check_deps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(time, "sleep", lambda s: None)      # the retry pauses
    return module


def _fake_uv(module, monkeypatch, replies):
    """Each uv call pops the next (returncode, stderr); a 0 writes a pin file."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        code, err = replies.pop(0)
        if code == 0:
            Path(cmd[cmd.index("-o") + 1]).write_text("torch==2.7.1+cu128\n")
        return types.SimpleNamespace(returncode=code, stderr=err, stdout="")

    monkeypatch.setattr(module.subprocess, "run", run)
    return calls


def test_a_transient_outage_is_retried(check_deps, monkeypatch, tmp_path):
    calls = _fake_uv(check_deps, monkeypatch, [(1, OUTAGE), (0, "")])
    ok, _ = check_deps._compile(tmp_path / "in", tmp_path / "ov", tmp_path / "out.txt", "cu128", "3.12",
                                wheel_only=True)
    assert ok and len(calls) == 2


def test_a_lasting_outage_is_reported_as_a_network_problem(check_deps, monkeypatch):
    _fake_uv(check_deps, monkeypatch, [(1, OUTAGE)] * check_deps.COMPILE_ATTEMPTS)
    ok, detail = check_deps.resolve("cu128", "3.12", "torch==2.7.1+cu128", "requirements-models.txt", "")
    assert not ok
    assert "network problem, not a pin problem" in detail
    assert "build a package from source" not in detail


def test_an_outage_in_the_wheel_only_pass_is_not_blamed_on_the_pins(check_deps, monkeypatch):
    _fake_uv(check_deps, monkeypatch, [(0, "")] + [(1, OUTAGE)] * check_deps.COMPILE_ATTEMPTS)
    ok, detail = check_deps.resolve("cu128", "3.12", "torch==2.7.1+cu128", "requirements-models.txt", "")
    assert not ok and "network problem" in detail


def test_a_real_resolution_failure_is_not_retried(check_deps, monkeypatch):
    calls = _fake_uv(check_deps, monkeypatch, [(1, "No solution found when resolving dependencies")])
    ok, detail = check_deps.resolve("cu128", "3.12", "torch==2.6.0+cu128", "requirements-models.txt", "")
    assert not ok and detail.startswith("does not resolve") and len(calls) == 1
