"""Shared test setup.

Three things about the app make naive tests flaky or wrong, and this file exists
to neutralise each one:

  * Settings() creates DATA_DIR/{audio,voices} the moment it is built, and the
    default is /data -- so the environment must point somewhere writable BEFORE
    any app module is imported. Settings are lru_cached, so every test also
    clears the cache to see its own environment.
  * The registry and app.main.queue are module-level singletons. JobQueue's
    semaphores bind to the first event loop that contends on them, and each
    TestClient runs its own loop, so a queue reused across tests can raise
    "bound to a different event loop". Both are rebuilt per test.
  * TestClient only runs the app lifespan inside a `with` block. Without it the
    job queue never gets a running loop and jobs never complete. The `api`
    fixture always uses `with`.

No test needs torch. Tests that exercise a torch import path install a fake one
with `fake_torch`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Must run before anything imports app.config.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="apg-test-")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("DEV_STUB", "1")
os.environ.setdefault("MODEL_IDLE_TIMEOUT", "0")      # no background sweeper unless a test asks
for var in ("API_KEY", "GPU_LOCK_FILE", "ELEVENLABS_API_KEY", "HF_TOKEN", "SA3_MODEL", "PROVIDERS"):
    os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Fresh data dir, settings, registry and shared provider state per test."""
    from app import config, registry
    from app.providers import elevenlabs, stable_audio

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    config.get_settings.cache_clear()
    monkeypatch.setattr(registry, "_registry", None)

    # Module-level state that outlives a single provider instance.
    monkeypatch.setattr(elevenlabs, "_auth_error", None)
    monkeypatch.setattr(elevenlabs, "_auth_failed_at", 0.0)
    monkeypatch.setattr(elevenlabs, "_auth_probe", False)
    stable_audio._SHARED.clear()
    stable_audio._HOLDERS.clear()
    stable_audio._CALL_LOCKS.clear()
    stable_audio._BUILD_LOCKS.clear()

    yield

    config.get_settings.cache_clear()


@pytest.fixture
def fake_torch(monkeypatch):
    """A stand-in torch module for code paths that only touch dtype names and
    the CUDA cache. Anything heavier belongs on the GPU smoke test."""
    torch = types.ModuleType("torch")
    torch.float16 = "torch.float16"
    torch.float32 = "torch.float32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch.manual_seed = lambda seed: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def use_providers(monkeypatch):
    """Replace the registry's provider set with the given instances."""
    from app import registry

    def _install(*providers):
        monkeypatch.setattr(registry, "_build", lambda: list(providers))
        monkeypatch.setattr(registry, "_registry", None)
        return registry.get_registry()

    return _install


@pytest.fixture
def api(monkeypatch):
    """A TestClient for the model service, with its lifespan running and a
    fresh job queue bound to this client's event loop."""
    from fastapi.testclient import TestClient

    from app import main
    from app.jobs import JobQueue

    monkeypatch.setattr(main, "queue", JobQueue(max_concurrent=1))
    with TestClient(main.app) as client:
        yield client
