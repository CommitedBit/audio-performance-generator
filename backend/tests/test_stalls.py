"""The idle sweep must never stall the API or another model's load.

Loading a model holds that provider's load lock for minutes. Before these
fixes, a sweep that reached a provider mid-load waited for that lock ON THE
EVENT LOOP, freezing every request until the load finished; and Stable Audio's
unload waited on a module-wide lock that a different checkpoint's build held.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from helpers import FakeProvider, reload_settings

from app.providers import stable_audio
from app.providers.base import Capability
from app.providers.stable_audio import StableAudio3Provider


def _finishes_within(fn, seconds: float) -> bool:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(seconds)
    return not t.is_alive()


def _start_blocked_load(provider) -> threading.Thread:
    provider.release_load.clear()
    t = threading.Thread(target=provider.load, daemon=True)
    t.start()
    deadline = time.monotonic() + 2
    while provider.load_calls == 0:
        assert time.monotonic() < deadline, "load never started"
        time.sleep(0.005)
    return t


def test_unload_if_idle_does_not_wait_for_a_load_in_progress():
    p = FakeProvider("p")
    loader = _start_blocked_load(p)
    try:
        assert _finishes_within(lambda: p.unload_if_idle(0), 0.5), "sweep blocked on the load lock"
    finally:
        p.release_load.set()
        loader.join(5)


def test_the_api_keeps_answering_while_a_sweep_meets_a_loading_model(use_providers, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main
    from app.jobs import JobQueue

    p = FakeProvider("loading", Capability.MUSIC)
    use_providers(p)
    monkeypatch.setattr(main, "queue", JobQueue())
    monkeypatch.setattr(main, "SWEEP_MIN_INTERVAL", 0.01)
    monkeypatch.setattr(main.settings, "model_idle_timeout", 1)
    loader = _start_blocked_load(p)
    try:
        with TestClient(main.app) as client:
            time.sleep(0.1)                           # several sweeps have fired by now
            status: list[int] = []
            assert _finishes_within(lambda: status.append(client.get("/health").status_code), 2.0), \
                "the event loop is frozen"
            assert status == [200]
            p.release_load.set()
    finally:
        p.release_load.set()
        loader.join(5)


async def test_sweep_runs_off_the_event_loop(monkeypatch):
    """Unloading frees GPU memory and runs gc, which can take seconds."""
    from app import main

    class SlowRegistry:
        def sweep_idle(self, timeout):
            time.sleep(0.5)
            return []

    monkeypatch.setattr(main, "get_registry", lambda: SlowRegistry())
    monkeypatch.setattr(main, "SWEEP_MIN_INTERVAL", 0.01)
    monkeypatch.setattr(main.settings, "model_idle_timeout", 1)
    sweeper = asyncio.create_task(main._sweep_idle_models())
    try:
        await asyncio.sleep(0.05)                     # the first sweep is now running
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.sleep(0.02)
        assert loop.time() - start < 0.3, "the sweep blocked the event loop"
    finally:
        sweeper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sweeper


def test_unload_runs_gc_before_emptying_the_cache(monkeypatch):
    """Without a collection, a model in a reference cycle keeps its VRAM."""
    import gc

    calls: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda *a: calls.append("gc") or 0)
    p = FakeProvider("p")
    p.load()
    p.unload()
    assert calls == ["gc"]


# -- Stable Audio 3: one checkpoint's build must not block another's ----------------

@pytest.fixture
def gated_builds(monkeypatch, fake_torch):
    """Builds block until their checkpoint's event is set."""
    gates: dict[str, threading.Event] = {}
    started: list[str] = []

    def fake_build(self, device, dtype, loader):
        started.append(self.hf_repo)
        gates.setdefault(self.hf_repo, threading.Event()).wait(5)
        return object()

    monkeypatch.setattr(StableAudio3Provider, "_build_pipeline", fake_build)
    monkeypatch.setattr(stable_audio, "_official_available", lambda: True)
    reload_settings(monkeypatch, HF_TOKEN="hf_test", SA3_MODEL=None)
    return gates, started


def _start_sa3_build(provider, gates, started) -> threading.Thread:
    gates[provider.hf_repo] = threading.Event()
    t = threading.Thread(target=provider.load, daemon=True)
    t.start()
    deadline = time.monotonic() + 2
    while provider.hf_repo not in started:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    return t


def test_unloading_one_checkpoint_does_not_wait_for_another_build(gated_builds):
    gates, started = gated_builds
    sfx, music = StableAudio3Provider(Capability.SFX), StableAudio3Provider(Capability.MUSIC)
    gates[sfx.hf_repo] = threading.Event()
    gates[sfx.hf_repo].set()
    sfx.load()
    builder = _start_sa3_build(music, gates, started)
    try:
        assert _finishes_within(sfx.unload, 0.5), "sfx unload waited for the music build"
        assert not sfx.loaded
    finally:
        gates[music.hf_repo].set()
        builder.join(5)


def test_different_checkpoints_load_in_parallel(gated_builds):
    gates, started = gated_builds
    sfx, music = StableAudio3Provider(Capability.SFX), StableAudio3Provider(Capability.MUSIC)
    builder = _start_sa3_build(music, gates, started)
    try:
        gates[sfx.hf_repo] = threading.Event()
        gates[sfx.hf_repo].set()
        assert _finishes_within(sfx.load, 0.5), "sfx load waited for the music build"
    finally:
        gates[music.hf_repo].set()
        builder.join(5)


def test_the_same_checkpoint_is_still_built_once(gated_builds, monkeypatch):
    gates, started = gated_builds
    reload_settings(monkeypatch, SA3_MODEL="medium")
    sfx, music = StableAudio3Provider(Capability.SFX), StableAudio3Provider(Capability.MUSIC)
    builder = _start_sa3_build(sfx, gates, started)
    waiter = threading.Thread(target=music.load, daemon=True)
    waiter.start()
    time.sleep(0.05)
    gates[sfx.hf_repo].set()
    builder.join(5)
    waiter.join(5)
    assert started == ["stabilityai/stable-audio-3-medium"]
    assert sfx.load() is music.load()
    assert stable_audio._HOLDERS[sfx._pipeline_key] == {sfx.id, music.id}
