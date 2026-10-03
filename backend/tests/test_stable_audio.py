"""Stable Audio 3's shared, refcounted pipeline and loader selection (providers.stable_audio)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from helpers import reload_settings

from app.providers import stable_audio
from app.providers.base import Capability, GenerateRequest
from app.providers.stable_audio import StableAudio3Provider


class FakeRunner:
    kind = "fake"
    sample_rate = 44100

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def run(self, prompt, negative_prompt, seconds, steps, cfg, seed):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        n = int(seconds * self.sample_rate)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        return np.stack([tone, tone]), self.sample_rate


@pytest.fixture
def builds(monkeypatch, fake_torch):
    """Count pipeline builds; every build returns a fresh FakeRunner."""
    built: list[tuple[str, FakeRunner]] = []

    def fake_build(self, device, dtype, loader):
        runner = FakeRunner()
        built.append((self.hf_repo, runner))
        return runner

    monkeypatch.setattr(StableAudio3Provider, "_build_pipeline", fake_build)
    monkeypatch.setattr(stable_audio, "_official_available", lambda: True)
    reload_settings(monkeypatch, HF_TOKEN="hf_test")
    return built


def _both(monkeypatch, sa3_model: str | None):
    reload_settings(monkeypatch, SA3_MODEL=sa3_model)
    return StableAudio3Provider(Capability.SFX), StableAudio3Provider(Capability.MUSIC)


def test_one_medium_checkpoint_is_shared_by_both_capabilities(builds, monkeypatch):
    sfx, music = _both(monkeypatch, "stabilityai/stable-audio-3-medium")
    assert sfx.load() is music.load()
    assert len(builds) == 1
    assert sfx._call_lock is music._call_lock


def test_shared_pipeline_survives_until_the_last_holder_unloads(builds, monkeypatch):
    sfx, music = _both(monkeypatch, "medium")
    sfx.load()
    music.load()
    key = sfx._pipeline_key
    sfx.unload()
    assert key in stable_audio._SHARED and stable_audio._HOLDERS[key] == {music.id}
    music.unload()
    assert key not in stable_audio._SHARED and key not in stable_audio._CALL_LOCKS
    sfx.load()
    assert len(builds) == 2                        # rebuilt only after it was really freed


def test_small_checkpoints_are_separate(builds, monkeypatch):
    sfx, music = _both(monkeypatch, None)
    assert sfx.load() is not music.load()
    assert [repo for repo, _ in builds] == ["stabilityai/stable-audio-3-small-sfx",
                                            "stabilityai/stable-audio-3-small-music"]


def test_calls_on_shared_weights_are_serialised(builds, monkeypatch):
    sfx, music = _both(monkeypatch, "medium")
    reqs = [(sfx, Capability.SFX), (music, Capability.MUSIC)] * 2
    threads = [threading.Thread(target=p.generate, args=(GenerateRequest("x", cap, seconds=1.0),))
               for p, cap in reqs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert builds[0][1].max_active == 1


def test_generate_clamps_length_and_measures_stereo(builds, monkeypatch):
    sfx, _ = _both(monkeypatch, None)
    result = sfx.generate(GenerateRequest("door slam", Capability.SFX, seconds=500))
    assert result.duration == pytest.approx(120.0)      # small tier max
    assert result.sample_rate == 44100


def test_model_name_mapping(monkeypatch):
    for value in ("medium", "stabilityai/stable-audio-3-medium"):
        reload_settings(monkeypatch, SA3_MODEL=value)
        p = StableAudio3Provider(Capability.MUSIC)
        assert (p.official_name, p.hf_repo, p.tier, p.max_seconds) == (
            "medium", "stabilityai/stable-audio-3-medium", "medium", 380.0)


def test_availability_needs_a_loader_and_a_token(monkeypatch):
    monkeypatch.setattr(stable_audio, "_official_available", lambda: True)
    reload_settings(monkeypatch, HF_TOKEN=None)
    p = StableAudio3Provider(Capability.SFX)
    assert not p.available() and "HF_TOKEN" in p.unavailable_reason()

    monkeypatch.setattr(stable_audio, "_official_available", lambda: False)
    monkeypatch.setattr(stable_audio, "_diffusers_problem", lambda: "diffusers is not installed")
    reload_settings(monkeypatch, HF_TOKEN="hf_test")
    assert not p.available()
    assert p.unavailable_reason().startswith("no usable loader")
