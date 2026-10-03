"""Provider registration, allowlisting and default routing (app.registry)."""
from __future__ import annotations

import pytest
from helpers import FakeProvider, reload_settings

from app import registry
from app.providers.base import Capability


def test_preference_rank_orders_local_then_other_local_then_cloud_then_stubs():
    ids = ["stub-voice", "elevenlabs-voice", "some-new-local", "musicgen", "chatterbox", "acestep"]
    ranked = sorted(ids, key=registry.preference_rank)
    assert ranked == ["chatterbox", "acestep", "musicgen", "some-new-local", "elevenlabs-voice", "stub-voice"]


def test_default_follows_preference_not_registration_order(use_providers):
    reg = use_providers(
        FakeProvider("stub-music", Capability.MUSIC),
        FakeProvider("musicgen", Capability.MUSIC),
        FakeProvider("acestep", Capability.MUSIC),
    )
    assert reg.default_for(Capability.MUSIC).id == "acestep"


def test_default_skips_unavailable(use_providers):
    reg = use_providers(
        FakeProvider("chatterbox", Capability.VOICE, is_available=False),
        FakeProvider("elevenlabs-voice", Capability.VOICE, remote=True),
    )
    assert reg.default_for(Capability.VOICE).id == "elevenlabs-voice"
    assert reg.default_for(Capability.SFX) is None


def test_resolve_errors(use_providers):
    reg = use_providers(
        FakeProvider("v", Capability.VOICE),
        FakeProvider("off", Capability.VOICE, is_available=False),
    )
    with pytest.raises(KeyError):
        reg.resolve(Capability.VOICE, "missing")
    with pytest.raises(ValueError):
        reg.resolve(Capability.MUSIC, "v")
    with pytest.raises(RuntimeError, match="disabled by the test"):
        reg.resolve(Capability.VOICE, "off")
    with pytest.raises(RuntimeError, match="no available provider for sfx"):
        reg.resolve(Capability.SFX, None)
    assert reg.resolve(Capability.VOICE, None).id == "v"


def test_providers_allowlist(monkeypatch, use_providers):
    reload_settings(monkeypatch, PROVIDERS="acestep, chatterbox")
    reg = use_providers(
        FakeProvider("chatterbox", Capability.VOICE),
        FakeProvider("acestep", Capability.MUSIC),
        FakeProvider("musicgen", Capability.MUSIC),
    )
    assert sorted(p.id for p in reg.all()) == ["acestep", "chatterbox"]


def test_dev_stub_adds_placeholders_after_real_providers(monkeypatch):
    reload_settings(monkeypatch, DEV_STUB="1")
    ids = [p.id for p in registry.get_registry().all()]
    assert ids[-3:] == ["stub-voice", "stub-music", "stub-sfx"]
    # Every real provider is registered (and listed) even when unavailable.
    for real in ("chatterbox", "acestep", "stable-audio-3-sfx", "stable-audio-3-music", "musicgen",
                 "elevenlabs-voice", "elevenlabs-sfx"):
        assert real in ids


def test_sweep_idle_unloads_only_idle_models(use_providers):
    busy = FakeProvider("busy", Capability.VOICE)
    idle = FakeProvider("idle", Capability.MUSIC)
    reg = use_providers(busy, idle)
    busy.load()
    idle.load()
    idle._last_used -= 100
    with busy.in_use():
        busy._last_used -= 100
        assert reg.sweep_idle(10) == ["idle"]
    assert busy.loaded and not idle.loaded
    assert reg.sweep_idle(0) == []
