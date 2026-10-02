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


def test_cloud_is_never_an_automatic_default(use_providers):
    """A local model failing must not silently route story text to a paid API."""
    reg = use_providers(
        FakeProvider("chatterbox", Capability.VOICE, is_available=False),
        FakeProvider("elevenlabs-voice", Capability.VOICE, remote=True),
    )
    assert reg.default_for(Capability.VOICE) is None
    with pytest.raises(RuntimeError, match="elevenlabs-voice can serve it if named explicitly"):
        reg.resolve(Capability.VOICE, None)
    assert reg.resolve(Capability.VOICE, "elevenlabs-voice").id == "elevenlabs-voice"
    assert reg.default_for(Capability.SFX) is None


def test_cloud_default_can_be_allowed(monkeypatch, use_providers):
    reload_settings(monkeypatch, ALLOW_CLOUD_DEFAULT="1")
    reg = use_providers(
        FakeProvider("chatterbox", Capability.VOICE, is_available=False),
        FakeProvider("elevenlabs-voice", Capability.VOICE, remote=True),
    )
    assert reg.default_for(Capability.VOICE).id == "elevenlabs-voice"


def test_local_still_beats_cloud_when_cloud_defaults_are_allowed(monkeypatch, use_providers):
    reload_settings(monkeypatch, ALLOW_CLOUD_DEFAULT="1")
    reg = use_providers(
        FakeProvider("elevenlabs-voice", Capability.VOICE, remote=True),
        FakeProvider("chatterbox", Capability.VOICE),
    )
    assert reg.default_for(Capability.VOICE).id == "chatterbox"


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


def test_no_placeholders_without_dev_stub_even_when_nothing_is_available(monkeypatch):
    """The old fallback served test tones on a box whose real models all failed."""
    reload_settings(monkeypatch, DEV_STUB="0")
    reg = registry.get_registry()
    assert not any(p.available() for p in reg.all())     # no model deps on the test host
    assert not any(p.id.startswith("stub") for p in reg.all())


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
