"""ElevenLabs credential-rejection state (providers.elevenlabs), with a fake httpx."""
from __future__ import annotations

import sys
import threading
import types

import pytest
from helpers import reload_settings

from app.providers import elevenlabs
from app.providers.base import Capability, GenerateRequest


class FakeHttpx(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("httpx")
        self.status = 200
        self.calls = 0
        self.gate: threading.Event | None = None

    def _reply(self):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        return types.SimpleNamespace(status_code=self.status, text="detail", content=b"ID3fake",
                                     json=lambda: {"voices": []}, raise_for_status=lambda: None)

    def post(self, url, **kwargs):
        return self._reply()

    def get(self, url, **kwargs):
        return self._reply()


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttpx()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    reload_settings(monkeypatch, ELEVENLABS_API_KEY="test-key", PROVIDER_RETRY_SECONDS="300")
    return fake


def _speak(p):
    return p.generate(GenerateRequest(prompt="hi", capability=Capability.VOICE))


def test_cloud_providers_are_remote_and_need_a_key(monkeypatch):
    reload_settings(monkeypatch, ELEVENLABS_API_KEY=None)
    voice, sfx = elevenlabs.ElevenLabsVoice(), elevenlabs.ElevenLabsSfx()
    assert voice.remote and sfx.remote
    assert not voice.available()
    assert voice.unavailable_reason() == "ELEVENLABS_API_KEY is not set"


def test_rejected_key_takes_both_providers_out_and_fails_fast(http):
    voice, sfx = elevenlabs.ElevenLabsVoice(), elevenlabs.ElevenLabsSfx()
    http.status = 401
    with pytest.raises(RuntimeError, match="ElevenLabs 401"):
        _speak(voice)
    assert not voice.available() and not sfx.available()
    assert "rejected the API key (401)" in sfx.unavailable_reason()
    with pytest.raises(RuntimeError, match="retrying in"):
        _speak(voice)
    assert http.calls == 1                         # the known-bad key cost no second call


def test_after_cooldown_only_one_caller_probes(http, monkeypatch):
    elevenlabs._record_auth_failure(401, "bad")
    reload_settings(monkeypatch, PROVIDER_RETRY_SECONDS="0")
    first = elevenlabs._auth_gate()
    second = elevenlabs._auth_gate()
    assert first == (None, True)
    assert second[0] and "re-checking" in second[0] and second[1] is False
    assert elevenlabs._auth_problem().endswith("(re-checking the key now)")


def test_successful_probe_clears_the_rejection(http, monkeypatch):
    elevenlabs._record_auth_failure(401, "bad")
    reload_settings(monkeypatch, PROVIDER_RETRY_SECONDS="0")
    http.status = 200
    voice = elevenlabs.ElevenLabsVoice()
    result = _speak(voice)
    assert result.mime == "audio/mpeg"
    assert elevenlabs._auth_problem() is None
    assert voice.available()


def test_inconclusive_probe_releases_the_claim(http, monkeypatch):
    elevenlabs._record_auth_failure(401, "bad")
    reload_settings(monkeypatch, PROVIDER_RETRY_SECONDS="0")
    http.status = 503
    with pytest.raises(RuntimeError, match="ElevenLabs 503"):
        _speak(elevenlabs.ElevenLabsVoice())
    assert elevenlabs._auth_gate() == (None, True)   # the next caller may probe again


def test_voices_never_wait_on_the_network(http):
    http.gate = threading.Event()                    # the refresh would hang forever
    voice = elevenlabs.ElevenLabsVoice()
    assert [v.id for v in voice.voices()] == [v.id for v in elevenlabs.FALLBACK_VOICES]
    http.gate.set()


def test_sfx_sends_the_requested_duration(http, monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(http, "post", lambda url, **kw: sent.append(kw["json"]) or http._reply())
    sfx = elevenlabs.ElevenLabsSfx()
    result = sfx.generate(GenerateRequest(prompt="rain", capability=Capability.SFX, seconds=5))
    assert sent[-1]["duration_seconds"] == 5.0
    assert result.duration == 5.0
