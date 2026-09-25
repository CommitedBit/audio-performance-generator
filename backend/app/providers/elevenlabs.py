"""ElevenLabs -- optional cloud fallback, kept for A/B comparison.

The key now lives in the server environment rather than browser localStorage,
which is the one unambiguous security win of moving to a backend. Without
ELEVENLABS_API_KEY set, this provider reports itself unavailable and never
appears in /v1/models.
"""
from __future__ import annotations

import logging
import threading
import time

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, VoiceInfo

log = logging.getLogger(__name__)

API_ROOT = "https://api.elevenlabs.io/v1"
TIMEOUT = 120.0

# /v1/models must never wait on the network. The gateway re-runs discovery on
# every generate request and abandons an upstream after DISCOVERY_TIMEOUT
# (10 s); fetching voices inline with a 15 s timeout meant a slow or
# unreachable ElevenLabs marked the whole models service down, and local
# Chatterbox / ACE-Step / Stable Audio generation then returned 503. Discovery
# now serves a cached list that a background thread refreshes.
VOICES_TTL = 600.0            # refresh a good list every 10 minutes
VOICES_RETRY = 60.0           # back off this long after a failed fetch
VOICES_FETCH_TIMEOUT = 10.0
# Served until the first fetch lands. It is the voice generate() defaults to,
# so it is always valid and the picker is never empty.
FALLBACK_VOICES = (
    VoiceInfo(id="21m00Tcm4TlvDq8ikWAM", name="Rachel (default)", description="premade"),
)


class ElevenLabsVoice(Provider):
    id = "elevenlabs-voice"
    name = "ElevenLabs (cloud)"
    capability = Capability.VOICE
    license = "commercial SaaS -- your ElevenLabs plan terms apply"
    requires_gpu = False
    description = "Cloud TTS. Sends text to ElevenLabs; requires an API key and network egress."

    def __init__(self) -> None:
        super().__init__()
        self._voice_lock = threading.Lock()
        self._voice_cache: list[VoiceInfo] = []
        self._voices_due = 0.0            # monotonic time of the next refresh
        self._voices_refreshing = False

    def available(self) -> bool:
        return bool(get_settings().elevenlabs_api_key)

    def unavailable_reason(self) -> str:
        return "" if self.available() else "ELEVENLABS_API_KEY is not set"

    def _load(self):
        return object()

    def voices(self) -> list[VoiceInfo]:
        """Return the cached voice list immediately; refresh it in the background."""
        if not self.available():
            return []
        with self._voice_lock:
            cached = list(self._voice_cache)
            if time.monotonic() >= self._voices_due and not self._voices_refreshing:
                # At most one fetch in flight, however often discovery runs.
                self._voices_refreshing = True
                threading.Thread(target=self._refresh_voices, name="elevenlabs-voices", daemon=True).start()
        return cached or list(FALLBACK_VOICES)

    def _refresh_voices(self) -> None:
        next_due = time.monotonic() + VOICES_RETRY
        try:
            import httpx

            r = httpx.get(
                f"{API_ROOT}/voices",
                headers={"xi-api-key": get_settings().elevenlabs_api_key or ""},
                timeout=VOICES_FETCH_TIMEOUT,
            )
            r.raise_for_status()
            fetched = [
                VoiceInfo(id=v["voice_id"], name=v.get("name", v["voice_id"]), description=v.get("category", ""))
                for v in r.json().get("voices", [])
            ]
            with self._voice_lock:
                self._voice_cache = fetched
            next_due = time.monotonic() + VOICES_TTL
        except Exception as exc:                       # noqa: BLE001
            # Keep serving the last good list (or the fallback) and retry later.
            log.warning("could not list ElevenLabs voices: %s", exc)
        finally:
            with self._voice_lock:
                self._voices_due = next_due
                self._voices_refreshing = False

    def params(self) -> list[ParamSpec]:
        return [
            ParamSpec("model_id", "str", "eleven_multilingual_v2", None, None, "ElevenLabs model"),
            ParamSpec("stability", "float", 0.5, 0.0, 1.0, ""),
            ParamSpec("similarity_boost", "float", 0.75, 0.0, 1.0, ""),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        import httpx

        key = get_settings().elevenlabs_api_key
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

        voice = req.voice_id or "21m00Tcm4TlvDq8ikWAM"
        body = {
            "text": req.prompt,
            "model_id": req.params.get("model_id", "eleven_multilingual_v2"),
            "voice_settings": {
                "stability": float(req.params.get("stability", 0.5)),
                "similarity_boost": float(req.params.get("similarity_boost", 0.75)),
            },
        }
        r = httpx.post(
            f"{API_ROOT}/text-to-speech/{voice}",
            headers={"xi-api-key": key, "Accept": "audio/mpeg", "Content-Type": "application/json"},
            json=body,
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            # The original client threw away status and body, making quota and
            # auth failures indistinguishable. Keep both.
            raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:400]}")

        audio = r.content
        return AudioResult(audio=audio, sample_rate=44100, duration=0.0, provider_id=self.id, mime="audio/mpeg")


class ElevenLabsSfx(Provider):
    id = "elevenlabs-sfx"
    name = "ElevenLabs SFX (cloud)"
    capability = Capability.SFX
    license = "commercial SaaS -- your ElevenLabs plan terms apply"
    requires_gpu = False
    description = "Cloud sound-effect generation."

    def available(self) -> bool:
        return bool(get_settings().elevenlabs_api_key)

    def unavailable_reason(self) -> str:
        return "" if self.available() else "ELEVENLABS_API_KEY is not set"

    def _load(self):
        return object()

    def params(self) -> list[ParamSpec]:
        return [
            ParamSpec("seconds", "float", 4.0, 0.5, 22.0, "Clip length"),
            ParamSpec("prompt_influence", "float", 0.3, 0.0, 1.0, ""),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        import httpx

        key = get_settings().elevenlabs_api_key
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

        body: dict = {"text": req.prompt, "prompt_influence": float(req.params.get("prompt_influence", 0.3))}
        seconds = req.seconds or req.params.get("seconds")
        if seconds:
            body["duration_seconds"] = float(seconds)

        r = httpx.post(
            f"{API_ROOT}/sound-generation",
            headers={"xi-api-key": key, "Accept": "audio/mpeg", "Content-Type": "application/json"},
            json=body,
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:400]}")

        return AudioResult(
            audio=r.content, sample_rate=44100, duration=float(seconds or 0.0),
            provider_id=self.id, mime="audio/mpeg",
        )
