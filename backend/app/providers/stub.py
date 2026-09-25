"""Dependency-free providers used for development and CI.

These let the whole API, the job queue and the frontend integration be built
and tested before a single multi-GB checkpoint is downloaded, and they keep the
container booting on a machine with no GPU.
"""
from __future__ import annotations

import hashlib
import math

from .base import (
    AudioResult,
    Capability,
    GenerateRequest,
    ParamSpec,
    Provider,
    VoiceInfo,
    pcm_to_wav,
)

SAMPLE_RATE = 24000


def _tone(seconds: float, seed: str, *, harmonics: int = 3) -> bytes:
    """A deterministic, prompt-derived tone. Audibly obvious as a placeholder."""
    import numpy as np

    digest = hashlib.sha256(seed.encode()).digest()
    base = 110.0 * (2 ** ((digest[0] % 24) / 12.0))
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    wave = np.zeros(n, dtype=np.float32)
    for h in range(1, harmonics + 1):
        wave += (0.5 ** h) * np.sin(2 * math.pi * base * h * t)

    attack = min(n, SAMPLE_RATE // 50)
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    env[-attack:] = np.linspace(1, 0, attack, dtype=np.float32)
    return pcm_to_wav(wave * env * 0.3, SAMPLE_RATE)


class StubVoice(Provider):
    id = "stub-voice"
    name = "Placeholder voice"
    capability = Capability.VOICE
    license = "n/a (generates a tone, not speech)"
    requires_gpu = False
    description = "Development placeholder. Emits a tone whose length tracks the text, so timeline wiring can be tested without a model."

    def _load(self):
        return object()

    def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id="stub-a", name="Placeholder A"),
            VoiceInfo(id="stub-b", name="Placeholder B"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        # ~14 characters per second is roughly conversational pace.
        seconds = req.seconds or max(0.6, min(30.0, len(req.prompt) / 14.0))
        audio = _tone(seconds, f"{req.voice_id or 'stub-a'}:{req.prompt}")
        return AudioResult(audio=audio, sample_rate=SAMPLE_RATE, duration=seconds, provider_id=self.id)


class StubMusic(Provider):
    id = "stub-music"
    name = "Placeholder music"
    capability = Capability.MUSIC
    license = "n/a (generates a tone, not music)"
    requires_gpu = False
    description = "Development placeholder for the music track."

    def _load(self):
        return object()

    def params(self) -> list[ParamSpec]:
        return [ParamSpec("seconds", "float", 10.0, 1.0, 120.0, "Clip length")]

    def generate(self, req: GenerateRequest) -> AudioResult:
        seconds = req.seconds or 10.0
        audio = _tone(seconds, f"music:{req.prompt}", harmonics=5)
        return AudioResult(audio=audio, sample_rate=SAMPLE_RATE, duration=seconds, provider_id=self.id)


class StubSfx(Provider):
    id = "stub-sfx"
    name = "Placeholder SFX"
    capability = Capability.SFX
    license = "n/a (generates a tone, not sfx)"
    requires_gpu = False
    description = "Development placeholder for the sfx track."

    def _load(self):
        return object()

    def params(self) -> list[ParamSpec]:
        return [ParamSpec("seconds", "float", 3.0, 0.2, 30.0, "Clip length")]

    def generate(self, req: GenerateRequest) -> AudioResult:
        seconds = req.seconds or 3.0
        audio = _tone(seconds, f"sfx:{req.prompt}", harmonics=1)
        return AudioResult(audio=audio, sample_rate=SAMPLE_RATE, duration=seconds, provider_id=self.id)
