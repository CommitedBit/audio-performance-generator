"""Test helpers: WAV builders, polling, and providers whose behaviour a test
controls exactly."""
from __future__ import annotations

import io
import threading
import time
import wave

from app.providers.base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, VoiceInfo


def reload_settings(monkeypatch, **env: str | None) -> None:
    """Apply env changes and drop the cached Settings so they take effect."""
    from app import config

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


def make_wav(seconds: float = 1.0, sample_rate: int = 24000, channels: int = 1, *, silent: bool = False) -> bytes:
    """A WAV of a 220 Hz tone (or silence)."""
    import numpy as np

    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = np.zeros(n, dtype=np.float32) if silent else 0.3 * np.sin(2 * np.pi * 220.0 * t)
    frames = (mono * 32767).astype(np.int16)
    if channels > 1:
        frames = np.repeat(frames[:, None], channels, axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames.tobytes())
    return buf.getvalue()


def wait_for_job(client, job_id: str, *, timeout: float = 10.0) -> dict:
    """Poll a job until it leaves queued/running."""
    deadline = time.monotonic() + timeout
    while True:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} still {job['status']} after {timeout}s")
        time.sleep(0.02)


class FakeProvider(Provider):
    """A provider whose load, output and availability are set by the test.

    `wav` is what generate() returns; `reported_duration` is the duration it
    CLAIMS, which the API is expected to ignore for WAV in favour of measuring.
    """

    def __init__(
        self,
        provider_id: str,
        capability: Capability = Capability.VOICE,
        *,
        remote: bool = False,
        seconds_range: tuple[float, float] | None = None,
        wav: bytes | None = None,
        reported_duration: float | None = None,
        is_available: bool = True,
        load_delay: float = 0.0,
        load_error: Exception | None = None,
        voices: list[VoiceInfo] | None = None,
    ) -> None:
        super().__init__()
        self.id = provider_id
        self.name = provider_id
        self.capability = capability
        self.remote = remote
        self.requires_gpu = not remote
        self._seconds_range = seconds_range
        self._wav = wav
        self._reported = reported_duration
        self._is_available = is_available
        self._load_delay = load_delay
        self._raise_on_load = load_error
        self._voices = voices or []
        self.load_calls = 0
        self.generate_calls: list[GenerateRequest] = []
        self.release_load = threading.Event()
        self.release_load.set()

    def available(self) -> bool:
        return self._is_available

    def unavailable_reason(self) -> str:
        return "" if self._is_available else "disabled by the test"

    def voices(self) -> list[VoiceInfo]:
        return self._voices

    def params(self) -> list[ParamSpec]:
        if self._seconds_range is None:
            return []
        lo, hi = self._seconds_range
        return [ParamSpec("seconds", "float", lo, lo, hi, "Clip length")]

    def _load(self):
        self.load_calls += 1
        if self._load_delay:
            time.sleep(self._load_delay)
        self.release_load.wait(10)
        if self._raise_on_load is not None:
            raise self._raise_on_load
        return object()

    def generate(self, req: GenerateRequest) -> AudioResult:
        self.load()
        self.generate_calls.append(req)
        seconds = req.seconds or 1.0
        audio = self._wav if self._wav is not None else make_wav(seconds)
        duration = self._reported if self._reported is not None else seconds
        return AudioResult(audio=audio, sample_rate=24000, duration=duration, provider_id=self.id)
