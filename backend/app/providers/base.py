"""Provider interface shared by local models and optional cloud backends."""
from __future__ import annotations

import abc
import enum
import io
import time
import wave
from dataclasses import dataclass, field


class Capability(str, enum.Enum):
    VOICE = "voice"
    MUSIC = "music"
    SFX = "sfx"


@dataclass(slots=True)
class VoiceInfo:
    id: str
    name: str
    description: str = ""
    cloned: bool = False


@dataclass(slots=True)
class ParamSpec:
    """Describes a tunable the UI should render for this provider."""
    name: str
    type: str                      # "float" | "int" | "str" | "bool"
    default: float | int | str | bool
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""


@dataclass(slots=True)
class GenerateRequest:
    prompt: str
    capability: Capability
    voice_id: str | None = None
    seconds: float | None = None
    seed: int | None = None
    params: dict = field(default_factory=dict)


@dataclass(slots=True)
class AudioResult:
    audio: bytes            # WAV bytes
    sample_rate: int
    duration: float
    provider_id: str
    mime: str = "audio/wav"


def pcm_to_wav(samples, sample_rate: int) -> bytes:
    """Encode float32/int16 mono or (channels, n) audio as 16-bit PCM WAV.

    Accepts a numpy array or any sequence of floats in [-1, 1]. Kept here so
    every provider returns one consistent container and the browser never has
    to guess at a codec.
    """
    import numpy as np

    arr = np.asarray(samples)
    if arr.ndim == 2:                      # (channels, n) -> interleaved
        channels = arr.shape[0]
        arr = arr.T.reshape(-1)
    else:
        channels = 1
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 1
    return frames / rate


class Provider(abc.ABC):
    """One generation backend for one capability.

    Models are loaded lazily on first generate() and may be unloaded to free
    VRAM. Implementations must tolerate load() being called repeatedly.
    """

    id: str = "unnamed"
    name: str = "Unnamed provider"
    capability: Capability = Capability.VOICE
    # Weights licence, shown in the UI. Kept deliberately explicit: several of
    # these models ship permissive CODE with non-commercial WEIGHTS.
    license: str = "unknown"
    requires_gpu: bool = True
    description: str = ""

    def __init__(self) -> None:
        self._model = None
        self._last_used: float = 0.0

    # -- capability reporting -------------------------------------------------

    def available(self) -> bool:
        """True if this provider's dependencies and weights are usable."""
        return True

    def unavailable_reason(self) -> str:
        return ""

    def voices(self) -> list[VoiceInfo]:
        return []

    def params(self) -> list[ParamSpec]:
        return []

    # -- lifecycle ------------------------------------------------------------

    @abc.abstractmethod
    def _load(self) -> object:
        """Construct and return the underlying model. Called once, lazily."""

    def load(self) -> object:
        if self._model is None:
            self._model = self._load()
        self._last_used = time.monotonic()
        return self._model

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def idle_seconds(self) -> float:
        if self._model is None or not self._last_used:
            return 0.0
        return time.monotonic() - self._last_used

    # -- generation -----------------------------------------------------------

    @abc.abstractmethod
    def generate(self, req: GenerateRequest) -> AudioResult:
        """Synchronous, blocking generation. Runs on a worker thread."""

    def info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "capability": self.capability.value,
            "license": self.license,
            "requires_gpu": self.requires_gpu,
            "description": self.description,
            "available": self.available(),
            "unavailable_reason": self.unavailable_reason(),
            "loaded": self.loaded,
            "voices": [
                {"id": v.id, "name": v.name, "description": v.description, "cloned": v.cloned}
                for v in self.voices()
            ],
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "minimum": p.minimum,
                    "maximum": p.maximum,
                    "description": p.description,
                }
                for p in self.params()
            ],
        }
