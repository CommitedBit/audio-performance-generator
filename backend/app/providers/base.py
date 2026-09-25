"""Provider interface shared by local models and optional cloud backends."""
from __future__ import annotations

import abc
import contextlib
import enum
import functools
import io
import threading
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


class SilentOutputError(RuntimeError):
    """Raised when a model returns audio that is silent or degenerate.

    Several failure modes in this stack produce a clean exit code and unusable
    audio rather than an error: an amplitude-collapsed decode, a deprecated
    TensorRT engine noise-wash, a model loaded onto the wrong device. A caller
    polling a job would see status=done and a clip of nothing. Cheap to check,
    and it converts a confusing silent failure into a legible one.
    """


def check_audio_sane(data: bytes, *, rms_floor: float = 1e-4) -> None:
    """Reject silent or DC-constant output. Raises SilentOutputError."""
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    if not frames:
        raise SilentOutputError("model returned zero audio frames")

    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if arr.size == 0:
        raise SilentOutputError("model returned zero audio samples")

    rms = float(np.sqrt(np.mean(arr**2)))
    if rms < rms_floor:
        raise SilentOutputError(
            f"output is effectively silent (rms={rms:.2e} < {rms_floor:.0e}); "
            "check the model loaded on the right device and produced real audio"
        )
    # A constant signal has energy but no information -- catches DC offset and
    # a stuck decoder, which an RMS check alone would pass.
    if float(np.std(arr)) < rms_floor:
        raise SilentOutputError(f"output is a constant signal (std={float(np.std(arr)):.2e})")


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
    # True for providers that run elsewhere (a cloud API). They never touch the
    # local GPU, so they skip the cross-service GPU slot.
    remote: bool = False
    requires_gpu: bool = True
    description: str = ""

    def __init__(self) -> None:
        self._model = None
        self._last_used: float = 0.0
        # Guards the check-and-load in load(), and the in-use count below. Per
        # instance, so two different models can still load in parallel.
        self._load_lock = threading.Lock()
        # Generations currently running on this provider's model.
        self._active = 0
        # Last load failure. Import checks cannot see a bad checkpoint, a
        # rejected credential or incompatible pins -- only a load attempt can --
        # so a failed load is recorded and reported until a retry succeeds.
        self._load_error: str | None = None
        self._load_failed_at = 0.0

    # -- capability reporting -------------------------------------------------
    #
    # A recorded load failure takes precedence over a provider's own checks, in
    # BOTH methods. Without it, a provider whose imports resolved but whose load
    # failed kept reporting available: discovery advertised it, default routing
    # kept choosing it, and every request re-ran the same doomed multi-GB load.
    # __init_subclass__ applies this to every override, so no provider -- present
    # or future -- has to remember to.

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "available" in cls.__dict__:
            inner_available = cls.__dict__["available"]

            @functools.wraps(inner_available)
            def available(self, _inner=inner_available) -> bool:
                return self.load_failure() is None and _inner(self)

            cls.available = available
        if "unavailable_reason" in cls.__dict__:
            inner_reason = cls.__dict__["unavailable_reason"]

            @functools.wraps(inner_reason)
            def unavailable_reason(self, _inner=inner_reason) -> str:
                # Checked first: several providers derive their reason text from
                # available(), and would otherwise blame a missing dependency.
                return self.load_failure() or _inner(self)

            cls.unavailable_reason = unavailable_reason

    def load_failure(self) -> str | None:
        """The recorded load failure while its cooldown runs, else None.

        Once the cooldown (PROVIDER_RETRY_SECONDS) elapses the provider reports
        available again, so the next request retries the load: one expensive
        attempt per window rather than one per request. A successful load
        clears the record.
        """
        if self._load_error is None:
            return None
        from ..config import get_settings

        remaining = get_settings().provider_retry_seconds - (time.monotonic() - self._load_failed_at)
        if remaining <= 0:
            return None
        return f"failed to load: {self._load_error} (retrying in {remaining:.0f}s)"

    def available(self) -> bool:
        """True if this provider's dependencies and weights are usable."""
        return self.load_failure() is None

    def unavailable_reason(self) -> str:
        return self.load_failure() or ""

    def voices(self) -> list[VoiceInfo]:
        return []

    def params(self) -> list[ParamSpec]:
        return []

    # -- lifecycle ------------------------------------------------------------

    @abc.abstractmethod
    def _load(self) -> object:
        """Construct and return the underlying model. Called once, lazily."""

    def load(self) -> object:
        # Double-checked: an already-loaded model returns without touching the
        # lock. Without the lock, two jobs for the same unloaded provider (with
        # MAX_CONCURRENT_JOBS > 1) both saw None and both ran _load(), building
        # two multi-GB copies at once -- an avoidable VRAM OOM -- and caching
        # whichever finished last while the other job ran on an orphan.
        model = self._model
        if model is None:
            with self._load_lock:
                if self._model is None:
                    # Checked here, under the lock, not only in available():
                    # jobs admitted before a failure was recorded are already
                    # queued past discovery, and without this each of them
                    # re-ran the same doomed multi-GB load in turn.
                    failure = self.load_failure()
                    if failure:
                        raise RuntimeError(failure)
                    try:
                        self._model = self._load()
                    except Exception as exc:
                        self._load_error = f"{type(exc).__name__}: {exc}"[:400]
                        self._load_failed_at = time.monotonic()
                        raise
                    self._load_error = None
                model = self._model
        self._last_used = time.monotonic()
        return model

    @contextlib.contextmanager
    def in_use(self):
        """Mark this provider busy for the length of one generation.

        The idle sweeper skips a busy provider, so a model is never unloaded
        out from under a running job. Before this, idleness was measured from
        load() -- the START of a job -- with no notion of work in flight, so a
        generation outlasting MODEL_IDLE_TIMEOUT could be unloaded mid-run, and
        with MAX_CONCURRENT_JOBS > 1 the next request would then load a second
        multi-GB copy while the first job still held the original.
        """
        with self._load_lock:
            self._active += 1
        try:
            yield self
        finally:
            with self._load_lock:
                self._active -= 1
                # Idle time counts from when the work finished.
                self._last_used = time.monotonic()

    def unload_if_idle(self, timeout: float) -> bool:
        """Unload only if no generation is using the model and it has sat idle
        past `timeout`. The check and the unload happen under the same lock
        that in_use() and load() take, so a job cannot start in between."""
        with self._load_lock:
            if self._active or self._model is None or not self._last_used:
                return False
            if time.monotonic() - self._last_used <= timeout:
                return False
            self.unload()
            return True

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
