"""Runtime configuration, all overridable by environment variable."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_device(requested: str) -> str:
    """Resolve DEVICE=auto to the best backend actually present.

    Kept dependency-free on purpose: importing torch here would make config
    import cost seconds and would crash the API on a machine without torch.
    """
    if requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings:
    def __init__(self) -> None:
        self.device: str = _resolve_device(os.getenv("DEVICE", "auto"))

        root = Path(os.getenv("DATA_DIR", "/data")).resolve()
        self.data_dir = root
        self.audio_dir = root / "audio"
        self.voices_dir = root / "voices"
        for d in (self.audio_dir, self.voices_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Which providers to construct at boot. "auto" enables every provider
        # whose dependencies import cleanly.
        self.providers: str = os.getenv("PROVIDERS", "auto")

        # Optional cloud fallback. Absent key simply means the provider is not
        # advertised by /v1/models.
        self.elevenlabs_api_key: str | None = os.getenv("ELEVENLABS_API_KEY") or None

        # A single GPU cannot safely run two diffusion/TTS models at once, so
        # generation is serialised. Raise only if you know your VRAM allows it.
        self.max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

        # Unload a model after this many seconds idle to free VRAM. 0 disables.
        self.model_idle_timeout: int = int(os.getenv("MODEL_IDLE_TIMEOUT", "600"))

        # After a provider fails to load, report it unavailable for this long
        # before the next request is allowed to retry the load.
        self.provider_retry_seconds: int = int(os.getenv("PROVIDER_RETRY_SECONDS", "300"))

        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")
            if o.strip()
        ]

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.dev_stub: bool = _bool("DEV_STUB", False)

    @property
    def on_gpu(self) -> bool:
        return self.device in {"cuda", "mps"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
