"""Builds the provider set at boot and routes requests to one of them.

A provider is *registered* if its class loads, and *available* if its
dependencies and credentials are actually present. Unavailable providers are
still listed by /v1/models, with a reason -- silently hiding them makes a
missing HF_TOKEN look like a bug in the UI.
"""
from __future__ import annotations

import logging

from .config import get_settings
from .providers.base import Capability, Provider

log = logging.getLogger(__name__)

# Preference among providers of the same capability, best first. The model
# service and the gateway both rank with this, so which provider is the default
# never depends on which service hosts it -- in the split topology the gateway
# used to take the first match in upstream order, which silently made Stable
# Audio the music default because ACE-Step lives on the second upstream.
PREFERENCE = ("chatterbox", "acestep", "stable-audio-3-sfx", "stable-audio-3-music", "musicgen")


def preference_rank(provider_id: str) -> tuple[int, int]:
    """Sort key: known local providers in PREFERENCE order, then any other local
    provider, then cloud fallbacks, then the placeholder stubs."""
    if provider_id.startswith("stub"):
        return (3, 0)
    if provider_id.startswith("elevenlabs"):
        return (2, 0)
    if provider_id in PREFERENCE:
        return (0, PREFERENCE.index(provider_id))
    return (1, 0)


def _build() -> list[Provider]:
    from .providers.stub import StubMusic, StubSfx, StubVoice

    providers: list[Provider] = []

    def try_add(factory, label: str) -> None:
        try:
            providers.append(factory())
        except Exception as exc:                       # noqa: BLE001
            # A provider that cannot even be constructed must not take the
            # whole API down with it.
            log.warning("provider %s could not be registered: %s", label, exc)

    from .providers.acestep import AceStepProvider
    from .providers.chatterbox import ChatterboxProvider
    from .providers.elevenlabs import ElevenLabsSfx, ElevenLabsVoice
    from .providers.musicgen import MusicGenProvider
    from .providers.stable_audio import StableAudio3Provider

    # Registration order is preference order: default_for() takes the first
    # available local provider, so the best pick per capability comes first.
    try_add(ChatterboxProvider, "chatterbox")
    try_add(AceStepProvider, "acestep")
    try_add(lambda: StableAudio3Provider(Capability.SFX), "stable-audio-3-sfx")
    try_add(lambda: StableAudio3Provider(Capability.MUSIC), "stable-audio-3-music")
    # Legacy, kept for comparison: CC-BY-NC weights and outclassed on quality.
    try_add(MusicGenProvider, "musicgen")
    try_add(ElevenLabsVoice, "elevenlabs-voice")
    try_add(ElevenLabsSfx, "elevenlabs-sfx")

    # Stubs last so a real provider always wins the default slot, but always
    # present so the API is usable before any weights exist.
    if get_settings().dev_stub or not any(p.available() for p in providers):
        providers.extend([StubVoice(), StubMusic(), StubSfx()])

    return providers


class Registry:
    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        for p in _build():
            self._providers[p.id] = p

        allow = get_settings().providers
        if allow != "auto":
            wanted = {x.strip() for x in allow.split(",") if x.strip()}
            self._providers = {k: v for k, v in self._providers.items() if k in wanted}

        for p in self._providers.values():
            log.info(
                "provider %-22s %-6s available=%s %s",
                p.id, p.capability.value, p.available(), p.unavailable_reason(),
            )

    def all(self) -> list[Provider]:
        return list(self._providers.values())

    def get(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def for_capability(self, capability: Capability) -> list[Provider]:
        return [p for p in self._providers.values() if p.capability is capability]

    def default_for(self, capability: Capability) -> Provider | None:
        """Prefer an available local provider; fall back to any available one."""
        candidates = [p for p in self.for_capability(capability) if p.available()]
        if not candidates:
            return None
        # sorted() is stable, so ties keep registration order.
        return sorted(candidates, key=lambda p: preference_rank(p.id))[0]

    def resolve(self, capability: Capability, provider_id: str | None) -> Provider:
        if provider_id:
            p = self.get(provider_id)
            if p is None:
                raise KeyError(f"unknown provider: {provider_id}")
            if p.capability is not capability:
                raise ValueError(f"provider {provider_id} does not serve {capability.value}")
            if not p.available():
                raise RuntimeError(f"provider {provider_id} unavailable: {p.unavailable_reason()}")
            return p

        p = self.default_for(capability)
        if p is None:
            raise RuntimeError(f"no available provider for {capability.value}")
        return p

    def sweep_idle(self, timeout: int) -> list[str]:
        """Unload models idle past `timeout` to return VRAM. Returns ids freed."""
        if timeout <= 0:
            return []
        freed = []
        for p in self._providers.values():
            idle = p.idle_seconds
            # Checked and unloaded atomically by the provider itself, and
            # skipped while a generation is still using the model.
            if p.unload_if_idle(timeout):
                log.info("unloaded %s after %.0fs idle", p.id, idle)
                freed.append(p.id)
        return freed


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
