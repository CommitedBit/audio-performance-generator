"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="Text to speak, or a description of the audio")
    provider: str | None = Field(None, description="Provider id; omitted means the server's default for this capability")
    voice_id: str | None = None
    # No global upper bound: providers advertise their own (ACE-Step 600 s,
    # Stable Audio 3 medium 380 s, ElevenLabs SFX 22 s) and the model service
    # validates against the resolved provider's `seconds` param. A single cap
    # here was wrong both ways -- it rejected lengths providers support and
    # accepted lengths they do not.
    seconds: float | None = Field(
        None, gt=0,
        description="Clip length; must fit the provider's advertised range. Ignored by providers that take no duration.",
    )
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class SpeechBody(GenerateBody):
    """OpenAI-ish shape so existing TTS clients mostly work unchanged.

    Either `prompt` or its OpenAI alias `input` must carry the text, so `prompt`
    is overridden as optional here -- inheriting it as required meant a plain
    {"input": ..., "voice": ...} request failed validation before the alias was
    ever consulted.
    """
    prompt: str | None = Field(None, max_length=5000, description="Text to speak")
    voice: str | None = Field(None, description="Alias for voice_id, for OpenAI compatibility")
    input: str | None = Field(None, max_length=5000, description="Alias for prompt, for OpenAI compatibility")

    def _first_text(self) -> str:
        """The first of `input` / `prompt` with real content.

        Whitespace does not count, so a blank `input` sent alongside a real
        `prompt` falls through to the prompt instead of shadowing it. The
        OpenAI alias wins only when both carry text. Validation and resolution
        both use this, so they cannot disagree.
        """
        for value in (self.input, self.prompt):
            if value and value.strip():
                return value
        return ""

    @model_validator(mode="after")
    def _require_text(self) -> SpeechBody:
        if not self._first_text():
            raise ValueError("either prompt or input must contain text")
        return self

    def resolved_prompt(self) -> str:
        return self._first_text()

    def resolved_voice(self) -> str | None:
        return self.voice or self.voice_id


class JobRef(BaseModel):
    id: str
    kind: str
    status: str
    audio_url: str | None = None


class VoiceRef(BaseModel):
    id: str
    name: str
    bytes: int
    created_at: float


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "degraded"]
    device: str
    providers_available: int
    providers_total: int
    gpu: dict[str, Any] | None = None
