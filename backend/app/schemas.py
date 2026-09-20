"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="Text to speak, or a description of the audio")
    provider: str | None = Field(None, description="Provider id; omitted means the server's default for this capability")
    voice_id: str | None = None
    seconds: float | None = Field(None, gt=0, le=300)
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class SpeechBody(GenerateBody):
    """OpenAI-ish shape so existing TTS clients mostly work unchanged."""
    voice: str | None = Field(None, description="Alias for voice_id, for OpenAI compatibility")
    input: str | None = Field(None, description="Alias for prompt, for OpenAI compatibility")

    def resolved_prompt(self) -> str:
        return self.input or self.prompt

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
