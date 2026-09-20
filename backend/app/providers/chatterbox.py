"""Chatterbox TTS (Resemble AI) -- the local 'voice' backend.

Zero-shot voice cloning: pass a reference sample and it mimics the speaker,
no fine-tuning. `exaggeration` controls emotional intensity, which is the
feature that makes it interesting versus a flat local TTS.
"""
from __future__ import annotations

import importlib.util
import logging

from .. import storage
from ..config import get_settings
from .base import (
    AudioResult,
    Capability,
    GenerateRequest,
    ParamSpec,
    Provider,
    VoiceInfo,
    pcm_to_wav,
)

log = logging.getLogger(__name__)


class ChatterboxProvider(Provider):
    id = "chatterbox"
    name = "Chatterbox (Resemble AI)"
    capability = Capability.VOICE
    # MIT for BOTH code and weights -- confirmed against the repo LICENSE, the
    # HF cardData license field and the PyPI package metadata. No separate
    # weights terms and no gated download, which is rare in this tier: every
    # clearly better-sounding expressive cloning model ships non-commercial
    # weights. That is why this is the default despite not topping the arena.
    license = "MIT (code and weights)"
    requires_gpu = False          # runs on CPU, just slowly
    description = "Expressive English TTS with zero-shot voice cloning from a short reference sample."

    def available(self) -> bool:
        return importlib.util.find_spec("chatterbox") is not None

    def unavailable_reason(self) -> str:
        if self.available():
            return ""
        return "chatterbox-tts is not installed in this image (pip install chatterbox-tts)"

    def _load(self):
        from chatterbox.tts import ChatterboxTTS

        device = get_settings().device
        log.info("loading Chatterbox on %s (first run downloads weights)", device)
        return ChatterboxTTS.from_pretrained(device=device)

    def voices(self) -> list[VoiceInfo]:
        # The built-in voice plus every uploaded clone reference.
        out = [VoiceInfo(id="default", name="Chatterbox default", description="Built-in voice")]
        for rec in storage.list_voice_references():
            out.append(
                VoiceInfo(id=rec["id"], name=rec.get("name", rec["id"]), description="Cloned", cloned=True)
            )
        return out

    def params(self) -> list[ParamSpec]:
        return [
            ParamSpec("exaggeration", "float", 0.5, 0.0, 2.0, "Emotional intensity; above ~1.0 gets unstable"),
            ParamSpec("cfg_weight", "float", 0.5, 0.0, 1.0, "Prompt adherence; lower is looser and often more natural"),
            ParamSpec("temperature", "float", 0.8, 0.05, 2.0, "Sampling randomness"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        model = self.load()

        kwargs: dict = {}
        if req.voice_id and req.voice_id != "default":
            ref = storage.voice_reference_path(req.voice_id)
            if ref is None:
                raise ValueError(f"unknown voice reference: {req.voice_id}")
            kwargs["audio_prompt_path"] = str(ref)

        for key in ("exaggeration", "cfg_weight", "temperature"):
            if key in req.params and req.params[key] is not None:
                kwargs[key] = float(req.params[key])

        text = req.prompt.strip()
        if not text:
            raise ValueError("prompt is empty")

        wav = model.generate(text, **kwargs)
        sr = int(getattr(model, "sr", 24000))

        # Chatterbox returns a torch tensor shaped (1, n).
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else wav
        if getattr(arr, "ndim", 1) == 2 and arr.shape[0] == 1:
            arr = arr[0]

        audio = pcm_to_wav(arr, sr)
        return AudioResult(
            audio=audio,
            sample_rate=sr,
            duration=len(arr) / sr,
            provider_id=self.id,
        )
