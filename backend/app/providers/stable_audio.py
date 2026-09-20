"""Stable Audio Open (Stability AI) via diffusers.

Registered for BOTH music and sfx: it handles short one-shot effects and
ambience well, so a VRAM-constrained box can run one checkpoint instead of two.
Instantiate it twice with different capabilities to serve both tracks.

Note: the HuggingFace repo is gated -- accept the licence on the model page and
supply HF_TOKEN, or the first load fails with a 401.
"""
from __future__ import annotations

import importlib.util
import logging
import os

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)


class StableAudioProvider(Provider):
    license = "Stability AI Community License -- free under a revenue threshold; read it before commercial use"
    requires_gpu = False
    description = "Text-to-audio for sound effects, ambience and short musical beds. 44.1kHz stereo."

    def __init__(self, capability: Capability = Capability.SFX, model_id: str | None = None) -> None:
        super().__init__()
        self.capability = capability
        self.id = f"stable-audio-{capability.value}"
        self.name = f"Stable Audio Open ({capability.value})"
        self.model_id = model_id or os.getenv("STABLE_AUDIO_MODEL", "stabilityai/stable-audio-open-1.0")

    def available(self) -> bool:
        return importlib.util.find_spec("diffusers") is not None

    def unavailable_reason(self) -> str:
        return "" if self.available() else "diffusers is not installed in this image"

    def _load(self):
        import torch
        from diffusers import StableAudioPipeline

        device = get_settings().device
        log.info("loading %s on %s (gated repo: needs HF_TOKEN)", self.model_id, device)
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableAudioPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        return pipe.to(device)

    def params(self) -> list[ParamSpec]:
        default_len = 4.0 if self.capability is Capability.SFX else 20.0
        return [
            ParamSpec("seconds", "float", default_len, 1.0, 47.0, "Clip length; the model tops out around 47s"),
            ParamSpec("steps", "int", 100, 10, 250, "Denoising steps; more is slower and slightly cleaner"),
            ParamSpec("guidance_scale", "float", 7.0, 1.0, 15.0, "Prompt adherence"),
            ParamSpec("negative_prompt", "str", "Low quality.", None, None, "What to steer away from"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        import torch

        pipe = self.load()
        device = get_settings().device

        prompt = req.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        seconds = float(req.seconds or req.params.get("seconds") or (4.0 if self.capability is Capability.SFX else 20.0))
        seconds = max(1.0, min(seconds, 47.0))

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(req.seed))

        result = pipe(
            prompt,
            negative_prompt=req.params.get("negative_prompt", "Low quality."),
            num_inference_steps=int(req.params.get("steps", 100)),
            guidance_scale=float(req.params.get("guidance_scale", 7.0)),
            audio_end_in_s=seconds,
            num_waveforms_per_prompt=1,
            generator=generator,
        )

        arr = result.audios[0].to(torch.float32).cpu().numpy()
        sr = int(pipe.vae.sampling_rate)
        n = arr.shape[-1]
        audio = pcm_to_wav(arr, sr)
        return AudioResult(audio=audio, sample_rate=sr, duration=n / sr, provider_id=self.id)
