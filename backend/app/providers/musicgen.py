"""MusicGen (Meta) via transformers -- local 'music' backend.

Deliberately uses transformers rather than the audiocraft package: audiocraft
pins old torch/numpy versions and fights every modern CUDA image. transformers
ships MusicGen natively and installs cleanly.

LICENCE WARNING, surfaced in the UI: the audiocraft CODE is MIT but the
MusicGen WEIGHTS have shipped under CC-BY-NC 4.0 (non-commercial).
"""
from __future__ import annotations

import importlib.util
import logging
import os

from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)

# MusicGen's audio tokeniser runs at 50 tokens per second of audio.
TOKENS_PER_SECOND = 50


class MusicGenProvider(Provider):
    id = "musicgen"
    name = "MusicGen (Meta)"
    capability = Capability.MUSIC
    license = "weights CC-BY-NC 4.0 (non-commercial) / code MIT -- verify on the model card"
    requires_gpu = False
    description = "Legacy. Superseded by ACE-Step (MIT) and Stable Audio 3 on both quality and licence; kept for comparison."

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        self.model_id = model_id or os.getenv("MUSICGEN_MODEL", "facebook/musicgen-medium")
        self._processor = None

    def available(self) -> bool:
        return importlib.util.find_spec("transformers") is not None

    def unavailable_reason(self) -> str:
        return "" if self.available() else "transformers is not installed in this image"

    def _load(self):
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        device = get_device()
        log.info("loading %s on %s (first run downloads weights)", self.model_id, device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = MusicgenForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype)
        return model.to(device)

    def unload(self) -> None:
        self._processor = None
        super().unload()

    def params(self) -> list[ParamSpec]:
        return [
            ParamSpec("seconds", "float", 10.0, 1.0, 120.0, "Clip length; cost scales linearly"),
            ParamSpec("guidance_scale", "float", 3.0, 1.0, 10.0, "Prompt adherence"),
            ParamSpec("temperature", "float", 1.0, 0.1, 2.0, "Sampling randomness"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        import torch

        model = self.load()
        processor = self._processor
        device = get_device()

        prompt = req.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        seconds = float(req.seconds or req.params.get("seconds") or 10.0)
        seconds = max(1.0, min(seconds, 120.0))
        max_new_tokens = int(seconds * TOKENS_PER_SECOND)

        if req.seed is not None:
            torch.manual_seed(int(req.seed))

        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                guidance_scale=float(req.params.get("guidance_scale", 3.0)),
                temperature=float(req.params.get("temperature", 1.0)),
            )

        sr = int(model.config.audio_encoder.sampling_rate)
        arr = tokens[0].detach().to(torch.float32).cpu().numpy()
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]

        audio = pcm_to_wav(arr, sr)
        return AudioResult(audio=audio, sample_rate=sr, duration=len(arr) / sr, provider_id=self.id)


def get_device() -> str:
    from ..config import get_settings
    return get_settings().device
