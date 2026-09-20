"""Stable Audio 3 (Stability AI) -- local 'sfx' and 'music' backend.

Supersedes Stable Audio Open 1.0, which is slower and materially worse on
Stability's own SFX benchmark. 44.1 kHz stereo, so it mixes against Chatterbox
voice without a resample step.

Checkpoint choice is a VRAM decision, not a taste one. Stability split the
small tier because at 459M "the inclusion of sound effects data degrades
musical coherence", while medium and large "handle both music and sound effect
generation within a single unified model":

    >=12 GB   ONE stable-audio-3-medium serves both music and sfx
              (~6.5 GB peak at 120s, 380s max length)
    8 GB      TWO specialists: small-sfx + small-music
              (~2.4 GB peak each, 120s max)

TWO OPERATIONAL TRAPS, both worth knowing before the first run:

1. FLASH ATTENTION 2 IS MANDATORY for the medium checkpoints, and its absence
   fails SILENTLY -- output collapses to static rather than raising. Requires
   compute capability >= 8.0 (A100/RTX 3090/RTX 4090 class). Turing (sm_75) and
   older cannot run medium at all. The small checkpoints do not list this
   requirement, which is a good reason to prefer them on older cards.

2. The weights are GATED on HuggingFace. Accept the terms on the model page and
   supply HF_TOKEN at RUNTIME -- do not bake a token into the image.
"""
from __future__ import annotations

import importlib.util
import logging
import os

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)

# Peak VRAM at max length, from Stability's published benchmark table.
PEAK_VRAM_GB = {"medium": 6.52, "small": 2.40}


class StableAudio3Provider(Provider):
    license = "Stability AI Community License - free below a $1M annual revenue threshold"
    requires_gpu = False          # the small checkpoints run on CPU
    description = "Text-to-audio for sound effects, ambience and instrumental beds. 44.1 kHz stereo."

    def __init__(self, capability: Capability = Capability.SFX, model_id: str | None = None) -> None:
        super().__init__()
        self.capability = capability
        self.id = f"stable-audio-3-{capability.value}"

        if model_id:
            self.model_id = model_id
        else:
            # SA3_MODEL pins one checkpoint for both tracks -- set it to
            # stabilityai/stable-audio-3-medium on a 12 GB+ card so a single
            # model serves music and sfx instead of loading two.
            shared = os.getenv("SA3_MODEL")
            default = f"stabilityai/stable-audio-3-small-{capability.value}"
            self.model_id = shared or default

        self.tier = "medium" if "medium" in self.model_id else "small"
        self.name = f"Stable Audio 3 {self.tier} ({capability.value})"
        self.max_seconds = 380.0 if self.tier == "medium" else 120.0

    def available(self) -> bool:
        # The official `stable_audio_3` package is preferred; diffusers documents
        # the medium checkpoints and works as a fallback.
        return (
            importlib.util.find_spec("stable_audio_3") is not None
            or importlib.util.find_spec("diffusers") is not None
        )

    def unavailable_reason(self) -> str:
        if not self.available():
            return "neither stable-audio-3 nor diffusers is installed in this image"
        if self.tier == "medium" and not _flash_attn_present():
            # Surfaced rather than left to corrupt audio silently.
            return "flash-attn is required for the medium checkpoint; without it output is static"
        if not os.getenv("HF_TOKEN"):
            return "HF_TOKEN is not set (stable-audio-3 weights are gated)"
        return ""

    def _load(self):
        import torch

        device = get_settings().device
        dtype = torch.float16 if device == "cuda" else torch.float32
        token = os.getenv("HF_TOKEN") or None

        if self.tier == "medium" and not _flash_attn_present():
            raise RuntimeError(
                "flash-attn is not installed and the Stable Audio 3 medium checkpoint "
                "silently produces static without it; install flash-attn or switch "
                "SA3_MODEL to a small checkpoint"
            )

        log.info("loading %s on %s (peak ~%.1f GB)", self.model_id, device, PEAK_VRAM_GB[self.tier])

        try:
            from diffusers import StableAudio3Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers does not expose StableAudio3Pipeline in this version; "
                "upgrade diffusers or install the official stable-audio-3 package"
            ) from exc

        pipe = StableAudio3Pipeline.from_pretrained(self.model_id, torch_dtype=dtype, token=token)
        return pipe.to(device)

    def params(self) -> list[ParamSpec]:
        default_len = 6.0 if self.capability is Capability.SFX else 30.0
        return [
            ParamSpec("seconds", "float", default_len, 1.0, self.max_seconds, "Clip length"),
            ParamSpec("steps", "int", 8, 4, 100, "Denoising steps; 8 is the tuned default"),
            ParamSpec("guidance_scale", "float", 7.0, 1.0, 15.0, "Prompt adherence"),
            ParamSpec("negative_prompt", "str", "", None, None, "What to steer away from"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        import torch

        pipe = self.load()
        device = get_settings().device

        prompt = req.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        default_len = 6.0 if self.capability is Capability.SFX else 30.0
        seconds = float(req.seconds or req.params.get("seconds") or default_len)
        seconds = max(1.0, min(seconds, self.max_seconds))

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(req.seed))

        result = pipe(
            prompt,
            negative_prompt=req.params.get("negative_prompt") or None,
            num_inference_steps=int(req.params.get("steps", 8)),
            guidance_scale=float(req.params.get("guidance_scale", 7.0)),
            audio_end_in_s=seconds,
            generator=generator,
        )

        audio = result.audios[0]
        arr = audio.to(torch.float32).cpu().numpy() if hasattr(audio, "to") else audio
        sr = int(getattr(getattr(pipe, "vae", None), "sampling_rate", 44100))
        n = arr.shape[-1]
        return AudioResult(audio=pcm_to_wav(arr, sr), sample_rate=sr, duration=n / sr, provider_id=self.id)


def _flash_attn_present() -> bool:
    return importlib.util.find_spec("flash_attn") is not None
