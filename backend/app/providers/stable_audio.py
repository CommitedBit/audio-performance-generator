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

FLASH ATTENTION IS OPTIONAL, despite what the README says. Verified against
stable_audio_3/models/transformer.py: the flash_attn import sits in a
try/except that prints "flash_attn not installed, disabling Flash Attention"
and sets the functions to None -- there is no raise. apply_attn then falls
through a four-tier cascade its own comments describe as math-equivalent:

    flex_attention with a band block mask
      -> chunked-halo masked SDPA  ("math-equivalent, ~30x faster than tier 4")
      -> full masked SDPA          (last resort, high memory)

The "output collapses to static" failure that the README's flash-attn warning
refers to is the bug PR #21 fixed in May 2026; only the documentation is stale.
So this provider does NOT require flash-attn, which matters on sm_120 where a
prebuilt wheel would pin the whole image to one torch build.

The weights are GATED on HuggingFace: accept the terms on the model page and
supply HF_TOKEN at RUNTIME. Never bake a token into the image.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)

# Peak VRAM at max length, from Stability's published benchmark table.
PEAK_VRAM_GB = {"medium": 6.52, "small": 2.40}

# One pipeline per (model id, device, dtype), shared by every provider instance
# that asks for it. With SA3_MODEL pinned to the medium checkpoint, the sfx and
# music providers point at the SAME weights; without sharing, using both would
# leave two copies resident and the second load could run the card out of
# VRAM -- the opposite of the one-model configuration SA3_MODEL exists for.
#
# Reference-counted by holder, because each provider unloads on its own idle
# timer: sfx going idle must not free a pipeline music is still using. The
# pipeline is dropped only when its last holder releases it. The small
# checkpoints have distinct ids per capability, so they never share.
#
# Sharing is safe because generation is serialised (MAX_CONCURRENT_JOBS=1): a
# diffusers pipeline holds scheduler state and must not run two calls at once.
# Raising that limit with a shared medium checkpoint would need a per-pipeline
# lock.
_SHARED: dict[tuple[str, str, str], object] = {}
_HOLDERS: dict[tuple[str, str, str], set[str]] = {}
_SHARED_LOCK = threading.Lock()


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

        # Key of the shared pipeline this instance holds while loaded.
        self._pipeline_key: tuple[str, str, str] | None = None
        self.tier = "medium" if "medium" in self.model_id else "small"
        self.name = f"Stable Audio 3 {self.tier} ({capability.value})"
        self.max_seconds = 380.0 if self.tier == "medium" else 120.0

    def _deps_present(self) -> bool:
        # The official `stable_audio_3` package is preferred; diffusers documents
        # the medium checkpoints and works as a fallback.
        return (
            importlib.util.find_spec("stable_audio_3") is not None
            or importlib.util.find_spec("diffusers") is not None
        )

    def available(self) -> bool:
        # The checkpoints are gated, so without a token every load fails on the
        # download. Reporting available anyway made this the sfx default and
        # broke every sfx request; the token is part of "can this run".
        return self._deps_present() and bool(os.getenv("HF_TOKEN"))

    def unavailable_reason(self) -> str:
        if not self._deps_present():
            return "neither stable-audio-3 nor diffusers is installed in this image"
        if not os.getenv("HF_TOKEN"):
            return "HF_TOKEN is not set (stable-audio-3 weights are gated)"
        return ""

    def _load(self):
        import torch

        device = get_settings().device
        dtype = torch.float16 if device == "cuda" else torch.float32
        key = (self.model_id, device, str(dtype))

        # Held for the whole build so two providers asking for the same weights
        # at once cannot both construct a copy.
        with _SHARED_LOCK:
            pipe = _SHARED.get(key)
            if pipe is not None:
                log.info("%s reusing the already-loaded %s pipeline", self.id, self.model_id)
            else:
                pipe = self._build_pipeline(device, dtype)
                _SHARED[key] = pipe
            _HOLDERS.setdefault(key, set()).add(self.id)

        self._pipeline_key = key
        return pipe

    def _build_pipeline(self, device: str, dtype):
        token = os.getenv("HF_TOKEN") or None

        if not _flash_attn_present():
            # Informational only: the SDPA fallback is math-equivalent, just slower.
            log.info("flash-attn absent; using the SDPA attention fallback (slower, same output)")

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

    def unload(self) -> None:
        if self._model is None:
            return
        key = getattr(self, "_pipeline_key", None)
        with _SHARED_LOCK:
            holders = _HOLDERS.get(key)
            if holders is not None:
                holders.discard(self.id)
                if not holders:
                    # Last holder out: drop the shared reference so it can be freed.
                    _HOLDERS.pop(key, None)
                    _SHARED.pop(key, None)
                else:
                    log.info("%s released %s; still held by %s", self.id, self.model_id, sorted(holders))
        self._pipeline_key = None
        # Drops this provider's reference and empties the CUDA cache. Harmless
        # while another holder keeps the pipeline alive: empty_cache only
        # returns blocks nothing is using.
        super().unload()

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
