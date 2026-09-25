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

import importlib.metadata
import importlib.util
import logging
import os
import re
import threading

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)

# Peak VRAM at max length, from Stability's published benchmark table.
PEAK_VRAM_GB = {"medium": 6.52, "small": 2.40}

# Two loaders, tried in order:
#
#   1. The official `stable_audio_3` package (StableAudioModel). Preferred, and
#      the one that works in the UNIFIED image: its inference path imports only
#      AutoConfig / AutoTokenizer / T5GemmaEncoderModel from transformers and
#      hf_hub_download / try_to_load_from_cache from huggingface-hub -- all
#      present in transformers 4.57.6 and huggingface-hub 0.36.2, checked
#      against the pinned sources. (Its declared >=5.8 / >=1.7.1 are looser
#      than what it actually calls.)
#   2. diffusers' StableAudio3Pipeline, a fallback. It first ships in diffusers
#      0.40.0, which requires huggingface-hub >=1.23 -- so it can never share
#      an image with ACE-Step, whose transformers <4.58 needs hub <1.0.
SA3_DIFFUSERS_MIN = (0, 40, 0)
HF_REPO_PREFIX = "stabilityai/stable-audio-3-"


def _official_available() -> bool:
    return importlib.util.find_spec("stable_audio_3") is not None


def _diffusers_problem() -> str:
    """Why the diffusers fallback cannot run here, or "" if it can.

    Checked from package metadata rather than by importing diffusers, which
    pulls in torch and is too slow for a discovery path called per request.
    """
    if importlib.util.find_spec("diffusers") is None:
        return "diffusers is not installed"
    try:
        version = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "diffusers is importable but has no package metadata"
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version)
    have = tuple(int(x or 0) for x in m.groups()) if m else (0, 0, 0)
    if have < SA3_DIFFUSERS_MIN:
        return f"diffusers {version} has no StableAudio3Pipeline (first in 0.40.0)"
    return ""


def _loader_problem() -> str:
    """Why no Stable Audio 3 loader can run here, or "" if one can."""
    if _official_available():
        return ""
    problem = _diffusers_problem()
    if not problem:
        return ""
    return f"no usable loader: the stable-audio-3 package is not installed and {problem}"


class _OfficialRunner:
    """StableAudioModel behind the call the provider makes."""

    kind = "stable-audio-3"

    def __init__(self, model) -> None:
        self.model = model
        self.sample_rate = int(model.model.sample_rate)

    def run(self, prompt, negative_prompt, seconds, steps, cfg, seed):
        # Returns [batch, channels, samples]. cfg_scale / negative_prompt only
        # take effect on the -base checkpoints; the post-trained ones ignore them.
        audio = self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=seconds,
            steps=steps,
            cfg_scale=cfg,
            batch_size=1,
            seed=-1 if seed is None else int(seed),
        )
        return audio[0], self.sample_rate


class _DiffusersRunner:
    """diffusers' StableAudio3Pipeline behind the same call."""

    kind = "diffusers"

    def __init__(self, pipe, device: str) -> None:
        self.pipe = pipe
        self.device = device
        self.sample_rate = int(getattr(getattr(pipe, "vae", None), "sampling_rate", 44100))

    def run(self, prompt, negative_prompt, seconds, steps, cfg, seed):
        import torch

        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            audio_end_in_s=seconds,
            generator=generator,
        )
        return result.audios[0], self.sample_rate


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
# A diffusers pipeline holds mutable scheduler state and must not run two calls
# at once. Each shared pipeline therefore carries its own call lock, so jobs on
# the SAME weights serialise even with MAX_CONCURRENT_JOBS raised, while jobs on
# different weights (the two small checkpoints) still run in parallel.
_SHARED: dict[tuple[str, str, str], object] = {}
_HOLDERS: dict[tuple[str, str, str], set[str]] = {}
_CALL_LOCKS: dict[tuple[str, str, str], threading.Lock] = {}
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
        # Replaced on load by the lock of the pipeline this instance holds.
        # Never cleared on unload, so a generate() racing an idle unload can
        # never find it missing; the next load always overwrites it first.
        self._call_lock = threading.Lock()
        # SA3_MODEL may be the HF repo id or the official short name
        # ("medium", "small-sfx"); each loader gets the form it expects.
        short = self.model_id.rsplit("/", 1)[-1]
        self.official_name = short[len("stable-audio-3-"):] if short.startswith("stable-audio-3-") else short
        self.hf_repo = self.model_id if "/" in self.model_id else HF_REPO_PREFIX + self.official_name
        self.tier = "medium" if "medium" in self.model_id else "small"
        self.name = f"Stable Audio 3 {self.tier} ({capability.value})"
        self.max_seconds = 380.0 if self.tier == "medium" else 120.0

    def _deps_present(self) -> bool:
        # Only a loader that can actually run counts: the official package, or
        # a diffusers new enough to have StableAudio3Pipeline.
        return _loader_problem() == ""

    def available(self) -> bool:
        # The checkpoints are gated, so without a token every load fails on the
        # download. Reporting available anyway made this the sfx default and
        # broke every sfx request; the token is part of "can this run".
        return self._deps_present() and bool(os.getenv("HF_TOKEN"))

    def unavailable_reason(self) -> str:
        problem = _loader_problem()
        if problem:
            return problem
        if not os.getenv("HF_TOKEN"):
            return "HF_TOKEN is not set (stable-audio-3 weights are gated)"
        return ""

    def _load(self):
        import torch

        device = get_settings().device
        dtype = torch.float16 if device == "cuda" else torch.float32
        loader = "stable-audio-3" if _official_available() else "diffusers"
        key = (f"{loader}:{self.hf_repo}", device, str(dtype))

        # Held for the whole build so two providers asking for the same weights
        # at once cannot both construct a copy.
        with _SHARED_LOCK:
            pipe = _SHARED.get(key)
            if pipe is not None:
                log.info("%s reusing the already-loaded %s pipeline", self.id, self.model_id)
            else:
                pipe = self._build_pipeline(device, dtype, loader)
                _SHARED[key] = pipe
                _CALL_LOCKS[key] = threading.Lock()
            _HOLDERS.setdefault(key, set()).add(self.id)
            self._call_lock = _CALL_LOCKS[key]

        self._pipeline_key = key
        return pipe

    def _build_pipeline(self, device: str, dtype, loader: str):
        if not _flash_attn_present():
            # Informational only: the SDPA fallback is math-equivalent, just slower.
            log.info("flash-attn absent; using the SDPA attention fallback (slower, same output)")
        log.info("loading %s via %s on %s (peak ~%.1f GB)", self.hf_repo, loader, device, PEAK_VRAM_GB[self.tier])

        if loader == "stable-audio-3":
            from stable_audio_3 import StableAudioModel

            # Downloads through huggingface-hub, which reads HF_TOKEN and HF_HOME
            # from the environment -- so the gated weights land on the models
            # volume. Half precision only on CUDA.
            model = StableAudioModel.from_pretrained(self.official_name, device=device, model_half=device == "cuda")
            return _OfficialRunner(model)

        from diffusers import StableAudio3Pipeline

        token = os.getenv("HF_TOKEN") or None
        pipe = StableAudio3Pipeline.from_pretrained(self.hf_repo, torch_dtype=dtype, token=token)
        return _DiffusersRunner(pipe.to(device), device)

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
                    _CALL_LOCKS.pop(key, None)
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
        pipe = self.load()

        prompt = req.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        default_len = 6.0 if self.capability is Capability.SFX else 30.0
        seconds = float(req.seconds or req.params.get("seconds") or default_len)
        seconds = max(1.0, min(seconds, self.max_seconds))

        # Held only around the model call: that is the part that mutates
        # shared scheduler state. Post-processing below works on the result.
        with self._call_lock:
            audio, sr = pipe.run(
                prompt,
                req.params.get("negative_prompt") or None,
                seconds,
                int(req.params.get("steps", 8)),
                float(req.params.get("guidance_scale", 7.0)),
                req.seed,
            )

        arr = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else audio
        n = arr.shape[-1]
        return AudioResult(audio=pcm_to_wav(arr, sr), sample_rate=sr, duration=n / sr, provider_id=self.id)


def _flash_attn_present() -> bool:
    return importlib.util.find_spec("flash_attn") is not None
