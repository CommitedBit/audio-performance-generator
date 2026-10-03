"""ACE-Step 1.5 -- local 'music' backend, MIT weights.

Why this is the music default over MusicGen: MIT weights (no CC-BY-NC rider),
10s-600s range, and explicit BPM/key control. It is also the fastest open music
model by a wide margin (~24x realtime on an RTX 3090).

Written against ACE-Step 1.5's actual API at the pinned commit (ca1e85fe9430):
`AceStepHandler.initialize_service` loads the DiT, `LLMHandler.initialize`
loads the optional planning LM, and `acestep.inference.generate_music` runs a
`GenerationParams` / `GenerationConfig` pair. (An earlier version of this file
called an `ACEStepPipeline` class from the 1.0 line; it does not exist in 1.5.)

Model selection is the DiT variant name ACE-Step calls `config_path`:
    acestep-v15-turbo      fastest; capped at 8 steps by ACE-Step itself
    acestep-v15-base       default here
    acestep-v15-sft
    acestep-v15-xl-turbo / -xl-base / -xl-sft   larger, higher quality
ACESTEP_MODEL takes the bare name or the "ACE-Step/<name>" form.

The planning LM ("thinking" / chain-of-thought metadata) is optional:
    ACESTEP_LM_MODEL   acestep-5Hz-lm-0.6B (default) | -1.7B | -4B | none
    ACESTEP_BACKEND    pt (default) | vllm
`pt` runs the LM on plain PyTorch. The vllm backend needs nano-vllm, which is
not installed and is reported broken on sm_120.

Checkpoints download under <project root>/checkpoints. The project root
defaults to $HF_HOME/acestep -- the persistent models volume -- rather than
ACE-Step's own default of the process working directory, which lives inside the
container and would re-download ~10 GB every time it is recreated.
ACESTEP_PROJECT_ROOT overrides it.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "acestep-v15-base"
DEFAULT_LM = "acestep-5Hz-lm-0.6B"
_LM_OFF = {"none", "off", "0", "false", "no"}


@dataclass
class _Loaded:
    dit: object
    llm: object
    lm_enabled: bool


class AceStepProvider(Provider):
    id = "acestep"
    name = "ACE-Step 1.5"
    capability = Capability.MUSIC
    license = "Apache-2.0 code, MIT weights - commercial use permitted"
    requires_gpu = True
    description = "Text-to-music with lyrics support and BPM/key control. 10s-600s. Song-oriented."

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        # `or`, not a getenv default: compose passes ${ACESTEP_MODEL:-}, which
        # sets the variable to "" when .env leaves it blank (as .env.example
        # does). A getenv default only applies when the variable is UNSET, so
        # it would have produced an empty model id for every default deploy.
        self.model_id = model_id or os.getenv("ACESTEP_MODEL") or DEFAULT_CONFIG
        # ACE-Step takes the bare variant name; accept the HF-style form too.
        self.config_path = self.model_id.rsplit("/", 1)[-1]
        self.backend = os.getenv("ACESTEP_BACKEND") or "pt"
        lm = (os.getenv("ACESTEP_LM_MODEL") or DEFAULT_LM).strip()
        self.lm_model: str | None = None if lm.lower() in _LM_OFF else lm
        # ACE-Step clamps turbo variants to 8 steps; others default to 30.
        self.is_turbo = "turbo" in self.config_path.lower()

    def available(self) -> bool:
        return importlib.util.find_spec("acestep") is not None

    def unavailable_reason(self) -> str:
        return "" if self.available() else "acestep is not installed in this image"

    def _project_root(self) -> Path:
        explicit = os.getenv("ACESTEP_PROJECT_ROOT")
        if explicit:
            return Path(explicit)
        return Path(os.getenv("HF_HOME") or "/models") / "acestep"

    def _load(self) -> _Loaded:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        device = get_settings().device
        root = self._project_root()
        root.mkdir(parents=True, exist_ok=True)
        log.info("loading ACE-Step %s on %s (lm=%s, backend=%s, root=%s)",
                 self.config_path, device, self.lm_model or "off", self.backend, root)

        dit = AceStepHandler()
        status, ok = dit.initialize_service(
            project_root=str(root),
            config_path=self.config_path,
            device=device,
            use_flash_attention=device == "cuda" and importlib.util.find_spec("flash_attn") is not None,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
            # Explicit, so the download path does not hinge on ACE-Step's
            # auto-detect, which probes Google and otherwise tries modelscope
            # first -- a package this image does not install.
            prefer_source="huggingface",
        )
        if not ok:
            raise RuntimeError(f"ACE-Step could not load {self.config_path}: {status}")

        llm = LLMHandler()
        lm_enabled = False
        if self.lm_model:
            status, ok = llm.initialize(
                checkpoint_dir=str(root / "checkpoints"),
                lm_model_path=self.lm_model,
                backend=self.backend,
                device=device,
                offload_to_cpu=False,
                dtype=None,
            )
            if not ok:
                raise RuntimeError(
                    f"ACE-Step LM {self.lm_model} (backend={self.backend}) failed to load: {status}; "
                    "set ACESTEP_LM_MODEL=none to run the DiT without it"
                )
            lm_enabled = True
        return _Loaded(dit=dit, llm=llm, lm_enabled=lm_enabled)

    def params(self) -> list[ParamSpec]:
        steps_default, steps_max = (8, 8) if self.is_turbo else (30, 200)
        return [
            ParamSpec("seconds", "float", 30.0, 10.0, 600.0, "Clip length"),
            ParamSpec("lyrics", "str", "", None, None, "Optional lyrics; blank gives an instrumental"),
            ParamSpec("inference_steps", "int", steps_default, 1, steps_max, "More steps is slower and slightly cleaner"),
            ParamSpec("guidance_scale", "float", 7.0, 1.0, 15.0, "Prompt adherence"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        import numpy as np
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        loaded: _Loaded = self.load()

        prompt = req.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        seconds = float(req.seconds or req.params.get("seconds") or 30.0)
        seconds = max(10.0, min(seconds, 600.0))
        lyrics = str(req.params.get("lyrics") or "").strip()
        steps = int(req.params.get("inference_steps") or (8 if self.is_turbo else 30))
        seeded = req.seed is not None
        lm = loaded.lm_enabled

        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            # ACE-Step's documented convention for an instrumental piece.
            lyrics=lyrics or "[Instrumental]",
            instrumental=not lyrics,
            duration=seconds,
            inference_steps=steps,
            guidance_scale=float(req.params.get("guidance_scale", 7.0)),
            seed=int(req.seed) if seeded else -1,
            # The planning features need the LM; with it off, run the DiT alone.
            thinking=lm,
            use_cot_metas=lm,
            use_cot_caption=lm,
            use_cot_language=lm,
        )
        # One take per request -- ACE-Step's default batch of 2 would double the
        # work for a clip the timeline never sees.
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=not seeded,
            seeds=[int(req.seed)] if seeded else None,
        )

        result = generate_music(loaded.dit, loaded.llm, params, config, save_dir=None)
        if not result.success:
            raise RuntimeError(f"ACE-Step generation failed: {result.error or result.status_message}")
        if not result.audios:
            raise RuntimeError("ACE-Step returned no audio")

        first = result.audios[0]
        tensor = first.get("tensor")
        if tensor is None:
            raise RuntimeError("ACE-Step returned an entry without an audio tensor")
        # [channels, samples], float32 -- and 48 kHz, not the 44.1 kHz an
        # earlier version assumed, which would have played ~9% slow and flat.
        sr = int(first.get("sample_rate", 48000))
        arr = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, dtype=np.float32)
        n = arr.shape[-1]
        return AudioResult(audio=pcm_to_wav(arr, sr), sample_rate=sr, duration=n / sr, provider_id=self.id)
