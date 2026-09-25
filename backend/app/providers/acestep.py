"""ACE-Step 1.5 -- local 'music' backend, MIT weights.

Why this is the music default over MusicGen: MIT weights (no CC-BY-NC rider),
10s-600s range, stem output, repaint/selective regeneration, and explicit
BPM/key control -- all of which map onto a timeline editor. It is also the
fastest open music model by a wide margin (~24x realtime on an RTX 3090).

VRAM tiers, from the project's own table:
    <=6 GB   2B turbo, DiT-only, INT8 + full CPU offload
    6-8 GB   2B turbo + 0.6B LM (PyTorch backend)
    8-16 GB  2B turbo/sft + 0.6B LM, or 1.7B LM at 12-16 GB (vLLM backend)
    16-20 GB 2B sft or XL turbo + 1.7B LM (XL needs CPU offload below 20 GB)
    20-24 GB XL turbo/sft + 1.7B LM, no offload
    >=24 GB  XL sft + 4B LM, best quality

Set ACESTEP_BACKEND=pt to use the PyTorch LM backend and avoid vLLM entirely.
vLLM pins torch and CUDA tightly and its compiled kernels break across
torch/CUDA combinations, so it is the main packaging hazard here.
"""
from __future__ import annotations

import importlib.util
import logging
import os

from ..config import get_settings
from .base import AudioResult, Capability, GenerateRequest, ParamSpec, Provider, pcm_to_wav

log = logging.getLogger(__name__)


class AceStepProvider(Provider):
    id = "acestep"
    name = "ACE-Step 1.5"
    capability = Capability.MUSIC
    license = "Apache-2.0 code, MIT weights - commercial use permitted"
    requires_gpu = True
    description = "Text-to-music with stems, repaint and BPM/key control. 10s-600s. Song-oriented."

    def __init__(self, model_id: str | None = None) -> None:
        super().__init__()
        # `or`, not a getenv default: compose passes ${ACESTEP_MODEL:-}, which
        # sets the variable to "" when .env leaves it blank (as .env.example
        # does). A getenv default only applies when the variable is UNSET, so
        # it would have produced an empty model id for every default deploy.
        self.model_id = model_id or os.getenv("ACESTEP_MODEL") or "ACE-Step/acestep-v15-base"
        self.backend = os.getenv("ACESTEP_BACKEND") or "pt"

    def available(self) -> bool:
        return importlib.util.find_spec("acestep") is not None

    def unavailable_reason(self) -> str:
        return "" if self.available() else "acestep is not installed in this image (EXTRAS=music)"

    def _load(self):
        # NOTE: ACE-Step 1.5 reorganised its entry points from the 1.0 line.
        # Both known module paths are tried so a version bump does not silently
        # break the provider; if neither resolves, the error names the problem
        # instead of surfacing as a generic AttributeError mid-generation.
        device = get_settings().device
        log.info("loading ACE-Step %s on %s (backend=%s)", self.model_id, device, self.backend)

        pipeline_cls = None
        for module_path, attr in (
            ("acestep.pipeline_ace_step", "ACEStepPipeline"),
            ("acestep.pipeline", "ACEStepPipeline"),
        ):
            try:
                module = __import__(module_path, fromlist=[attr])
                pipeline_cls = getattr(module, attr)
                break
            except (ImportError, AttributeError):
                continue

        if pipeline_cls is None:
            raise RuntimeError(
                "could not locate ACEStepPipeline in the installed acestep package; "
                "check the entry point for your installed version"
            )

        return pipeline_cls(checkpoint_dir=None, dtype="bfloat16", device=device)

    def params(self) -> list[ParamSpec]:
        return [
            ParamSpec("seconds", "float", 30.0, 10.0, 600.0, "Clip length"),
            ParamSpec("lyrics", "str", "", None, None, "Optional lyrics; blank gives an instrumental"),
            ParamSpec("infer_steps", "int", 27, 10, 100, "More steps is slower and slightly cleaner"),
            ParamSpec("guidance_scale", "float", 15.0, 1.0, 30.0, "Prompt adherence"),
        ]

    def generate(self, req: GenerateRequest) -> AudioResult:
        pipeline = self.load()

        prompt = req.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        seconds = float(req.seconds or req.params.get("seconds") or 30.0)
        seconds = max(10.0, min(seconds, 600.0))

        result = pipeline(
            prompt=prompt,
            lyrics=str(req.params.get("lyrics", "")),
            audio_duration=seconds,
            infer_step=int(req.params.get("infer_steps", 27)),
            guidance_scale=float(req.params.get("guidance_scale", 15.0)),
            manual_seeds=str(req.seed) if req.seed is not None else None,
        )

        arr, sr = _as_array(result)
        audio = pcm_to_wav(arr, sr)
        n = arr.shape[-1] if getattr(arr, "ndim", 1) > 1 else len(arr)
        return AudioResult(audio=audio, sample_rate=sr, duration=n / sr, provider_id=self.id)


def _as_array(result):
    """Normalise the several shapes ACE-Step has returned across versions."""
    import numpy as np

    sr = 44100
    audio = result

    if isinstance(result, tuple) and len(result) == 2:
        audio, sr = result
    elif isinstance(result, dict):
        audio = result.get("audio", result.get("waveform"))
        sr = int(result.get("sample_rate", sr))
    elif isinstance(result, list) and result:
        audio = result[0]

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio)
    if arr.ndim == 3:            # (batch, channels, n)
        arr = arr[0]
    return arr, int(sr)
