"""Provider calls into the model libraries, checked against their pinned sources.

ACE-Step, Stable Audio 3, Chatterbox and diffusers need CUDA torch, so neither
dev nor CI installs them, and nothing else checks that a provider still calls
them the way the pinned version defines. A renamed keyword would surface only
on the GPU box, as a TypeError on the first generation.

scripts/snapshot_contracts.py records the signatures, config fields and
attributes the providers use, from the exact pinned sources, into
tests/contracts/. These tests build fake modules from those files and run each
provider's real _load() and generate() against them:

  * every call is bound with inspect.Signature against the recorded signature,
    and a keyword that would land in a **kwargs the source never reads by name
    counts as a mismatch -- binding alone would accept any typo there;
  * the fake behind the call sees each argument under the name of the pinned
    parameter it bound to, not where the provider put it -- so arguments
    passed out of order, or a pin that moves a parameter, put the wrong value
    under a name the test checks;
  * every attribute read off a returned object, and every key read off a
    returned dict (an `in` test included), must be one the source defines;
  * each test asserts the exact arguments the provider sends, so one dropped
    from a call fails even where the library's default would have bound.

A mismatch raises (TypeError for a call, AttributeError or KeyError for a
read, ImportError for a name) and is also recorded, so a provider that
swallows the exception -- getattr(obj, name, default) does -- still fails.

The snapshots are only as good as their pins, so the last tests check that the
refs they were taken at are the refs compose and pyproject.toml install, and
that each is pinned exactly, not by a floor -- except diffusers, whose
snapshot is of the floor the repo declares.
"""
from __future__ import annotations

import collections.abc
import importlib.machinery
import inspect
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from helpers import make_wav, reload_settings

from app import storage
from app.providers import stable_audio
from app.providers.acestep import AceStepProvider
from app.providers.base import Capability, GenerateRequest, wav_duration
from app.providers.chatterbox import ChatterboxProvider
from app.providers.stable_audio import StableAudio3Provider

CONTRACTS = Path(__file__).parent / "contracts"
REPO = Path(__file__).resolve().parents[2]


# -- fakes built from a snapshot ---------------------------------------------------


class _Src(str):
    """A default value's source text, shown as written in signature errors."""

    def __repr__(self) -> str:
        return str(self)


def _signature(params: list[dict], *, drop_first: bool = False) -> inspect.Signature:
    if drop_first:
        params = params[1:]
    return inspect.Signature([
        inspect.Parameter(
            p["name"],
            getattr(inspect.Parameter, p["kind"]),
            default=inspect.Parameter.empty if p["default"] is None else _Src(p["default"]),
        )
        for p in params
    ])


def _by_name(bound: inspect.BoundArguments) -> tuple[list, dict]:
    """A bound call's arguments, each under the pinned parameter it landed in.

    A fake's impl gets these, not the provider's raw arguments: given the raw
    ones it read a positional argument under its own parameter name, so a
    provider passing arguments out of order, or a pin that reordered them,
    still passed. Positional-only and *args values stay positional; everything
    else, a **kwargs included, goes by name.
    """
    args: list = []
    kwargs: dict = {}
    for name, value in bound.arguments.items():
        kind = bound.signature.parameters[name].kind
        if kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(value)
        else:
            kwargs[name] = value
    return args, kwargs


class Contracts:
    """Builds fakes from tests/contracts/*.json and collects every mismatch."""

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.violations: list[str] = []

    def violation(self, message: str, exc_type: type[Exception]) -> Exception:
        self.violations.append(message)
        return exc_type(message)

    def library(self, name: str) -> Library:
        return Library(self, name)


# Not in a snapshot means one of two things, and the fix differs: the pinned
# source lacks it (the provider is wrong), or it was never listed in
# scripts/snapshot_contracts.py (list it, re-run, and see which).
_UNRECORDED = "not in the snapshot of the pinned source (absent upstream, or unlisted in scripts/snapshot_contracts.py)"


class _Strict:
    """Base of every fake library object: only snapshot attributes exist."""

    _fake_qualname = ""
    _fake_attributes: frozenset[str] = frozenset()
    _fake_absent: frozenset[str] = frozenset()       # recorded as null: not in the pinned source
    _fake_contracts: Contracts

    def __getattr__(self, name):                      # called only when lookup fails
        if name.startswith("__") or name.startswith("_fake"):
            raise AttributeError(name)
        if name in self._fake_attributes:
            raise AttributeError(f"test_contracts did not set {self._fake_qualname}.{name}")
        if name in self._fake_absent:
            raise self._fake_contracts.violation(
                f"provider calls {self._fake_qualname}.{name}, which the pinned source does not define", AttributeError)
        raise self._fake_contracts.violation(
            f"provider reads {self._fake_qualname}.{name}: {_UNRECORDED}", AttributeError)

    def __setattr__(self, name, value):
        if not name.startswith("_fake") and name not in self._fake_attributes:
            raise self._fake_contracts.violation(f"{self._fake_qualname}.{name} is set: {_UNRECORDED}", AttributeError)
        object.__setattr__(self, name, value)


class _StrictDict(dict):
    """A returned dict whose readable keys are those the source builds it with.

    Every way of naming a key is checked, not only [] and get(): `"x" in d` is
    the dict form of getattr(obj, name, default), and left unchecked, a
    misspelt key behind it quietly takes the provider's fallback value.
    """

    def __init__(self, contracts: Contracts, what: str, keys: list[str], values: dict) -> None:
        super().__init__(values)
        self._contracts, self._what, self._keys = contracts, what, frozenset(keys)

    def _check(self, key) -> None:
        if key not in self._keys:
            raise self._contracts.violation(
                f"provider reads {key!r} from {self._what}, whose keys in the pinned source are {sorted(self._keys)}",
                KeyError,
            )

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._check(key)
        return super().__contains__(key)

    def keys(self):
        # dict's own keys view answers `in` without calling __contains__.
        return collections.abc.KeysView(self)

    def pop(self, key, *default):
        self._check(key)
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        self._check(key)
        return super().setdefault(key, default)


class Library:
    def __init__(self, contracts: Contracts, name: str) -> None:
        self.contracts = contracts
        self.name = name
        self.data = json.loads((CONTRACTS / f"{name}.json").read_text())

    def entry(self, module: str, name: str, kind: str) -> dict | None:
        try:
            entry = self.data["modules"][module][name]
        except KeyError:
            raise AssertionError(
                f"{module}.{name} is not in tests/contracts/{self.name}.json: add it to LIBRARIES in "
                "scripts/snapshot_contracts.py and re-run the script") from None
        assert entry is None or entry["kind"] == kind, f"{module}.{name} is a {entry['kind']}, not a {kind}"
        return entry

    def check(self, qualname: str, sig: inspect.Signature, consumed, args, kwargs) -> inspect.BoundArguments:
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as exc:
            raise self.contracts.violation(f"{qualname}: {exc} (pinned signature: {sig})", TypeError) from None
        var_kw = next((p.name for p in sig.parameters.values() if p.kind is p.VAR_KEYWORD), None)
        unread = sorted(set(bound.arguments.get(var_kw, {})) - set(consumed or ())) if var_kw else []
        if unread:
            raise self.contracts.violation(
                f"{qualname}: {unread} would land in **{var_kw}, which the pinned source never reads by name",
                TypeError,
            )
        return bound

    # Each builder returns None for a symbol the pinned source lacks, so it is
    # left off the fake module and the provider's import of it fails.

    def function(self, module: str, name: str, impl):
        entry = self.entry(module, name, "function")
        if entry is None:
            return None
        qualname, sig = f"{module}.{name}", _signature(entry["params"])

        def fake(*args, **kwargs):
            args, kwargs = _by_name(self.check(qualname, sig, entry.get("kwargs_consumed"), args, kwargs))
            return impl(*args, **kwargs)

        return fake

    def cls(self, module: str, name: str, impl: type | None = None) -> type | None:
        entry = self.entry(module, name, "class")
        if entry is None:
            return None
        qualname = f"{module}.{name}"
        given = {k: v for k, v in vars(impl).items()
                 if callable(v) or isinstance(v, (classmethod, staticmethod))} if impl else {}
        extra = set(given) - set(entry["methods"])
        assert not extra, f"the fake {qualname} defines {sorted(extra)}, which its snapshot does not record"
        namespace = {
            "_fake_qualname": qualname,
            "_fake_attributes": frozenset(entry["attributes"]),
            "_fake_absent": frozenset(m for m, spec in entry["methods"].items() if spec is None),
            "_fake_contracts": self.contracts,
        }
        for method, spec in entry["methods"].items():
            if spec is not None:      # absent upstream: left off, so a call hits __getattr__
                namespace[method] = self._method(f"{qualname}.{method}", spec, given.get(method))
        return type(name, (_Strict,), namespace)

    def _method(self, qualname: str, spec: dict, impl):
        binding = spec["binding"]
        if impl is not None:
            kind = ("classmethod" if isinstance(impl, classmethod)
                    else "staticmethod" if isinstance(impl, staticmethod) else "method")
            assert kind == binding, f"the fake {qualname} is a {kind}; the pinned source has a {binding}"
            impl = getattr(impl, "__func__", impl)
        elif qualname.endswith(".__init__"):
            impl = lambda obj, *args, **kwargs: None      # noqa: E731
        else:
            def impl(*args, **kwargs):
                raise NotImplementedError(f"test_contracts has no fake for {qualname}")
        sig = _signature(spec["params"], drop_first=binding in {"method", "classmethod"})
        consumed = spec.get("kwargs_consumed")

        def bound_call(owner, *args, **kwargs):
            args, kwargs = _by_name(self.check(qualname, sig, consumed, args, kwargs))
            return impl(owner, *args, **kwargs)

        def static_call(*args, **kwargs):
            args, kwargs = _by_name(self.check(qualname, sig, consumed, args, kwargs))
            return impl(*args, **kwargs)

        if binding == "staticmethod":
            return staticmethod(static_call)
        return classmethod(bound_call) if binding == "classmethod" else bound_call

    def make(self, cls: type, **attributes):
        """An instance of a fake class without running its __init__ -- for
        objects a library builds internally, which a provider never constructs."""
        obj = object.__new__(cls)
        for name, value in attributes.items():
            setattr(obj, name, value)
        return obj

    def dataclass(self, module: str, name: str) -> type | None:
        entry = self.entry(module, name, "dataclass")
        if entry is None:
            return None
        qualname = f"{module}.{name}"
        fields = entry["fields"]
        params = [
            inspect.Parameter(
                f["name"],
                inspect.Parameter.KEYWORD_ONLY if f["kw_only"] else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if f["default"] is None else _Src(f["default"]),
            )
            for f in fields if f["init"]
        ]
        # dataclasses puts keyword-only fields last; sorted() is stable.
        sig = inspect.Signature(sorted(params, key=lambda p: p.kind))
        library = self

        def __init__(obj, *args, **kwargs):
            bound = library.check(qualname, sig, None, args, kwargs)
            # The fields the caller passed, as opposed to ones left at a default.
            object.__setattr__(obj, "_fake_given", frozenset(bound.arguments))
            for f in fields:
                object.__setattr__(obj, f["name"], bound.arguments.get(f["name"], _Src(f["default"] or "")))

        return type(name, (_Strict,), {
            "__init__": __init__,
            "_fake_qualname": qualname,
            "_fake_attributes": frozenset(f["name"] for f in fields),
            "_fake_contracts": self.contracts,
        })

    def dict_literal(self, module: str, function: str, var: str, **values) -> _StrictDict:
        keys = self.entry(module, function, "function")["dict_literals"][var]
        unknown = set(values) - set(keys)
        assert not unknown, f"{module}.{function}() never puts {sorted(unknown)} in {var}"
        return _StrictDict(self.contracts, f"the {var} entries {module}.{function}() returns", keys, values)

    def keys(self, module: str, name: str) -> list[str]:
        return self.entry(module, name, "dict")["keys"]

    def install(self, fakes: dict[str, dict[str, object]]) -> None:
        """Put a fake module in sys.modules for every module the snapshot
        records (and their parent packages), holding the given fakes."""
        recorded = self.data["modules"]
        missing = {f"{m}.{n}" for m, names in fakes.items() for n in names if m not in recorded or n not in recorded[m]}
        assert not missing, f"no snapshot entry for {sorted(missing)}"
        made: dict[str, types.ModuleType] = {}
        for module in sorted(recorded):
            parts = module.split(".")
            for depth in range(1, len(parts) + 1):
                name = ".".join(parts[:depth])
                if name not in made:
                    made[name] = self._module(name, recorded.get(name, {}))
                    if depth > 1:
                        setattr(made[".".join(parts[:depth - 1])], parts[depth - 1], made[name])
        for module, names in fakes.items():
            for name, fake in names.items():
                if fake is not None:
                    setattr(made[module], name, fake)
        for name, mod in made.items():
            self.contracts.monkeypatch.setitem(sys.modules, name, mod)

    def _module(self, name: str, recorded: dict) -> types.ModuleType:
        mod = types.ModuleType(name)
        # find_spec() reads __spec__ off a module already in sys.modules, and
        # the providers' availability checks go through find_spec().
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        mod.__path__ = []
        contracts, library = self.contracts, self.name

        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            if attr not in recorded:
                raise contracts.violation(f"provider imports {name}.{attr}: {_UNRECORDED}", AttributeError)
            if recorded[attr] is None:
                raise contracts.violation(
                    f"provider imports {name}.{attr}, which the pinned {library} source does not define",
                    AttributeError)
            raise AttributeError(f"test_contracts has no fake for {name}.{attr}")

        mod.__getattr__ = __getattr__
        return mod


@pytest.fixture
def contracts(monkeypatch):
    c = Contracts(monkeypatch)
    yield c
    # Also fails a test whose provider caught the exception a mismatch raised.
    assert not c.violations, "calls that do not match the pinned sources:\n  " + "\n  ".join(c.violations)


def _tone(channels: int, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / 1000.0
    return np.tile(0.3 * np.sin(2 * np.pi * t), (channels, 1)).astype(np.float32)


# -- ACE-Step 1.5 ----------------------------------------------------------------


@pytest.fixture
def acestep(contracts, monkeypatch, tmp_path):
    lib = contracts.library("acestep")
    # The calls each fake saw. Set `failure` to GenerationResult fields to make
    # generate_music report a failed run instead.
    ace = types.SimpleNamespace(initialize_service=[], initialize=[], generate_music=[], failure=None)

    class AceStepHandler:
        def initialize_service(self, **kwargs):
            ace.initialize_service.append(kwargs)
            return "ready", True

    class LLMHandler:
        def initialize(self, **kwargs):
            ace.initialize.append(kwargs)
            return "ready", True

    Handler = lib.cls("acestep.handler", "AceStepHandler", AceStepHandler)
    LLM = lib.cls("acestep.llm_inference", "LLMHandler", LLMHandler)
    Params = lib.dataclass("acestep.inference", "GenerationParams")
    Config = lib.dataclass("acestep.inference", "GenerationConfig")
    GenerationResult = lib.dataclass("acestep.inference", "GenerationResult")

    def generate_music(**call):
        ace.generate_music.append(call)
        # Two handlers, then params and config: each checked under the pinned
        # parameter it bound to, so passing them in another order, or a pin
        # that reorders them, fails here instead of reading the wrong object.
        for name, cls in (("dit_handler", Handler), ("llm_handler", LLM), ("params", Params), ("config", Config)):
            if not isinstance(call.get(name), cls):
                raise contracts.violation(
                    f"acestep.inference.generate_music: {name} got {type(call.get(name)).__name__}; "
                    f"expected {cls.__name__}", TypeError)
        if ace.failure is not None:
            # As the pinned generate_music reports a failed run
            # (acestep/inference.py:920-926 and 1109-1115).
            return GenerationResult(audios=[], extra_outputs={}, success=False, **ace.failure)
        audio = lib.dict_literal("acestep.inference", "generate_music", "audio_dict",
                                 path="", key="take-0", params={},
                                 tensor=_tone(2, int(call["params"].duration * 48000)), sample_rate=48000)
        return GenerationResult(audios=[audio], status_message="ok", success=True, error=None)

    lib.install({
        "acestep.handler": {"AceStepHandler": Handler},
        "acestep.llm_inference": {"LLMHandler": LLM},
        "acestep.inference": {
            "GenerationParams": Params,
            "GenerationConfig": Config,
            "GenerationResult": GenerationResult,
            "generate_music": lib.function("acestep.inference", "generate_music", generate_music),
        },
    })
    reload_settings(monkeypatch, ACESTEP_PROJECT_ROOT=str(tmp_path / "acestep"),
                    ACESTEP_MODEL=None, ACESTEP_LM_MODEL=None, ACESTEP_BACKEND=None)
    return ace


ACE_DIT_KEYWORDS = {"project_root", "config_path", "device", "use_flash_attention", "compile_model",
                    "offload_to_cpu", "offload_dit_to_cpu", "quantization", "prefer_source"}
ACE_PARAMS_KEYWORDS = {"task_type", "caption", "lyrics", "instrumental", "duration", "inference_steps",
                       "guidance_scale", "seed", "thinking", "use_cot_metas", "use_cot_caption", "use_cot_language"}
ACE_GENERATE_ARGUMENTS = {"dit_handler", "llm_handler", "params", "config", "save_dir"}


def test_acestep_with_the_planning_lm(acestep):
    provider = AceStepProvider()
    result = provider.generate(GenerateRequest("warm ambient piano", Capability.MUSIC, seconds=12, seed=7,
                                               params={"lyrics": "la la", "inference_steps": 20}))

    assert result.sample_rate == 48000
    assert wav_duration(result.audio) == pytest.approx(12.0)
    (dit,) = acestep.initialize_service
    assert set(dit) == ACE_DIT_KEYWORDS
    assert (dit["config_path"], dit["device"], dit["prefer_source"]) == ("acestep-v15-base", "cpu", "huggingface")
    (lm,) = acestep.initialize
    assert set(lm) == {"checkpoint_dir", "lm_model_path", "backend", "device", "offload_to_cpu", "dtype"}
    assert (lm["lm_model_path"], lm["backend"], lm["device"]) == ("acestep-5Hz-lm-0.6B", "pt", "cpu")
    (call,) = acestep.generate_music
    assert set(call) == ACE_GENERATE_ARGUMENTS and call["save_dir"] is None
    params, config = call["params"], call["config"]
    assert params._fake_given == ACE_PARAMS_KEYWORDS
    assert (params.thinking, params.use_cot_metas, params.use_cot_caption, params.use_cot_language) == (True,) * 4
    assert (params.caption, params.lyrics, params.instrumental) == ("warm ambient piano", "la la", False)
    assert (params.duration, params.seed, params.inference_steps) == (12.0, 7, 20)
    assert config._fake_given == {"batch_size", "use_random_seed", "seeds"}
    assert (config.batch_size, config.use_random_seed, config.seeds) == (1, False, [7])


def test_acestep_without_the_lm(acestep, monkeypatch):
    reload_settings(monkeypatch, ACESTEP_LM_MODEL="none", ACESTEP_MODEL="ACE-Step/acestep-v15-turbo")
    provider = AceStepProvider()
    result = provider.generate(GenerateRequest("drum loop", Capability.MUSIC, seconds=10))

    assert wav_duration(result.audio) == pytest.approx(10.0)
    assert acestep.initialize == []                          # the LM is never loaded
    (dit,) = acestep.initialize_service
    assert set(dit) == ACE_DIT_KEYWORDS and dit["config_path"] == "acestep-v15-turbo"
    (call,) = acestep.generate_music
    assert set(call) == ACE_GENERATE_ARGUMENTS
    params, config = call["params"], call["config"]
    assert params._fake_given == ACE_PARAMS_KEYWORDS
    assert (params.thinking, params.use_cot_metas, params.use_cot_caption, params.use_cot_language) == (False,) * 4
    assert (params.lyrics, params.instrumental, params.inference_steps, params.seed) == ("[Instrumental]", True, 8, -1)
    assert (config.use_random_seed, config.seeds) == (True, None)


@pytest.mark.parametrize(("failure", "reason"), [
    # generate_music's except branch: `error` carries the exception text.
    ({"status_message": "Error: CUDA out of memory", "error": "CUDA out of memory"}, "CUDA out of memory"),
    # A failed DiT run copies `error` from a dict that may not have one, so
    # the status message is then the only reason given.
    ({"status_message": "DiT produced no latents", "error": None}, "DiT produced no latents"),
])
def test_acestep_failed_generation(acestep, failure, reason):
    # Without this, the provider's reads of .error and .status_message never
    # run under the strict fake, and a misspelt one would surface on the GPU
    # box as an AttributeError in place of the reason the run failed.
    acestep.failure = failure
    with pytest.raises(RuntimeError, match=f"^ACE-Step generation failed: {re.escape(reason)}$"):
        AceStepProvider().generate(GenerateRequest("drum loop", Capability.MUSIC, seconds=10))


@pytest.mark.parametrize("read", [
    lambda entry, key: entry[key],
    lambda entry, key: entry.get(key, 0),
    lambda entry, key: key in entry,
    lambda entry, key: key in entry.keys(),             # noqa: SIM118 -- the form under test
    lambda entry, key: entry.pop(key, 0),
    lambda entry, key: entry.setdefault(key, 0),
], ids=["getitem", "get", "in", "in-keys", "pop", "setdefault"])
def test_acestep_audio_entry_checks_every_key_read(monkeypatch, read):
    # The provider reads the entry with .get() today. Rewritten as
    # `int(entry["sr"]) if "sr" in entry else 48000`, a misspelt key took the
    # fallback rate and every provider test still passed.
    contracts = Contracts(monkeypatch)        # not the fixture: the violation below is meant
    entry = contracts.library("acestep").dict_literal(
        "acestep.inference", "generate_music", "audio_dict", sample_rate=48000)
    read(entry, "sample_rate")
    assert contracts.violations == []
    with pytest.raises(KeyError, match="provider reads 'sr' from the audio_dict entries"):
        read(entry, "sr")
    assert len(contracts.violations) == 1


# -- Stable Audio 3 ----------------------------------------------------------------


@pytest.fixture
def sa3_official(contracts, fake_torch, monkeypatch):
    lib = contracts.library("stable_audio_3")
    known = lib.keys("stable_audio_3.model_configs", "models")
    Wrapper = lib.cls("stable_audio_3.models.diffusion", "ConditionedDiffusionModelWrapper")
    loaded: list[tuple] = []
    calls: list[dict] = []

    class StableAudioModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            # As the real one does (stable_audio_3/model.py): only names in
            # model_configs load.
            if model_name not in known:
                raise contracts.violation(f"StableAudioModel.from_pretrained({model_name!r}): not one of {known}",
                                          ValueError)
            loaded.append((model_name, kwargs))
            return lib.make(FakeModel, model=lib.make(Wrapper, sample_rate=44100))

        def generate(self, **kwargs):
            calls.append(kwargs)
            return _tone(2, int(kwargs["duration"] * 44100))[None]          # [batch, channels, samples]

    FakeModel = lib.cls("stable_audio_3", "StableAudioModel", StableAudioModel)
    lib.install({
        "stable_audio_3": {"StableAudioModel": FakeModel},
        "stable_audio_3.models.diffusion": {"ConditionedDiffusionModelWrapper": Wrapper},
    })
    reload_settings(monkeypatch, HF_TOKEN="hf_test", SA3_MODEL=None)
    return loaded, calls


@pytest.mark.parametrize(("capability", "sa3_model", "official_name"), [
    (Capability.SFX, None, "small-sfx"),
    (Capability.MUSIC, None, "small-music"),
    (Capability.MUSIC, "stabilityai/stable-audio-3-medium", "medium"),
    (Capability.SFX, "medium", "medium"),
])
def test_stable_audio_official_loader(sa3_official, monkeypatch, capability, sa3_model, official_name):
    loaded, calls = sa3_official
    reload_settings(monkeypatch, SA3_MODEL=sa3_model)
    provider = StableAudio3Provider(capability)
    assert stable_audio._official_available()                # the real check sees the fake package
    result = provider.generate(GenerateRequest("door creak", capability, seconds=2, seed=5,
                                               params={"negative_prompt": "music", "steps": 12}))

    assert loaded == [(official_name, {"device": "cpu", "model_half": False})]
    assert result.sample_rate == 44100
    assert wav_duration(result.audio) == pytest.approx(2.0)
    (call,) = calls
    assert set(call) == {"prompt", "negative_prompt", "duration", "steps", "cfg_scale", "batch_size", "seed"}
    assert (call["prompt"], call["negative_prompt"], call["duration"]) == ("door creak", "music", 2.0)
    assert (call["seed"], call["steps"], call["batch_size"]) == (5, 12, 1)


def test_stable_audio_official_loader_unseeded(sa3_official):
    _, calls = sa3_official
    StableAudio3Provider(Capability.SFX).generate(GenerateRequest("rain", Capability.SFX, seconds=1))
    assert calls[0]["seed"] == -1 and calls[0]["negative_prompt"] is None


@pytest.fixture
def sa3_diffusers(contracts, fake_torch, monkeypatch):
    lib = contracts.library("diffusers")
    # None in sys.modules makes find_spec() report the package absent, so the
    # provider falls back to diffusers whatever happens to be installed.
    monkeypatch.setitem(sys.modules, "stable_audio_3", None)
    Output = lib.dataclass("diffusers.pipelines.pipeline_utils", "AudioPipelineOutput")
    Vae = lib.cls("diffusers.models.autoencoders.autoencoder_same", "AutoencoderSAME")
    calls: dict[str, list] = {"from_pretrained": [], "to": [], "__call__": []}

    class StableAudio3Pipeline:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            calls["from_pretrained"].append((pretrained_model_name_or_path, kwargs))
            # Not SA3's real 44.1 kHz: a value the provider's 44100 fallback
            # cannot produce, so the test proves the rate is read off the VAE.
            return lib.make(cls, vae=lib.make(Vae, sampling_rate=48000))

        def to(self, *args, **kwargs):
            calls["to"].append(args)
            return self

        def __call__(self, **kwargs):
            calls["__call__"].append(kwargs)
            sr = self.vae.sampling_rate
            return Output(audios=_tone(2, int(kwargs.get("duration", 10.0) * sr))[None])

    class Generator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    monkeypatch.setattr(fake_torch, "Generator", Generator, raising=False)
    lib.install({
        "diffusers": {"StableAudio3Pipeline": lib.cls("diffusers", "StableAudio3Pipeline", StableAudio3Pipeline)},
        "diffusers.models.autoencoders.autoencoder_same": {"AutoencoderSAME": Vae},
        "diffusers.pipelines.pipeline_utils": {"AudioPipelineOutput": Output},
    })
    reload_settings(monkeypatch, HF_TOKEN="hf_test", SA3_MODEL=None)
    return calls


def test_stable_audio_diffusers_loader(sa3_diffusers):
    provider = StableAudio3Provider(Capability.SFX)
    assert not stable_audio._official_available()
    result = provider.generate(GenerateRequest("glass breaking", Capability.SFX, seconds=3, seed=11,
                                               params={"steps": 16, "guidance_scale": 4.0}))

    assert result.sample_rate == 48000
    assert wav_duration(result.audio) == pytest.approx(3.0)
    ((repo, kwargs),) = sa3_diffusers["from_pretrained"]
    assert repo == "stabilityai/stable-audio-3-small-sfx"
    assert kwargs == {"torch_dtype": "torch.float32", "token": "hf_test"}
    assert sa3_diffusers["to"] == [("cpu",)]
    (call,) = sa3_diffusers["__call__"]
    # The provider passes the prompt positionally, so this is where a pin that
    # moves `prompt` out of the first slot shows up.
    assert set(call) == {"prompt", "negative_prompt", "num_inference_steps", "guidance_scale", "duration", "generator"}
    assert (call["prompt"], call["negative_prompt"], call["duration"]) == ("glass breaking", None, 3.0)
    assert (call["num_inference_steps"], call["guidance_scale"]) == (16, 4.0)
    assert call["generator"].seed == 11


# -- Chatterbox ----------------------------------------------------------------------


@pytest.fixture
def chatterbox(contracts, fake_torch, monkeypatch):
    lib = contracts.library("chatterbox")
    calls: dict[str, list] = {"from_pretrained": [], "generate": [], "manual_seed": []}

    class ChatterboxTTS:
        @classmethod
        def from_pretrained(cls, device):
            calls["from_pretrained"].append(device)
            return lib.make(cls, sr=24000)

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return _tone(1, 24000)                                     # (1, n), as Chatterbox returns

    monkeypatch.setattr(fake_torch, "manual_seed", lambda seed: calls["manual_seed"].append(seed))
    lib.install({"chatterbox.tts": {"ChatterboxTTS": lib.cls("chatterbox.tts", "ChatterboxTTS", ChatterboxTTS)}})
    return calls


def test_chatterbox_default_voice(chatterbox):
    provider = ChatterboxProvider()
    assert provider.available()                              # the real check sees the fake package
    result = provider.generate(GenerateRequest("Hello there.", Capability.VOICE, voice_id="default", seed=3,
                                               params={"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.9}))

    assert result.sample_rate == 24000
    assert wav_duration(result.audio) == pytest.approx(1.0)
    assert chatterbox["from_pretrained"] == ["cpu"]
    # `text` goes in positionally: checked here under the name it bound to.
    assert chatterbox["generate"] == [
        {"text": "Hello there.", "exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.9}]
    assert chatterbox["manual_seed"] == [3]


def test_chatterbox_cloned_voice(chatterbox):
    ref = storage.save_voice_reference(make_wav(1.0), "narrator")
    ChatterboxProvider().generate(GenerateRequest("Once upon a time.", Capability.VOICE, voice_id=ref["id"]))

    assert chatterbox["generate"] == [
        {"text": "Once upon a time.", "audio_prompt_path": str(storage.voice_reference_path(ref["id"]))}]
    assert chatterbox["manual_seed"] == []


# -- the snapshots and the pins they were taken at ------------------------------------


def _snapshots() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(CONTRACTS.glob("*.json"))}


def test_snapshots_record_where_they_came_from():
    snaps = _snapshots()
    assert set(snaps) == {"acestep", "stable_audio_3", "chatterbox", "diffusers"}
    for name, snap in snaps.items():
        source = snap["source"]
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), name
        # A ref edited by hand, without re-running the script, no longer
        # matches the archive the signatures were read from.
        assert source["ref"] in source["url"], f"{name}: ref {source['ref']} is not the archive {source['url']}"


# Libraries the repo pins only by a floor, so their snapshot is of that floor.
# Every other one must be pinned exactly: a `>=` in any one place lets that
# image's --no-deps install pull a newer release than the snapshot was taken
# from, while these tests still pass against the old one.
FLOOR_PINNED = {"diffusers"}


def _library_pins() -> dict[str, tuple[str, str]]:
    """library -> (how a requirement spec names it, the pin it must carry).

    A GitHub archive is pinned by commit, `@ref`; a PyPI release by `==ref`,
    or by `>=ref` if it is in FLOOR_PINNED.
    """
    out = {}
    for name, snap in _snapshots().items():
        source = snap["source"]
        m = re.match(r"https://github\.com/([^/]+/[^/]+)/archive/", source["url"])
        if m:
            out[name] = (m.group(1), "@" + source["ref"])
        else:
            out[name] = (source["distribution"], (">=" if name in FLOOR_PINNED else "==") + source["ref"])
    return out


def _pins_in(specs: list[str]) -> dict[str, list[str]]:
    """library -> every pin these requirement specs give it: `==version`,
    `>=version` or `@ref`, operator included so a loosened pin differs.

    Read from `name==version`, `name>=version`, and
    `git+https://github.com/owner/repo@ref` with or without a `name @ `
    prefix. A spec naming a snapshotted library in any other form fails: it
    would otherwise be skipped, and the test would pass without comparing it.
    """
    by_key = {key.lower(): name for name, (key, _) in _library_pins().items()}
    found: dict[str, list[str]] = {}
    for spec in specs:
        git = re.fullmatch(r"(?:[\w.-]+\s*@\s*)?git\+https://github\.com/([^@\s]+?)(?:\.git)?@(\S+)", spec)
        plain = re.fullmatch(r"([\w.-]+)\s*(==|>=)\s*([\w.]+)", spec)
        if git and git.group(1).lower() in by_key:
            found.setdefault(by_key[git.group(1).lower()], []).append("@" + git.group(2))
        elif plain and plain.group(1).lower() in by_key:
            found.setdefault(by_key[plain.group(1).lower()], []).append(plain.group(2) + plain.group(3))
        else:
            named = re.match(r"[\w.-]+", spec)
            assert not (named and named.group(0).lower() in by_key), f"cannot read the pin in {spec!r}"
    return found


def _compose_models(path: Path) -> list[str]:
    """Every spec in every MODELS build arg, read as text: a plain or folded
    (>-) YAML scalar, which is how both compose files write it."""
    lines = path.read_text().splitlines()
    specs: list[str] = []
    for i, line in enumerate(lines):
        m = re.match(r"(\s*)MODELS:\s*(.*?)\s*$", line)
        if not m:
            continue
        indent, value = len(m.group(1)), m.group(2)
        if value in {">", ">-", "|", "|-"}:
            block = []
            for nxt in lines[i + 1:]:
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                block.append(nxt.strip())
            value = " ".join(block)
        specs += value.strip("'\"").split()
    return specs


def _pyproject_extras(path: Path) -> list[str]:
    text = path.read_text()
    section = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text, re.M | re.S)
    assert section, "no [project.optional-dependencies] in pyproject.toml"
    body = "\n".join(line.split("#", 1)[0] for line in section.group(1).splitlines())
    return re.findall(r'"([^"]+)"', body)


def _requirements(path: Path) -> list[str]:
    return [line.split("#", 1)[0].split(";", 1)[0].strip() for line in path.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


# Every place the repo pins a snapshotted library, with the libraries each must
# pin -- so a pin moved out of a place, or written in a form the parser skips,
# fails here instead of leaving that place unchecked.
PIN_SITES = {
    "compose.gpu.yml": (_compose_models, {"acestep", "stable_audio_3", "chatterbox"}),
    "compose.gpu.split.yml": (_compose_models, {"acestep", "stable_audio_3", "chatterbox"}),
    "backend/pyproject.toml": (_pyproject_extras, {"stable_audio_3", "chatterbox", "diffusers"}),
    # The split topology's models image: the only one where the diffusers
    # loader can run. (requirements-models.txt allows an older diffusers on
    # purpose -- the unified image never uses that loader.)
    "backend/requirements-models-voice-sfx.txt": (_requirements, {"diffusers"}),
}


@pytest.mark.parametrize("site", sorted(PIN_SITES))
def test_pins_agree_with_the_snapshots(site):
    read, expected = PIN_SITES[site]
    pins = _pins_in(read(REPO / site))
    assert set(pins) == expected, f"{site} pins {sorted(pins)}; expected {sorted(expected)}"
    wanted = {name: pin for name, (_, pin) in _library_pins().items()}
    for name, found in pins.items():
        assert set(found) == {wanted[name]}, (
            f"{site} pins {name} as {sorted(set(found))}, but tests/contracts/{name}.json was taken at "
            f"{wanted[name]!r}, and every place must pin it so (`>=` only for FLOOR_PINNED): change the pin "
            "everywhere, then re-run scripts/snapshot_contracts.py")


def test_diffusers_floor_matches_the_provider_check():
    ref = _snapshots()["diffusers"]["source"]["ref"]
    assert tuple(int(x) for x in ref.split(".")) == stable_audio.SA3_DIFFUSERS_MIN
