#!/usr/bin/env python3
"""Snapshot the third-party APIs the model providers call, from the pinned sources.

The providers in backend/app/providers/{acestep,stable_audio,chatterbox}.py call
libraries that need CUDA torch, so neither dev nor CI can import them. Their
calls were once checked against the pinned sources by throwaway scripts that
are gone, and Chatterbox's never were. This records what the pinned sources
actually define -- every function, method, constructor and config field the
providers touch, with each parameter's name, kind and default -- into
backend/tests/contracts/<library>.json. backend/tests/test_contracts.py builds
fakes from those files and runs the providers' real load and generate paths
against them, offline.

Nothing here imports the libraries or torch: each source archive is downloaded
at its exact pin and parsed with `ast`. Names are resolved the way Python would
resolve them -- through re-exports, relative imports and base classes (C3
order) -- so a method inherited from a mixin is found where it is defined.

The snapshots record only what the source defines. A symbol a provider uses
that the source lacks is written as null and a requested attribute that is
missing is simply absent, so the contract test, not this script, reports the
mismatch.

Output is deterministic: sorted keys, source text rather than ast.unparse
(which differs between Python versions), and no timestamps. Run it twice and
the files are byte-identical; --check regenerates in memory and fails if the
committed snapshots differ.

When a pin changes, change it everywhere it appears (test_contracts.py checks
the compose files and pyproject.toml agree), update LIBRARIES below, re-run
this, and let the contract test say which provider calls broke.

usage: scripts/snapshot_contracts.py            (needs network; stdlib only)
       scripts/snapshot_contracts.py --check
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "backend" / "tests" / "contracts"


# -- what to record ------------------------------------------------------------
#
# One entry per name a provider imports or reads. Classes list only the methods
# and attributes a provider uses: a provider reaching for anything else then
# finds it missing from the snapshot, and the contract test fails until that
# use has been checked against the source here too.

@dataclass(frozen=True)
class Function:
    name: str
    # Local `name = {...}` dict literals whose keys a provider reads off the
    # function's result (ACE-Step returns its audio entries as plain dicts).
    dict_literals: tuple[str, ...] = ()


@dataclass(frozen=True)
class Class:
    name: str
    methods: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Dataclass:
    """Every field, in order: the generated __init__ needs all of them to bind."""
    name: str


@dataclass(frozen=True)
class DictKeys:
    """Keys of a module-level dict literal, for an argument checked against it."""
    name: str


@dataclass(frozen=True)
class Library:
    name: str                 # snapshot file stem
    distribution: str
    ref: str                  # the pin, exactly as the repo writes it
    package: str              # top-level import package inside the archive
    uses: dict[str, tuple]    # import path -> names a provider uses from it
    github: str = ""          # "owner/repo" for a commit archive
    pypi: str = ""            # PyPI project for an sdist


LIBRARIES = [
    Library(
        name="acestep",
        distribution="ACE-Step-1.5",
        github="ace-step/ACE-Step-1.5",
        ref="ca1e85fe9430",
        package="acestep",
        uses={
            "acestep.handler": (Class("AceStepHandler", methods=("__init__", "initialize_service")),),
            "acestep.llm_inference": (Class("LLMHandler", methods=("__init__", "initialize")),),
            "acestep.inference": (
                Dataclass("GenerationParams"),
                Dataclass("GenerationConfig"),
                Dataclass("GenerationResult"),
                Function("generate_music", dict_literals=("audio_dict",)),
            ),
        },
    ),
    Library(
        name="stable_audio_3",
        distribution="stable-audio-3",
        github="Stability-AI/stable-audio-3",
        ref="779434a90819",
        package="stable_audio_3",
        uses={
            "stable_audio_3": (
                Class("StableAudioModel", methods=("from_pretrained", "generate"), attributes=("model",)),
            ),
            # The class of StableAudioModel.model, whose sample_rate the provider reads.
            "stable_audio_3.models.diffusion": (Class("ConditionedDiffusionModelWrapper", attributes=("sample_rate",)),),
            # from_pretrained raises ValueError for any model name not in here.
            "stable_audio_3.model_configs": (DictKeys("models"),),
        },
    ),
    Library(
        name="chatterbox",
        distribution="chatterbox-tts",
        pypi="chatterbox-tts",
        ref="0.1.7",
        package="chatterbox",
        uses={
            "chatterbox.tts": (
                Class("ChatterboxTTS", methods=("from_pretrained", "generate"), attributes=("sr",)),
            ),
        },
    ),
    # The Stable Audio 3 provider's fallback loader. Not pinned exactly: the
    # repo declares a floor (diffusers>=0.40.0, the first release with
    # StableAudio3Pipeline), and the floor is what this snapshots.
    Library(
        name="diffusers",
        distribution="diffusers",
        pypi="diffusers",
        ref="0.40.0",
        package="diffusers",
        uses={
            "diffusers": (
                Class("StableAudio3Pipeline", methods=("from_pretrained", "to", "__call__"), attributes=("vae",)),
            ),
            # The class of StableAudio3Pipeline.vae, whose sampling_rate the
            # provider reads (a config field, forwarded as an attribute).
            "diffusers.models.autoencoders.autoencoder_same": (Class("AutoencoderSAME", attributes=("sampling_rate",)),),
            # What StableAudio3Pipeline.__call__ returns.
            "diffusers.pipelines.pipeline_utils": (Dataclass("AudioPipelineOutput"),),
        },
    ),
]


# -- fetching ------------------------------------------------------------------

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "snapshot_contracts.py"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch(lib: Library) -> tuple[str, bytes]:
    """(url, archive bytes) for the library at its pin."""
    if lib.github:
        url = f"https://github.com/{lib.github}/archive/{lib.ref}.tar.gz"
        return url, _get(url)
    meta = json.loads(_get(f"https://pypi.org/pypi/{lib.pypi}/{lib.ref}/json"))
    sdists = [u for u in meta["urls"] if u["packagetype"] == "sdist"]
    if len(sdists) != 1:
        raise SystemExit(f"{lib.pypi}=={lib.ref}: expected one sdist on PyPI, found {len(sdists)}")
    url, expected = sdists[0]["url"], sdists[0]["digests"]["sha256"]
    data = _get(url)
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"{url}: sha256 does not match the digest PyPI publishes")
    return url, data


# -- the source tree -----------------------------------------------------------

class Source:
    """The package's modules, read straight out of the archive."""

    def __init__(self, archive: bytes, package: str) -> None:
        files: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    files[member.name] = tar.extractfile(member).read().decode("utf-8")
        inits = sorted((p for p in files if p.endswith(f"/{package}/__init__.py")), key=lambda p: (p.count("/"), p))
        if not inits:
            raise SystemExit(f"no {package}/__init__.py in the archive")
        if len(inits) > 1 and inits[0].count("/") == inits[1].count("/"):
            raise SystemExit(f"ambiguous package root for {package}: {inits[:2]}")
        # Paths are recorded relative to the archive's top directory, so they
        # match the file paths in the upstream repo or sdist at that ref.
        self.top = inits[0].split("/", 1)[0] + "/"
        base = inits[0][: -len(f"{package}/__init__.py")]
        self.paths: dict[str, str] = {}
        self.text: dict[str, str] = {}
        for path, text in files.items():
            if not path.startswith(base + package + "/"):
                continue
            parts = path[len(base):-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module = ".".join(parts)
            self.paths[module] = path[len(self.top):]
            self.text[module] = text
        self._trees: dict[str, ast.Module] = {}

    def tree(self, module: str) -> ast.Module:
        if module not in self._trees:
            self._trees[module] = ast.parse(self.text[module], filename=self.paths[module])
        return self._trees[module]

    def is_package(self, module: str) -> bool:
        return self.paths[module].endswith("__init__.py")

    def segment(self, module: str, node: ast.AST | None) -> str | None:
        """The node's exact source text, whitespace-collapsed."""
        if node is None:
            return None
        text = ast.get_source_segment(self.text[module], node)
        return " ".join(text.split()) if text is not None else None

    def line(self, module: str, lineno: int) -> str:
        return self.text[module].splitlines()[lineno - 1].strip()


def _module_level(stmts):
    """Statements that run at import time: through if/try/with, not into defs."""
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for attr in ("body", "orelse", "finalbody"):
            yield from _module_level(getattr(stmt, attr, []))
        for handler in getattr(stmt, "handlers", []):
            yield from _module_level(handler.body)


def _defines(stmt: ast.stmt, name: str) -> bool:
    """True if this statement binds `name` itself (not by importing it)."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name == name
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return isinstance(stmt.target, ast.Name) and stmt.target.id == name
    return False


@dataclass
class Found:
    module: str
    node: ast.AST


class Resolver:
    def __init__(self, src: Source) -> None:
        self.src = src

    def _absolute(self, module: str, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        parts = module.split(".")
        if not self.src.is_package(module):
            parts = parts[:-1]
        parts = parts[: len(parts) - (node.level - 1)]
        return ".".join(parts + ([node.module] if node.module else []))

    def resolve(self, module: str, name: str, seen: frozenset = frozenset()) -> Found | None:
        """Where `from module import name` really leads, inside this package.

        Explicit definitions and imports only. Star imports are ignored: in
        diffusers they pull in the dummy placeholder classes that stand in for
        the real ones when torch is missing.
        """
        if (module, name) in seen or module not in self.src.text:
            return None
        seen = seen | {(module, name)}
        defs, imports = [], []
        for stmt in _module_level(self.src.tree(module).body):
            if _defines(stmt, name):
                defs.append(stmt)
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name and alias.name != "*":
                        imports.append((self._absolute(module, stmt), alias.name))
        if defs:
            return Found(module, defs[-1])
        results: dict[tuple, Found] = {}
        for source_module, source_name in imports:
            found = self.resolve(source_module, source_name, seen)
            submodule = f"{source_module}.{source_name}"
            if found is None and submodule in self.src.text:
                found = Found(submodule, self.src.tree(submodule))
            if found is not None:
                results[(found.module, getattr(found.node, "lineno", 0))] = found
        if len(results) > 1:
            raise SystemExit(f"{module}.{name} resolves to more than one definition: {sorted(results)}")
        return next(iter(results.values()), None)

    def _base(self, module: str, expr: ast.expr) -> Found | None:
        if isinstance(expr, ast.Name):
            return self.resolve(module, expr.id)
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            mod = self.resolve(module, expr.value.id)
            if mod is not None and isinstance(mod.node, ast.Module):
                return self.resolve(mod.module, expr.attr)
        return None                        # outside the package (torch.nn.Module, ...)

    def mro(self, cls: Found) -> list[Found]:
        """C3 linearisation over the bases defined inside the package.

        Bases outside it (torch.nn.Module, ...) are left out: a method only they
        define is then reported missing rather than guessed at.
        """
        bases = [b for b in (self._base(cls.module, e) for e in cls.node.bases)
                 if b is not None and isinstance(b.node, ast.ClassDef)]
        seqs = [s for s in [*(self.mro(b) for b in bases), bases] if s]
        out = [cls]
        while seqs:
            tails = {_key(f) for s in seqs for f in s[1:]}
            head = next((s[0] for s in seqs if _key(s[0]) not in tails), None)
            if head is None:
                raise SystemExit(f"inconsistent MRO for {cls.node.name}")
            out.append(head)
            seqs = [rest for rest in ([f for f in s if _key(f) != _key(head)] for s in seqs) if rest]
        return out


def _key(found: Found) -> tuple[str, int]:
    return found.module, found.node.lineno


# -- recording -----------------------------------------------------------------

_KINDS = {
    "posonlyargs": "POSITIONAL_ONLY",
    "args": "POSITIONAL_OR_KEYWORD",
    "vararg": "VAR_POSITIONAL",
    "kwonlyargs": "KEYWORD_ONLY",
    "kwarg": "VAR_KEYWORD",
}


def _decorator_names(src: Source, module: str, node) -> list[str]:
    return [src.segment(module, d) for d in node.decorator_list]


def _params(src: Source, module: str, fn) -> list[dict]:
    a = fn.args
    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    out = []

    def param(arg, kind, default):
        return {
            "name": arg.arg,
            "kind": kind,
            "annotation": src.segment(module, arg.annotation),
            # Source text of the default, or null when the parameter has none.
            "default": src.segment(module, default),
        }

    for i, arg in enumerate(positional):
        out.append(param(arg, _KINDS["posonlyargs"] if i < len(a.posonlyargs) else _KINDS["args"], defaults[i]))
    if a.vararg:
        out.append(param(a.vararg, _KINDS["vararg"], None))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append(param(arg, _KINDS["kwonlyargs"], default))
    if a.kwarg:
        out.append(param(a.kwarg, _KINDS["kwarg"], None))
    return out


def _kwargs_consumed(fn) -> list[str]:
    """Names a **kwargs function reads out of its catch-all by literal key.

    Binding cannot check a keyword that lands in **kwargs; this list can. A
    name the function never reads is at best forwarded somewhere unchecked and
    at worst ignored, so the contract test treats it as a mismatch.
    """
    if not fn.args.kwarg:
        return []
    kw = fn.args.kwarg.arg
    names = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == kw
                and node.func.attr in {"pop", "get", "setdefault"}
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value)
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == kw
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            names.add(node.slice.value)
    return sorted(names)


def _callable(src: Source, found: Found, binding: str) -> dict:
    fn = found.node
    entry = {
        "binding": binding,
        "defined_in": src.paths[found.module],
        "line": fn.lineno,
        "decorators": _decorator_names(src, found.module, fn),
        "params": _params(src, found.module, fn),
        "returns": src.segment(found.module, fn.returns),
    }
    if fn.args.kwarg:
        entry["kwargs_consumed"] = _kwargs_consumed(fn)
    return entry


def _dict_keys(src: Source, module: str, value: ast.expr, what: str) -> list[str]:
    if not isinstance(value, ast.Dict) or not all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                                  for k in value.keys):
        raise SystemExit(f"{what} is not a dict literal with string keys")
    return sorted(k.value for k in value.keys)


def record_function(src: Source, found: Found, use: Function) -> dict:
    entry = {"kind": "function", **_callable(src, found, "function")}
    if use.dict_literals:
        literals = {}
        for var in use.dict_literals:
            keysets = {
                tuple(_dict_keys(src, found.module, node.value, f"{use.name}.{var}"))
                for node in ast.walk(found.node)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets)
            }
            if len(keysets) != 1:
                raise SystemExit(f"{use.name}: expected one `{var} = {{...}}`, found {len(keysets)} shapes")
            literals[var] = list(keysets.pop())
        entry["dict_literals"] = literals
    return entry


def _binding(node) -> str:
    names = {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
    if "staticmethod" in names:
        return "staticmethod"
    if "classmethod" in names:
        return "classmethod"
    return "method"


def _attribute_evidence(src: Source, mro: list[Found], attr: str) -> dict | None:
    """Where the class (or a base) defines `attr`, in the three ways these
    libraries do it: `self.attr = ...`; a class-body assignment; or, in
    diffusers, a component passed to self.register_modules(attr=...) or a
    parameter of an @register_to_config __init__ -- diffusers stores those in
    the config, and ModelMixin.__getattr__ forwards config names to the object.
    """
    for cls in mro:
        module = cls.module
        for stmt in cls.node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _defines(stmt, attr):
                return {"defined_in": src.paths[module], "line": stmt.lineno, "via": "class attribute",
                        "source": src.line(module, stmt.lineno)}
        methods = [s for s in cls.node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        # __init__ first: an attribute it sets exists on every instance.
        methods.sort(key=lambda m: (m.name != "__init__", m.lineno))
        for method in methods:
            if method.name == "__init__" and any(src.segment(module, d) == "register_to_config"
                                                 for d in method.decorator_list):
                for arg in method.args.args + method.args.kwonlyargs:
                    if arg.arg == attr:
                        return {"defined_in": src.paths[module], "line": arg.lineno, "via": "register_to_config",
                                "source": src.line(module, arg.lineno)}
            for node in ast.walk(method):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    targets = [node.target]
                for t in targets:
                    if (isinstance(t, ast.Attribute) and t.attr == attr
                            and isinstance(t.value, ast.Name) and t.value.id == "self"):
                        return {"defined_in": src.paths[module], "line": node.lineno,
                                "via": f"self.{attr} in {method.name}", "source": src.line(module, node.lineno)}
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "register_modules"):
                    for kw in node.keywords:
                        if kw.arg == attr:
                            return {"defined_in": src.paths[module], "line": kw.value.lineno,
                                    "via": "register_modules", "source": src.line(module, kw.value.lineno)}
    return None


def record_class(src: Source, resolver: Resolver, found: Found, use: Class) -> dict:
    mro = resolver.mro(found)
    entry = {
        "kind": "class",
        "defined_in": src.paths[found.module],
        "line": found.node.lineno,
        "methods": {},
        "attributes": {},
    }
    for name in use.methods:
        for cls in mro:
            method = next((s for s in cls.node.body
                           if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == name), None)
            if method is not None:
                entry["methods"][name] = _callable(src, Found(cls.module, method), _binding(method))
                break
        else:
            # Recorded as missing rather than raised: the contract test should
            # be the one to report that a provider calls something absent.
            print(f"  ! {found.node.name}.{name} is not defined in the pinned source", file=sys.stderr)
            entry["methods"][name] = None
    for attr in use.attributes:
        evidence = _attribute_evidence(src, mro, attr)
        if evidence is None:
            print(f"  ! {found.node.name}.{attr} is not defined in the pinned source", file=sys.stderr)
        else:
            entry["attributes"][attr] = evidence
    return entry


def _is_dataclass(src: Source, module: str, node: ast.ClassDef) -> dict | None:
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if src.segment(module, target) in {"dataclass", "dataclasses.dataclass"}:
            options = {kw.arg: src.segment(module, kw.value) for kw in d.keywords} if isinstance(d, ast.Call) else {}
            return options
    return None


def record_dataclass(src: Source, resolver: Resolver, found: Found) -> dict:
    options = _is_dataclass(src, found.module, found.node)
    if options is None:
        raise SystemExit(f"{found.node.name} is not a @dataclass")
    fields_by_name: dict[str, dict] = {}
    # Base dataclass fields come first, in MRO-reversed order, as dataclasses does.
    for cls in reversed(resolver.mro(found)):
        cls_options = _is_dataclass(src, cls.module, cls.node)
        if cls_options is None:
            continue
        kw_only = cls_options.get("kw_only") == "True"
        for stmt in cls.node.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            annotation = src.segment(cls.module, stmt.annotation)
            if annotation.startswith(("ClassVar", "typing.ClassVar", "tp.ClassVar")):
                continue
            value = stmt.value
            init = True
            if (isinstance(value, ast.Call) and src.segment(cls.module, value.func) in {"field", "dataclasses.field"}):
                kws = {kw.arg: kw.value for kw in value.keywords}
                init = not (isinstance(kws.get("init"), ast.Constant) and kws["init"].value is False)
            fields_by_name[stmt.target.id] = {
                "name": stmt.target.id,
                "annotation": annotation,
                "default": src.segment(cls.module, value),
                "init": init,
                "kw_only": kw_only,
                "line": stmt.lineno,
            }
    return {
        "kind": "dataclass",
        "defined_in": src.paths[found.module],
        "line": found.node.lineno,
        "options": options,
        "fields": list(fields_by_name.values()),
    }


def snapshot(lib: Library) -> dict:
    url, archive = fetch(lib)
    src = Source(archive, lib.package)
    resolver = Resolver(src)
    modules: dict[str, dict] = {}
    for module, uses in lib.uses.items():
        entries = modules.setdefault(module, {})
        for use in uses:
            found = resolver.resolve(module, use.name)
            if found is None:
                print(f"  ! {module}.{use.name} is not defined in the pinned source", file=sys.stderr)
                entries[use.name] = None
                continue
            node = found.node
            if isinstance(use, Function) and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entries[use.name] = record_function(src, found, use)
            elif isinstance(use, Class) and isinstance(node, ast.ClassDef):
                entries[use.name] = record_class(src, resolver, found, use)
            elif isinstance(use, Dataclass) and isinstance(node, ast.ClassDef):
                entries[use.name] = record_dataclass(src, resolver, found)
            elif isinstance(use, DictKeys) and isinstance(node, (ast.Assign, ast.AnnAssign)):
                entries[use.name] = {
                    "kind": "dict",
                    "defined_in": src.paths[found.module],
                    "line": node.lineno,
                    "keys": _dict_keys(src, found.module, node.value, f"{module}.{use.name}"),
                }
            else:
                raise SystemExit(f"{module}.{use.name}: expected a {type(use).__name__}, found {type(node).__name__}")
    return {
        "library": lib.name,
        "source": {
            "distribution": lib.distribution,
            "ref": lib.ref,
            "url": url,
            "sha256": hashlib.sha256(archive).hexdigest(),
        },
        "generated_by": "scripts/snapshot_contracts.py",
        "modules": modules,
    }


def render(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and fail if the committed snapshots differ")
    args = parser.parse_args()

    stale = []
    for lib in LIBRARIES:
        print(f"{lib.distribution} @ {lib.ref}", file=sys.stderr)
        text = render(snapshot(lib))
        path = OUT_DIR / f"{lib.name}.json"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.relative_to(ROOT))
        else:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"  wrote {path.relative_to(ROOT)}", file=sys.stderr)
    if stale:
        print("stale snapshots (re-run scripts/snapshot_contracts.py): " + ", ".join(map(str, stale)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
