#!/usr/bin/env python3
"""Resolve every GPU image's dependency set the way backend/Dockerfile installs it.

Catches, without a GPU or a multi-GB build:
  * a requirement set that no longer resolves (conflicting pins);
  * a package that would have to be BUILT from an sdist on the GPU box -- the
    failure mode that once broke the image (onnx needing cmake);
  * a TORCH_PIN the override cannot satisfy on the image's own index;
  * a toolchain that cannot run on Blackwell (sm_120 needs CUDA >= 12.8, cu128+).

Build args are read from `docker compose config`, so this checks what compose
actually builds -- not a copy of it that could drift.

The Dockerfile installs `.` and the requirement set in two steps; this resolves
them together, which is stricter: uv does not re-check already-installed
packages, so a conflict between the API's deps and a model set would otherwise
slip through.

usage: scripts/check_deps.py            (needs docker compose and uv)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
TOPOLOGIES = {
    "unified": ["compose.gpu.yml"],
    "split": ["compose.gpu.yml", "compose.gpu.split.yml"],
}
PLATFORM = "x86_64-manylinux_2_28"
# An index that is down answers with these. uv retries each request itself; a
# whole compile is retried after a pause before the index is declared
# unreachable. Without this a 503 from download.pytorch.org surfaced as "the
# image would build a package from source" -- a failure about the code that was
# really about the network.
NETWORK_ERRORS = ("Failed to fetch", "Request failed after", "HTTP status server error",
                  "error sending request", "operation timed out", "Connection reset")
COMPILE_ATTEMPTS = 3
RETRY_PAUSE = 20.0


class IndexUnreachable(Exception):
    """A package index could not be reached; says nothing about the pins."""


# Pure-Python packages published only as sdists; building them needs no
# toolchain. Anything else that would need building is an error.
SDIST_ALLOWED = ["antlr4-python3-runtime"]


def dockerfile_defaults() -> dict[str, str]:
    """ARG defaults, for build args compose does not set."""
    out = {}
    for line in (BACKEND / "Dockerfile").read_text().splitlines():
        m = re.match(r"\s*ARG\s+(\w+)=(.*)$", line)
        if m:
            out.setdefault(m.group(1), m.group(2).strip().strip('"'))
    return out


def gpu_images() -> dict[tuple, list[str]]:
    """Distinct GPU build configurations -> where each is used."""
    defaults = dockerfile_defaults()
    images: dict[tuple, list[str]] = {}
    for topology, files in TOPOLOGIES.items():
        cmd = ["docker", "compose"] + [a for f in files for a in ("-f", f)] + ["config", "--format", "json"]
        cfg = json.loads(subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout)
        for name, svc in cfg["services"].items():
            args = {**defaults, **((svc.get("build") or {}).get("args") or {})}
            if not svc.get("build") or args.get("BASE") != "cuda":
                continue
            key = (args["CUDA_TAG"], args["TORCH_INDEX"], args["PYTHON_VERSION"], args["TORCH_PIN"],
                   args["MODEL_REQUIREMENTS"], args.get("EXTRAS", ""))
            images.setdefault(key, []).append(f"{topology}/{name}")
    return images


def blackwell_problems(cuda_tag: str, torch_index: str, torch_pin: str) -> list[str]:
    problems = []
    m = re.match(r"(\d+)\.(\d+)", cuda_tag)
    if not m or (int(m.group(1)), int(m.group(2))) < (12, 8):
        problems.append(f"CUDA_TAG {cuda_tag} is older than 12.8 (no sm_120)")
    m = re.match(r"cu(\d+)$", torch_index)
    if not m or int(m.group(1)) < 128:
        problems.append(f"TORCH_INDEX {torch_index} is older than cu128 (no sm_120)")
    for spec in torch_pin.split():
        local = spec.split("+", 1)[1] if "+" in spec else ""
        if local and local != torch_index:
            problems.append(f"TORCH_PIN {spec} does not match TORCH_INDEX {torch_index}")
    return problems


def _compile(inputs: Path, overrides: Path, out: Path, torch_index: str, python: str,
             *, wheel_only: bool) -> tuple[bool, str]:
    cmd = [
        "uv", "pip", "compile", str(inputs), "--overrides", str(overrides),
        "--python-version", python, "--python-platform", PLATFORM,
        "--index-strategy", "unsafe-best-match",
        "--extra-index-url", f"https://download.pytorch.org/whl/{torch_index}",
        "--no-header", "--no-annotate", "--quiet", "-o", str(out),
    ]
    if wheel_only:
        cmd += ["--only-binary", ":all:", *[a for p in SDIST_ALLOWED for a in ("--no-binary", p)]]
    for attempt in range(1, COMPILE_ATTEMPTS + 1):
        r = subprocess.run(cmd, cwd=BACKEND, capture_output=True, text=True)
        err = (r.stderr or r.stdout).strip()
        if r.returncode == 0 or not any(marker in err for marker in NETWORK_ERRORS):
            return r.returncode == 0, err
        if attempt < COMPILE_ATTEMPTS:
            print(f"      index unreachable (attempt {attempt}/{COMPILE_ATTEMPTS}); retrying in "
                  f"{RETRY_PAUSE * attempt:.0f}s", file=sys.stderr)
            time.sleep(RETRY_PAUSE * attempt)
    raise IndexUnreachable(err)


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def resolve(torch_index: str, python: str, torch_pin: str, requirements: str, extras: str) -> tuple[bool, str]:
    """Resolve as the image installs, then again wheel-only; both must agree.

    Comparing matters more than the wheel-only resolve succeeding: when a
    package exists only as an sdist, a wheel-only resolver does not fail -- it
    quietly backtracks to older versions of whatever needed it (omegaconf
    2.3.1 -> 2.0.6 here), "passing" while describing a different environment
    from the one the image gets.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        overrides = tmp / "overrides.txt"
        overrides.write_text("\n".join(torch_pin.split()) + "\n")
        # Absolute paths: entries in a requirements file resolve relative to
        # that file, which lives in a temp dir.
        project = f"audio-performance-generator-api[{extras}] @ {BACKEND.as_uri()}" if extras else str(BACKEND)
        inputs = tmp / "inputs.txt"
        # The pins are inputs as well as overrides: the Dockerfile installs
        # them outright first, and an override alone never adds a package.
        inputs.write_text("\n".join([project, f"-r {BACKEND / requirements}", *torch_pin.split()]) + "\n")

        try:
            ok, err = _compile(inputs, overrides, tmp / "image.txt", torch_index, python, wheel_only=False)
            if not ok:
                return False, f"does not resolve:\n{err}"
            image = _pins(tmp / "image.txt")
            ok, err = _compile(inputs, overrides, tmp / "wheels.txt", torch_index, python, wheel_only=True)
        except IndexUnreachable as exc:
            return False, ("a package index stayed unreachable after "
                           f"{COMPILE_ATTEMPTS} attempts -- a network problem, not a pin problem; "
                           f"re-run the check:\n{exc}")
        wheels = _pins(tmp / "wheels.txt") if ok else {}

    if not ok or wheels != image:
        changed = sorted(
            f"{name}: image gets {image.get(name)}, wheels-only {wheels.get(name)}"
            for name in set(image) | set(wheels) if image.get(name) != wheels.get(name)
        )
        detail = "\n".join(changed) if ok else err
        return False, ("the image would build a package from source (add it to SDIST_ALLOWED only if it is "
                       f"pure Python):\n{detail}")

    report = []
    for spec in torch_pin.split():
        name, want = spec.split("==", 1)
        got = image.get(name.lower())
        if got != want:
            return False, f"{name} resolved to {got}, but TORCH_PIN asks for {want}"
        report.append(f"{name}=={got}")
    return True, ", ".join(report) + f" ({len(image)} packages)"


def main() -> int:
    failed = 0
    for (cuda_tag, torch_index, python, torch_pin, requirements, extras), used_by in gpu_images().items():
        label = f"{requirements} [{', '.join(used_by)}]"
        problems = blackwell_problems(cuda_tag, torch_index, torch_pin)
        ok, detail = resolve(torch_index, python, torch_pin, requirements, extras)
        if problems or not ok:
            failed += 1
            print(f"FAIL  {label}")
            for p in problems:
                print(f"      {p}")
            if not ok:
                print("      " + detail.replace("\n", "\n      "))
        else:
            print(f"ok    {label}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
