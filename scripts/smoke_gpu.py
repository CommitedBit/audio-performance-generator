#!/usr/bin/env python3
"""First real run on the GPU box: build, start, and generate one clip per track
type with each required local model, timing every load and recording VRAM.

    scripts/smoke_gpu.py              # unified topology (compose.gpu.yml)
    scripts/smoke_gpu.py --split      # + compose.gpu.split.yml
    scripts/smoke_gpu.py --no-build   # reuse images already built
    scripts/smoke_gpu.py --quick      # skip the restart / cold-from-disk round
    scripts/smoke_gpu.py --dev        # exercise this script on the local stub stack

Runs on the VM, or from the Mac with DOCKER_CONTEXT=gpu: every request goes
through `docker compose exec`, so the loopback-only ports need no tunnel, and
the API key is read from the gateway container's own environment rather than
passed on this machine's command line. Needs only python3 (stdlib) and docker.

Three rounds per model, so the timings separate the costs:
    first  download + load + generate    (a fresh volume pulls ~31 GB)
    warm   generate only                 (model resident)
    cold   load from disk + generate     (after restarting the model services)
so download ~= first - cold and load ~= cold - warm.

Writes smoke-results/<timestamp>/: the clips, report.txt, results.json (the
input for VRAM budgeting) and, on any failure, the compose logs.
"""
from __future__ import annotations

import argparse
import array
import datetime as dt
import io
import json
import math
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234
CASES = [
    # capability, provider, request body, expected duration range (s)
    ("voice", "chatterbox",
     {"input": "The lighthouse keeper climbed the stairs one last time, and the lamp was already lit."},
     (1.0, 30.0)),
    ("music", "acestep", {"prompt": "slow ambient piano, warm, gentle, 70 bpm", "seconds": 20}, (19.0, 21.0)),
    ("sfx", "stable-audio-3-sfx", {"prompt": "a heavy wooden door creaks open slowly", "seconds": 4}, (3.5, 4.5)),
]
FIRST_TIMEOUT = 60 * 60          # a first run may download tens of GB
LATER_TIMEOUT = 20 * 60
RMS_FLOOR = 1e-3


class Compose:
    def __init__(self, files: list[str]) -> None:
        self.base = ["docker", "compose"] + [a for f in files for a in ("-f", f)]

    def run(self, *args: str, check: bool = True, timeout: float | None = None,
            input: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([*self.base, *args], cwd=ROOT, capture_output=True, check=check,
                              timeout=timeout, input=input)

    def api(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> tuple[int, bytes]:
        """One HTTP call from inside the gateway container."""
        script = ('exec curl -sS -w "\\n%{http_code}" -X "$1" -H "Content-Type: application/json" '
                  '-H "X-API-Key: ${API_KEY:-}" --max-time "$2" "http://127.0.0.1:8000$3" ${4:+--data-binary @-}')
        args = ["exec", "-T", "gateway", "sh", "-c", script, "sh", method, str(int(timeout)), path]
        data = None
        if body is not None:
            args.append("1")
            data = json.dumps(body).encode()
        r = self.run(*args, check=False, timeout=timeout + 30, input=data)
        if r.returncode != 0 and not r.stdout:
            raise RuntimeError(f"{method} {path}: {r.stderr.decode(errors='replace').strip()}")
        payload, _, status = r.stdout.rpartition(b"\n")
        return int(status or 0), payload

    def api_json(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> tuple[int, dict]:
        status, payload = self.api(method, path, body, timeout)
        try:
            return status, json.loads(payload or b"{}")
        except json.JSONDecodeError:
            return status, {"raw": payload[:500].decode(errors="replace")}


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.failed = False
        self.lines: list[str] = []

    def say(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)
        (self.out / "report.txt").write_text("\n".join(self.lines) + "\n")

    def ok(self, msg: str) -> None:
        self.say(f"ok    {msg}")

    def fail(self, msg: str) -> None:
        self.failed = True
        self.say(f"FAIL  {msg}")


class VramSampler(threading.Thread):
    """Streams nvidia-smi from the models container; survives its restarts."""

    def __init__(self, compose: Compose) -> None:
        super().__init__(daemon=True)
        self.compose = compose
        self.samples: list[tuple[float, float, float]] = []    # (time, used GiB, total GiB)
        self.stop = threading.Event()

    def run(self) -> None:
        cmd = [*self.compose.base, "exec", "-T", "models", "nvidia-smi",
               "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-lms", "1000"]
        while not self.stop.is_set():
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:
                if self.stop.is_set():
                    break
                try:
                    used, total = (float(x) / 1024 for x in line.split(","))
                except ValueError:
                    continue
                self.samples.append((time.monotonic(), used, total))
            proc.kill()
            self.stop.wait(2)

    def peak(self, start: float, end: float) -> float | None:
        window = [used for t, used, _ in self.samples if start <= t <= end]
        return round(max(window), 2) if window else None

    def latest(self) -> float | None:
        return round(self.samples[-1][1], 2) if self.samples else None


def analyse_wav(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels, rate, width, n = wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getnframes()
        frames = wf.readframes(n)
    if width != 2:
        return {"channels": channels, "sample_rate": rate, "duration": n / rate, "rms": None}
    samples = array.array("h", frames)
    step = max(1, len(samples) // 400_000)          # plenty for an RMS estimate
    picked = samples[::step]
    rms = math.sqrt(sum(s * s for s in picked) / max(1, len(picked))) / 32768.0
    return {"channels": channels, "sample_rate": rate, "duration": round(n / rate, 3), "rms": round(rms, 5),
            "peak": round(max((abs(s) for s in picked), default=0) / 32768.0, 4)}


def wait_live(compose: Compose, report: Report, timeout: float = 600) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, body = compose.api_json("GET", "/health", timeout=15)
            if status == 200:
                return True
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        time.sleep(5)
    report.fail(f"gateway /health did not answer within {timeout:.0f}s")
    return False


def check_ready(compose: Compose, report: Report, when: str, wait: float = 180) -> bool:
    """Poll /health/ready. The gateway answers before the model services
    finish starting, and they report "missing" until they do."""
    deadline = time.monotonic() + wait
    while True:
        status, body = compose.api_json("GET", "/health/ready", timeout=30)
        if status == 200:
            report.ok(f"/health/ready {when}")
            return True
        if time.monotonic() > deadline:
            report.fail(f"/health/ready {when}: {status} {'; '.join(body.get('problems', [])) or body}")
            return False
        time.sleep(5)


def run_case(compose: Compose, report: Report, vram: VramSampler, out: Path,
             case: tuple, round_name: str, timeout: float) -> dict:
    capability, provider, body, (lo, hi) = case
    body = {**body, "provider": provider, "seed": SEED}
    path = "/v1/audio/speech?wait=false" if capability == "voice" else f"/v1/audio/{capability}"
    label = f"{round_name:5} {capability:5} {provider}"
    result: dict = {"round": round_name, "capability": capability, "provider": provider}

    start = time.monotonic()
    status, job = compose.api_json("POST", path, body, timeout=120)
    if status not in (200, 202) or "id" not in job:
        report.fail(f"{label}: submit returned {status}: {job.get('detail', job)}")
        return result
    job_id = job["id"]
    while job.get("status") in ("queued", "running"):
        if time.monotonic() - start > timeout:
            report.fail(f"{label}: job {job_id} still {job['status']} after {timeout:.0f}s")
            return result
        time.sleep(3)
        status, job = compose.api_json("GET", f"/v1/jobs/{job_id}", timeout=30)
        if status != 200:
            report.fail(f"{label}: polling returned {status}: {job.get('detail', job)}")
            return result
    # Server-side timestamps: exact, unlike this loop's 3 s polling. `seconds`
    # covers load + generate; queue wait is reported separately.
    if job.get("finished_at") and job.get("started_at"):
        elapsed = job["finished_at"] - job["started_at"]
        result["queued_s"] = round(job["started_at"] - job["created_at"], 2)
    else:
        elapsed = time.monotonic() - start
    result.update(seconds=round(elapsed, 2), vram_peak_gb=vram.peak(start, time.monotonic()),
                  vram_after_gb=vram.latest())

    if job.get("status") != "done":
        report.fail(f"{label}: job {job['status']}: {job.get('error')}")
        return result
    served_by = job.get("meta", {}).get("provider")
    if served_by != provider:
        report.fail(f"{label}: served by {served_by}, expected {provider}")
        return result

    status, audio = compose.api("GET", job["audio_url"], timeout=120)
    if status != 200 or not audio:
        report.fail(f"{label}: audio fetch returned {status}")
        return result
    suffix = ".mp3" if job.get("meta", {}).get("mime") == "audio/mpeg" else ".wav"
    clip = out / f"{capability}-{round_name}{suffix}"
    clip.write_bytes(audio)
    result["file"] = clip.name
    if suffix == ".wav":
        info = analyse_wav(audio)
        result.update(info)
        problems = []
        if not lo <= info["duration"] <= hi:
            problems.append(f"duration {info['duration']}s outside {lo}-{hi}s")
        if info["rms"] is not None and info["rms"] < RMS_FLOOR:
            problems.append(f"rms {info['rms']} below {RMS_FLOOR}")
        if problems:
            report.fail(f"{label}: {'; '.join(problems)}")
            return result
    result["ok"] = True
    report.ok(f"{label}: {elapsed:6.1f}s  {result.get('duration')}s audio  rms {result.get('rms')}  "
              f"vram peak {result['vram_peak_gb']} GiB")
    return result


def preflight(compose: Compose, report: Report, image_tag: str) -> bool:
    report.say("== preflight")
    r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", f"nvidia/cuda:{image_tag}", "nvidia-smi",
                        "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        report.fail(f"no GPU inside containers (fix the NVIDIA container toolkit first): {r.stderr.strip()[-300:]}")
        return False
    report.ok(f"GPU visible in containers: {r.stdout.strip()}")

    root = subprocess.run(["docker", "info", "--format", "{{.DockerRootDir}}"], capture_output=True, text=True)
    df = subprocess.run(["docker", "run", "--rm", "-v", f"{root.stdout.strip()}:/r:ro", "busybox", "df", "-Pk", "/r"],
                        capture_output=True, text=True)
    try:
        free_gb = int(df.stdout.splitlines()[-1].split()[3]) / 1024**2
        (report.ok if free_gb >= 80 else report.fail)(
            f"{free_gb:.0f} GB free under Docker's root (weights ~31 GB + images ~20 GB; want >= 80)")
    except (IndexError, ValueError):
        report.say("??    could not read free disk space")

    cfg = json.loads(compose.run("config", "--format", "json").stdout)
    env = cfg["services"]["models"].get("environment") or {}
    if env.get("HF_TOKEN"):
        report.ok("HF_TOKEN is set for the models service")
    else:
        report.fail("HF_TOKEN is empty: Stable Audio 3 weights are gated (set it in .env)")
    sa3 = env.get("SA3_MODEL") or "stabilityai/stable-audio-3-small-sfx"
    report.say(f"note  accept the licence for https://huggingface.co/{sa3 if '/' in sa3 else 'stabilityai/stable-audio-3-' + sa3}"
               " with the account that owns HF_TOKEN, or its download is refused")
    return not report.failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="store_true", help="use the split topology")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild images")
    ap.add_argument("--quick", action="store_true", help="skip the restart / cold-from-disk round")
    ap.add_argument("--dev", action="store_true",
                    help="run against docker-compose.yml's placeholder stack: tests this script, not the models")
    a = ap.parse_args()

    if a.dev:
        files = ["docker-compose.yml"]
        CASES[:] = [(cap, f"stub-{cap}", body, rng) for cap, _, body, rng in CASES]
    else:
        files = ["compose.gpu.yml"] + (["compose.gpu.split.yml"] if a.split else [])
    compose = Compose(files)
    out = ROOT / "smoke-results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    report = Report(out)
    report.say(f"smoke run {out.name}: {' + '.join(files)}")

    if not a.dev:
        cfg = json.loads(compose.run("config", "--format", "json").stdout)
        cuda_tag = cfg["services"]["models"]["build"]["args"]["CUDA_TAG"].replace("cudnn-runtime", "base")
        if not preflight(compose, report, cuda_tag):
            return 1

    report.say("== build and start")
    up = ["up", "-d"] + ([] if a.no_build else ["--build"])
    started = time.monotonic()
    r = compose.run(*up, check=False)
    (out / "compose-up.log").write_bytes(r.stdout + r.stderr)
    if r.returncode != 0:
        report.fail(f"docker compose {' '.join(up)} failed; see compose-up.log")
        return 1
    report.ok(f"stack up in {time.monotonic() - started:.0f}s")
    if not wait_live(compose, report):
        return finish(compose, report, out, [])

    vram = VramSampler(compose)
    if not a.dev:
        vram.start()
    results: list[dict] = []
    try:
        # Readiness first: every required model must at least be installable here.
        if check_ready(compose, report, "before any load"):
            report.say("== round 1: download + load + generate")
            results += [run_case(compose, report, vram, out, c, "first", FIRST_TIMEOUT) for c in CASES]
            report.say("== round 2: generate with the model resident")
            results += [run_case(compose, report, vram, out, c, "warm", LATER_TIMEOUT) for c in CASES]
            if not a.quick:
                report.say("== round 3: restart the model services, load from disk")
                services = ["stub"] if a.dev else ["models"] + (["music"] if a.split else [])
                compose.run("restart", *services)
                if wait_live(compose, report) and check_ready(compose, report, "after the restart"):
                    results += [run_case(compose, report, vram, out, c, "cold", LATER_TIMEOUT) for c in CASES]
            check_ready(compose, report, "after every model has loaded", wait=0)
    finally:
        vram.stop.set()
    return finish(compose, report, out, results)


def finish(compose: Compose, report: Report, out: Path, results: list[dict]) -> int:
    status, models = compose.api_json("GET", "/v1/models", timeout=30)
    loaded = sorted(p["id"] for p in models.get("providers", []) if p.get("loaded"))
    report.say(f"loaded at the end: {', '.join(loaded) or 'none'}")

    by = {(r["round"], r["provider"]): r for r in results}
    report.say("")
    def cell(value) -> str:
        return "-" if value is None else str(value)

    report.say(f"{'model':22} {'first s':>8} {'warm s':>7} {'cold s':>7} {'~load s':>8} {'peak GiB':>9}")
    for _, provider, _, _ in CASES:
        first, warm, cold = (by.get((n, provider), {}).get("seconds") for n in ("first", "warm", "cold"))
        load = round(cold - warm, 2) if cold is not None and warm is not None else None
        peaks = [r["vram_peak_gb"] for r in results if r["provider"] == provider and r.get("vram_peak_gb")]
        peak = max(peaks) if peaks else None
        report.say(f"{provider:22} {cell(first):>8} {cell(warm):>7} {cell(cold):>7} {cell(load):>8} {cell(peak):>9}")
    (out / "results.json").write_text(json.dumps({"results": results}, indent=2))

    if report.failed:
        logs = compose.run("logs", "--no-color", "--timestamps", check=False)
        (out / "compose-logs.txt").write_bytes(logs.stdout + logs.stderr)
        report.say(f"\nFAILED. Logs saved to {out / 'compose-logs.txt'}")
        return 1
    report.say(f"\nPASSED. Listen to the clips in {out} -- automated checks cannot judge quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
