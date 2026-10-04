#!/usr/bin/env python3
"""First real run on the GPU box: build, start, and generate one clip per track
type with each required local model, timing every load and recording VRAM.

    scripts/smoke_gpu.py              # unified topology (compose.gpu.yml)
    scripts/smoke_gpu.py --split      # + compose.gpu.split.yml
    scripts/smoke_gpu.py --no-build   # reuse images already built
    scripts/smoke_gpu.py --quick      # skip the restart / cold-from-disk round
    scripts/smoke_gpu.py --dev        # exercise this script on the local stub stack
    scripts/smoke_gpu.py --idle-check # also check an idle unload gives the VRAM back

Runs on the VM, or from the Mac with DOCKER_CONTEXT=gpu: every request goes
through `docker compose exec`, so the loopback-only ports need no tunnel, and
the API key is read from the gateway container's own environment rather than
passed on this machine's command line. Needs only python3 (stdlib) and docker.

Three rounds per model, so the timings separate the costs:
    first  download + load + generate    (a fresh volume pulls ~31 GB)
    warm   generate only                 (model resident)
    cold   load from disk + generate     (after restarting the model services)
so download ~= first - cold and load ~= cold - warm.

--idle-check (opt-in) restarts the model services and takes a baseline before
any model loads, waits after the last round until /v1/models lists nothing
loaded, and then requires each process's PyTorch allocations back within
IDLE_TORCH_SLACK_GIB of the baseline, and the card's VRAM within
IDLE_VRAM_SLACK_GIB per model service. The production MODEL_IDLE_TIMEOUT
(1800 s) is far too long to wait for, so give the run a short one; compose
passes the shell's value through:
    MODEL_IDLE_TIMEOUT=60 python3 scripts/smoke_gpu.py --idle-check
The stack keeps that timeout until it is next started without it. With --dev
the stubs load and unload like real models but hold no VRAM, so only the
unload is checked.

Writes smoke-results/<timestamp>/, or the new directory --out names: the
clips, report.txt, results.json (the input for VRAM budgeting) and, on any
failure, the compose logs.
"""
from __future__ import annotations

import argparse
import array
import datetime as dt
import io
import json
import math
import shlex
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
# The gateway answers /health/ready and /v1/models from discovery up to this
# many seconds old (MODELS_CACHE_SECONDS in backend/app/gateway.py; a copy, as
# this script cannot import app, pinned by backend/tests/test_smoke_gpu.py).
# Right after `up` recreates a model service, or `restart` replaces one, the
# gateway can still describe the OLD process: ready, with its models loaded.
GATEWAY_CACHE_S = 3

# --idle-check waits for the model service's own idle sweep
# (_sweep_idle_models in backend/app/main.py): it sleeps
# max(SWEEP_MIN_INTERVAL, timeout // 4), then unloads every model idle for longer
# than MODEL_IDLE_TIMEOUT. A copy, because this script runs with nothing
# installed and cannot import app; backend/tests/test_smoke_gpu.py fails if the
# two drift apart, which would make the check give up on an unload that is
# merely on schedule.
SWEEP_MIN_INTERVAL = 30
# Above this the check refuses to start. It would otherwise sit for half an hour
# at the production default before learning anything.
IDLE_TIMEOUT_CEILING = 300
# Waited on top of MODEL_IDLE_TIMEOUT + one sweep interval: the sweep's own run
# (gc and empty_cache per model), the gateway's discovery cache, and this
# script's 5 s polling.
IDLE_SLACK = 60
# Two allowances for what an unload may leave behind, because one figure cannot
# tell a model left resident from what every process keeps anyway.
#
# PyTorch's allocator, per model-service process, read from that service's own
# /health (torch_allocated_gb). A model unloaded in name only keeps all its
# weights here, and the smallest checkpoint, SA3 small (459M parameters, see
# stable_audio.py), is about 0.85 GiB at half precision. What legitimately stays
# is small: the cuBLAS workspaces PyTorch keeps per thread, MiB to tens of MiB
# each. So this is the check that catches a model left resident.
IDLE_TORCH_SLACK_GIB = 0.25
# The whole card as nvidia-smi sees it, per model-service process (the split
# topology runs two). The baseline includes each process's CUDA context
# (take_baseline asks every service's /health, which creates it); this covers
# the kernel images and library handles a process picks up while generating
# and keeps until it exits, which is per process. SA3 small fits under it, so
# this check alone cannot catch a model left resident; it catches memory held
# outside PyTorch's allocator.
#
# Neither is measured on the 5090 yet: the report prints both residuals, so the
# first run can confirm or tune them.
IDLE_VRAM_SLACK_GIB = 1.0
# nvidia-smi trails the frees by a reading or two, and /v1/models reports a
# model unloaded as soon as the sweep drops it, before its gc.collect and
# empty_cache have run.
VRAM_SETTLE = 30


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

    def wait_sample(self, since: float, timeout: float = 30) -> float | None:
        """The newest reading taken after `since`, waiting up to `timeout` for
        one. A stale reading would describe the GPU before the event."""
        deadline = time.monotonic() + timeout
        while True:
            if self.samples and self.samples[-1][0] >= since:
                return round(self.samples[-1][1], 2)
            if time.monotonic() > deadline:
                return None
            time.sleep(1)


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


def wait_out_gateway_cache() -> None:
    """Call after anything that may replace a model-service process, before
    asking the gateway about it (see GATEWAY_CACHE_S). A run chained within
    seconds of one with another MODEL_IDLE_TIMEOUT got "ready" and the old
    process's loaded models from the cache, while `up` was recreating the
    service; every submit then failed with a 502."""
    time.sleep(GATEWAY_CACHE_S + 1)


def start_stack(compose: Compose, report: Report, out: Path, build: bool) -> bool:
    report.say("== build and start")
    up = ["up", "-d"] + (["--build"] if build else [])
    started = time.monotonic()
    r = compose.run(*up, check=False)
    (out / "compose-up.log").write_bytes(r.stdout + r.stderr)
    if r.returncode != 0:
        report.fail(f"docker compose {' '.join(up)} failed; see compose-up.log")
        return False
    report.ok(f"stack up in {time.monotonic() - started:.0f}s")
    # `up` recreates a service whose image or settings changed, e.g. a run with
    # a different MODEL_IDLE_TIMEOUT from the last one.
    wait_out_gateway_cache()
    return True


def restart_models(compose: Compose, report: Report, services: list[str], when: str) -> bool:
    """Restart the model services, then wait until the stack is ready again."""
    r = compose.run("restart", *services, check=False)
    if r.returncode != 0:
        report.fail(f"docker compose restart {' '.join(services)} failed: "
                    f"{r.stderr.decode(errors='replace').strip()[-300:]}")
        return False
    wait_out_gateway_cache()
    return wait_live(compose, report) and check_ready(compose, report, when)


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


def model_services(dev: bool, split: bool) -> list[str]:
    """The services that run models: restarted for the cold round, and swept
    for idle models."""
    return ["stub"] if dev else ["models"] + (["music"] if split else [])


def sweep_interval(timeout: int) -> int:
    """Seconds between the model service's idle sweeps (see SWEEP_MIN_INTERVAL)."""
    return max(SWEEP_MIN_INTERVAL, timeout // 4)


def loaded_providers(compose: Compose) -> list[str] | None:
    """The providers /v1/models reports loaded, or None when that is unknown.
    A model service that is down drops its providers from the list, which
    would otherwise read as "nothing loaded"."""
    try:
        status, models = compose.api_json("GET", "/v1/models", timeout=30)
    except (RuntimeError, subprocess.TimeoutExpired):
        return None
    if status != 200 or models.get("upstreams_down"):
        return None
    return sorted(p["id"] for p in models.get("providers", []) if p.get("loaded"))


def started_at(compose: Compose, services: list[str]) -> list[str]:
    """Each model-service container's start time. A restart changes it, and a
    restart also empties VRAM, which must not pass for an idle unload."""
    ids = compose.run("ps", "-q", *services, check=False).stdout.decode().split()
    if not ids:
        return []
    r = subprocess.run(["docker", "inspect", "--format", "{{.Name}} {{.State.StartedAt}}", *ids],
                       capture_output=True, text=True)
    return sorted(r.stdout.splitlines())


def service_health(compose: Compose, service: str) -> dict | None:
    """A model service's own /health, asked inside its container: the gateway's
    /health carries no GPU or job figures. Calling it also creates the
    process's CUDA context (torch.cuda.mem_get_info), which otherwise appears
    whenever the container healthcheck first runs."""
    try:
        r = compose.run("exec", "-T", service, "curl", "-fsS", "--max-time", "15",
                        "http://127.0.0.1:8000/health", check=False, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def torch_allocated(health: dict | None) -> float | None:
    """GiB PyTorch's allocator holds in that process, or None when /health
    reports no figure (no GPU, or an image built before the field existed)."""
    return ((health or {}).get("gpu") or {}).get("torch_allocated_gb")


class IdleCheck:
    """--idle-check: after the rounds, wait for the idle sweep to unload every
    model, then require the VRAM to come back down.

    The rounds only record peaks. Whether an unload really frees the memory is
    a separate question -- a model still referenced from a cycle or a library
    cache unloads in name only -- and the VRAM budget (M3) rests on the answer.
    """

    def __init__(self, compose: Compose, report: Report, vram: VramSampler, services: list[str],
                 dev: bool) -> None:
        self.compose, self.report, self.vram, self.services, self.dev = compose, report, vram, services, dev
        self.timeout = 0
        self.baseline_at: float | None = None
        self.result: dict = {"services": services, "vram_measured": not dev, "unload_observed": False,
                             "vram_ok": None}

    def configure(self, rerun: str) -> bool:
        """Read the timeout compose gives each model service. Runs before the
        build, so a timeout too long to wait for fails in seconds, not after
        an hour of first-run downloads."""
        cfg = json.loads(self.compose.run("config", "--format", "json").stdout)
        timeouts: dict[str, int] = {}
        for svc in self.services:
            raw = (cfg["services"][svc].get("environment") or {}).get("MODEL_IDLE_TIMEOUT")
            if raw is None:
                self.report.fail(f"idle check: compose gives {svc} no MODEL_IDLE_TIMEOUT, so it runs at the app's "
                                 "default and this run cannot shorten it; add "
                                 "MODEL_IDLE_TIMEOUT: ${MODEL_IDLE_TIMEOUT:-1800} to its environment")
                return False
            try:
                timeouts[svc] = int(raw)
            except ValueError:
                self.report.fail(f"idle check: MODEL_IDLE_TIMEOUT={raw!r} for {svc} is not a whole number of seconds")
                return False
        if min(timeouts.values()) <= 0:
            self.report.fail(f"idle check: MODEL_IDLE_TIMEOUT={min(timeouts.values())} disables the idle sweep, "
                             "so nothing would unload")
            return False
        self.timeout = max(timeouts.values())
        interval = sweep_interval(self.timeout)
        bound = self.timeout + interval + IDLE_SLACK
        if self.timeout > IDLE_TIMEOUT_CEILING:
            self.report.fail(
                f"idle check: MODEL_IDLE_TIMEOUT is {self.timeout}s, so an unload could take up to {bound}s, and "
                f"this check waits for timeouts up to {IDLE_TIMEOUT_CEILING}s only. Set a short one in the shell "
                f"for this run (compose passes it through): MODEL_IDLE_TIMEOUT=60 {rerun}")
            return False
        self.result.update(model_idle_timeout_s=self.timeout, sweep_interval_s=interval, wait_bound_s=bound,
                           vram_tolerance_gib=round(IDLE_VRAM_SLACK_GIB * len(self.services), 2),
                           torch_allocated_tolerance_gib=IDLE_TORCH_SLACK_GIB)
        self.report.ok(f"idle check: MODEL_IDLE_TIMEOUT {self.timeout}s and a sweep every {interval}s, so every "
                       f"model should unload within {bound}s of the last generation")
        return True

    def take_baseline(self) -> None:
        """VRAM in model-service processes that have never loaded a model:
        what an unload should return to.

        The services are restarted first. `up -d` keeps a running container,
        so a rerun would otherwise inherit the process the last run left: its
        models swept, so nothing reads as loaded, yet whatever those unloads
        left resident -- the leak this check looks for -- sits in the baseline,
        and the same leak then passes."""
        self.report.say("== idle check: restart the model services for a clean baseline")
        if not restart_models(self.compose, self.report, self.services, "after the idle check's restart"):
            return
        deadline = time.monotonic() + 30
        loaded = loaded_providers(self.compose)
        while loaded is None and time.monotonic() < deadline:
            time.sleep(5)
            loaded = loaded_providers(self.compose)
        if loaded is None:
            self.report.fail("idle check: /v1/models did not answer for every model service, so there is no baseline")
            return
        if loaded:
            self.report.fail(f"idle check: {', '.join(loaded)} loaded since the restart, before the first round, so "
                             "there is no clean baseline; is something else using this stack?")
            return
        allocated: dict[str, float | None] = {}
        for svc in self.services:
            health = service_health(self.compose, svc)
            if health is None:
                self.report.fail(f"idle check: {svc}'s own /health did not answer, so there is no baseline")
                return
            # Counted from the restart. A job means a model may be loading in
            # this process already, which /v1/models shows only once it is done.
            tracked = (health.get("jobs") or {}).get("tracked")
            if tracked:
                self.report.fail(f"idle check: {svc} has taken {tracked} job(s) since the restart, before the first "
                                 "round, so there is no clean baseline; is something else using this stack?")
                return
            allocated[svc] = torch_allocated(health)
        # After every /health above, so the reading includes each CUDA context.
        probed_at = time.monotonic()
        if self.dev:
            self.baseline_at = probed_at
            self.report.say("note  idle check: VRAM not measured (--dev has no GPU); only the unload is checked")
            return
        missing = [svc for svc, gib in allocated.items() if gib is None]
        if missing:
            self.report.fail(f"idle check: /health of {', '.join(missing)} gives no torch_allocated_gb, so a model "
                             "left resident could pass; rebuild the images (run without --no-build)")
            return
        base = self.vram.wait_sample(since=probed_at)
        if base is None:
            self.report.fail("idle check: nvidia-smi in the models container gave no reading within 30s")
            return
        self.baseline_at = probed_at
        self.result.update(vram_baseline_gib=base, torch_allocated_baseline_gib=allocated)
        self.report.ok(f"idle check: VRAM baseline {base} GiB with nothing loaded; PyTorch holds "
                       + ", ".join(f"{gib} GiB in {svc}" for svc, gib in allocated.items()))

    def wait_for_unload(self) -> None:
        if self.baseline_at is None:
            self.report.say("skip  idle check: no baseline to compare with (see above)")
            return
        bound = self.result["wait_bound_s"]
        self.report.say(f"== idle check: wait up to {bound}s for the idle sweep to unload every model")
        start = time.monotonic()
        before = started_at(self.compose, self.services)
        first: list[str] | None = None
        pending: list[str] = []
        while True:
            loaded = loaded_providers(self.compose)
            if loaded is not None:
                if first is None:
                    first = loaded
                    if not first:
                        self.report.fail("idle check: no model was loaded after the rounds, so there was no "
                                         "unload to observe")
                        return
                if not loaded:
                    break
                pending = loaded
            if time.monotonic() - start > bound:
                what = ", ".join(pending) + " still loaded" if pending else "no answer from every model service"
                self.report.fail(f"idle check: {what} {bound}s after the last generation (MODEL_IDLE_TIMEOUT "
                                 f"{self.timeout}s + one {self.result['sweep_interval_s']}s sweep + {IDLE_SLACK}s)")
                return
            time.sleep(5)
        unloaded_at = time.monotonic()
        if started_at(self.compose, self.services) != before:
            self.report.fail("idle check: a model service restarted during the wait; its models went with the "
                             "process, not with the idle sweep")
            return
        waited = round(unloaded_at - start)
        self.result.update(unloaded=first, unload_wait_s=waited, unload_observed=True)
        self.report.ok(f"idle check: {', '.join(first)} unloaded {waited}s after the last generation")
        if self.dev:
            self.report.say("note  idle check: VRAM not measured (--dev has no GPU)")
            return

        base = self.result["vram_baseline_gib"]
        torch_base = self.result["torch_allocated_baseline_gib"]
        allowed = self.result["vram_tolerance_gib"]

        def read() -> tuple[float | None, float | None, dict[str, float | None]]:
            """(VRAM now, its residual, each process's PyTorch residual); None where unread."""
            after = self.vram.wait_sample(since=unloaded_at)
            held = {svc: torch_allocated(service_health(self.compose, svc)) for svc in self.services}
            return (after, None if after is None else round(after - base, 2),
                    {svc: None if gib is None else round(gib - torch_base[svc], 2) for svc, gib in held.items()})

        after, residual, torch_residual = read()
        while (residual is None or residual > allowed
               or any(r is None or r > IDLE_TORCH_SLACK_GIB for r in torch_residual.values())):
            if time.monotonic() >= unloaded_at + VRAM_SETTLE:
                break
            time.sleep(2)
            after, residual, torch_residual = read()
        peak = self.vram.peak(self.baseline_at, unloaded_at)
        self.result.update(vram_peak_gib=peak, vram_after_unload_gib=after, vram_residual_gib=residual,
                           torch_allocated_residual_gib=torch_residual)

        over, unknown = [], []
        for svc, r in torch_residual.items():
            if r is None:
                unknown.append(f"{svc}'s own /health gave no PyTorch figure after the unload")
            elif r > IDLE_TORCH_SLACK_GIB:
                over.append(f"PyTorch in {svc} still holds {r:.2f} GiB more than at the baseline (allowed "
                            f"{IDLE_TORCH_SLACK_GIB} GiB): a model stayed referenced after its unload")
        if residual is None:
            unknown.append("nvidia-smi gave no reading after the unload")
        elif residual > allowed:
            where = "" if over else ", outside PyTorch's allocator"
            over.append(f"VRAM {after} GiB after the unload, {residual:.2f} GiB above the {base} GiB baseline "
                        f"(allowed {allowed} GiB): memory stayed resident{where}")
        self.result["vram_ok"] = False if over else None if unknown else True
        for problem in over + unknown:
            self.report.fail(f"idle check: {problem}")
        if not over and not unknown:
            self.report.ok(f"idle check: VRAM back to {after} GiB after the unload, +{residual} GiB on the {base} GiB "
                           f"baseline (allowed +{allowed}, peak {peak}); PyTorch "
                           + ", ".join(f"+{r} GiB in {svc}" for svc, r in torch_residual.items())
                           + f" (allowed +{IDLE_TORCH_SLACK_GIB} each)")


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


def output_dir(out: str | None) -> Path:
    """Where this run writes: --out, or a new timestamped smoke-results/ directory."""
    if out:
        return Path(out)
    return ROOT / "smoke-results" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="store_true", help="use the split topology")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild images")
    ap.add_argument("--quick", action="store_true", help="skip the restart / cold-from-disk round")
    ap.add_argument("--dev", action="store_true",
                    help="run against docker-compose.yml's placeholder stack: tests this script, not the models")
    ap.add_argument("--idle-check", action="store_true",
                    help="after the rounds, wait for the idle sweep to unload every model and check the VRAM comes "
                         f"back down; needs MODEL_IDLE_TIMEOUT <= {IDLE_TIMEOUT_CEILING} in the environment")
    ap.add_argument("--out", metavar="DIR",
                    help="write this run's files to DIR, which must not exist yet "
                         "(default: smoke-results/<timestamp>)")
    a = ap.parse_args()
    out = output_dir(a.out)
    # scripts/check.sh deletes a passing dev run's --out directory, so never
    # adopt one that may already hold someone's results.
    if out.exists():
        ap.error(f"--out {out} already exists")

    if a.dev:
        files = ["docker-compose.yml"]
        CASES[:] = [(cap, f"stub-{cap}", body, rng) for cap, _, body, rng in CASES]
    else:
        files = ["compose.gpu.yml"] + (["compose.gpu.split.yml"] if a.split else [])
    compose = Compose(files)
    services = model_services(a.dev, a.split)
    out.mkdir(parents=True)
    report = Report(out)
    report.say(f"smoke run {out.name}: {' + '.join(files)}")

    vram = VramSampler(compose)
    idle = IdleCheck(compose, report, vram, services, a.dev) if a.idle_check else None
    if idle is not None:
        report.say("== idle check: settings")
        if not idle.configure(rerun=f"python3 scripts/smoke_gpu.py {shlex.join(sys.argv[1:])}"):
            return 1
        # A short timeout also unloads models BETWEEN rounds, whenever one sits
        # idle while another loads, so "warm" may include a reload and no peak
        # shows every model resident at once.
        report.say("note  idle check: the short timeout can unload models between rounds, so the timings and "
                   "peaks below are not resident-model figures; take those from a run without --idle-check")

    if not a.dev:
        cfg = json.loads(compose.run("config", "--format", "json").stdout)
        cuda_tag = cfg["services"]["models"]["build"]["args"]["CUDA_TAG"].replace("cudnn-runtime", "base")
        if not preflight(compose, report, cuda_tag):
            return 1

    if not start_stack(compose, report, out, build=not a.no_build):
        return 1
    if not wait_live(compose, report):
        return finish(compose, report, out, [], idle)

    if not a.dev:
        vram.start()
    results: list[dict] = []
    try:
        # Readiness first: every required model must at least be installable here.
        if check_ready(compose, report, "before any load"):
            if idle is not None:
                idle.take_baseline()
            report.say("== round 1: download + load + generate")
            results += [run_case(compose, report, vram, out, c, "first", FIRST_TIMEOUT) for c in CASES]
            report.say("== round 2: generate with the model resident")
            results += [run_case(compose, report, vram, out, c, "warm", LATER_TIMEOUT) for c in CASES]
            if not a.quick:
                report.say("== round 3: restart the model services, load from disk")
                if restart_models(compose, report, services, "after the restart"):
                    results += [run_case(compose, report, vram, out, c, "cold", LATER_TIMEOUT) for c in CASES]
            check_ready(compose, report, "after every model has loaded", wait=0)
            if idle is not None:
                idle.wait_for_unload()
    finally:
        vram.stop.set()
    return finish(compose, report, out, results, idle)


def finish(compose: Compose, report: Report, out: Path, results: list[dict], idle: IdleCheck | None) -> int:
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
    payload: dict = {"results": results}
    if idle is not None:
        payload["idle_check"] = idle.result
        r = idle.result
        report.say("")
        if r["vram_measured"]:
            report.say(f"idle check VRAM GiB: baseline {cell(r.get('vram_baseline_gib'))}  peak "
                       f"{cell(r.get('vram_peak_gib'))}  after unload {cell(r.get('vram_after_unload_gib'))}  "
                       f"residual {cell(r.get('vram_residual_gib'))}  allowed +{cell(r.get('vram_tolerance_gib'))}")
            held = r.get("torch_allocated_residual_gib") or {}
            report.say("idle check PyTorch residual GiB: "
                       + ("  ".join(f"{svc} {cell(gib)}" for svc, gib in held.items()) or "-")
                       + f"  allowed +{r['torch_allocated_tolerance_gib']} each")
        else:
            report.say("idle check VRAM: not measured (--dev has no GPU); unload "
                       + ("observed" if r["unload_observed"] else "NOT observed"))
        if idle.timeout:
            report.say(f"note  the stack keeps MODEL_IDLE_TIMEOUT={idle.timeout} until it is started without it: "
                       f"{' '.join(compose.base)} up -d")
    (out / "results.json").write_text(json.dumps(payload, indent=2))

    if report.failed:
        logs = compose.run("logs", "--no-color", "--timestamps", check=False)
        (out / "compose-logs.txt").write_bytes(logs.stdout + logs.stderr)
        report.say(f"\nFAILED. Logs saved to {out / 'compose-logs.txt'}")
        return 1
    report.say(f"\nPASSED. Listen to the clips in {out} -- automated checks cannot judge quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
