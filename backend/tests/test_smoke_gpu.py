"""scripts/smoke_gpu.py: the schedules it waits on, and --idle-check's verdicts.

The script runs with nothing installed, so it cannot import app and keeps its
own copies of two timings. The idle check waits MODEL_IDLE_TIMEOUT plus one
sweep interval (plus slack) for the idle sweep to unload every model: a copy
shorter than the real interval fails a working unload that is merely on
schedule. And after replacing a model-service process the script waits out the
gateway's discovery cache: a copy shorter than the real cache trusts a "ready"
that describes the old process. The first tests compare each copy with the app.

The rest drive the script against a fake stack, sampler and clock. Its VRAM
verdicts only run on the GPU box, which no CI can reach, and each one guards a
way the check could otherwise pass without proving anything.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest
from helpers import FakeProvider

from app.providers.base import Capability

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "smoke_gpu.py"
RERUN = "python3 scripts/smoke_gpu.py --idle-check"
GIB = 1024**3


def _smoke_gpu():
    spec = importlib.util.spec_from_file_location("smoke_gpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Slept(Exception):
    """Ends the sweep loop at its first sleep, before it sweeps anything."""


# 60: the floor wins. 121: timeout // 4 is 30, where true division would give
# 30.25. 300 (the check's ceiling) and 1800 (the default): timeout // 4 wins.
@pytest.mark.parametrize("timeout", [60, 121, 300, 1800])
def test_sweep_interval_matches_the_model_service(monkeypatch, timeout):
    from app import main

    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)
        raise _Slept

    # Only main's view of asyncio: patching asyncio.sleep itself would reach
    # the event loop machinery too.
    monkeypatch.setattr(main, "asyncio", types.SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread))
    monkeypatch.setattr(main.settings, "model_idle_timeout", timeout)
    with pytest.raises(_Slept):
        asyncio.run(main._sweep_idle_models())
    assert slept == [_smoke_gpu().sweep_interval(timeout)]


def test_gateway_cache_copy_matches_the_gateway():
    # The literal default, not app.gateway.MODELS_CACHE_SECONDS: that one
    # follows whatever environment the tests run in.
    m = re.search(r'os\.getenv\("MODELS_CACHE_SECONDS", "([\d.]+)"\)', (ROOT / "backend/app/gateway.py").read_text())
    assert m, "gateway.py no longer reads MODELS_CACHE_SECONDS with a literal default"
    assert float(m.group(1)) == _smoke_gpu().GATEWAY_CACHE_S
    # The default is what runs only while no compose file overrides it.
    for rel in ("docker-compose.yml", "compose.gpu.yml", "compose.gpu.split.yml"):
        assert "MODELS_CACHE_SECONDS" not in (ROOT / rel).read_text(), rel


# -- fakes ----------------------------------------------------------------------

class FakeClock:
    """time.monotonic/time.sleep for the script: sleeping advances the clock.
    The fakes below log what the script asked for, and when, in `events`."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.events: list[tuple[str, float]] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def log(self, what: str) -> None:
        self.events.append((what, self.now))

    def names(self) -> list[str]:
        return [what for what, _ in self.events]

    def first(self, what: str) -> float:
        return next(t for w, t in self.events if w == what)


def probe(allocated: float | None = 0.0, tracked: int = 0) -> dict:
    """A model service's own /health. None for `allocated` means no GPU."""
    gpu = None if allocated is None else {"torch_allocated_gb": allocated, "torch_reserved_gb": allocated}
    return {"jobs": {"tracked": tracked}, "gpu": gpu}


class FakeCompose:
    """The stack. /v1/models answers from a script of loaded-provider lists
    (the last one repeats; None stands for a model service that is down), and
    each service's own /health from `probes` (likewise; None: no answer)."""

    base = ["docker", "compose", "-f", "compose.gpu.yml"]

    def __init__(self, loaded: list, timeout: str = "60", probes: dict | None = None,
                 clock: FakeClock | None = None, restart_rc: int = 0) -> None:
        self.loaded = list(loaded)
        self.timeout = timeout
        self.probes = {svc: list(seq) for svc, seq in (probes or {"models": [probe()]}).items()}
        self.clock = clock
        self.restart_rc = restart_rc

    def _log(self, what: str) -> None:
        if self.clock is not None:
            self.clock.log(what)

    def api_json(self, method, path, body=None, timeout=60):
        self._log(path)
        if path in ("/health", "/health/ready"):
            return 200, {}
        assert (method, path) == ("GET", "/v1/models")
        now = self.loaded.pop(0) if len(self.loaded) > 1 else self.loaded[0]
        if now is None:
            return 200, {"providers": [], "upstreams_down": ["models"]}
        providers = [{"id": p, "loaded": True} for p in now] + [{"id": "never-used", "loaded": False}]
        return 200, {"providers": providers, "upstreams_down": []}

    def run(self, *args, check=True, timeout=None, input=None):
        done = types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if args[0] == "config":
            env = {"MODEL_IDLE_TIMEOUT": self.timeout}
            cfg = {"services": {svc: {"environment": env} for svc in self.probes}}
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(cfg).encode(), stderr=b"")
        self._log(" ".join(args[:3]) if args[0] == "exec" else args[0])
        if args[0] == "restart":
            done.returncode = self.restart_rc
            done.stderr = b"no such service" if self.restart_rc else b""
        elif args[0] == "exec":
            seq = self.probes[args[2]]
            health = seq.pop(0) if len(seq) > 1 else seq[0]
            if health is None:
                done.returncode, done.stderr = 7, b"curl: (7) Failed to connect"
            else:
                done.stdout = json.dumps(health).encode()
        # Anything else is `up` or `ps -q`: no container ids, so started_at()
        # never shells out to docker.
        return done


class FakeVram:
    """VRAM readings in order (the last repeats) and a fixed peak, in GiB.
    Logs each request at the time it asks the reading to postdate."""

    def __init__(self, *readings: float | None, peak: float = 20.0, clock: FakeClock | None = None) -> None:
        self.readings = list(readings)
        self.peak_gib = peak
        self.clock = clock

    def wait_sample(self, since, timeout=30):
        if self.clock is not None:
            self.clock.events.append(("nvidia-smi", since))
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]

    def peak(self, start, end):
        return self.peak_gib


@pytest.fixture
def smoke(monkeypatch):
    module = _smoke_gpu()
    monkeypatch.setattr(module, "time", FakeClock())
    return module


def _run(smoke, tmp_path, compose, vram, dev=False):
    report = smoke.Report(tmp_path)
    check = smoke.IdleCheck(compose, report, vram, list(compose.probes), dev)
    assert check.configure(RERUN)
    check.take_baseline()
    check.wait_for_unload()
    return check.result, report


def _fails(report) -> list[str]:
    return [line for line in report.lines if line.startswith("FAIL")]


# -- the gateway's cache --------------------------------------------------------

def test_a_new_stack_is_not_asked_whether_it_is_ready_within_the_gateway_cache(smoke, tmp_path):
    # `up` may have just recreated the model service (a run with a different
    # MODEL_IDLE_TIMEOUT), and the gateway's cache still describes the old one.
    clock = smoke.time
    compose = FakeCompose([[]], clock=clock)
    report = smoke.Report(tmp_path)
    assert smoke.start_stack(compose, report, tmp_path, build=False)
    assert smoke.check_ready(compose, report, "before any load")
    assert clock.first("/health/ready") - clock.first("up") > smoke.GATEWAY_CACHE_S


def test_a_restarted_stack_is_not_asked_whether_it_is_ready_within_the_gateway_cache(smoke, tmp_path):
    clock = smoke.time
    compose = FakeCompose([[]], clock=clock)
    assert smoke.restart_models(compose, smoke.Report(tmp_path), ["models"], "after the restart")
    assert clock.first("/health/ready") - clock.first("restart") > smoke.GATEWAY_CACHE_S


def test_a_failed_restart_is_a_failure_not_a_crash(smoke, tmp_path):
    report = smoke.Report(tmp_path)
    assert not smoke.restart_models(FakeCompose([[]], restart_rc=1), report, ["models"], "after the restart")
    assert report.failed and "no such service" in report.lines[-1]


# -- the baseline ---------------------------------------------------------------

@pytest.mark.parametrize("services", [["models"], ["models", "music"]])
def test_the_baseline_comes_from_restarted_processes_with_their_cuda_context(smoke, tmp_path, services):
    # A rerun reuses the container the last run left, leak and all, unless the
    # check restarts it. And a process has a CUDA context only once its own
    # /health has run; the baseline must come after that, not race the
    # container healthcheck.
    clock = smoke.time
    compose = FakeCompose([[], ["acestep"], []], probes={svc: [probe()] for svc in services}, clock=clock)
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5, 2.0, clock=clock))
    assert not report.failed, report.lines
    names = clock.names()
    baseline = names.index("nvidia-smi")
    assert names.index("restart") < names.index("/v1/models") < baseline
    for svc in services:
        assert names.index("restart") < names.index(f"exec -T {svc}") < baseline
        # A reading from before the /health would predate the context.
        assert clock.first("nvidia-smi") >= clock.first(f"exec -T {svc}")
    assert result["torch_allocated_baseline_gib"] == {svc: 0.0 for svc in services}


def test_a_failed_restart_takes_no_baseline(smoke, tmp_path):
    compose = FakeCompose([[], ["acestep"], []], restart_rc=1)
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5))
    assert report.failed and not result["unload_observed"] and "vram_baseline_gib" not in result


def test_a_model_loaded_before_the_baseline_fails_it(smoke, tmp_path):
    # Even after the restart: something else generating on this stack would
    # put a model in the baseline and let a model that never unloads pass.
    compose = FakeCompose([["acestep"]])
    result, report = _run(smoke, tmp_path, compose, FakeVram(9.0, 9.0))
    assert report.failed and "vram_baseline_gib" not in result


def test_a_process_that_has_taken_a_job_gives_no_baseline(smoke, tmp_path):
    # A job still loading its model shows nowhere in /v1/models yet.
    compose = FakeCompose([[], ["acestep"], []], probes={"models": [probe(tracked=1)]})
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5))
    assert report.failed and "taken 1 job(s)" in _fails(report)[0]
    assert "vram_baseline_gib" not in result


def test_a_service_whose_own_health_does_not_answer_gives_no_baseline(smoke, tmp_path):
    compose = FakeCompose([[], ["acestep"], []], probes={"models": [probe()], "music": [None]})
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5))
    assert report.failed and "music's own /health did not answer" in _fails(report)[0]


def test_an_image_without_allocator_figures_gives_no_baseline(smoke, tmp_path):
    # Built before /health reported torch_allocated_gb: only nvidia-smi would
    # be left, and the smallest model fits inside its allowance.
    compose = FakeCompose([[], ["acestep"], []], probes={"models": [probe(allocated=None)]})
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5))
    assert report.failed and "rebuild the images" in _fails(report)[0]


# -- the verdict after the unload ----------------------------------------------

def test_vram_back_near_the_baseline_passes(smoke, tmp_path):
    compose = FakeCompose([[], ["acestep", "chatterbox"], ["acestep"], []],
                          probes={"models": [probe(0.0), probe(0.03)]})
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5, 2.3))
    assert not report.failed, report.lines
    assert result["unload_observed"] and result["vram_ok"] is True
    assert result["unloaded"] == ["acestep", "chatterbox"]
    assert (result["vram_baseline_gib"], result["vram_peak_gib"], result["vram_after_unload_gib"]) == (1.5, 20.0, 2.3)
    assert result["vram_residual_gib"] == 0.8 and result["torch_allocated_residual_gib"] == {"models": 0.03}
    assert "+0.8 GiB" in report.lines[-1] and "+0.03 GiB in models" in report.lines[-1]


def test_vram_that_stays_up_after_the_unload_fails(smoke, tmp_path):
    # Every model reports unloaded and PyTorch let go, yet 6 GiB never came
    # back: memory held outside PyTorch's allocator.
    compose = FakeCompose([[], ["acestep"], []])
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5, 7.5))
    assert report.failed and result["vram_ok"] is False
    assert "stayed resident, outside PyTorch's allocator" in report.lines[-1]


@pytest.mark.parametrize("services", [["models"], ["models", "music"]])
def test_a_model_left_referenced_fails_inside_the_device_allowance(smoke, tmp_path, services):
    # SA3 small (459M parameters, ~0.85 GiB at half precision) left
    # referenced after its unload. nvidia-smi's residual stays inside its
    # allowance -- 1 GiB, or 2 GiB with --split's two processes -- so only
    # the per-process PyTorch figure can catch it.
    leak = 0.86
    probes = {svc: [probe(0.0), probe(leak if svc == "models" else 0.0)] for svc in services}
    compose = FakeCompose([[], ["stable-audio-3-sfx"], []], probes=probes)
    device_after = 0.5 + 0.9 * len(services)
    result, report = _run(smoke, tmp_path, compose, FakeVram(0.5, device_after))
    assert report.failed and result["vram_ok"] is False
    assert result["vram_residual_gib"] <= result["vram_tolerance_gib"]
    assert _fails(report) == [
        f"FAIL  idle check: PyTorch in models still holds {leak:.2f} GiB more than at the baseline (allowed "
        f"{smoke.IDLE_TORCH_SLACK_GIB} GiB): a model stayed referenced after its unload"]


def test_a_model_that_never_unloads_fails_at_the_bound(smoke, tmp_path):
    compose = FakeCompose([[], ["acestep"]])
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5))
    assert report.failed and not result["unload_observed"]
    assert "acestep still loaded 150s after the last generation" in report.lines[-1]


def test_a_service_that_is_down_is_not_an_unload(smoke, tmp_path):
    # A down service drops its providers from /v1/models, which looks exactly
    # like "nothing loaded" unless upstreams_down is checked.
    compose = FakeCompose([[], ["acestep"], None])
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5, 1.5))
    assert report.failed and not result["unload_observed"]


def test_a_restart_is_not_an_unload(smoke, tmp_path, monkeypatch):
    # A restarted process frees its VRAM too, without the idle sweep.
    starts = iter([["/models-1 2026-10-03T10:00:00Z"], ["/models-1 2026-10-03T10:02:00Z"]])
    monkeypatch.setattr(smoke, "started_at", lambda compose, services: next(starts))
    compose = FakeCompose([[], ["acestep"], []])
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5, 1.5))
    assert report.failed and not result["unload_observed"]
    assert "restarted" in report.lines[-1]


def test_nothing_loaded_after_the_rounds_proves_nothing(smoke, tmp_path):
    compose = FakeCompose([[], []])
    result, report = _run(smoke, tmp_path, compose, FakeVram(1.5, 1.5))
    assert report.failed and not result["unload_observed"]


def test_dev_reports_vram_as_not_measured_rather_than_passing_it(smoke, tmp_path):
    compose = FakeCompose([[], ["stub-voice"], []], probes={"stub": [probe(allocated=None)]})
    result, report = _run(smoke, tmp_path, compose, FakeVram(None), dev=True)
    assert not report.failed and result["unload_observed"]
    assert result["vram_measured"] is False and result["vram_ok"] is None
    assert "vram_after_unload_gib" not in result


@pytest.mark.parametrize(("timeout", "message"), [
    ("1800", "MODEL_IDLE_TIMEOUT=60 " + RERUN),
    ("0", "disables the idle sweep"),
    ("soon", "not a whole number"),
])
def test_configure_refuses_a_timeout_it_cannot_wait_for(smoke, tmp_path, timeout, message):
    report = smoke.Report(tmp_path)
    check = smoke.IdleCheck(FakeCompose([[]], timeout=timeout), report, FakeVram(1.0), ["models"], False)
    assert not check.configure(RERUN)
    assert report.failed and message in report.lines[-1]


# -- what the check reads from a real model service ------------------------------

def test_the_figures_the_check_reads_are_the_ones_health_reports(api, use_providers, monkeypatch):
    # The script parses the model service's /health by field name; a rename on
    # either side would leave it reading None, which fails every GPU run.
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, mem_get_info=lambda: (20 * GIB, 32 * GIB),
        get_device_name=lambda index: "NVIDIA GeForce RTX 5090",
        memory_allocated=lambda: int(1.5 * GIB), memory_reserved=lambda: 2 * GIB)
    monkeypatch.setitem(sys.modules, "torch", torch)
    use_providers(FakeProvider("acestep", Capability.MUSIC))
    health = api.get("/health").json()
    assert _smoke_gpu().torch_allocated(health) == 1.5
    assert health["gpu"]["torch_reserved_gb"] == 2.0
    assert health["jobs"]["tracked"] == 0
