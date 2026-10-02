"""Process-wide RNG scope (app.rng) and cross-process GPU slots (app.gpu_lock)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

from helpers import reload_settings

from app.gpu_lock import gpu_slot
from app.rng import _RWLock, rng_scope

BACKEND = Path(__file__).resolve().parents[1]


# -- rng_scope ------------------------------------------------------------------

def _in_thread(fn) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def test_unseeded_jobs_share_the_scope():
    lock = _RWLock()
    inside = threading.Barrier(2, timeout=2)

    def reader():
        with lock.shared():
            inside.wait()                     # both readers must be inside at once

    threads = [_in_thread(reader), _in_thread(reader)]
    for t in threads:
        t.join(3)
    assert not any(t.is_alive() for t in threads)


def test_seeded_job_excludes_everything_else():
    lock = _RWLock()
    events: list[str] = []
    release = threading.Event()
    holding = threading.Event()

    def writer():
        with lock.exclusive():
            holding.set()
            release.wait(2)
            events.append("writer done")

    def reader():
        with lock.shared():
            events.append("reader in")

    w = _in_thread(writer)
    holding.wait(2)
    r = _in_thread(reader)
    time.sleep(0.05)
    assert events == []                      # reader is blocked by the writer
    release.set()
    w.join(2)
    r.join(2)
    assert events == ["writer done", "reader in"]


def test_waiting_writer_blocks_new_readers():
    """Writer preference: a stream of unseeded jobs cannot starve a seeded one."""
    lock = _RWLock()
    events: list[str] = []
    first_in = threading.Event()
    release_first = threading.Event()

    def long_reader():
        with lock.shared():
            first_in.set()
            release_first.wait(2)
        events.append("reader1 out")

    def writer():
        with lock.exclusive():
            events.append("writer in")

    def late_reader():
        with lock.shared():
            events.append("reader2 in")

    r1 = _in_thread(long_reader)
    first_in.wait(2)
    w = _in_thread(writer)
    time.sleep(0.05)
    r2 = _in_thread(late_reader)
    time.sleep(0.05)
    assert events == []                      # reader2 queued behind the waiting writer
    release_first.set()
    for t in (r1, w, r2):
        t.join(2)
    assert events.index("writer in") < events.index("reader2 in")


def test_rng_scope_picks_the_mode():
    with rng_scope(seeded=False), rng_scope(seeded=False):
        pass                                 # shared scopes nest
    with rng_scope(seeded=True):
        pass


# -- gpu_slot -------------------------------------------------------------------

def test_gpu_slot_is_a_no_op_without_a_lock_file(monkeypatch):
    monkeypatch.delenv("GPU_LOCK_FILE", raising=False)
    with gpu_slot(on_wait=lambda: (_ for _ in ()).throw(AssertionError("must not wait"))):
        pass


def _hold_slot_in_subprocess(prefix: str, seconds: float) -> subprocess.Popen:
    """Hold one slot from ANOTHER process, as a second model container would."""
    script = textwrap.dedent(f"""
        import sys, time
        from app.gpu_lock import gpu_slot
        with gpu_slot():
            print("held", flush=True)
            time.sleep({seconds})
    """)
    env = {**os.environ, "GPU_LOCK_FILE": prefix, "PYTHONPATH": str(BACKEND)}
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_gpu_slot_waits_for_another_process(tmp_path, monkeypatch):
    prefix = str(tmp_path / "locks" / "gpu")
    reload_settings(monkeypatch, GPU_LOCK_FILE=prefix, MAX_CONCURRENT_JOBS="1")
    other = _hold_slot_in_subprocess(prefix, 0.5)
    waits: list[float] = []
    start = time.monotonic()
    try:
        with gpu_slot(on_wait=lambda: waits.append(time.monotonic())):
            elapsed = time.monotonic() - start
    finally:
        other.wait(5)
    assert len(waits) == 1                   # told the job once that it is waiting
    assert elapsed >= 0.3


def test_gpu_slot_count_follows_max_concurrent_jobs(tmp_path, monkeypatch):
    prefix = str(tmp_path / "gpu")
    reload_settings(monkeypatch, GPU_LOCK_FILE=prefix, MAX_CONCURRENT_JOBS="2")
    other = _hold_slot_in_subprocess(prefix, 0.5)
    try:
        start = time.monotonic()
        with gpu_slot(on_wait=lambda: (_ for _ in ()).throw(AssertionError("second slot was free"))):
            assert time.monotonic() - start < 0.2
    finally:
        other.wait(5)
