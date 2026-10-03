"""The in-process job queue (app.jobs): lanes, cancellation, errors, eviction."""
from __future__ import annotations

import asyncio
import threading

from app.jobs import REMOTE_CONCURRENCY, JobQueue, JobStatus


def _blocking(gate: threading.Event, result: dict | None = None):
    def fn(job):
        gate.wait(5)
        return result or {"audio_id": "a"}
    return fn


async def _until(predicate, timeout: float = 5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


async def test_local_lane_is_serialised():
    q = JobQueue(max_concurrent=1)
    gate = threading.Event()
    first = q.submit("music", _blocking(gate))
    second = q.submit("music", _blocking(gate))
    await _until(lambda: first.status is JobStatus.RUNNING)
    await asyncio.sleep(0.05)
    assert second.status is JobStatus.QUEUED
    gate.set()
    await q.wait_for(second, timeout=5)
    assert (first.status, second.status) == (JobStatus.DONE, JobStatus.DONE)
    assert second.started_at >= first.finished_at


async def test_remote_lane_does_not_wait_behind_local_work():
    q = JobQueue(max_concurrent=1)
    gate = threading.Event()
    local = q.submit("music", _blocking(gate))
    await _until(lambda: local.status is JobStatus.RUNNING)
    remote = q.submit("speech", lambda job: {"audio_id": "r"}, remote=True)
    await q.wait_for(remote, timeout=5)
    assert remote.status is JobStatus.DONE
    assert local.status is JobStatus.RUNNING
    assert remote.lane == "remote" and local.lane == "local"
    gate.set()
    await q.wait_for(local, timeout=5)


async def test_remote_lane_is_bounded():
    q = JobQueue(max_concurrent=1)
    gate = threading.Event()
    jobs = [q.submit("speech", _blocking(gate), remote=True) for _ in range(REMOTE_CONCURRENCY + 1)]
    await _until(lambda: sum(j.status is JobStatus.RUNNING for j in jobs) == REMOTE_CONCURRENCY)
    await asyncio.sleep(0.05)
    assert sum(j.status is JobStatus.RUNNING for j in jobs) == REMOTE_CONCURRENCY
    gate.set()
    for j in jobs:
        await q.wait_for(j, timeout=5)


async def test_cancel_only_works_while_queued():
    q = JobQueue(max_concurrent=1)
    gate = threading.Event()
    ran: list[str] = []
    running = q.submit("music", _blocking(gate))
    queued = q.submit("music", lambda job: ran.append("ran") or {})
    await _until(lambda: running.status is JobStatus.RUNNING)
    assert not q.cancel(running.id)
    assert q.cancel(queued.id)
    assert queued.status is JobStatus.CANCELLED
    assert not q.cancel(queued.id)
    gate.set()
    await q.wait_for(running, timeout=5)
    await asyncio.sleep(0.05)
    assert ran == []


async def test_errors_keep_the_exception_type():
    q = JobQueue()

    def fails(job):
        raise MemoryError()

    def fails_with_text(job):
        raise ValueError("bad prompt")

    a, b = q.submit("music", fails), q.submit("music", fails_with_text)
    await q.wait_for(a, timeout=5)
    await q.wait_for(b, timeout=5)
    assert (a.status, a.error) == (JobStatus.ERROR, "MemoryError")
    assert b.error == "ValueError: bad prompt"


async def test_wait_for_timeout_returns_the_live_job():
    q = JobQueue()
    gate = threading.Event()
    job = q.submit("music", _blocking(gate))
    returned = await q.wait_for(job, timeout=0.05)
    assert returned is job and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
    gate.set()
    await q.wait_for(job, timeout=5)


async def test_result_meta_is_merged_and_listed_newest_first():
    q = JobQueue()
    a = q.submit("speech", lambda job: {"audio_id": "x", "meta": {"duration": 1.5}}, meta={"provider": "p"})
    await q.wait_for(a, timeout=5)
    b = q.submit("speech", lambda job: {"audio_id": "y"})
    await q.wait_for(b, timeout=5)
    assert a.meta["provider"] == "p" and a.meta["duration"] == 1.5
    assert [j["id"] for j in q.list(10)] == [b.id, a.id]
    assert q.stats()["by_status"] == {"done": 2}


async def test_eviction_never_drops_live_jobs():
    q = JobQueue(max_concurrent=1, retain=2)
    gate = threading.Event()
    live = q.submit("music", _blocking(gate))
    await _until(lambda: live.status is JobStatus.RUNNING)
    for _ in range(3):
        q.submit("speech", lambda job: {}, remote=True)
    assert q.get(live.id) is live
    gate.set()
    await q.wait_for(live, timeout=5)


async def test_eviction_keeps_order_and_bound_behind_a_live_job():
    """A long job at the head of the list must neither move nor loosen the bound.

    _evict used to re-append a live oldest job and stop: the running job jumped
    to the top of GET /v1/jobs as if it were the newest, and one job more than
    `retain` stayed in memory. The position cycles as jobs arrive, so the state
    is checked after every submission, not just at the end.
    """
    q = JobQueue(max_concurrent=1, retain=3)
    gate = threading.Event()
    live = q.submit("music", _blocking(gate))
    await _until(lambda: live.status is JobStatus.RUNNING)
    for n in range(6):
        job = q.submit("speech", lambda job: {"audio_id": "x"}, remote=True)
        await q.wait_for(job, timeout=5)
        listed = q.list(50)
        created = [j["created_at"] for j in listed]
        assert created == sorted(created, reverse=True), f"out of order after submission {n}"
        assert len(q._jobs) <= 3, f"{len(q._jobs)} jobs kept after submission {n}; retain is 3"
        assert listed[-1]["id"] == live.id                   # the live job is kept, as the oldest
    gate.set()
    await q.wait_for(live, timeout=5)
