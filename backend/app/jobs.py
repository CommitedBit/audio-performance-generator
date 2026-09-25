"""In-process job queue for generation work.

Music generation can take minutes, which is far too long to hold an HTTP
request open through a proxy. Callers enqueue a job, get an id back, and poll.
Short work (TTS) can still await the job inline via `wait_for`.

Generation is serialised by a semaphore because a single GPU cannot safely run
two audio models concurrently -- raise MAX_CONCURRENT_JOBS only if VRAM allows.
"""
from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    audio_id: str | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)
    # "local" jobs share the GPU-sized semaphore; "remote" (cloud) jobs run in a
    # separate lane so they never wait behind local inference.
    lane: str = "local"
    _future: asyncio.Future | None = field(default=None, repr=False, compare=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "audio_id": self.audio_id,
            "error": self.error,
            "meta": self.meta,
            "queue_position": self.meta.get("queue_position"),
        }


# Concurrent cloud requests allowed in the remote lane.
REMOTE_CONCURRENCY = 4


class JobQueue:
    def __init__(self, max_concurrent: int = 1, retain: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._retain = retain
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        # Cloud providers use no local GPU, so MAX_CONCURRENT_JOBS must not gate
        # them. They were queued behind local jobs: an ElevenLabs line could sit
        # behind a 600 s ACE-Step job and blow past the 180 s inline speech wait.
        # Bounded anyway, so a burst cannot open unlimited outbound requests.
        self._remote_sem = asyncio.Semaphore(REMOTE_CONCURRENCY)

    def submit(self, kind: str, fn: Callable[[Job], dict], meta: dict | None = None,
               *, remote: bool = False) -> Job:
        """Queue `fn`, which runs on a worker thread and returns a result dict.

        `fn` receives its own Job so it can report progress.
        """
        job = Job(id=uuid.uuid4().hex, kind=kind, meta=meta or {}, lane="remote" if remote else "local")
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._evict()
        job._future = asyncio.ensure_future(self._run(job, fn))
        return job

    async def _run(self, job: Job, fn: Callable[[Job], dict]) -> dict:
        # Position counts only jobs waiting in the same lane.
        waiting = [j for j in self._jobs.values() if j.status is JobStatus.QUEUED and j.lane == job.lane]
        job.meta["queue_position"] = max(0, len(waiting) - 1)
        async with self._sem if job.lane == "local" else self._remote_sem:
            if job.status is JobStatus.CANCELLED:
                return {}
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.meta["queue_position"] = 0
            try:
                # Providers are synchronous and release the GIL inside torch,
                # so a thread is the right place for them.
                result = await asyncio.to_thread(fn, job)
                job.audio_id = result.get("audio_id")
                job.meta.update(result.get("meta", {}))
                job.status = JobStatus.DONE
                job.progress = 1.0
                return result
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                raise
            except Exception as exc:                     # noqa: BLE001
                log.exception("job %s (%s) failed", job.id, job.kind)
                job.status = JobStatus.ERROR
                # The message reaches the UI, so keep the type name -- a bare
                # str(exc) on a CUDA OOM is unhelpfully empty.
                job.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                return {}
            finally:
                job.finished_at = time.time()

    async def wait_for(self, job: Job, timeout: float | None = None) -> Job:
        if job._future is not None:
            # A timeout is the normal path for slow work: the caller then gets
            # the job back to poll instead of holding the connection open.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(job._future), timeout=timeout)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status in {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED}:
            return False
        # A job already inside torch cannot be interrupted; cancelling only
        # helps while it is still queued. Say so rather than pretending.
        if job.status is JobStatus.QUEUED and job._future is not None:
            job.status = JobStatus.CANCELLED
            job._future.cancel()
            return True
        return False

    def list(self, limit: int = 50) -> list[dict]:
        return [self._jobs[i].public() for i in reversed(self._order[-limit:]) if i in self._jobs]

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for j in self._jobs.values():
            counts[j.status.value] = counts.get(j.status.value, 0) + 1
        return {"max_concurrent": self._max_concurrent, "tracked": len(self._jobs), "by_status": counts}

    def _evict(self) -> None:
        while len(self._order) > self._retain:
            old = self._order.pop(0)
            job = self._jobs.get(old)
            if job and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                self._order.append(old)     # never evict live work
                return
            self._jobs.pop(old, None)
