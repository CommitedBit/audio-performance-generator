"""Cross-process GPU slots, so separate model containers share one card safely.

Each model service serialises its own generations on an in-process semaphore
(MAX_CONCURRENT_JOBS). That is enough when one container owns the GPU, but the
split topology runs `models` and `music` as separate processes with separate
semaphores -- so an ACE-Step job could run at the same moment as a voice or SFX
job on the same card, and MAX_CONCURRENT_JOBS=1 limited each container rather
than the GPU. This module adds a limit both containers can see.

The slots are lock files on a volume every GPU service mounts, taken with
flock(2), which works across processes and across containers sharing a host
filesystem. There are MAX_CONCURRENT_JOBS of them, so the setting now caps
inference on the GPU as a whole. A crashed holder releases its slot
automatically: the kernel drops the lock with the file descriptor.

Enabled by GPU_LOCK_FILE (a path prefix; slots are <prefix>.0, <prefix>.1, ...).
Unset, this is a no-op -- the unified topology has one process, where the
in-process semaphore already does the job.
"""
from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from .config import get_settings

log = logging.getLogger(__name__)

POLL_SECONDS = 0.25


@contextlib.contextmanager
def gpu_slot(*, on_wait: Callable[[], None] | None = None) -> Iterator[None]:
    """Hold one GPU slot for the duration of the block, waiting if none is free."""
    prefix = os.getenv("GPU_LOCK_FILE")
    if not prefix:
        yield
        return

    slots = max(1, get_settings().max_concurrent_jobs)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    held = None
    waited = False
    while held is None:
        for i in range(slots):
            f = open(f"{prefix}.{i}", "a+")  # noqa: SIM115 -- must stay open while held
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            held = f
            break
        if held is None:
            if not waited:
                waited = True
                log.info("all %d GPU slot(s) busy in another service; waiting", slots)
                if on_wait:
                    on_wait()
            time.sleep(POLL_SECONDS)

    try:
        yield
    finally:
        try:
            fcntl.flock(held, fcntl.LOCK_UN)
        finally:
            held.close()
