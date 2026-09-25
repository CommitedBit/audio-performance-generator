"""Keeps seeded generations reproducible when several run at once.

Every local stack here draws from PyTorch's process-wide random generator, and
none of them can be handed an isolated one: transformers' generate() samples
with torch.multinomial on the global generator and takes no generator argument
(MusicGen); stable-audio-3's generate() calls torch.manual_seed() on every call;
ACE-Step's LM seeds it globally; Chatterbox samples from it throughout. With
MAX_CONCURRENT_JOBS > 1, any job running beside a seeded one reseeds or advances
the generator under it, so the same seed stopped reproducing the same audio.

The fix is a reader/writer scope: a seeded generation holds it exclusively, so
nothing else touches the generator between its seeding and its last draw;
unseeded generations share it, since they promise no reproducibility and may
overlap each other freely. Writers are preferred, so a steady flow of unseeded
jobs cannot starve a seeded one. With MAX_CONCURRENT_JOBS=1 -- the default --
nothing ever contends and this changes nothing.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator


class _RWLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextlib.contextmanager
    def shared(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if not self._readers:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


_GLOBAL_RNG = _RWLock()


def rng_scope(*, seeded: bool):
    """Exclusive for a seeded generation, shared otherwise."""
    return _GLOBAL_RNG.exclusive() if seeded else _GLOBAL_RNG.shared()
