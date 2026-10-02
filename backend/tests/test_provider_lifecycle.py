"""Provider load/unload/in-use bookkeeping and load-failure cooldown (providers.base)."""
from __future__ import annotations

import threading
import time

import pytest
from helpers import FakeProvider, reload_settings

from app.providers.base import Capability, Provider, check_audio_sane, pcm_to_wav, wav_duration


def test_concurrent_loads_build_the_model_once():
    p = FakeProvider("p", load_delay=0.05)
    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(p.load())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert p.load_calls == 1
    assert len({id(r) for r in results}) == 1


def test_load_failure_is_recorded_and_blocks_retries_during_cooldown(monkeypatch):
    reload_settings(monkeypatch, PROVIDER_RETRY_SECONDS="300")
    p = FakeProvider("p", load_error=OSError("bad checkpoint"))
    with pytest.raises(OSError):
        p.load()
    assert not p.available()
    assert "failed to load: OSError: bad checkpoint" in p.unavailable_reason()
    with pytest.raises(RuntimeError, match="failed to load"):
        p.load()
    assert p.load_calls == 1                      # the doomed load did not re-run


def test_load_retries_after_cooldown_and_success_clears_the_failure(monkeypatch):
    reload_settings(monkeypatch, PROVIDER_RETRY_SECONDS="0")
    p = FakeProvider("p", load_error=OSError("flaky"))
    with pytest.raises(OSError):
        p.load()
    p._raise_on_load = None
    assert p.available()                          # cooldown of 0 has already elapsed
    p.load()
    assert p.load_calls == 2
    assert p._load_error is None
    assert p.unavailable_reason() == ""


def test_jobs_queued_behind_a_failing_load_do_not_rerun_it(monkeypatch):
    """The failure check runs under the load lock, so a waiter sees it."""
    reload_settings(monkeypatch, PROVIDER_RETRY_SECONDS="300")
    p = FakeProvider("p", load_error=OSError("boom"))
    p.release_load.clear()
    errors: list[BaseException] = []

    def attempt():
        try:
            p.load()
        except BaseException as exc:                  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=attempt)
    first.start()
    while p.load_calls == 0:
        time.sleep(0.005)
    second = threading.Thread(target=attempt)
    second.start()
    time.sleep(0.05)                                  # second is now waiting on the lock
    p.release_load.set()
    first.join()
    second.join()
    assert p.load_calls == 1
    assert sorted(type(e).__name__ for e in errors) == ["OSError", "RuntimeError"]


def test_load_failure_overrides_subclass_available():
    """__init_subclass__ wraps overrides, so a provider cannot forget the check."""

    class Optimistic(Provider):
        id = "optimistic"

        def available(self):
            return True

        def unavailable_reason(self):
            return "never shown"

        def _load(self):
            raise ValueError("nope")

        def generate(self, req):
            raise NotImplementedError

    p = Optimistic()
    assert p.available()
    with pytest.raises(ValueError):
        p.load()
    assert not p.available()
    assert p.unavailable_reason().startswith("failed to load: ValueError: nope")


def test_unload_if_idle_respects_in_use_and_timeout():
    p = FakeProvider("p")
    assert not p.unload_if_idle(0)                    # never loaded
    p.load()
    with p.in_use():
        p._last_used -= 1000
        assert not p.unload_if_idle(10)              # busy
    assert not p.unload_if_idle(10)                  # in_use exit reset the idle clock
    p._last_used -= 1000
    assert p.idle_seconds >= 1000
    assert p.unload_if_idle(10)
    assert not p.loaded
    assert p.idle_seconds == 0.0


def test_pcm_to_wav_handles_channels_first_stereo():
    import numpy as np

    stereo = np.zeros((2, 4800), dtype=np.float32)
    stereo[0] = 0.5
    data = pcm_to_wav(stereo, 48000)
    assert abs(wav_duration(data) - 0.1) < 1e-9


def test_check_audio_sane_rejects_silence_and_dc():
    import numpy as np

    from app.providers.base import SilentOutputError

    with pytest.raises(SilentOutputError, match="silent"):
        check_audio_sane(pcm_to_wav(np.zeros(1000, dtype=np.float32), 24000))
    with pytest.raises(SilentOutputError, match="constant"):
        check_audio_sane(pcm_to_wav(np.full(1000, 0.5, dtype=np.float32), 24000))
    check_audio_sane(pcm_to_wav(np.sin(np.linspace(0, 100, 1000)).astype(np.float32), 24000))


def test_info_shape(use_providers):
    p = FakeProvider("p", Capability.SFX, seconds_range=(1, 9))
    info = p.info()
    assert info["capability"] == "sfx"
    assert info["params"][0] == {"name": "seconds", "type": "float", "default": 1, "minimum": 1,
                                 "maximum": 9, "description": "Clip length"}
