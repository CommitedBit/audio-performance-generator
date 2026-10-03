"""A job reports which phase it is in while it runs.

job.progress used to jump from 0 straight to 1 and the only message was
"generating with <id>" -- set BEFORE a multi-minute first load -- so a music job
showed nothing useful for minutes. The phases are real (loading, generating,
saving); the progress values mark them and do not claim to measure the model's
own work, which would need hooks into each model library.
"""
from __future__ import annotations

import threading
import time

from helpers import FakeProvider, wait_for_job

from app.main import PROGRESS
from app.providers.base import Capability


class GatedGenerate(FakeProvider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.release_generate = threading.Event()

    def generate(self, req):
        self.load()
        self.release_generate.wait(10)
        return super().generate(req)


def _until(api, job_id, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        job = api.get(f"/v1/jobs/{job_id}").json()
        if predicate(job):
            return job
        assert time.monotonic() < deadline, f"job never reached the expected phase: {job}"
        time.sleep(0.01)


def test_a_first_load_is_reported_before_generating(api, use_providers):
    p = GatedGenerate("acestep", Capability.MUSIC, seconds_range=(1, 60))
    p.release_load.clear()
    use_providers(p)
    job_id = api.post("/v1/audio/music", json={"prompt": "x", "seconds": 2}).json()["id"]

    job = _until(api, job_id, lambda j: j["status"] == "running")
    assert job["message"] == "loading acestep" and job["progress"] == PROGRESS["loading"]
    p.release_load.set()
    job = _until(api, job_id, lambda j: j["message"].startswith("generating"))
    assert job["message"] == "generating with acestep" and job["progress"] == PROGRESS["generating"]
    p.release_generate.set()
    job = wait_for_job(api, job_id)
    assert (job["status"], job["progress"]) == ("done", 1.0)


def test_a_loaded_model_goes_straight_to_generating(api, use_providers):
    p = GatedGenerate("acestep", Capability.MUSIC, seconds_range=(1, 60))
    p.load()
    use_providers(p)
    job_id = api.post("/v1/audio/music", json={"prompt": "x", "seconds": 2}).json()["id"]
    job = _until(api, job_id, lambda j: j["status"] == "running" and j["message"])
    assert job["message"] == "generating with acestep"
    p.release_generate.set()
    assert wait_for_job(api, job_id)["status"] == "done"


def test_a_cloud_provider_never_reports_loading(api, use_providers):
    p = GatedGenerate("elevenlabs-voice", Capability.VOICE, remote=True)
    use_providers(p)
    job_id = api.post("/v1/audio/speech?wait=false", json={"input": "hi", "provider": "elevenlabs-voice"}).json()["id"]
    job = _until(api, job_id, lambda j: j["status"] == "running" and j["message"])
    assert job["message"] == "generating with elevenlabs-voice"
    p.release_generate.set()
    assert wait_for_job(api, job_id)["status"] == "done"
