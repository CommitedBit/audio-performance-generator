"""The model service's HTTP surface (app.main), end to end through the job queue."""
from __future__ import annotations

import contextlib
import io
import wave

from helpers import FakeProvider, make_wav, wait_for_job

from app.providers.base import Capability


def _wav_info(data: bytes) -> tuple[int, int, float]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getframerate(), wf.getnframes() / wf.getframerate()


# -- health and discovery -----------------------------------------------------

def test_health_reports_jobs_and_providers(api):
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["providers_available"] >= 3       # the dev stubs
    assert body["jobs"]["max_concurrent"] == 1


def test_models_lists_stubs_with_defaults_in_dev_mode(api):
    body = api.get("/v1/models").json()
    ids = {p["id"] for p in body["providers"]}
    assert {"stub-voice", "stub-music", "stub-sfx"} <= ids
    assert body["defaults"] == {"voice": "stub-voice", "music": "stub-music", "sfx": "stub-sfx"}
    unavailable = [p for p in body["providers"] if not p["available"]]
    # Real providers without their deps are still listed, with a reason.
    assert all(p["unavailable_reason"] for p in unavailable)


# -- speech ---------------------------------------------------------------------

def test_speech_waits_inline_and_measures_the_stored_file(api):
    r = api.post("/v1/audio/speech", json={"input": "Hello there, this is a test line."})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "done"
    assert job["id"] and ":" not in job["id"]
    audio = api.get(job["audio_url"])
    assert audio.status_code == 200
    _, _, measured = _wav_info(audio.content)
    assert abs(job["meta"]["duration"] - measured) < 1e-6


def test_speech_openai_aliases(api, use_providers):
    voice = FakeProvider("fake-voice", Capability.VOICE)
    use_providers(voice)
    r = api.post("/v1/audio/speech", json={"input": "from input", "voice": "v1", "prompt": "from prompt"})
    assert r.status_code == 200
    req = voice.generate_calls[-1]
    assert req.prompt == "from input"
    assert req.voice_id == "v1"


def test_speech_blank_input_falls_through_to_prompt(api, use_providers):
    voice = FakeProvider("fake-voice", Capability.VOICE)
    use_providers(voice)
    assert api.post("/v1/audio/speech", json={"input": "   ", "prompt": "real text"}).status_code == 200
    assert voice.generate_calls[-1].prompt == "real text"


def test_speech_without_text_is_rejected(api):
    assert api.post("/v1/audio/speech", json={"input": "  "}).status_code == 422
    assert api.post("/v1/audio/speech", json={}).status_code == 422


def test_speech_drops_seconds_for_a_provider_without_a_duration_param(api, use_providers):
    voice = FakeProvider("fake-voice", Capability.VOICE)        # no seconds ParamSpec
    use_providers(voice)
    r = api.post("/v1/audio/speech", json={"input": "short", "seconds": 25, "params": {"seconds": 99}})
    assert r.status_code == 200
    req = voice.generate_calls[-1]
    assert req.seconds is None
    assert "seconds" not in req.params


def test_speech_without_wait_returns_a_job(api):
    r = api.post("/v1/audio/speech?wait=false", json={"input": "later"})
    assert r.status_code in (200, 202)
    assert wait_for_job(api, r.json()["id"])["status"] == "done"


# -- music / sfx ----------------------------------------------------------------

def test_music_is_queued_then_completes_at_the_requested_length(api):
    r = api.post("/v1/audio/music", json={"prompt": "calm piano", "seconds": 2})
    assert r.status_code == 202
    job = wait_for_job(api, r.json()["id"])
    assert job["status"] == "done"
    assert abs(job["meta"]["duration"] - 2.0) < 0.01
    assert job["audio_url"] == f"/v1/audio/{job['audio_id']}"


def test_seconds_outside_the_provider_range_is_rejected(api):
    r = api.post("/v1/audio/sfx", json={"prompt": "door", "seconds": 31})     # stub-sfx max is 30
    assert r.status_code == 422
    assert "stub-sfx" in r.json()["detail"]
    assert api.post("/v1/audio/sfx", json={"prompt": "door", "seconds": 0}).status_code == 422


def test_seconds_in_params_is_validated_too(api):
    r = api.post("/v1/audio/music", json={"prompt": "x", "params": {"seconds": "abc"}})
    assert r.status_code == 422
    r = api.post("/v1/audio/music", json={"prompt": "x", "params": {"seconds": 500}})
    assert r.status_code == 422


def test_unknown_provider_is_404_and_wrong_capability_is_400(api):
    assert api.post("/v1/audio/music", json={"prompt": "x", "provider": "nope"}).status_code == 404
    assert api.post("/v1/audio/music", json={"prompt": "x", "provider": "stub-sfx"}).status_code == 400


def test_unavailable_provider_is_503(api, use_providers):
    use_providers(FakeProvider("off", Capability.MUSIC, is_available=False))
    r = api.post("/v1/audio/music", json={"prompt": "x", "provider": "off"})
    assert r.status_code == 503
    assert "disabled by the test" in r.json()["detail"]
    assert api.post("/v1/audio/music", json={"prompt": "x"}).status_code == 503


# -- what the worker does with provider output ------------------------------------

def test_wav_duration_is_measured_not_trusted(api, use_providers):
    # Stereo 3 s, but the provider claims 0.01 s (the wrong-axis bug).
    music = FakeProvider("fake-music", Capability.MUSIC, seconds_range=(1, 60),
                         wav=make_wav(3.0, 44100, channels=2), reported_duration=0.01)
    use_providers(music)
    job = wait_for_job(api, api.post("/v1/audio/music", json={"prompt": "x"}).json()["id"])
    assert job["status"] == "done"
    assert abs(job["meta"]["duration"] - 3.0) < 0.01
    channels, rate, _ = _wav_info(api.get(job["audio_url"]).content)
    assert (channels, rate) == (2, 44100)


def test_silent_output_fails_the_job(api, use_providers):
    use_providers(FakeProvider("quiet", Capability.SFX, seconds_range=(1, 10), wav=make_wav(1.0, silent=True)))
    job = wait_for_job(api, api.post("/v1/audio/sfx", json={"prompt": "x"}).json()["id"])
    assert job["status"] == "error"
    assert job["error"].startswith("SilentOutputError")
    assert job["audio_id"] is None


def test_speech_error_is_reported_inline(api, use_providers):
    use_providers(FakeProvider("broken", Capability.VOICE, load_error=OSError("weights missing")))
    r = api.post("/v1/audio/speech", json={"input": "hi"})
    assert r.status_code == 500
    assert "weights missing" in r.json()["detail"]


def test_local_jobs_take_the_gpu_slot_and_rng_scope_remote_jobs_do_not(api, use_providers, monkeypatch):
    from app import main

    slots: list[str] = []
    scopes: list[bool] = []

    @contextlib.contextmanager
    def fake_slot(on_wait=None):
        slots.append("held")
        yield

    def fake_scope(*, seeded):
        scopes.append(seeded)
        return contextlib.nullcontext()

    monkeypatch.setattr(main, "gpu_slot", fake_slot)
    monkeypatch.setattr(main, "rng_scope", fake_scope)
    use_providers(FakeProvider("local", Capability.VOICE), FakeProvider("cloud", Capability.VOICE, remote=True))

    assert api.post("/v1/audio/speech", json={"input": "a", "provider": "local", "seed": 7}).status_code == 200
    assert (slots, scopes) == (["held"], [True])
    assert api.post("/v1/audio/speech", json={"input": "b", "provider": "local"}).status_code == 200
    assert (slots, scopes) == (["held", "held"], [True, False])
    assert api.post("/v1/audio/speech", json={"input": "c", "provider": "cloud", "seed": 7}).status_code == 200
    assert (slots, scopes) == (["held", "held"], [True, False])


# -- jobs -----------------------------------------------------------------------

def test_job_list_has_audio_urls_and_unknown_jobs_404(api):
    api.post("/v1/audio/speech", json={"input": "one"})
    jobs = api.get("/v1/jobs").json()["jobs"]
    assert jobs and jobs[0]["audio_url"].startswith("/v1/audio/")
    assert api.get("/v1/jobs/does-not-exist").status_code == 404
    assert api.delete("/v1/jobs/does-not-exist").status_code == 404


def test_finished_job_cannot_be_cancelled(api):
    job = api.post("/v1/audio/speech", json={"input": "done already"}).json()
    r = api.delete(f"/v1/jobs/{job['id']}")
    assert r.status_code == 409
    assert "done" in r.json()["detail"]


# -- audio and voice storage ----------------------------------------------------

def test_audio_ids_are_validated_and_deletable(api):
    job = api.post("/v1/audio/speech", json={"input": "keep me"}).json()
    audio_id = job["audio_id"]
    assert api.get("/v1/audio/not..valid").status_code == 400
    assert api.get("/v1/audio/" + "0" * 32).status_code == 404
    assert api.delete(f"/v1/audio/{audio_id}").json() == {"deleted": audio_id}
    assert api.get(f"/v1/audio/{audio_id}").status_code == 404
    assert api.delete(f"/v1/audio/{audio_id}").status_code == 404


def test_voice_reference_upload_list_delete(api):
    files = {"file": ("narrator.wav", make_wav(1.0), "audio/wav")}
    voice = api.post("/v1/voices", files=files, data={"name": "Narrator"}).json()
    assert voice["name"] == "Narrator"
    assert [v["id"] for v in api.get("/v1/voices").json()["voices"]] == [voice["id"]]
    usage = api.get("/v1/storage").json()
    assert usage["voice_files"] == 1
    assert api.delete(f"/v1/voices/{voice['id']}").status_code == 200
    assert api.get("/v1/voices").json()["voices"] == []


def test_voice_upload_validation(api):
    assert api.post("/v1/voices", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 422
    assert api.post("/v1/voices", files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 415
