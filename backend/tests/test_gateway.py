"""The aggregating gateway (app.gateway), against fake upstream services."""
from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from app import gateway


def _provider(pid: str, capability: str, available: bool = True, reason: str = "") -> dict:
    return {"id": pid, "capability": capability, "available": available, "unavailable_reason": reason,
            "remote": pid.startswith("elevenlabs")}


class Upstreams:
    """Fake model services keyed by hostname, served through httpx.MockTransport."""

    def __init__(self) -> None:
        self.providers: dict[str, list[dict]] = {}
        self.jobs: dict[str, dict[str, dict]] = {}
        self.down: set[str] = set()
        self.posts: list[tuple[str, str, dict, dict]] = []   # (service, path, query, body)
        self.post_reply: tuple[int, dict] = (202, {"id": "abc", "status": "queued", "audio_url": None})
        self.discoveries = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        service = request.url.host
        if service in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if request.method == "GET" and path == "/v1/models":
            self.discoveries += 1
            return httpx.Response(200, json={"providers": self.providers.get(service, [])})
        if request.method == "GET" and path == "/v1/jobs":
            return httpx.Response(200, json={"jobs": list(self.jobs.get(service, {}).values())})
        if path.startswith("/v1/jobs/"):
            job = self.jobs.get(service, {}).get(path.rsplit("/", 1)[-1])
            if job is None:
                return httpx.Response(404, json={"detail": "unknown job"})
            if request.method == "DELETE":
                return httpx.Response(409, json={"detail": f"job is {job['status']} and can no longer be cancelled"})
            return httpx.Response(200, json=job)
        if request.method == "POST":
            query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
            self.posts.append((service, path, query, json.loads(request.content)))
            status, body = self.post_reply
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"detail": "no route"})


@pytest.fixture
def ups(monkeypatch):
    fake = Upstreams()
    transport = httpx.MockTransport(fake.handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=transport, **kw))
    monkeypatch.setattr(gateway, "UPSTREAMS", {"models": "http://models:8000", "music": "http://music:8000"})
    monkeypatch.setattr(gateway, "_LAST_OWNER", {})
    monkeypatch.setattr(gateway, "_REQUIRED_SEEN", {})
    monkeypatch.setattr(gateway, "_CACHE", None)
    monkeypatch.setattr(gateway, "MODELS_CACHE_SECONDS", 0.0)     # tests change upstreams between calls
    monkeypatch.setattr(gateway, "REQUIRED_PROVIDERS", ["chatterbox", "acestep", "stable-audio-3-sfx"])
    fake.providers = {
        "models": [
            _provider("chatterbox", "voice"),
            _provider("stable-audio-3-sfx", "sfx"),
            _provider("stable-audio-3-music", "music"),
            _provider("elevenlabs-voice", "voice"),
        ],
        "music": [_provider("acestep", "music")],
    }
    return fake


@pytest.fixture
def gw(ups):
    with TestClient(gateway.app) as client:
        yield client


# -- discovery ------------------------------------------------------------------

def test_models_are_merged_and_tagged_with_their_service(gw):
    body = gw.get("/v1/models").json()
    services = {p["id"]: p["service"] for p in body["providers"]}
    assert services["acestep"] == "music" and services["chatterbox"] == "models"
    assert body["upstreams_down"] == []


def test_defaults_rank_across_services_not_by_upstream_order(gw):
    # stable-audio-3-music is on the FIRST upstream, acestep on the second.
    assert gw.get("/v1/models").json()["defaults"] == {
        "voice": "chatterbox", "music": "acestep", "sfx": "stable-audio-3-sfx"}


def test_duplicate_provider_ids_keep_the_first_service(gw, ups):
    ups.providers["music"].append(_provider("chatterbox", "voice"))
    owners = [p["service"] for p in gw.get("/v1/models").json()["providers"] if p["id"] == "chatterbox"]
    assert owners == ["models"]


def test_health_reflects_upstreams(gw, ups):
    assert gw.get("/health").json()["status"] == "ok"
    ups.down.add("music")
    body = gw.get("/health").json()
    assert body["status"] == "degraded"
    assert body["upstreams"] == {"models": "up", "music": "down"}
    # It remembers which service served acestep, so it can say which one is gone.
    assert body["required"]["acestep"] == {"status": "missing", "reason": "its service music is unreachable"}
    ups.down.add("models")
    assert gw.get("/health").json()["status"] == "down"


def test_health_reports_capabilities_across_services(gw, ups):
    ups.providers["models"][0] = _provider("chatterbox", "voice", False, "load failed")
    body = gw.get("/health").json()
    assert body["capabilities"]["voice"] == {"status": "cloud", "available": ["elevenlabs-voice"]}
    assert body["capabilities"]["music"]["status"] == "ok"
    assert body["required"]["chatterbox"] == {"status": "unavailable", "reason": "load failed"}
    assert body["status"] == "degraded"


# -- readiness --------------------------------------------------------------------

def test_ready_when_every_required_provider_is_available(gw):
    r = gw.get("/health/ready")
    assert (r.status_code, r.json()["ready"], r.json()["problems"]) == (200, True, [])


def test_not_ready_when_a_required_model_is_replaced_by_a_fallback(gw, ups):
    """MusicGen standing in for a broken ACE-Step still serves music -- not ready."""
    ups.providers["music"] = [_provider("acestep", "music", False, "ImportError: nano-vllm"),
                              _provider("musicgen", "music")]
    r = gw.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["problems"] == ["acestep unavailable: ImportError: nano-vllm"]
    assert gw.get("/health").json()["capabilities"]["music"]["status"] == "ok"


def test_not_ready_when_a_required_service_is_down(gw, ups):
    gw.get("/v1/models")
    ups.down.add("music")
    r = gw.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["problems"] == ["acestep missing: its service music is unreachable"]


def test_readiness_without_a_required_list(gw, ups, monkeypatch):
    monkeypatch.setattr(gateway, "REQUIRED_PROVIDERS", [])
    assert gw.get("/health/ready").status_code == 200
    ups.providers["models"] = [p for p in ups.providers["models"] if p["capability"] != "sfx"]
    ups.providers["models"].append(_provider("stable-audio-3-sfx", "sfx", False, "HF_TOKEN is not set"))
    r = gw.get("/health/ready")
    assert (r.status_code, r.json()["problems"]) == (503, ["sfx: no available provider"])


def test_readiness_needs_no_api_key(gw, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert gw.get("/health/ready").status_code == 200
    assert gw.get("/v1/models").status_code == 401


def test_required_provider_changes_are_logged_once(gw, ups, caplog):
    with caplog.at_level("INFO", logger="gateway"):
        gw.get("/v1/models")
        ups.providers["music"] = [_provider("acestep", "music", False, "CUDA OOM")]
        gw.get("/v1/models")
        gw.get("/v1/models")
        ups.providers["music"] = [_provider("acestep", "music")]
        gw.get("/v1/models")
    messages = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "gateway"]
    assert messages == [
        ("ERROR", "required provider acestep is unavailable: CUDA OOM"),
        ("INFO", "required provider acestep is available again"),
    ]


def test_required_providers_parsing(monkeypatch):
    monkeypatch.delenv("REQUIRED_PROVIDERS", raising=False)
    assert gateway._parse_required() == ["chatterbox", "acestep", "stable-audio-3-sfx"]
    monkeypatch.setenv("REQUIRED_PROVIDERS", " stub-voice , stub-sfx ")
    assert gateway._parse_required() == ["stub-voice", "stub-sfx"]
    for off in ("none", "", "  "):
        monkeypatch.setenv("REQUIRED_PROVIDERS", off)
        assert gateway._parse_required() == []


# -- discovery cache ----------------------------------------------------------------

def test_discovery_is_cached_briefly(gw, ups, monkeypatch):
    monkeypatch.setattr(gateway, "MODELS_CACHE_SECONDS", 60.0)
    for _ in range(5):
        gw.post("/v1/audio/sfx", json={"prompt": "x"})
    gw.get("/health")
    assert ups.discoveries == 2                   # one fan-out to two upstreams
    monkeypatch.setattr(gateway, "MODELS_CACHE_SECONDS", 0.0)
    gw.get("/v1/models")
    assert ups.discoveries == 4


# -- cloud is never an automatic default ---------------------------------------------

def test_cloud_is_not_picked_automatically(gw, ups):
    ups.providers["models"][0] = _provider("chatterbox", "voice", False, "load failed")
    assert gw.get("/v1/models").json()["defaults"]["voice"] is None
    r = gw.post("/v1/audio/speech", json={"input": "secret story"})
    assert r.status_code == 503
    assert "elevenlabs-voice can serve it if named explicitly" in r.json()["detail"]
    assert ups.posts == []                        # nothing was sent anywhere
    assert gw.post("/v1/audio/speech", json={"input": "ok", "provider": "elevenlabs-voice"}).status_code == 202


def test_cloud_default_can_be_allowed(gw, ups, monkeypatch):
    monkeypatch.setattr(gateway.settings, "allow_cloud_default", True)
    ups.providers["models"][0] = _provider("chatterbox", "voice", False, "load failed")
    assert gw.get("/v1/models").json()["defaults"]["voice"] == "elevenlabs-voice"


# -- generation routing -----------------------------------------------------------

def test_speech_is_forwarded_with_a_resolved_provider_and_voice(gw, ups):
    r = gw.post("/v1/audio/speech", json={"input": "hello", "voice": "v1"})
    assert r.status_code == 202
    assert r.json()["id"] == "models:abc"
    service, path, query, body = ups.posts[-1]
    assert (service, path, query) == ("models", "/v1/audio/speech", {"wait": "true"})
    assert body["prompt"] == "hello" and body["provider"] == "chatterbox" and body["voice_id"] == "v1"
    assert "voice" not in body and "input" not in body


def test_speech_wait_flag_is_passed_through(gw, ups):
    gw.post("/v1/audio/speech?wait=false", json={"prompt": "x"})
    assert ups.posts[-1][2] == {"wait": "false"}


def test_music_goes_to_the_service_that_owns_the_default(gw, ups):
    gw.post("/v1/audio/music", json={"prompt": "strings", "seconds": 30})
    service, path, _, body = ups.posts[-1]
    assert (service, path, body["provider"], body["seconds"]) == ("music", "/v1/audio/music", "acestep", 30)


def test_explicit_provider_errors(gw, ups):
    assert gw.post("/v1/audio/sfx", json={"prompt": "x", "provider": "nope"}).status_code == 404
    assert gw.post("/v1/audio/sfx", json={"prompt": "x", "provider": "acestep"}).status_code == 400
    ups.providers["models"][1] = _provider("stable-audio-3-sfx", "sfx", False, "HF_TOKEN is not set")
    r = gw.post("/v1/audio/sfx", json={"prompt": "x", "provider": "stable-audio-3-sfx"})
    assert r.status_code == 503 and "HF_TOKEN" in r.json()["detail"]
    assert gw.post("/v1/audio/sfx", json={"prompt": "x"}).status_code == 503     # no default left


def test_known_provider_on_a_down_service_is_503_not_404(gw, ups):
    gw.get("/v1/models")                          # learn that acestep lives on `music`
    ups.down.add("music")
    r = gw.post("/v1/audio/music", json={"prompt": "x", "provider": "acestep"})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    # Names the owner it remembers, not just "something is down".
    assert "served by music" in r.json()["detail"]


def test_never_seen_provider_while_a_service_is_down_is_503(gw, ups):
    ups.down.add("music")
    r = gw.post("/v1/audio/music", json={"prompt": "x", "provider": "brand-new"})
    assert r.status_code == 503 and r.headers["retry-after"] == "5"


def test_upstream_errors_keep_their_status(gw, ups):
    ups.post_reply = (422, {"detail": "seconds must be 10-600 for acestep (got 5)"})
    r = gw.post("/v1/audio/music", json={"prompt": "x", "seconds": 5})
    assert r.status_code == 422
    assert r.json()["detail"] == "music: seconds must be 10-600 for acestep (got 5)"


def test_unreachable_upstream_during_forward_is_502(gw, ups, monkeypatch):
    # Discovery succeeded, then the service died before the forward.
    monkeypatch.setattr(gateway, "_resolve_service", lambda cap, pid: _resolved("music", "acestep"))
    ups.down.add("music")
    assert gw.post("/v1/audio/music", json={"prompt": "x"}).status_code == 502


async def _resolved(service, provider):
    return service, provider


# -- jobs -------------------------------------------------------------------------

def test_job_routes_by_prefix_and_relays_upstream_status(gw, ups):
    ups.jobs["music"] = {"j1": {"id": "j1", "status": "done", "created_at": 1.0, "audio_url": "/v1/audio/a"}}
    assert gw.get("/v1/jobs/music:j1").json()["id"] == "music:j1"
    r = gw.get("/v1/jobs/music:gone")
    assert (r.status_code, r.json()["detail"]) == (404, "unknown job")
    assert gw.delete("/v1/jobs/music:j1").status_code == 409
    assert gw.get("/v1/jobs/j1").status_code == 400                # no service prefix
    assert gw.get("/v1/jobs/elsewhere:j1").status_code == 404      # unknown service
    ups.down.add("music")
    assert gw.get("/v1/jobs/music:j1").status_code == 502


def test_job_list_aggregates_newest_first(gw, ups):
    ups.jobs["models"] = {"a": {"id": "a", "created_at": 1.0}, "c": {"id": "c", "created_at": 3.0}}
    ups.jobs["music"] = {"b": {"id": "b", "created_at": 2.0}}
    body = gw.get("/v1/jobs").json()
    assert [j["id"] for j in body["jobs"]] == ["models:c", "music:b", "models:a"]
    ups.down.add("music")
    body = gw.get("/v1/jobs").json()
    assert body["upstreams_down"] == ["music"]
    assert [j["id"] for j in body["jobs"]] == ["models:c", "models:a"]


# -- configuration ----------------------------------------------------------------

def test_upstreams_parsing(monkeypatch):
    monkeypatch.setenv("UPSTREAMS", " a=http://a:8000/ , b:c=http://x, junk, =http://y, d= ")
    assert gateway._parse_upstreams() == {"a": "http://a:8000"}


# -- the real model service behind the real gateway -----------------------------------

@pytest.fixture
def stack(monkeypatch):
    """Gateway -> app.main over ASGI, sharing one data dir: the dev topology in-process."""
    from app import main
    from app.jobs import JobQueue

    monkeypatch.setattr(main, "queue", JobQueue())
    real = httpx.AsyncClient
    transport = httpx.ASGITransport(app=main.app)
    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda *a, **kw: real(*a, transport=transport, **kw))
    monkeypatch.setattr(gateway, "UPSTREAMS", {"stub": "http://stub:8000"})
    monkeypatch.setattr(gateway, "_LAST_OWNER", {})
    monkeypatch.setattr(gateway, "_REQUIRED_SEEN", {})
    monkeypatch.setattr(gateway, "_CACHE", None)
    monkeypatch.setattr(gateway, "MODELS_CACHE_SECONDS", 0.0)
    monkeypatch.setattr(gateway, "REQUIRED_PROVIDERS", ["stub-voice", "stub-music", "stub-sfx"])
    with TestClient(gateway.app) as client:
        yield client


def test_dev_stack_end_to_end(stack):
    health = stack.get("/health").json()
    assert health["status"] == "stub"
    assert health["capabilities"]["voice"]["status"] == "stub"
    assert stack.get("/health/ready").status_code == 200

    models = stack.get("/v1/models").json()
    assert all("remote" in p for p in models["providers"])
    assert models["defaults"]["voice"] == "stub-voice"
    assert next(p for p in models["providers"] if p["id"] == "elevenlabs-voice")["remote"] is True

    job = stack.post("/v1/audio/speech", json={"input": "End to end."}).json()
    assert job["id"].startswith("stub:") and job["status"] == "done"
    assert stack.get(job["audio_url"]).content[:4] == b"RIFF"
    assert stack.get(f"/v1/jobs/{job['id']}").json()["audio_id"] == job["audio_id"]
