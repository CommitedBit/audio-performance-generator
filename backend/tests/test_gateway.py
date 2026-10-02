"""The aggregating gateway (app.gateway), against fake upstream services."""
from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

from app import gateway


def _provider(pid: str, capability: str, available: bool = True, reason: str = "") -> dict:
    return {"id": pid, "capability": capability, "available": available, "unavailable_reason": reason}


class Upstreams:
    """Fake model services keyed by hostname, served through httpx.MockTransport."""

    def __init__(self) -> None:
        self.providers: dict[str, list[dict]] = {}
        self.jobs: dict[str, dict[str, dict]] = {}
        self.down: set[str] = set()
        self.posts: list[tuple[str, str, dict, dict]] = []   # (service, path, query, body)
        self.post_reply: tuple[int, dict] = (202, {"id": "abc", "status": "queued", "audio_url": None})

    def handler(self, request: httpx.Request) -> httpx.Response:
        service = request.url.host
        if service in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if request.method == "GET" and path == "/v1/models":
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
    ups.down.add("models")
    assert gw.get("/health").json()["status"] == "down"


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
