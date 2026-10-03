"""Optional shared-secret auth (app.auth), on both apps."""
from __future__ import annotations

import pytest


@pytest.fixture(params=["main", "gateway"])
def client(request, api, monkeypatch):
    if request.param == "main":
        yield api
        return
    from fastapi.testclient import TestClient

    from app import gateway

    monkeypatch.setattr(gateway, "UPSTREAMS", {})
    with TestClient(gateway.app) as c:
        yield c


def test_no_key_configured_means_open(client):
    assert client.get("/v1/storage").status_code == 200


def test_key_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.get("/v1/storage").status_code == 401
    assert client.get("/v1/storage", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/storage", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_health_and_preflight_stay_open(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.get("/health").status_code == 200
    preflight = client.options(
        "/v1/storage",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code != 401
