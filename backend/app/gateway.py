"""Aggregating gateway over the per-model backends.

WHY THIS EXISTS: the three model stacks declare mutually incompatible
dependencies and cannot share a Python environment. Verified from their own
metadata:

    chatterbox-tts 0.1.7   torch==2.6.0   transformers==5.2.0   py<3.14
    ACE-Step-1.5           torch==2.10.0+cu128                  py>=3.11,<3.13
    stable-audio-3         torch==2.7.1   transformers>=5.8.0

torch pins are exact and mutually exclusive, and the transformers constraints
contradict outright. So each model runs in its own container, and this gateway
presents them to the frontend as one API. The UI is unchanged: it still sees a
single /v1/models and a single set of generation endpoints.

Audio and voice references need no routing -- every service shares one data
volume, so the gateway reads them straight off disk. Only jobs are routed, by
prefixing the job id with its upstream key, which keeps routing stateless
across restarts.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import storage
from .auth import api_key_middleware
from .config import get_settings
from .schemas import GenerateBody, SpeechBody

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("gateway")

# UPSTREAMS="tts=http://tts:8000,music=http://music:8000,sfx=http://sfx:8000"
def _parse_upstreams() -> dict[str, str]:
    raw = os.getenv("UPSTREAMS", "")
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, url = part.split("=", 1)
        key, url = key.strip(), url.strip().rstrip("/")
        if key and url and ":" not in key:      # ':' is the job-id delimiter
            out[key] = url
    return out


UPSTREAMS = _parse_upstreams()
DISCOVERY_TIMEOUT = float(os.getenv("DISCOVERY_TIMEOUT", "10"))
# Generous: a TTS request waits inline, and the first one may load weights.
FORWARD_TIMEOUT = float(os.getenv("FORWARD_TIMEOUT", "600"))

app = FastAPI(
    title="Audio Performance Generator Gateway",
    version="0.1.0",
)
app.middleware("http")(api_key_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _get_json(client: httpx.AsyncClient, key: str, path: str) -> dict | None:
    try:
        r = await client.get(f"{UPSTREAMS[key]}{path}", timeout=DISCOVERY_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:                       # noqa: BLE001
        # One dead backend must not take down discovery for the others.
        log.warning("upstream %s %s failed: %s", key, path, exc)
        return None


async def _gather_models() -> tuple[list[dict], dict[str, str], list[str]]:
    """Fan out to every upstream and merge. Returns (providers, owner_map, down)."""
    providers: list[dict] = []
    owner: dict[str, str] = {}
    down: list[str] = []

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_get_json(client, k, "/v1/models") for k in UPSTREAMS),
            return_exceptions=False,
        )

    for key, data in zip(UPSTREAMS, results, strict=True):
        if data is None:
            down.append(key)
            continue
        for p in data.get("providers", []):
            # Stubs from several upstreams would collide on id; keep the first
            # and note which service actually serves each provider.
            if p["id"] in owner:
                continue
            owner[p["id"]] = key
            providers.append({**p, "service": key})

    return providers, owner, down


def _defaults(providers: list[dict]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for cap in ("voice", "music", "sfx"):
        usable = [p for p in providers if p["capability"] == cap and p["available"]]
        local = [p for p in usable if not p["id"].startswith(("elevenlabs", "stub"))]
        non_stub = [p for p in usable if not p["id"].startswith("stub")]
        pick = (local or non_stub or usable)
        out[cap] = pick[0]["id"] if pick else None
    return out


@app.get("/health")
async def health():
    providers, _, down = await _gather_models()
    available = sum(1 for p in providers if p["available"])
    return {
        "status": "ok" if available and not down else ("degraded" if available else "down"),
        "device": "gateway",
        "providers_available": available,
        "providers_total": len(providers),
        "gpu": None,
        "upstreams": {k: ("down" if k in down else "up") for k in UPSTREAMS},
    }


@app.get("/v1/models")
async def list_models():
    providers, _, down = await _gather_models()
    return {
        "device": "multi-service",
        "providers": providers,
        "defaults": _defaults(providers),
        "upstreams_down": down,
    }


async def _resolve_service(capability: str, provider_id: str | None) -> tuple[str, str]:
    """Pick the upstream that serves this request. Returns (service_key, provider_id)."""
    providers, owner, _ = await _gather_models()

    if provider_id:
        if provider_id not in owner:
            raise HTTPException(404, f"unknown provider: {provider_id}")
        match = next(p for p in providers if p["id"] == provider_id)
        if match["capability"] != capability:
            raise HTTPException(400, f"provider {provider_id} does not serve {capability}")
        if not match["available"]:
            raise HTTPException(503, f"provider {provider_id} unavailable: {match['unavailable_reason']}")
        return owner[provider_id], provider_id

    chosen = _defaults(providers).get(capability)
    if not chosen:
        raise HTTPException(503, f"no available provider for {capability}")
    return owner[chosen], chosen


def _tag_job(service: str, payload: dict) -> dict:
    """Prefix the job id so later polls route without server-side state."""
    if payload.get("id"):
        payload["id"] = f"{service}:{payload['id']}"
    # audio_url needs no rewrite: the data volume is shared, so the gateway
    # serves /v1/audio/{id} itself.
    return payload


async def _forward(service: str, path: str, body: dict) -> JSONResponse:
    url = f"{UPSTREAMS[service]}{path}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=body, timeout=FORWARD_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, f"{service} timed out after {FORWARD_TIMEOUT}s") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"cannot reach {service}: {exc}") from exc

    try:
        payload = r.json()
    except ValueError:
        raise HTTPException(502, f"{service} returned a non-JSON response ({r.status_code})") from None

    if r.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise HTTPException(r.status_code, f"{service}: {detail}")

    return JSONResponse(_tag_job(service, payload), status_code=r.status_code)


@app.post("/v1/audio/speech")
async def speech(body: SpeechBody, wait: bool = Query(True)):
    prompt = body.resolved_prompt()
    if not prompt or not prompt.strip():
        raise HTTPException(422, "prompt/input is empty")
    service, provider = await _resolve_service("voice", body.provider)
    payload = body.model_dump(exclude_none=True)
    payload.update({"prompt": prompt, "provider": provider, "voice_id": body.resolved_voice()})
    payload.pop("voice", None)
    payload.pop("input", None)
    return await _forward(service, f"/v1/audio/speech?wait={str(wait).lower()}", payload)


async def _generate(capability: str, body: GenerateBody):
    service, provider = await _resolve_service(capability, body.provider)
    payload = body.model_dump(exclude_none=True)
    payload["provider"] = provider
    return await _forward(service, f"/v1/audio/{capability}", payload)


@app.post("/v1/audio/music")
async def music(body: GenerateBody):
    return await _generate("music", body)


@app.post("/v1/audio/sfx")
async def sfx(body: GenerateBody):
    return await _generate("sfx", body)


def _split_job_id(job_id: str) -> tuple[str, str]:
    if ":" not in job_id:
        raise HTTPException(400, "job id must be prefixed with its service")
    service, real_id = job_id.split(":", 1)
    if service not in UPSTREAMS:
        raise HTTPException(404, f"unknown service: {service}")
    return service, real_id


async def _proxy_job(method: str, job_id: str):
    """Relay a job request, preserving the upstream's own status and detail.

    Unlike discovery, which deliberately collapses failures so one dead backend
    cannot break /v1/models, a job request has exactly one upstream and its
    answer is meaningful: a 404 after that service restarted means "this job
    is gone", not "cannot reach". Only transport failures become 502/504.
    """
    service, real_id = _split_job_id(job_id)
    url = f"{UPSTREAMS[service]}/v1/jobs/{real_id}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.request(method, url, timeout=DISCOVERY_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise HTTPException(504, f"{service} timed out after {DISCOVERY_TIMEOUT}s") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"cannot reach {service}: {exc}") from exc

    try:
        payload = r.json()
    except ValueError:
        raise HTTPException(502, f"{service} returned a non-JSON response ({r.status_code})") from None

    if r.status_code >= 400:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise HTTPException(r.status_code, detail)
    return _tag_job(service, payload)


@app.get("/v1/jobs")
async def list_jobs(limit: int = Query(50, ge=1, le=200)):
    """Recent jobs across every upstream, newest first.

    Each id is prefixed with its service, exactly as the single-job routes
    expect, so an entry from this list can be polled or cancelled directly.
    Like /v1/models, one unreachable upstream degrades the list instead of
    failing it, and is named in `upstreams_down`.
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_get_json(client, k, f"/v1/jobs?limit={limit}") for k in UPSTREAMS)
        )
    jobs: list[dict] = []
    down: list[str] = []
    for key, data in zip(UPSTREAMS, results, strict=True):
        if data is None:
            down.append(key)
            continue
        jobs.extend(_tag_job(key, dict(j)) for j in data.get("jobs", []))
    jobs.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    return {"jobs": jobs[:limit], "upstreams_down": down}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    return await _proxy_job("GET", job_id)


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    return await _proxy_job("DELETE", job_id)


# -- served straight off the shared data volume; no routing needed -------------

@app.get("/v1/audio/{audio_id}")
def get_audio(audio_id: str):
    try:
        path = storage.audio_path(audio_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if path is None:
        raise HTTPException(404, "unknown audio id")
    meta = storage.audio_meta(audio_id) or {}
    return FileResponse(
        path,
        media_type=meta.get("mime", "audio/wav"),
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.delete("/v1/audio/{audio_id}")
def remove_audio(audio_id: str):
    try:
        ok = storage.delete_audio(audio_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not ok:
        raise HTTPException(404, "unknown audio id")
    return {"deleted": audio_id}


@app.get("/v1/voices")
def list_voices():
    return {"voices": storage.list_voice_references()}


@app.post("/v1/voices")
async def add_voice(file: UploadFile = File(...), name: str = Form("")):
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty upload")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "reference sample too large (max 25 MB)")
    suffix = "." + (file.filename or "ref.wav").rsplit(".", 1)[-1].lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(415, f"unsupported audio type: {suffix}")
    return storage.save_voice_reference(data, name or (file.filename or "Reference"), suffix=suffix)


@app.delete("/v1/voices/{voice_id}")
def remove_voice(voice_id: str):
    try:
        ok = storage.delete_voice_reference(voice_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not ok:
        raise HTTPException(404, "unknown voice id")
    return {"deleted": voice_id}


@app.get("/v1/storage")
def storage_usage():
    return storage.usage()
