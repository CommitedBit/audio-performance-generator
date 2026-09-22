"""Optional shared-secret auth.

Off by default: with API_KEY unset every request passes, which is right for a
loopback-bound single-user box. Set API_KEY before exposing the service on a
LAN or through a tunnel -- an open endpoint means anyone who can reach it can
monopolise the GPU and read or upload voice-clone reference samples.

Deliberately a static shared secret rather than user accounts: this is a
single-user tool, and a login system would be more surface, not less.

Implemented as middleware rather than an app-level dependency so /health can be
exempt -- container healthchecks and orchestration probes must not need a key,
or the container reports unhealthy forever.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse

# Probes and CORS preflight must work without a key.
OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


async def api_key_middleware(request: Request, call_next):
    expected = os.getenv("API_KEY") or ""
    if not expected or request.url.path in OPEN_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    supplied = request.headers.get("x-api-key") or ""
    # compare_digest: constant time, so a wrong key cannot be recovered by
    # timing how long the rejection takes.
    if not supplied or not hmac.compare_digest(supplied, expected):
        return JSONResponse({"detail": "missing or invalid X-API-Key"}, status_code=401)

    return await call_next(request)
