"""Per-capability health, shared by the model services and the gateway.

The old /health said "ok" whenever any provider was available, which the
placeholder tone generators always are -- so a GPU box whose real models had
all failed reported healthy while serving test tones. Health is now judged per
capability, and only a LOCAL, real provider counts as ok:

    ok     a local model can serve it
    cloud  only a cloud provider (ElevenLabs) can -- works, but sends data out
    stub   only a placeholder can (development)
    down   nothing can

Works on provider summaries (dicts), because the gateway only ever sees
providers as JSON from its upstreams.
"""
from __future__ import annotations

import logging

CAPABILITIES = ("voice", "music", "sfx")


def is_stub(provider_id: str) -> bool:
    return provider_id.startswith("stub")


def summary(provider) -> dict:
    """The fields health needs, from a Provider instance."""
    return {
        "id": provider.id,
        "capability": provider.capability.value,
        "available": provider.available(),
        "remote": provider.remote,
        "unavailable_reason": provider.unavailable_reason(),
    }


def capability_status(providers: list[dict], capability: str) -> dict | None:
    """Status of one capability, or None if nothing here serves it at all."""
    served = [p for p in providers if p["capability"] == capability]
    if not served:
        return None
    usable = [p for p in served if p["available"]]
    if any(not p.get("remote") and not is_stub(p["id"]) for p in usable):
        status = "ok"
    elif any(p.get("remote") for p in usable):
        status = "cloud"
    elif usable:
        status = "stub"
    else:
        status = "down"
    return {"status": status, "available": [p["id"] for p in usable]}


def summarize(providers: list[dict]) -> tuple[str, dict[str, dict]]:
    """(overall status, per-capability status) for the capabilities present."""
    caps = {}
    for cap in CAPABILITIES:
        status = capability_status(providers, cap)
        if status is not None:
            caps[cap] = status
    statuses = {c["status"] for c in caps.values()}
    if not caps or statuses == {"down"}:
        overall = "down"
    elif statuses == {"ok"}:
        overall = "ok"
    elif statuses == {"stub"}:
        overall = "stub"                     # a dev stack, working as intended
    else:
        overall = "degraded"
    return overall, caps


def log_problems(providers: list[dict], log: logging.Logger) -> None:
    """Say loudly, once at startup, which capabilities cannot run locally.

    Only capabilities this service actually serves are checked, so a split
    service that hosts music alone does not report voice and sfx missing.
    """
    _, caps = summarize(providers)
    for cap, status in caps.items():
        if status["status"] == "ok":
            continue
        reasons = "; ".join(
            f"{p['id']}: {p['unavailable_reason'] or 'available'}" for p in providers if p["capability"] == cap
        )
        if status["status"] == "stub":
            log.warning("%s is served by a placeholder only (DEV_STUB): %s", cap, reasons)
        elif status["status"] == "cloud":
            log.warning("%s has no local model; only the cloud provider can serve it: %s", cap, reasons)
        else:
            log.error("%s has NO available provider: %s", cap, reasons)
