"""Defaults that live in several files must agree.

The same setting is defaulted in the code, the compose files, .env.example and
the proxies in front of the gateway. They drifted before: the idle timeout was
600 s in code and 1800 s everywhere else, and the gateway's forward timeout
was 900 s in compose while nginx gave up at 600 s -- so a slow request ended in
a bare nginx 504 instead of the gateway's own error naming the slow service.
"""
from __future__ import annotations

import re
from pathlib import Path

from helpers import reload_settings

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _compose_default(rel: str, var: str) -> str:
    m = re.search(rf"{var}: \$\{{{var}:-([^}}]*)\}}", _read(rel))
    assert m, f"{var} has no ${{{var}:-default}} in {rel}"
    return m.group(1)


def _env_example(var: str) -> str:
    m = re.search(rf"^{var}=(.*)$", _read(".env.example"), re.M)
    assert m, f"{var} missing from .env.example"
    return m.group(1).strip()


def test_idle_timeout_default_agrees_everywhere(monkeypatch):
    reload_settings(monkeypatch, MODEL_IDLE_TIMEOUT=None)
    from app.config import get_settings

    code = str(get_settings().model_idle_timeout)
    assert code == _env_example("MODEL_IDLE_TIMEOUT")
    for rel in ("compose.gpu.yml", "compose.gpu.split.yml"):
        assert _compose_default(rel, "MODEL_IDLE_TIMEOUT") == code, rel


def test_forward_timeout_default_agrees_and_the_proxies_outlast_it():
    m = re.search(r'os\.getenv\("FORWARD_TIMEOUT", "(\d+)"\)', _read("backend/app/gateway.py"))
    assert m, "gateway.py no longer reads FORWARD_TIMEOUT with a literal default"
    forward = int(m.group(1))
    assert int(_compose_default("compose.gpu.yml", "FORWARD_TIMEOUT")) == forward
    assert int(_env_example("FORWARD_TIMEOUT")) == forward

    nginx = _read("frontend/nginx.conf")
    for directive in ("proxy_read_timeout", "proxy_send_timeout"):
        value = int(re.search(rf"{directive}\s+(\d+)s;", nginx).group(1))
        assert value > forward, f"nginx {directive} {value}s must outlast the gateway's {forward}s"

    vite = _read("frontend/vite.config.ts")
    for key in ("timeout", "proxyTimeout"):
        ms = int(re.search(rf"\b{key}: ([\d_]+),", vite).group(1).replace("_", ""))
        assert ms > forward * 1000, f"vite {key} {ms} ms must outlast the gateway's {forward}s"
