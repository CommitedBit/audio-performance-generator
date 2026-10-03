"""On-disk storage for generated audio and voice-clone reference samples."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .config import get_settings

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe(audio_id: str) -> str:
    """Reject anything that could escape the data directory.

    Ids come from URL path params, so this is the trust boundary.
    """
    if not _ID_RE.match(audio_id):
        raise ValueError(f"invalid id: {audio_id!r}")
    return audio_id


def save_audio(data: bytes, *, suffix: str = ".wav", meta: dict | None = None) -> str:
    s = get_settings()
    audio_id = uuid.uuid4().hex
    (s.audio_dir / f"{audio_id}{suffix}").write_bytes(data)
    record = {"id": audio_id, "suffix": suffix, "bytes": len(data), "created_at": time.time(), **(meta or {})}
    (s.audio_dir / f"{audio_id}.json").write_text(json.dumps(record, indent=2))
    return audio_id


def audio_path(audio_id: str) -> Path | None:
    s = get_settings()
    audio_id = _safe(audio_id)
    for p in s.audio_dir.glob(f"{audio_id}.*"):
        if p.suffix != ".json":
            return p
    return None


def audio_meta(audio_id: str) -> dict | None:
    s = get_settings()
    p = s.audio_dir / f"{_safe(audio_id)}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def delete_audio(audio_id: str) -> bool:
    s = get_settings()
    audio_id = _safe(audio_id)
    found = False
    for p in s.audio_dir.glob(f"{audio_id}.*"):
        p.unlink(missing_ok=True)
        found = True
    return found


# -- voice clone references ---------------------------------------------------

def save_voice_reference(data: bytes, name: str, *, suffix: str = ".wav") -> dict:
    s = get_settings()
    voice_id = uuid.uuid4().hex
    (s.voices_dir / f"{voice_id}{suffix}").write_bytes(data)
    record = {
        "id": voice_id,
        "name": name or f"Voice {voice_id[:6]}",
        "suffix": suffix,
        "bytes": len(data),
        "created_at": time.time(),
    }
    (s.voices_dir / f"{voice_id}.json").write_text(json.dumps(record, indent=2))
    return record


def voice_reference_path(voice_id: str) -> Path | None:
    s = get_settings()
    voice_id = _safe(voice_id)
    for p in s.voices_dir.glob(f"{voice_id}.*"):
        if p.suffix != ".json":
            return p
    return None


def list_voice_references() -> list[dict]:
    s = get_settings()
    out = []
    for p in sorted(s.voices_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def delete_voice_reference(voice_id: str) -> bool:
    s = get_settings()
    voice_id = _safe(voice_id)
    found = False
    for p in s.voices_dir.glob(f"{voice_id}.*"):
        p.unlink(missing_ok=True)
        found = True
    return found


def usage() -> dict:
    s = get_settings()
    def total(d: Path) -> tuple[int, int]:
        files = [p for p in d.glob("*") if p.suffix != ".json"]
        return len(files), sum(p.stat().st_size for p in files)
    n_audio, b_audio = total(s.audio_dir)
    n_voice, b_voice = total(s.voices_dir)
    return {
        "audio_files": n_audio,
        "audio_bytes": b_audio,
        "voice_files": n_voice,
        "voice_bytes": b_voice,
        "data_dir": str(s.data_dir),
    }
