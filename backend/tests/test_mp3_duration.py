"""MP3 length is measured from the stream, like WAV length.

ElevenLabs returns mp3, and only WAV was measured server-side: ElevenLabsVoice
reported 0.0 s and ElevenLabsSfx the length it was ASKED for. The timeline
copes (it decodes when the length is 0), but every other API consumer got the
wrong number -- including the planned director, which lays out lines by their
measured length.
"""
from __future__ import annotations

import pytest
from helpers import FakeProvider, wait_for_job

from app.providers.base import AudioResult, Capability, SilentOutputError, mp3_duration

# MPEG-1 Layer III, no CRC, 128 kbps, 44.1 kHz: 1152 samples and 417 bytes per
# frame (418 when padded).
MPEG1_128K_44K = bytes([0xFF, 0xFB, 0x90, 0x00])
MPEG1_128K_44K_PADDED = bytes([0xFF, 0xFB, 0x92, 0x00])
# MPEG-2 Layer III, 64 kbps, 24 kHz: 576 samples, 72*64000/24000 = 192 bytes.
MPEG2_64K_24K = bytes([0xFF, 0xF3, 0x84, 0x00])


def _frames(header: bytes, size: int, count: int) -> bytes:
    return (header + bytes(size - 4)) * count


def test_cbr_frames():
    assert mp3_duration(_frames(MPEG1_128K_44K, 417, 100)) == pytest.approx(100 * 1152 / 44100)


def test_padded_frames_and_an_id3_tag_are_handled():
    tag_body = bytes(300)
    size = len(tag_body)
    synchsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    id3 = b"ID3\x04\x00\x00" + synchsafe + tag_body
    data = id3 + _frames(MPEG1_128K_44K_PADDED, 418, 40) + _frames(MPEG1_128K_44K, 417, 10)
    assert mp3_duration(data) == pytest.approx(50 * 1152 / 44100)


def test_mpeg2_frames():
    assert mp3_duration(_frames(MPEG2_64K_24K, 192, 125)) == pytest.approx(125 * 576 / 24000)


def test_junk_before_the_first_frame_is_skipped():
    assert mp3_duration(b"\x00\x01garbage" + _frames(MPEG1_128K_44K, 417, 20)) == pytest.approx(20 * 1152 / 44100)


def test_no_frames_is_rejected():
    with pytest.raises(ValueError):
        mp3_duration(b"not an mp3 at all" * 50)


class Mp3Provider(FakeProvider):
    """Returns mp3 and, like ElevenLabsVoice, reports no length of its own."""

    def __init__(self, data: bytes) -> None:
        super().__init__("cloud-voice", Capability.VOICE, remote=True)
        self._mp3 = data

    def generate(self, req):
        return AudioResult(audio=self._mp3, sample_rate=44100, duration=0.0, provider_id=self.id, mime="audio/mpeg")


def test_api_reports_the_measured_mp3_length(api, use_providers):
    use_providers(Mp3Provider(_frames(MPEG1_128K_44K, 417, 100)))
    job = wait_for_job(api, api.post("/v1/audio/speech?wait=false", json={"input": "hi", "provider": "cloud-voice"}).json()["id"])
    assert job["status"] == "done"
    assert job["meta"]["duration"] == pytest.approx(100 * 1152 / 44100, abs=1e-3)
    assert job["meta"]["mime"] == "audio/mpeg"


def test_an_mp3_with_no_audio_fails_the_job(api, use_providers):
    use_providers(Mp3Provider(b"<html>quota exceeded</html>"))
    job = wait_for_job(api, api.post("/v1/audio/speech?wait=false", json={"input": "hi", "provider": "cloud-voice"}).json()["id"])
    assert job["status"] == "error"
    assert job["error"].startswith(SilentOutputError.__name__)
