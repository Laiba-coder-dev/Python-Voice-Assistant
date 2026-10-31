"""Unit tests for ElevenLabs speech-to-text (network fully mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from elevenlabs.core.api_error import ApiError

from voice_assistant.config import Settings
from voice_assistant.errors import (
    AuthenticationError,
    EmptyAudioError,
    RateLimitError,
    SpeechRecognitionError,
)
from voice_assistant.speech_to_text import SpeechToText, TranscriptResult


class _FakeConvert:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class _FakeSpeechToText:
    def __init__(self, convert):
        self.convert = convert


class _FakeClient:
    def __init__(self, convert):
        self.speech_to_text = _FakeSpeechToText(convert)


def _settings():
    return Settings(elevenlabs_api_key="test-key")


def test_transcribe_pcm_returns_text_and_metadata():
    response = SimpleNamespace(
        text="  Hello world.  ", language_code="eng", language_probability=0.99
    )
    convert = _FakeConvert(response=response)
    stt = SpeechToText(_FakeClient(convert), _settings())

    pcm = np.zeros(16000, dtype=np.int16)  # 1 second
    result = stt.transcribe_pcm(pcm, 16000)

    assert isinstance(result, TranscriptResult)
    assert result.text == "Hello world."
    assert result.language_code == "eng"
    assert result.duration_seconds == pytest.approx(1.0)
    # The file argument should be a (name, bytes, mime) tuple.
    file_arg = convert.kwargs["file"]
    assert file_arg[0] == "speech.wav"
    assert file_arg[1][:4] == b"RIFF"
    assert convert.kwargs["model_id"] == "scribe_v2"


def test_transcribe_pcm_too_short_raises_empty_audio():
    convert = _FakeConvert(response=SimpleNamespace(text="x"))
    stt = SpeechToText(_FakeClient(convert), _settings())

    with pytest.raises(EmptyAudioError):
        stt.transcribe_pcm(np.zeros(100, dtype=np.int16), 16000)


def test_transcribe_multichannel_response_is_joined():
    response = SimpleNamespace(
        transcripts=[
            SimpleNamespace(text="First part.", language_code="eng", language_probability=0.9),
            SimpleNamespace(text="Second part.", language_code="eng", language_probability=0.9),
        ]
    )
    stt = SpeechToText(_FakeClient(_FakeConvert(response=response)), _settings())
    result = stt.transcribe_bytes(b"RIFFxxxx", filename="a.wav")
    assert result.text == "First part. Second part."


def test_transcribe_maps_auth_error():
    error = ApiError(status_code=401, body="unauthorized")
    stt = SpeechToText(_FakeClient(_FakeConvert(error=error)), _settings())
    with pytest.raises(AuthenticationError):
        stt.transcribe_bytes(b"RIFFxxxx")


def test_transcribe_maps_rate_limit():
    error = ApiError(status_code=429, body="slow down")
    stt = SpeechToText(_FakeClient(_FakeConvert(error=error)), _settings())
    with pytest.raises(RateLimitError):
        stt.transcribe_bytes(b"RIFFxxxx")


def test_transcribe_async_webhook_response_raises():
    response = SimpleNamespace(message="queued")
    stt = SpeechToText(_FakeClient(_FakeConvert(response=response)), _settings())
    with pytest.raises(SpeechRecognitionError):
        stt.transcribe_bytes(b"RIFFxxxx")
