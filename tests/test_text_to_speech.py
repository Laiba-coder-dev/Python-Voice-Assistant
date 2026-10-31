"""Unit tests for ElevenLabs text-to-speech (network fully mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from elevenlabs.core.api_error import ApiError

from voice_assistant.config import Settings
from voice_assistant.errors import AuthenticationError, RateLimitError
from voice_assistant.text_to_speech import TextToSpeech, split_for_speech


class _FakeTTS:
    def __init__(self, pieces=(b"a", b"b"), error=None):
        self.pieces = list(pieces)
        self.error = error
        self.convert_calls = []
        self.stream_calls = []

    def convert(self, **kwargs):
        self.convert_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return iter(self.pieces)

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return iter(self.pieces)


class _FakeVoices:
    def search(self, page_size=100):
        return SimpleNamespace(
            voices=[
                SimpleNamespace(voice_id="v1", name="Voice One", labels={"accent": "us"}),
                SimpleNamespace(voice_id="v2", name="Voice Two", labels=None),
            ]
        )


class _FakeClient:
    def __init__(self, tts):
        self.text_to_speech = tts
        self.voices = _FakeVoices()


def _settings():
    return Settings(elevenlabs_api_key="test-key", tts_output_format="pcm_24000")


def test_synthesize_concatenates_pcm():
    tts = _FakeTTS(pieces=(b"aa", b"bb"))
    service = TextToSpeech(_FakeClient(tts), _settings())
    assert service.synthesize("Hello there.") == b"aabb"
    assert tts.convert_calls[0]["voice_id"] == _settings().tts_voice_id
    assert tts.convert_calls[0]["output_format"] == "pcm_24000"


def test_synthesize_stream_yields_pieces():
    tts = _FakeTTS(pieces=(b"x", b"y"))
    service = TextToSpeech(_FakeClient(tts), _settings())
    assert b"".join(service.synthesize_stream("Hi.")) == b"xy"


def test_long_text_is_split_into_multiple_calls():
    tts = _FakeTTS()
    service = TextToSpeech(_FakeClient(tts), _settings())
    long_text = " ".join(["This is a sentence number %d." % i for i in range(200)])
    service.synthesize(long_text)
    assert len(tts.convert_calls) > 1


def test_empty_text_produces_no_calls():
    tts = _FakeTTS()
    service = TextToSpeech(_FakeClient(tts), _settings())
    assert service.synthesize("   ") == b""
    assert tts.convert_calls == []


def test_synthesize_maps_auth_error():
    error = ApiError(status_code=401, body="nope")
    service = TextToSpeech(_FakeClient(_FakeTTS(error=error)), _settings())
    with pytest.raises(AuthenticationError):
        service.synthesize("hello")


def test_synthesize_maps_rate_limit():
    error = ApiError(status_code=429, body="slow")
    service = TextToSpeech(_FakeClient(_FakeTTS(error=error)), _settings())
    with pytest.raises(RateLimitError):
        service.synthesize("hello")


def test_list_voices():
    service = TextToSpeech(_FakeClient(_FakeTTS()), _settings())
    voices = service.list_voices()
    assert voices[0]["id"] == "v1"
    assert voices[0]["name"] == "Voice One"


def test_split_for_speech_respects_max_chars():
    text = "Hello there. " * 100
    chunks = split_for_speech(text, max_chars=50)
    assert chunks
    assert all(len(c) <= 50 for c in chunks)


def test_split_for_speech_long_single_sentence():
    word = "supercalifragilistic " * 40
    chunks = split_for_speech(word.strip(), max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_split_for_speech_empty():
    assert split_for_speech("   ") == []
