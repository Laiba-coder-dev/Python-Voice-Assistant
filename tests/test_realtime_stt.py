"""Unit tests for the realtime transcription wrapper (WebSocket mocked)."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

from elevenlabs.realtime.connection import RealtimeEvents

from voice_assistant.config import Settings
from voice_assistant.errors import AuthenticationError, SpeechRecognitionError
from voice_assistant.realtime_stt import RealtimeTranscriber


class FakeConnection:
    def __init__(self):
        self.handlers = {}
        self.sent = []
        self.commits = 0
        self.closed = False

    def on(self, event, callback):
        self.handlers.setdefault(event, []).append(callback)

    async def send(self, data):
        self.sent.append(data)

    async def commit(self):
        self.commits += 1

    async def close(self):
        self.closed = True


class FakeRealtime:
    def __init__(self, connection):
        self.connection = connection
        self.options = None

    async def connect(self, options):
        self.options = options
        return self.connection


class FakeClient:
    def __init__(self, realtime):
        self.speech_to_text = SimpleNamespace(realtime=realtime)


def _settings():
    return Settings(elevenlabs_api_key="test", sample_rate=16000)


def test_realtime_connects_and_parses_events():
    connection = FakeConnection()
    realtime = FakeRealtime(connection)
    committed = []
    partials = []
    errors = []
    opened = []

    transcriber = RealtimeTranscriber(
        FakeClient(realtime),
        _settings(),
        on_partial=partials.append,
        on_committed=committed.append,
        on_error=errors.append,
        on_open=lambda: opened.append(True),
    )

    asyncio.run(transcriber.start())
    assert realtime.options["model_id"] == "scribe_v2"
    assert realtime.options["sample_rate"] == 16000

    for cb in connection.handlers[RealtimeEvents.OPEN]:
        cb({})
    for cb in connection.handlers[RealtimeEvents.PARTIAL_TRANSCRIPT]:
        cb({"transcript": "hel"})
    for cb in connection.handlers[RealtimeEvents.COMMITTED_TRANSCRIPT]:
        cb({"transcript": "hello"})

    assert opened == [True]
    assert partials == ["hel"]
    assert committed == ["hello"]


def test_realtime_sends_base64_audio_and_commits():
    connection = FakeConnection()
    transcriber = RealtimeTranscriber(FakeClient(FakeRealtime(connection)), _settings())

    asyncio.run(transcriber.start())
    asyncio.run(transcriber.send_audio(b"\x01\x02\x03"))
    asyncio.run(transcriber.commit())
    asyncio.run(transcriber.close())

    assert base64.b64decode(connection.sent[0]["audio_base_64"]) == b"\x01\x02\x03"
    assert connection.commits == 1
    assert connection.closed is True


def test_realtime_auth_error_is_classified():
    connection = FakeConnection()
    errors = []
    transcriber = RealtimeTranscriber(
        FakeClient(FakeRealtime(connection)), _settings(), on_error=errors.append
    )
    asyncio.run(transcriber.start())

    for cb in connection.handlers[RealtimeEvents.AUTH_ERROR]:
        cb({"message_type": "auth_error"})

    assert len(errors) == 1
    assert isinstance(errors[0], AuthenticationError)


def test_realtime_transcriber_error_is_classified():
    connection = FakeConnection()
    errors = []
    transcriber = RealtimeTranscriber(
        FakeClient(FakeRealtime(connection)), _settings(), on_error=errors.append
    )
    asyncio.run(transcriber.start())

    for cb in connection.handlers[RealtimeEvents.TRANSCRIBER_ERROR]:
        cb({"message_type": "transcriber_error", "message": "bad audio"})

    assert isinstance(errors[0], SpeechRecognitionError)
