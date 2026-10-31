"""Unit tests for the conversation engine and the voice-assistant state machine."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from voice_assistant.assistant import (
    ConversationEngine,
    SessionState,
    VoiceAssistant,
)
from voice_assistant.config import Settings
from voice_assistant.errors import (
    ConfigurationError,
    ConversationError,
    EmptyAudioError,
    SpeechRecognitionError,
)
from voice_assistant.llm import OfflineBackend
from voice_assistant.speech_to_text import TranscriptResult


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakePlayer:
    def __init__(self):
        self.played = []
        self.stopped = False

    @property
    def is_playing(self):
        return False

    def stop(self):
        self.stopped = True

    def play_stream(self, chunks, sample_rate=None):
        self.played.append((b"".join(chunks), sample_rate))

    def play(self, data, sample_rate=None):
        self.played.append((data, sample_rate))


class FakeTTS:
    def __init__(self, error=None):
        self.error = error
        self.texts = []

    def synthesize_stream(self, text):
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        yield b"pcm-bytes"


class FakeSTT:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def transcribe_pcm(self, pcm, sample_rate):
        self.calls.append((pcm, sample_rate))
        if self.error is not None:
            raise self.error
        return self.result or TranscriptResult(text="")


def _make_assistant(*, stt=None, tts=None, player=None, engine=None, settings=None):
    settings = settings or Settings(
        elevenlabs_api_key="test", llm_provider="offline", echo_guard=True
    )
    engine = engine or ConversationEngine(backend=OfflineBackend(), assistant_name="Test")
    events = {
        "states": [],
        "users": [],
        "assistants": [],
        "errors": [],
        "infos": [],
    }
    assistant = VoiceAssistant(
        settings,
        client=object(),
        stt=stt,
        tts=tts or FakeTTS(),
        engine=engine,
        recorder=object(),
        player=player or FakePlayer(),
        on_state=lambda s: events["states"].append(s),
        on_user=lambda t: events["users"].append(t),
        on_assistant=lambda t: events["assistants"].append(t),
        on_error=lambda t: events["errors"].append(t),
        on_info=lambda t: events["infos"].append(t),
    )
    return assistant, events


# --------------------------------------------------------------------------- #
# ConversationEngine
# --------------------------------------------------------------------------- #
def test_engine_stores_history():
    engine = ConversationEngine(backend=OfflineBackend())
    reply = engine.respond("Hello")
    assert reply
    assert len(engine.history) == 2
    assert engine.history[0]["role"] == "user"
    assert engine.history[1]["role"] == "assistant"


def test_engine_trims_history():
    engine = ConversationEngine(backend=OfflineBackend(), max_history_turns=1)
    for _ in range(4):
        engine.respond("Hello")
    assert len(engine.history) <= 2


def test_engine_rolls_back_failed_turn():
    class _Boom:
        name = "boom"

        def generate(self, messages):
            raise ConversationError("nope")

    engine = ConversationEngine(backend=_Boom())
    with pytest.raises(ConversationError):
        engine.respond("Hello")
    assert engine.history == []


def test_engine_reset():
    engine = ConversationEngine(backend=OfflineBackend())
    engine.respond("Hello")
    engine.reset()
    assert engine.history == []


# --------------------------------------------------------------------------- #
# VoiceAssistant
# --------------------------------------------------------------------------- #
def test_handle_text_full_turn():
    player = FakePlayer()
    assistant, events = _make_assistant(player=player)

    reply = assistant.handle_text("Hello")

    assert reply is not None
    assert events["users"] == ["Hello"]
    assert events["assistants"] == [reply]
    assert SessionState.THINKING in events["states"]
    assert SessionState.SPEAKING in events["states"]
    assert events["states"][-1] == SessionState.IDLE
    assert player.played and player.played[0][0] == b"pcm-bytes"


def test_handle_text_without_speaking():
    player = FakePlayer()
    assistant, _ = _make_assistant(player=player)
    assistant.set_speaking_enabled(False)

    assistant.handle_text("Hello")
    assert player.played == []


def test_handle_pcm_transcribes_and_responds():
    stt = FakeSTT(result=TranscriptResult(text="who are you?"))
    assistant, events = _make_assistant(stt=stt)

    assistant.handle_pcm(np.zeros(16000, dtype=np.int16), 16000)

    assert events["users"] == ["who are you?"]
    assert events["assistants"]
    assert SessionState.TRANSCRIBING in events["states"]


def test_handle_pcm_empty_transcript_is_graceful():
    stt = FakeSTT(result=TranscriptResult(text="   "))
    assistant, events = _make_assistant(stt=stt)

    result = assistant.handle_pcm(np.zeros(16000, dtype=np.int16), 16000)

    assert result is None
    assert events["users"] == []
    assert events["infos"]


def test_handle_pcm_empty_audio_error_is_graceful():
    stt = FakeSTT(error=EmptyAudioError("too short"))
    assistant, events = _make_assistant(stt=stt)

    result = assistant.handle_pcm(np.zeros(16000, dtype=np.int16), 16000)

    assert result is None
    assert events["infos"]
    assert events["errors"] == []


def test_handle_pcm_recognition_error_reports():
    stt = FakeSTT(error=SpeechRecognitionError("boom"))
    assistant, events = _make_assistant(stt=stt)

    assistant.handle_pcm(np.zeros(16000, dtype=np.int16), 16000)

    assert events["errors"]
    assert events["users"] == []


def test_echo_guard_ignores_own_speech():
    assistant, events = _make_assistant()
    assistant._last_spoken = "The capital of France is Paris."

    assert assistant._is_echo("the capital of France is Paris") is True
    assert assistant._is_echo("What's the weather today?") is False

    # A matching transcript is ignored entirely.
    events["users"].clear()
    assistant.handle_text("The capital of France is Paris.")
    assert events["users"] == []


def test_missing_api_key_raises_configuration_error():
    assistant = VoiceAssistant(Settings(elevenlabs_api_key=None), client=object())
    with pytest.raises(ConfigurationError):
        assistant.handle_text("Hello")


def test_stop_and_close_are_idempotent():
    player = FakePlayer()
    assistant, _ = _make_assistant(player=player)

    assistant.stop()
    assert assistant.stop_event.is_set()
    assert player.stopped

    assistant.close()
    assistant.close()  # must not raise


def test_tts_failure_reports_error_but_keeps_state_clean():
    assistant, events = _make_assistant(tts=FakeTTS(error=SpeechRecognitionError("tts down")))

    reply = assistant.handle_text("Hello")

    assert reply is not None  # the text reply still succeeded
    assert events["errors"]


def test_stop_event_stops_run_loop():
    # Provide a recorder whose record_utterance blocks on the stop event.
    class BlockingRecorder:
        def record_utterance(self, stop_event, timeout=None):
            stop_event.wait(2.0)
            return None

    assistant, _ = _make_assistant()
    assistant._recorder = BlockingRecorder()

    stop_event = threading.Event()
    thread = threading.Thread(target=assistant.run, args=(stop_event,), daemon=True)
    thread.start()
    stop_event.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert assistant.state == SessionState.STOPPED
