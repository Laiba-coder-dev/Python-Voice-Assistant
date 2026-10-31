"""Conversation management and the voice-assistant state machine.

:class:`ConversationEngine` owns the message history and delegates reply
generation to a pluggable backend (see :mod:`voice_assistant.llm`).

:class:`VoiceAssistant` wires the four independent concerns together - audio
capture, transcription, response generation and speech synthesis - and drives
the state machine::

    IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> SPEAKING -> LISTENING ...

It is deliberately UI-agnostic: the CLI and the Streamlit app both consume the
same callbacks and methods.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Callable, Optional

from elevenlabs import ElevenLabs

from .audio_handler import (
    AudioDeviceError,
    AudioError,
    MicrophoneRecorder,
    PCMPlayer,
    decode_audio,
)
from .config import ConfigError, Settings
from .errors import (
    ConfigurationError,
    EmptyAudioError,
    SpeechSynthesisError,
    VoiceAssistantError,
)
from .llm import LLMBackend, build_backend
from .logging_utils import get_logger
from .speech_to_text import SpeechToText, TranscriptResult
from .text_to_speech import TextToSpeech

log = get_logger(__name__)


class SessionState(str, Enum):
    """Visible states of the assistant."""

    IDLE = "Idle"
    LISTENING = "Listening"
    TRANSCRIBING = "Transcribing"
    THINKING = "Thinking"
    SPEAKING = "Speaking"
    ERROR = "Error"
    STOPPED = "Stopped"


# Callback type aliases.
StateCallback = Callable[[SessionState], None]
TextCallback = Callable[[str], None]
MessageCallback = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Conversation engine
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ConversationEngine:
    """Maintains history and produces context-aware replies."""

    backend: LLMBackend
    assistant_name: str = "Aria"
    system_prompt: str = ""
    max_history_turns: int = 12
    _messages: list[dict[str, str]] = field(default_factory=list, repr=False)
    _system: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._system = self._build_system_message()

    def _build_system_message(self) -> str:
        now = datetime.now()
        parts = [
            f"You are {self.assistant_name}, a voice assistant.",
            self.system_prompt.strip(),
            "The user is speaking to you out loud and will hear your reply "
            "spoken. Replies must be plain prose suitable for text to speech.",
            f"Current date and time: {now.strftime('%A, %d %B %Y, %H:%M')}.",
        ]
        return "\n".join(part for part in parts if part)

    def _request_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self._system}, *self._messages]

    def respond(self, user_text: str) -> str:
        """Return the assistant's reply to ``user_text``."""
        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("Cannot respond to empty text.")

        self._messages.append({"role": "user", "content": user_text})
        self._trim_history()
        try:
            reply = self.backend.generate(self._request_messages())
        except Exception:
            # Do not keep an unmatched user turn if generation failed.
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            raise
        self._messages.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply

    def _trim_history(self) -> None:
        limit = max(1, self.max_history_turns) * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

    def reset(self) -> None:
        self._messages.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._messages)


# --------------------------------------------------------------------------- #
# Voice assistant
# --------------------------------------------------------------------------- #
class VoiceAssistant:
    """Drives the full listen -> transcribe -> think -> speak loop."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: ElevenLabs | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        engine: ConversationEngine | None = None,
        recorder: MicrophoneRecorder | None = None,
        player: PCMPlayer | None = None,
        on_state: Optional[StateCallback] = None,
        on_user: Optional[TextCallback] = None,
        on_assistant: Optional[TextCallback] = None,
        on_error: Optional[MessageCallback] = None,
        on_info: Optional[MessageCallback] = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._stt = stt
        self._tts = tts
        self._engine = engine
        self._recorder = recorder
        self._player = player

        self._on_state = on_state
        self._on_user = on_user
        self._on_assistant = on_assistant
        self._on_error = on_error
        self._on_info = on_info

        self._turn_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._last_spoken = ""
        self._ready = False
        self._closed = False
        self._state = SessionState.IDLE
        self._running = False
        self._speak_enabled = True

    # -- properties ----------------------------------------------------- #
    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def engine(self) -> ConversationEngine:
        self._ensure_ready()
        assert self._engine is not None
        return self._engine

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    # -- lifecycle ------------------------------------------------------ #
    def _ensure_ready(self) -> None:
        if self._ready:
            return
        settings = self.settings
        try:
            settings.require_elevenlabs()
        except ConfigError as exc:
            raise ConfigurationError(str(exc)) from exc

        if self._client is None:
            self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        assert self._client is not None
        if self._stt is None:
            self._stt = SpeechToText(self._client, settings)
        if self._tts is None:
            self._tts = TextToSpeech(self._client, settings)
        if self._engine is None:
            self._engine = ConversationEngine(
                backend=build_backend(settings),
                assistant_name=settings.assistant_name,
                system_prompt=settings.system_prompt,
                max_history_turns=settings.max_history_turns,
            )
        if self._recorder is None:
            self._recorder = MicrophoneRecorder(
                sample_rate=settings.sample_rate,
                device=settings.input_device,
                threshold_factor=settings.vad_threshold_factor,
                min_threshold=settings.vad_min_threshold,
                silence_duration=settings.silence_duration,
                min_speech_duration=settings.min_speech_duration,
                max_record_seconds=settings.max_record_seconds,
            )
        if self._player is None:
            self._player = PCMPlayer(
                sample_rate=settings.tts_sample_rate,
                device=settings.output_device,
            )
        self._ready = True
        log.info("Assistant ready: %s", settings.public_summary())

    def run(self, stop_event: threading.Event | None = None) -> None:
        """Continuously listen and respond until ``stop_event`` is set."""
        if stop_event is not None:
            self._stop_event = stop_event
        try:
            self._ensure_ready()
        except VoiceAssistantError as exc:
            self._fail(exc)
            return
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            return

        self._running = True
        self._set_state(SessionState.LISTENING)
        self._emit_info(
            f"{self.settings.assistant_name} is listening. Speak whenever you're ready."
        )
        try:
            while not self._stop_event.is_set():
                try:
                    audio = self._recorder.record_utterance(
                        self._stop_event, timeout=self.settings.listen_timeout
                    )
                except AudioDeviceError as exc:
                    self._fail(exc)
                    break
                if self._stop_event.is_set():
                    break
                if audio is None:
                    continue
                self.handle_pcm(audio, self.settings.sample_rate)
        except Exception as exc:  # noqa: BLE001 - never crash the loop silently
            self._fail(exc)
        finally:
            self._running = False
            self.close()
            self._set_state(SessionState.STOPPED)

    # -- per-turn handlers ---------------------------------------------- #
    def handle_pcm(self, pcm, sample_rate: int) -> str | None:
        """Transcribe and respond to a captured utterance."""
        self._ensure_ready()
        with self._turn_lock:
            self._set_state(SessionState.TRANSCRIBING)
            try:
                result: TranscriptResult = self._stt.transcribe_pcm(pcm, sample_rate)
            except EmptyAudioError:
                self._emit_info("I didn't catch any speech in that recording.")
                self._idle_state()
                return None
            except VoiceAssistantError as exc:
                self._fail(exc)
                self._idle_state(SessionState.LISTENING if self._running else SessionState.IDLE)
                return None

            text = result.text.strip()
            if not text:
                self._emit_info("I couldn't make out what was said.")
                self._idle_state()
                return None
            return self._respond(text)

    def handle_audio_bytes(self, data: bytes) -> str | None:
        """Decode an uploaded audio clip and process it."""
        self._ensure_ready()
        try:
            pcm, sample_rate = decode_audio(data)
        except AudioError as exc:
            self._fail(exc)
            self._idle_state()
            return None
        return self.handle_pcm(pcm, sample_rate)

    def handle_text(self, text: str, *, speak: bool = True) -> str | None:
        """Process a typed message (the optional text-input path)."""
        self._ensure_ready()
        text = (text or "").strip()
        if not text:
            return None
        with self._turn_lock:
            return self._respond(text, speak=speak)

    def _respond(self, user_text: str, *, speak: bool = True) -> str | None:
        if self._is_echo(user_text):
            log.debug("Ignoring suspected echo of the assistant's own speech.")
            self._idle_state(SessionState.LISTENING if self._running else SessionState.IDLE)
            return None

        self._emit_user(user_text)
        self._set_state(SessionState.THINKING)
        try:
            assert self._engine is not None
            reply = self._engine.respond(user_text)
        except VoiceAssistantError as exc:
            self._fail(exc)
            self._idle_state(SessionState.LISTENING if self._running else SessionState.IDLE)
            return None
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            self._idle_state(SessionState.LISTENING if self._running else SessionState.IDLE)
            return None

        self._emit_assistant(reply)
        if speak and self._speak_enabled:
            self._speak(reply)
        self._idle_state(SessionState.LISTENING if self._running else SessionState.IDLE)
        return reply

    def _speak(self, text: str) -> None:
        if not text.strip():
            return
        self._last_spoken = text
        self._set_state(SessionState.SPEAKING)
        try:
            assert self._tts is not None and self._player is not None
            self._player.play_stream(
                self._tts.synthesize_stream(text), self.settings.tts_sample_rate
            )
        except SpeechSynthesisError as exc:
            self._fail(exc)
        except AudioDeviceError as exc:
            # Reply generation still succeeded; only playback failed.
            self._fail(exc, fatal=False)
        except VoiceAssistantError as exc:
            self._fail(exc)
    # -- echo guard ----------------------------------------------------- #
    def _is_echo(self, user_text: str) -> bool:
        if not self.settings.echo_guard or not self._last_spoken:
            return False
        candidate = _normalise(user_text)
        spoken = _normalise(self._last_spoken)
        if not candidate or not spoken:
            return False
        if len(candidate) >= 12 and (
            candidate in spoken or spoken.startswith(candidate)
        ):
            return True
        return SequenceMatcher(None, candidate, spoken).ratio() > 0.85

    # -- controls ------------------------------------------------------- #
    def stop(self) -> None:
        """Signal the loop to finish and stop any audio playback."""
        self._stop_event.set()
        if self._player is not None:
            self._player.stop()

    def close(self) -> None:
        """Release audio resources. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._player is not None:
            self._player.stop()
        if self._engine is not None:
            closer = getattr(self._engine.backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - best effort
                    pass

    def reset_conversation(self) -> None:
        if self._engine is not None:
            self._engine.reset()

    def set_speaking_enabled(self, enabled: bool) -> None:
        """Enable/disable audio playback (text output still works)."""
        self._speak_enabled = enabled
        if not enabled and self._player is not None:
            self._player.stop()

    # -- state / event helpers ----------------------------------------- #
    def _idle_state(self, state: SessionState | None = None) -> None:
        if state is None:
            state = SessionState.LISTENING if self._running else SessionState.IDLE
        self._set_state(state)

    def _set_state(self, state: SessionState) -> None:
        self._state = state
        self._safe_call(self._on_state, state)

    def _emit_user(self, text: str) -> None:
        self._safe_call(self._on_user, text)

    def _emit_assistant(self, text: str) -> None:
        self._safe_call(self._on_assistant, text)

    def _emit_info(self, message: str) -> None:
        log.info(message)
        self._safe_call(self._on_info, message)

    def _fail(self, exc: Exception, *, fatal: bool = True) -> None:
        message = _human_message(exc)
        log.error("Assistant error%s: %s", " (fatal)" if fatal else "", message)
        self._set_state(SessionState.ERROR)
        self._safe_call(self._on_error, message)

    @staticmethod
    def _safe_call(callback, *args) -> None:  # noqa: ANN001
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001 - a broken UI callback must not stop us
            log.exception("Error in UI callback %r", callback)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .,!?;:")


def _human_message(exc: Exception) -> str:
    from .errors import (
        AuthenticationError,
        RateLimitError,
        ServiceUnavailableError,
    )

    if isinstance(exc, AuthenticationError):
        return str(exc)
    if isinstance(exc, RateLimitError):
        return str(exc)
    if isinstance(exc, ServiceUnavailableError):
        return str(exc)
    if isinstance(exc, ConfigurationError):
        return str(exc)
    if isinstance(exc, VoiceAssistantError):
        return str(exc)
    if isinstance(exc, AudioDeviceError):
        return str(exc)
    return f"Unexpected error: {exc}"
