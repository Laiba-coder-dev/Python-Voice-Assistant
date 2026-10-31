"""Low-latency streaming transcription via Scribe v2 Realtime.

This is an opt-in alternative to the record-and-transcribe workflow in
:mod:`voice_assistant.speech_to_text`.  It opens a WebSocket, streams raw PCM
from the microphone and relies on the server's voice-activity detector to
commit each utterance (``commit_strategy="vad"``), which minimises latency.

Enable it with ``STT_MODE=realtime``.  It requires the ``websockets`` package
(installed with the ElevenLabs SDK when realtime support is available) and an
account/model that exposes ``scribe_v2_realtime``.
"""

from __future__ import annotations

import base64
import queue
import threading
from typing import Callable, Optional

from elevenlabs import ElevenLabs
from elevenlabs.realtime.connection import RealtimeEvents
from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy

from .audio_handler import AudioDeviceError
from .config import Settings
from .errors import (
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    SpeechRecognitionError,
)
from .logging_utils import get_logger

log = get_logger(__name__)

_TEXT_EVENT = "transcript"


class RealtimeTranscriber:
    """Manages a single ElevenLabs realtime transcription connection."""

    def __init__(
        self,
        client: ElevenLabs,
        settings: Settings,
        *,
        on_partial: Optional[Callable[[str], None]] = None,
        on_committed: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_open: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self.on_partial = on_partial
        self.on_committed = on_committed
        self.on_error = on_error
        self.on_open = on_open
        self.on_close = on_close
        self._connection = None

    # -- connection ----------------------------------------------------- #
    async def start(self) -> None:
        settings = self._settings
        options: dict = {
            "model_id": settings.stt_model_id,
            "audio_format": AudioFormat.PCM_16000
            if settings.sample_rate == 16_000
            else settings.sample_rate,
            "sample_rate": settings.sample_rate,
            "commit_strategy": CommitStrategy.VAD,
        }
        if settings.stt_language_code:
            options["language_code"] = settings.stt_language_code
        if settings.stt_keyterms:
            options["keyterms"] = list(settings.stt_keyterms)
        if settings.stt_no_verbatim:
            options["no_verbatim"] = True

        try:
            self._connection = await self._client.speech_to_text.realtime.connect(
                options
            )
        except Exception as exc:  # noqa: BLE001
            raise SpeechRecognitionError(
                f"Could not open the realtime transcription socket: {exc}"
            ) from exc

        conn = self._connection
        conn.on(RealtimeEvents.OPEN, lambda _data: self._call(self.on_open))
        conn.on(
            RealtimeEvents.PARTIAL_TRANSCRIPT,
            lambda data: self._call(self.on_partial, _text(data)),
        )
        conn.on(
            RealtimeEvents.COMMITTED_TRANSCRIPT,
            lambda data: self._call(self.on_committed, _text(data)),
        )
        conn.on(RealtimeEvents.CLOSE, lambda _data: self._call(self.on_close))
        for event in (
            RealtimeEvents.ERROR,
            RealtimeEvents.AUTH_ERROR,
            RealtimeEvents.QUOTA_EXCEEDED,
            RealtimeEvents.RATE_LIMITED,
            RealtimeEvents.COMMIT_THROTTLED,
            RealtimeEvents.TRANSCRIBER_ERROR,
            RealtimeEvents.INVALID_REQUEST,
            RealtimeEvents.INPUT_ERROR,
            RealtimeEvents.RESOURCE_EXHAUSTED,
        ):
            conn.on(event, self._handle_error_event)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self._connection is None or not pcm_bytes:
            return
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        await self._connection.send({"audio_base_64": encoded})

    async def commit(self) -> None:
        if self._connection is not None:
            await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._connection = None

    # -- callback plumbing --------------------------------------------- #
    @staticmethod
    def _call(callback, *args) -> None:  # noqa: ANN001
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            log.exception("Error in realtime callback %r", callback)

    def _handle_error_event(self, data: dict) -> None:
        message = _text(data) or str(data)
        error = _classify_realtime_error(data, message)
        self._call(self.on_error, error)


def _classify_realtime_error(data: dict, message: str) -> Exception:
    kind = str(data.get("message_type", ""))
    if kind in {"auth_error"}:
        return AuthenticationError(
            "ElevenLabs rejected the realtime transcription credentials."
        )
    if kind in {"quota_exceeded", "rate_limited"}:
        return RateLimitError(
            "Realtime transcription hit a rate or quota limit; try again shortly."
        )
    if kind in {"transcriber_error", "invalid_request", "input_error"}:
        return SpeechRecognitionError(f"Realtime transcription error: {message}")
    return ServiceUnavailableError(f"Realtime transcription error: {message}")


def _text(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("transcript", _TEXT_EVENT, "text"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


# --------------------------------------------------------------------------- #
# Async driver: continuous mic -> realtime STT -> assistant
# --------------------------------------------------------------------------- #
async def run_realtime_assistant(assistant, stop_event: threading.Event) -> None:  # noqa: ANN001
    """Continuously stream microphone audio and respond to committed speech.

    The assistant's LLM + TTS run in a thread executor so the WebSocket keeps
    receiving audio while the assistant is thinking/speaking.  The microphone is
    muted (sending paused) during playback to avoid transcribing our own voice.
    """
    import asyncio

    import sounddevice as sd

    from .assistant import SessionState

    settings: Settings = assistant.settings
    assistant._ensure_ready()
    assistant._running = True
    assistant._set_state(SessionState.LISTENING)

    committed: "asyncio.Queue[str]" = asyncio.Queue()
    speaking = threading.Event()
    loop = asyncio.get_running_loop()

    def on_committed(text: str) -> None:
        text = (text or "").strip()
        if not text or speaking.is_set():
            return
        committed.put_nowait(text)

    transcriber = RealtimeTranscriber(
        assistant._client,
        settings,
        on_partial=lambda text: assistant._emit_info(text) if text else None,
        on_committed=on_committed,
        on_error=lambda exc: assistant._fail(exc, fatal=False),
    )

    try:
        await transcriber.start()
    except Exception as exc:  # noqa: BLE001
        assistant._fail(exc)
        assistant._running = False
        assistant.close()
        return

    # --- microphone capture (PortAudio thread -> thread-safe queue) ---
    mic_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

    def _mic_callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if speaking.is_set():
            return
        try:
            mic_queue.put_nowait(bytes(indata))
        except queue.Full:  # pragma: no cover - drop if we fall behind
            pass

    try:
        mic_stream = sd.RawInputStream(
            samplerate=settings.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=max(1, int(settings.sample_rate * 0.1)),
            device=settings.input_device,
            callback=_mic_callback,
        )
        mic_stream.start()
    except Exception as exc:  # noqa: BLE001
        assistant._fail(
            AudioDeviceError(f"Could not open the microphone: {exc}")
        )
        await transcriber.close()
        assistant._running = False
        assistant.close()
        return

    async def _pump_audio() -> None:
        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(
                    loop.run_in_executor(None, mic_queue.get), timeout=0.25
                )
            except asyncio.TimeoutError:
                continue
            await transcriber.send_audio(chunk)

    pump_task = asyncio.create_task(_pump_audio())

    try:
        while not stop_event.is_set():
            try:
                text = await asyncio.wait_for(committed.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            speaking.set()
            if assistant._player is not None:
                assistant._player.stop()
            await loop.run_in_executor(None, assistant.handle_text, text)
            speaking.clear()
    finally:
        stop_event.set()
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:  # noqa: BLE001
            pass
        await transcriber.close()
        assistant._running = False
        assistant.close()
        assistant._set_state(SessionState.STOPPED)
