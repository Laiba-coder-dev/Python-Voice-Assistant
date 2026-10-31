"""Optional full-duplex mode powered by ElevenLabs Conversational AI (Agents).

Unlike the pipeline mode (record -> Scribe -> LLM -> TTS), Agents mode hands the
whole conversation to an ElevenLabs agent you configure in the dashboard: the
agent handles speech recognition, turn-taking, the LLM and speech synthesis in a
single low-latency, interruptible session.

It is entirely optional and selected with ``ASSISTANT_MODE=agents`` plus an
``ELEVENLABS_AGENT_ID``.  Pipeline mode remains the default because it needs no
dashboard setup and gives you full control over every stage.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import AudioInterface, Conversation

from .assistant import SessionState
from .config import Settings
from .errors import ConfigurationError, ServiceUnavailableError
from .logging_utils import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16_000
INPUT_FRAMES = 4_000  # 250 ms @ 16 kHz, matching ElevenLabs' recommendation


def _default_audio_interface() -> AudioInterface:
    """Prefer the official interface; fall back to sounddevice if PyAudio is absent."""
    try:
        from elevenlabs.conversational_ai.default_audio_interface import (
            DefaultAudioInterface,
        )

        return DefaultAudioInterface()
    except Exception as exc:  # noqa: BLE001 - pyaudio missing
        log.info("PyAudio unavailable (%s); using sounddevice audio interface.", exc)
        return SoundDeviceAudioInterface()


class SoundDeviceAudioInterface(AudioInterface):
    """A minimal PyAudio-free audio interface built on sounddevice."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._partial = b""
        self._stop = threading.Event()
        self._in_stream = None
        self._out_stream = None

    def start(self, input_callback) -> None:  # noqa: ANN001
        import sounddevice as sd

        def _in_callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if not self._stop.is_set():
                input_callback(bytes(indata))

        def _out_callback(outdata, frames, time_info, status) -> None:  # noqa: ANN001
            need = frames * outdata.shape[1] * 2
            buffer = self._partial
            while len(buffer) < need:
                try:
                    buffer += self._queue.get_nowait()
                except queue.Empty:
                    break
            chunk, self._partial = buffer[:need], buffer[need:]
            if len(chunk) < need:
                chunk += b"\x00" * (need - len(chunk))
            outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(
                frames, outdata.shape[1]
            )

        self._in_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=INPUT_FRAMES,
            callback=_in_callback,
        )
        self._out_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1_000,
            callback=_out_callback,
        )
        self._in_stream.start()
        self._out_stream.start()

    def output(self, audio: bytes) -> None:
        if audio:
            self._queue.put(audio)

    def interrupt(self) -> None:
        self._partial = b""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001 - best effort
                    pass
        self._in_stream = None
        self._out_stream = None


def run_agents_conversation(
    settings: Settings,
    *,
    on_user=None,  # noqa: ANN001
    on_agent=None,  # noqa: ANN001
    on_error=None,  # noqa: ANN001
    on_state=None,  # noqa: ANN001
) -> None:
    """Run a blocking ElevenLabs Agents session until it ends.

    Two connection styles are supported and the code picks whichever the agent
    allows:

    * **Authenticated** (``requires_auth=True``) mints a short-lived signed URL
      server-side.  It needs the API key to carry the ``convai_write``
      permission.
    * **Public** (``requires_auth=False``) connects straight to the agent over
      the WebSocket using just its ``agent_id``.  Public agents accept this, so
      it works even with a restricted key that lacks ``convai_write``.

    ``ELEVENLABS_AGENT_PUBLIC=true`` chooses the public path first; otherwise
    the authenticated path is tried first.  Either way the other path is used
    as an automatic fallback, so a session starts as long as *one* of them is
    permitted.
    """
    if not settings.agents_agent_id:
        raise ConfigurationError(
            "Agents mode needs ELEVENLABS_AGENT_ID (create an agent at "
            "https://elevenlabs.io/app/conversational-ai)."
        )
    try:
        settings.require_elevenlabs()
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(str(exc)) from exc

    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    audio_interface = _default_audio_interface()

    def _safe(callback, *args) -> None:  # noqa: ANN001
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            log.exception("Error in agents callback %r", callback)

    def _build(requires_auth: bool) -> Conversation:
        return Conversation(
            client=client,
            agent_id=settings.agents_agent_id,
            requires_auth=requires_auth,
            audio_interface=audio_interface,
            callback_user_transcript=lambda text: _safe(on_user, text),
            callback_agent_response=lambda text: _safe(on_agent, text),
        )

    # Try the configured style first, then fall back to the other one.
    public_first = bool(settings.agents_agent_public)
    order = [not public_first, public_first]

    conversation: Conversation | None = None
    last_error: Exception | None = None

    _safe(on_state, SessionState.LISTENING)
    for requires_auth in order:
        candidate = _build(requires_auth)
        try:
            candidate.start_session()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.info(
                "Agent session did not start (requires_auth=%s): %s",
                requires_auth,
                exc,
            )
            try:
                candidate.end_session()
            except Exception:  # noqa: BLE001 - nothing was opened
                pass
            continue
        conversation = candidate
        last_error = None
        break

    if conversation is None:
        _safe(on_state, SessionState.ERROR)
        _safe(
            on_error,
            ServiceUnavailableError(
                f"The ElevenLabs agent session failed: {last_error}"
            ),
        )
        _safe(on_state, SessionState.STOPPED)
        return

    try:
        conversation.wait_for_session_end()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    except Exception as exc:  # noqa: BLE001
        _safe(on_state, SessionState.ERROR)
        _safe(
            on_error,
            ServiceUnavailableError(f"The ElevenLabs agent session failed: {exc}"),
        )
    finally:
        try:
            conversation.end_session()
        except Exception:  # noqa: BLE001
            pass
        _safe(on_state, SessionState.STOPPED)
