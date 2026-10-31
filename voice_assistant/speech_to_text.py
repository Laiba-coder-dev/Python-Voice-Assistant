"""ElevenLabs speech-to-text integration (Scribe).

The default, universally available path uploads a recorded WAV clip to
``/v1/speech-to-text`` (model ``scribe_v2``).  A low-latency WebSocket path
(``scribe_v2_realtime``) is provided separately in
:mod:`voice_assistant.realtime_stt` for users whose account supports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError

from .audio_handler import pcm_to_wav_bytes
from .config import Settings
from .errors import (
    AuthenticationError,
    EmptyAudioError,
    RateLimitError,
    ServiceUnavailableError,
    SpeechRecognitionError,
)
from .logging_utils import get_logger

log = get_logger(__name__)

#: Scribe rejects audio shorter than ~100 ms.
MIN_AUDIO_SECONDS = 0.1


@dataclass(slots=True)
class TranscriptResult:
    """A transcription plus a little metadata."""

    text: str
    language_code: str | None = None
    language_probability: float | None = None
    duration_seconds: float | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _classify_api_error(exc: ApiError, what: str) -> Exception:
    status = exc.status_code
    if status in (401, 403):
        return AuthenticationError(
            "ElevenLabs rejected the API key while trying to transcribe "
            f"({status}). Check ELEVENLABS_API_KEY."
        )
    if status == 429:
        return RateLimitError(
            "ElevenLabs rate limit reached while transcribing. Wait a moment "
            "and try again."
        )
    if status is not None and status >= 500:
        return ServiceUnavailableError(
            f"ElevenLabs reported a server error ({status}) during {what}."
        )
    return SpeechRecognitionError(f"Speech recognition failed: {exc}")


class SpeechToText:
    """Thin, testable wrapper around the ElevenLabs transcription endpoint."""

    def __init__(self, client: ElevenLabs, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def transcribe_pcm(self, pcm, sample_rate: int) -> TranscriptResult:
        """Transcribe raw mono int16 PCM."""
        duration = len(pcm) / float(sample_rate) if sample_rate else 0.0
        if duration < MIN_AUDIO_SECONDS:
            raise EmptyAudioError("That recording was too short to understand.")
        wav = pcm_to_wav_bytes(pcm, sample_rate)
        result = self.transcribe_bytes(wav, filename="speech.wav")
        result.duration_seconds = duration
        return result

    def transcribe_bytes(
        self, audio_bytes: bytes, filename: str = "speech.wav"
    ) -> TranscriptResult:
        """Transcribe an encoded audio file (WAV/MP3/...)."""
        if not audio_bytes:
            raise EmptyAudioError("No audio was captured.")

        settings = self._settings
        kwargs: dict[str, Any] = {
            "model_id": settings.stt_model_id,
            "file": (filename, audio_bytes, "audio/wav"),
        }
        if settings.stt_language_code:
            kwargs["language_code"] = settings.stt_language_code
        if settings.stt_no_verbatim:
            kwargs["no_verbatim"] = True
        if settings.stt_keyterms:
            kwargs["keyterms"] = list(settings.stt_keyterms)

        try:
            response = self._client.speech_to_text.convert(**kwargs)
        except ApiError as exc:
            raise _classify_api_error(exc, "transcription") from exc
        except Exception as exc:  # noqa: BLE001 - network/serialization issues
            raise ServiceUnavailableError(
                f"Could not reach ElevenLabs for transcription: {exc}"
            ) from exc

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Any) -> TranscriptResult:
        """Normalise the three possible response shapes into a transcript."""
        # Single-channel chunk response.
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return TranscriptResult(
                text=text.strip(),
                language_code=getattr(response, "language_code", None),
                language_probability=getattr(response, "language_probability", None),
            )

        # Multichannel response (unexpected here, but handle gracefully).
        transcripts = getattr(response, "transcripts", None)
        if transcripts:
            joined = " ".join(
                (getattr(t, "text", "") or "").strip() for t in transcripts
            ).strip()
            first = transcripts[0]
            return TranscriptResult(
                text=joined,
                language_code=getattr(first, "language_code", None),
                language_probability=getattr(first, "language_probability", None),
            )

        # Async/webhook acknowledgement - no inline text.
        message = getattr(response, "message", None)
        if message:
            raise SpeechRecognitionError(
                f"Transcription was queued asynchronously ({message}); inline "
                "results are required, so disable webhook mode."
            )

        raise SpeechRecognitionError(
            "ElevenLabs returned an unexpected transcription response."
        )
