"""ElevenLabs text-to-speech integration.

Responses are synthesised as raw 16-bit PCM (default ``pcm_24000``) so they can
be played immediately through :class:`~voice_assistant.audio_handler.PCMPlayer`
without decoding.  Long replies are split on sentence boundaries and synthesised
chunk-by-chunk; :meth:`TextToSpeech.synthesize_stream` yields those chunks as
they arrive so playback can begin before the whole reply is generated.
"""

from __future__ import annotations

import re
from typing import Iterator

from elevenlabs import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.types.voice_settings import VoiceSettings

from .config import Settings
from .errors import (
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    SpeechSynthesisError,
)
from .logging_utils import get_logger

log = get_logger(__name__)

#: Roughly how many characters to send per TTS request when splitting output.
MAX_CHUNK_CHARS = 800

# Sentence-ish boundaries, keeping the delimiter with the sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def split_for_speech(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into speakable chunks no longer than ``max_chars``.

    Splitting happens on sentence boundaries where possible; individual
    sentences longer than ``max_chars`` are split on word boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
            current = ""
        # A single sentence can still exceed the limit.
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        current = sentence
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _classify_api_error(exc: ApiError, what: str) -> Exception:
    status = exc.status_code
    if status in (401, 403):
        return AuthenticationError(
            "ElevenLabs rejected the API key during speech synthesis "
            f"({status}). Check ELEVENLABS_API_KEY."
        )
    if status == 429:
        return RateLimitError(
            "ElevenLabs rate limit reached during speech synthesis. Wait a "
            "moment and try again."
        )
    if status is not None and status >= 500:
        return ServiceUnavailableError(
            f"ElevenLabs reported a server error ({status}) during {what}."
        )
    return SpeechSynthesisError(f"Speech synthesis failed: {exc}")


class TextToSpeech:
    """Wrapper around ElevenLabs voice generation."""

    def __init__(self, client: ElevenLabs, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    # -- configuration -------------------------------------------------- #
    def voice_settings(self) -> VoiceSettings:
        settings = self._settings
        return VoiceSettings(
            stability=settings.tts_stability,
            similarity_boost=settings.tts_similarity_boost,
            style=settings.tts_style,
            use_speaker_boost=settings.tts_speaker_boost,
            speed=settings.tts_speed,
        )

    def _request_kwargs(self, text: str) -> dict:
        return {
            "voice_id": self._settings.tts_voice_id,
            "text": text,
            "model_id": self._settings.tts_model_id,
            "output_format": self._settings.tts_output_format,
            "voice_settings": self.voice_settings(),
        }

    # -- synthesis ------------------------------------------------------ #
    def synthesize(self, text: str) -> bytes:
        """Return the complete PCM audio for ``text`` (blocking)."""
        audio = bytearray()
        for chunk in split_for_speech(text):
            audio.extend(self._synthesize_chunk(chunk))
        return bytes(audio)

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Yield PCM byte chunks for ``text`` as they are generated."""
        chunks = split_for_speech(text)
        if not chunks:
            return
        for index, chunk in enumerate(chunks):
            try:
                stream = self._client.text_to_speech.stream(
                    **self._request_kwargs(chunk)
                )
                for piece in stream:
                    if piece:
                        yield piece
            except ApiError as exc:
                # Fail forward: if we already spoke part of the reply, say so.
                if index > 0:
                    log.warning("TTS failed after partial audio: %s", exc)
                raise _classify_api_error(exc, "speech synthesis") from exc
            except Exception as exc:  # noqa: BLE001
                raise ServiceUnavailableError(
                    f"Could not reach ElevenLabs for speech synthesis: {exc}"
                ) from exc

    def _synthesize_chunk(self, chunk: str) -> bytes:
        try:
            generator = self._client.text_to_speech.convert(
                **self._request_kwargs(chunk)
            )
            return b"".join(piece for piece in generator if piece)
        except ApiError as exc:
            raise _classify_api_error(exc, "speech synthesis") from exc
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailableError(
                f"Could not reach ElevenLabs for speech synthesis: {exc}"
            ) from exc

    # -- voice catalogue ------------------------------------------------ #
    def list_voices(self, page_size: int = 100) -> list[dict[str, str]]:
        """Return available voices as ``[{"id", "name", "labels"}, ...]``."""
        try:
            response = self._client.voices.search(page_size=page_size)
        except ApiError as exc:
            raise _classify_api_error(exc, "voice listing") from exc
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailableError(
                f"Could not fetch voices from ElevenLabs: {exc}"
            ) from exc

        voices = getattr(response, "voices", None) or []
        result: list[dict[str, str]] = []
        for voice in voices:
            voice_id = getattr(voice, "voice_id", None)
            name = getattr(voice, "name", None) or "Unnamed voice"
            labels = getattr(voice, "labels", None) or {}
            descriptor = ", ".join(str(v) for v in labels.values()) if labels else ""
            if voice_id:
                result.append({"id": voice_id, "name": name, "labels": descriptor})
        return result
