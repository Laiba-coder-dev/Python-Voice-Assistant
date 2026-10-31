"""Application-wide error types.

Having a small, predictable hierarchy lets the UI show helpful messages and
decide whether a retry makes sense, without every module inventing its own
exception names.
"""

from __future__ import annotations


class VoiceAssistantError(Exception):
    """Base class for all expected assistant errors."""


class ConfigurationError(VoiceAssistantError):
    """The application is misconfigured (missing keys, bad values)."""


class AuthenticationError(VoiceAssistantError):
    """The remote service rejected our credentials."""


class RateLimitError(VoiceAssistantError):
    """The remote service is rate limiting or the quota is exhausted."""


class ServiceUnavailableError(VoiceAssistantError):
    """A network/timeout/server problem prevented the request."""


class SpeechRecognitionError(VoiceAssistantError):
    """Speech-to-text failed for a non-empty recording."""


class EmptyAudioError(SpeechRecognitionError):
    """The recording was too short or contained no intelligible speech."""


class SpeechSynthesisError(VoiceAssistantError):
    """Text-to-speech failed."""


class ConversationError(VoiceAssistantError):
    """Response generation failed."""
