"""Configuration and environment-variable handling.

All tunables live in a single :class:`Settings` object which is populated from
environment variables (optionally via a local ``.env`` file).  Nothing in this
module ever writes secrets anywhere, and API keys are marked ``repr=False`` so a
stray ``print(settings)`` will not leak credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root is the directory that contains the ``voice_assistant`` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# ``override=False`` means real environment variables win over the .env file.
load_dotenv(ENV_FILE, override=False)


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _get(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - trivial
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - trivial
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _get_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = _get(name)
    if raw is None:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


#: A small set of stable, publicly available ElevenLabs voices used to seed the
#: voice picker in the UI.  The live list is fetched from the API when possible.
KNOWN_VOICES: dict[str, str] = {
    "George – Warm Narrator": "JBFqnCBsd6RMkjVDRZzb",
    "Sarah – Mature & Reassuring": "EXAVITQu4vr4xnSDxMaL",
    "Laura – Young & Social": "FGY2WhTYpPnrIDTdsKH5",
    "Alice – Clear & Engaging": "Xb7hH8MSUJpSbSDYk0k2",
    "Brian – Deep & Comforting": "nPczCjzI2devNBz1zQrb",
    "Daniel – Steady Broadcaster": "onwK4e9ZLuTAKqWW03F9",
    "Lily – Velvety Actress": "pFZP5JQG7iQjIQuC4Bku",
    "Callum – Husky Trickster": "N2lVS1w4EtoT3dr4eOWO",
    "Jessica – Expressive & Warm": "cgSgspJ2msm6clMCkdW9",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Your answers are spoken "
    "aloud, so keep them natural, concise and easy to follow by ear. Prefer "
    "short sentences and two to four sentences per answer unless the user "
    "explicitly asks for detail. Never use markdown, bullet points, code "
    "blocks, emoji or stage directions in your replies. If a question is "
    "ambiguous, ask a brief clarifying question. Remember what was said "
    "earlier in the conversation and use it to answer follow-up questions. If "
    "you do not know something, say so plainly instead of making it up."
)


@dataclass(slots=True)
class Settings:
    """All runtime configuration for the assistant."""

    # ------------------------------------------------------------------ API /
    elevenlabs_api_key: str | None = field(default=None, repr=False)

    # ------------------------------------------------- Speech recognition
    stt_model_id: str = "scribe_v2"
    stt_mode: str = "batch"  # "batch" (reliable, default) or "realtime"
    stt_language_code: str | None = None  # e.g. "eng"; None = auto-detect
    stt_no_verbatim: bool = False
    stt_keyterms: list[str] = field(default_factory=list)

    # ------------------------------------------------- Text to speech
    tts_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George
    tts_model_id: str = "eleven_turbo_v2_5"
    tts_output_format: str = "pcm_24000"
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.75
    tts_style: float = 0.0
    tts_speaker_boost: bool = True
    tts_speed: float = 1.0

    # ------------------------------------------------- Audio capture
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16_000
    vad_threshold_factor: float = 2.5
    vad_min_threshold: float = 0.01
    silence_duration: float = 0.8
    min_speech_duration: float = 0.25
    max_record_seconds: float = 20.0
    listen_timeout: float | None = None  # None = listen until stopped
    echo_guard: bool = True

    # ------------------------------------------------- Response generation
    llm_provider: str = "auto"  # auto | openai | offline
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 400
    llm_timeout: float = 30.0

    # ------------------------------------------------- Conversation
    assistant_name: str = "Aria"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history_turns: int = 12
    log_transcripts: bool = False

    # ------------------------------------------------- ElevenLabs Agents mode
    agents_agent_id: str | None = None
    #: When True, connect to the agent as a *public* agent (no signed URL).
    #: Public agents accept the bare ``agent_id`` over the WebSocket, so this
    #: works even for API keys without the ``convai_write`` permission.
    agents_agent_public: bool = False

    # ------------------------------------------------------------------ helpers
    @property
    def elevenlabs_configured(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def tts_sample_rate(self) -> int:
        """Sample rate implied by :attr:`tts_output_format` (e.g. pcm_24000)."""
        return _sample_rate_from_format(self.tts_output_format)

    def resolved_llm_provider(self) -> str:
        """Resolve ``auto`` to a concrete backend based on available keys."""
        provider = (self.llm_provider or "auto").strip().lower()
        if provider == "auto":
            if self.llm_api_key:
                return "openai"
            if self.agents_agent_id:
                return "elevenlabs"
            return "offline"
        return provider

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        problems: list[str] = []
        if not self.elevenlabs_api_key:
            problems.append(
                "ELEVENLABS_API_KEY is not set. Speech recognition and speech "
                "synthesis require an ElevenLabs API key."
            )
        if self.stt_mode not in {"batch", "realtime"}:
            problems.append("STT_MODE must be 'batch' or 'realtime'.")
        if self.sample_rate not in {8_000, 16_000, 22_050, 24_000, 44_100, 48_000}:
            problems.append(
                f"SAMPLE_RATE={self.sample_rate} is unusual; 16000 is recommended."
            )
        if self.tts_speed <= 0:
            problems.append("TTS_SPEED must be greater than 0.")
        if self.max_history_turns < 1:
            problems.append("MAX_HISTORY_TURNS must be at least 1.")
        provider = self.resolved_llm_provider()
        if provider not in {"openai", "offline", "elevenlabs"}:
            problems.append(
                f"LLM_PROVIDER={self.llm_provider!r} is not supported "
                "(use 'auto', 'openai', 'elevenlabs' or 'offline')."
            )
        if provider == "openai" and not self.llm_api_key:
            problems.append(
                "LLM_PROVIDER=openai requires LLM_API_KEY (or OPENAI_API_KEY)."
            )
        if provider == "elevenlabs" and not self.agents_agent_id:
            problems.append(
                "LLM_PROVIDER=elevenlabs requires ELEVENLABS_AGENT_ID "
                "(create an agent at https://elevenlabs.io/app/conversational-ai)."
            )
        return problems

    def require_elevenlabs(self) -> None:
        """Raise :class:`ConfigError` if ElevenLabs is not configured."""
        if not self.elevenlabs_api_key:
            raise ConfigError(
                "Missing ELEVENLABS_API_KEY. Create a .env file (see "
                ".env.example) or export the variable before starting."
            )

    def public_summary(self) -> dict[str, object]:
        """A redacted view of the configuration, safe to display or log."""
        return {
            "assistant_name": self.assistant_name,
            "stt_model_id": self.stt_model_id,
            "stt_mode": self.stt_mode,
            "stt_language_code": self.stt_language_code,
            "tts_model_id": self.tts_model_id,
            "tts_voice_id": self.tts_voice_id,
            "tts_output_format": self.tts_output_format,
            "llm_provider": self.resolved_llm_provider(),
            "llm_model": self.llm_model,
            "sample_rate": self.sample_rate,
            "elevenlabs_configured": self.elevenlabs_configured,
        }

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance from environment variables."""
        return cls(
            elevenlabs_api_key=_get("ELEVENLABS_API_KEY"),
            stt_model_id=_get("STT_MODEL_ID", "scribe_v2") or "scribe_v2",
            stt_mode=(_get("STT_MODE", "batch") or "batch").lower(),
            stt_language_code=_get("STT_LANGUAGE_CODE"),
            stt_no_verbatim=_get_bool("STT_NO_VERBATIM", False),
            stt_keyterms=_get_list("STT_KEYTERMS"),
            tts_voice_id=_get("TTS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
            or "JBFqnCBsd6RMkjVDRZzb",
            tts_model_id=_get("TTS_MODEL_ID", "eleven_turbo_v2_5")
            or "eleven_turbo_v2_5",
            tts_output_format=_get("TTS_OUTPUT_FORMAT", "pcm_24000")
            or "pcm_24000",
            tts_stability=_get_float("TTS_STABILITY", 0.5),
            tts_similarity_boost=_get_float("TTS_SIMILARITY_BOOST", 0.75),
            tts_style=_get_float("TTS_STYLE", 0.0),
            tts_speaker_boost=_get_bool("TTS_SPEAKER_BOOST", True),
            tts_speed=_get_float("TTS_SPEED", 1.0),
            input_device=_get_int("INPUT_DEVICE", -1) if _get("INPUT_DEVICE") else None,
            output_device=_get_int("OUTPUT_DEVICE", -1)
            if _get("OUTPUT_DEVICE")
            else None,
            sample_rate=_get_int("SAMPLE_RATE", 16_000),
            vad_threshold_factor=_get_float("VAD_THRESHOLD_FACTOR", 2.5),
            vad_min_threshold=_get_float("VAD_MIN_THRESHOLD", 0.01),
            silence_duration=_get_float("SILENCE_DURATION", 0.8),
            min_speech_duration=_get_float("MIN_SPEECH_DURATION", 0.25),
            max_record_seconds=_get_float("MAX_RECORD_SECONDS", 20.0),
            listen_timeout=_get_float("LISTEN_TIMEOUT", 0.0) or None,
            echo_guard=_get_bool("ECHO_GUARD", True),
            llm_provider=(_get("LLM_PROVIDER", "auto") or "auto").lower(),
            llm_api_key=_get("LLM_API_KEY") or _get("OPENAI_API_KEY"),
            llm_base_url=_get("LLM_BASE_URL", "https://api.openai.com/v1")
            or "https://api.openai.com/v1",
            llm_model=_get("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            llm_temperature=_get_float("LLM_TEMPERATURE", 0.7),
            llm_max_tokens=_get_int("LLM_MAX_TOKENS", 400),
            llm_timeout=_get_float("LLM_TIMEOUT", 30.0),
            assistant_name=_get("ASSISTANT_NAME", "Aria") or "Aria",
            system_prompt=_get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
            or DEFAULT_SYSTEM_PROMPT,
            max_history_turns=_get_int("MAX_HISTORY_TURNS", 12),
            log_transcripts=_get_bool("LOG_TRANSCRIPTS", False),
            agents_agent_id=_get("ELEVENLABS_AGENT_ID"),
            agents_agent_public=_get_bool("ELEVENLABS_AGENT_PUBLIC", False),
        )


def _sample_rate_from_format(output_format: str) -> int:
    """Extract the sample rate from an ElevenLabs output format string.

    ``pcm_24000`` -> 24000, ``mp3_44100_128`` -> 44100, unknown -> 24000.
    """
    parts = (output_format or "").split("_")
    for part in parts[1:]:
        if part.isdigit() and int(part) >= 8_000:
            return int(part)
    return 24_000
