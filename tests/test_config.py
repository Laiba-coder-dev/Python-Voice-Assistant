"""Unit tests for configuration handling."""

from __future__ import annotations

from voice_assistant.config import Settings, _sample_rate_from_format


def test_sample_rate_from_format():
    assert _sample_rate_from_format("pcm_24000") == 24000
    assert _sample_rate_from_format("mp3_44100_128") == 44100
    assert _sample_rate_from_format("pcm_16000") == 16000
    assert _sample_rate_from_format("nonsense") == 24000


def test_secret_values_are_hidden_from_repr():
    settings = Settings(
        elevenlabs_api_key="EL-SECRET-XYZ",
        llm_api_key="LLM-SECRET-ABC",
    )
    text = repr(settings)
    assert "EL-SECRET-XYZ" not in text
    assert "LLM-SECRET-ABC" not in text


def test_validate_reports_missing_elevenlabs_key():
    settings = Settings(elevenlabs_api_key=None)
    problems = settings.validate()
    assert any("ELEVENLABS_API_KEY" in p for p in problems)


def test_validate_accepts_offline_without_llm_key():
    settings = Settings(elevenlabs_api_key="x", llm_provider="offline")
    assert settings.validate() == []


def test_validate_flags_openai_without_key():
    settings = Settings(elevenlabs_api_key="x", llm_provider="openai")
    problems = settings.validate()
    assert any("LLM_API_KEY" in p for p in problems)


def test_resolved_provider_defaults():
    assert Settings(elevenlabs_api_key="x", llm_api_key="k").resolved_llm_provider() == "openai"
    assert Settings(elevenlabs_api_key="x").resolved_llm_provider() == "offline"
    assert (
        Settings(elevenlabs_api_key="x", llm_provider="offline").resolved_llm_provider()
        == "offline"
    )


def test_from_env_reads_values(monkeypatch):
    for name in ("ELEVENLABS_API_KEY", "TTS_VOICE_ID", "LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-env")
    monkeypatch.setenv("TTS_VOICE_ID", "voice-123")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SAMPLE_RATE", "16000")

    settings = Settings.from_env()
    assert settings.elevenlabs_api_key == "from-env"
    assert settings.tts_voice_id == "voice-123"
    assert settings.sample_rate == 16000


def test_require_elevenlabs_raises():
    from voice_assistant.config import ConfigError

    settings = Settings(elevenlabs_api_key=None)
    try:
        settings.require_elevenlabs()
    except ConfigError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ConfigError")
