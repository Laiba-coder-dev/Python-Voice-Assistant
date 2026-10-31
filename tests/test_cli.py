"""Unit tests for CLI parsing and Agents-mode validation."""

from __future__ import annotations

import pytest

from voice_assistant import agents_mode
from voice_assistant.agents_mode import run_agents_conversation
from voice_assistant.cli import build_parser, main
from voice_assistant.config import Settings
from voice_assistant.errors import ConfigurationError


class FakeConversation:
    """Records how the agent session was constructed/used."""

    instances: list["FakeConversation"] = []
    fail_auth: bool = False

    def __init__(self, **kwargs):  # noqa: ANN003
        self.requires_auth = kwargs["requires_auth"]
        self.started = False
        self.ended = False
        FakeConversation.instances.append(self)

    def start_session(self):
        if self.requires_auth and FakeConversation.fail_auth:
            raise RuntimeError("missing_permissions: convai_write")
        self.started = True

    def wait_for_session_end(self):
        return None

    def end_session(self):
        self.ended = True


@pytest.fixture
def fake_agents(monkeypatch):
    FakeConversation.instances = []
    FakeConversation.fail_auth = False
    monkeypatch.setattr(agents_mode, "ElevenLabs", lambda **kwargs: object())
    monkeypatch.setattr(agents_mode, "_default_audio_interface", lambda: object())
    monkeypatch.setattr(agents_mode, "Conversation", FakeConversation)
    return FakeConversation


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.mode == "pipeline"
    assert args.text is None
    assert args.list_devices is False


def test_check_command_without_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert main(["--check"]) == 1


def test_check_command_with_key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--check"]) == 0


def test_list_devices_does_not_crash():
    assert main(["--list-devices"]) in (0, 1)


def test_agents_mode_requires_agent_id():
    with pytest.raises(ConfigurationError):
        run_agents_conversation(Settings(elevenlabs_api_key="x"))


def test_agents_mode_requires_api_key():
    with pytest.raises(ConfigurationError):
        run_agents_conversation(
            Settings(elevenlabs_api_key=None, agents_agent_id="agent-1")
        )


def test_agents_public_setting_from_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_AGENT_PUBLIC", "true")
    assert Settings.from_env().agents_agent_public is True
    monkeypatch.setenv("ELEVENLABS_AGENT_PUBLIC", "false")
    assert Settings.from_env().agents_agent_public is False


def test_agents_mode_public_first(fake_agents):
    run_agents_conversation(
        Settings(
            elevenlabs_api_key="x",
            agents_agent_id="agent-1",
            agents_agent_public=True,
        )
    )
    assert [c.requires_auth for c in fake_agents.instances] == [False]
    assert fake_agents.instances[0].started is True


def test_agents_mode_falls_back_to_public_when_signed_url_fails(fake_agents):
    fake_agents.fail_auth = True
    run_agents_conversation(
        Settings(
            elevenlabs_api_key="x",
            agents_agent_id="agent-1",
            agents_agent_public=False,
        )
    )
    assert [c.requires_auth for c in fake_agents.instances] == [True, False]
    assert fake_agents.instances[0].started is False
    assert fake_agents.instances[1].started is True

