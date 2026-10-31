"""Unit tests for the LLM backends (HTTP mocked)."""

from __future__ import annotations

import httpx
import pytest

from voice_assistant.config import Settings
from voice_assistant.errors import (
    AuthenticationError,
    ConversationError,
    RateLimitError,
    ServiceUnavailableError,
)
from voice_assistant.llm import (
    ElevenLabsAgentBackend,
    OfflineBackend,
    OpenAICompatibleBackend,
    build_backend,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _messages(text: str):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]


# --------------------------------------------------------------------------- #
# Offline backend
# --------------------------------------------------------------------------- #
def test_offline_greeting():
    reply = OfflineBackend().generate(_messages("Hello there"))
    assert "listening" in reply.lower()


def test_offline_identity():
    reply = OfflineBackend().generate(_messages("who are you?"))
    assert "assistant" in reply.lower()


def test_offline_arithmetic():
    reply = OfflineBackend().generate(_messages("what is 12 times 4"))
    assert "48" in reply


def test_offline_repeat_last_answer():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there friend."},
        {"role": "user", "content": "say that again"},
    ]
    assert OfflineBackend().generate(messages) == "Hello there friend."


def test_offline_fallback_mentions_offline_mode():
    reply = OfflineBackend().generate(_messages("Tell me about quantum physics"))
    assert "offline" in reply.lower()


def test_offline_empty_message():
    assert "didn't catch" in OfflineBackend().generate(_messages("")).lower()


# --------------------------------------------------------------------------- #
# OpenAI-compatible backend
# --------------------------------------------------------------------------- #
def _backend():
    return OpenAICompatibleBackend(api_key="k", base_url="https://example.test/v1")


def test_openai_success(monkeypatch):
    payload = {"choices": [{"message": {"content": "  Hello!  "}}]}
    monkeypatch.setattr(
        "voice_assistant.llm.httpx.post",
        lambda *a, **k: _FakeResponse(200, payload),
    )
    assert _backend().generate(_messages("hi")) == "Hello!"


def test_openai_auth_error(monkeypatch):
    monkeypatch.setattr(
        "voice_assistant.llm.httpx.post",
        lambda *a, **k: _FakeResponse(401, {"error": {"message": "bad key"}}),
    )
    with pytest.raises(AuthenticationError):
        _backend().generate(_messages("hi"))


def test_openai_rate_limit(monkeypatch):
    monkeypatch.setattr(
        "voice_assistant.llm.httpx.post",
        lambda *a, **k: _FakeResponse(429, {}),
    )
    with pytest.raises(RateLimitError):
        _backend().generate(_messages("hi"))


def test_openai_server_error(monkeypatch):
    monkeypatch.setattr(
        "voice_assistant.llm.httpx.post",
        lambda *a, **k: _FakeResponse(503, {}),
    )
    with pytest.raises(ServiceUnavailableError):
        _backend().generate(_messages("hi"))


def test_openai_bad_request(monkeypatch):
    monkeypatch.setattr(
        "voice_assistant.llm.httpx.post",
        lambda *a, **k: _FakeResponse(400, {"message": "bad request"}),
    )
    with pytest.raises(ConversationError):
        _backend().generate(_messages("hi"))


def test_openai_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "voice_assistant.llm.httpx.post",
        lambda *a, **k: _FakeResponse(200, {"unexpected": True}),
    )
    with pytest.raises(ConversationError):
        _backend().generate(_messages("hi"))


def test_openai_timeout(monkeypatch):
    def _boom(*a, **k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("voice_assistant.llm.httpx.post", _boom)
    with pytest.raises(ServiceUnavailableError):
        _backend().generate(_messages("hi"))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_build_backend_offline_without_key():
    backend = build_backend(Settings(elevenlabs_api_key="x"))
    assert isinstance(backend, OfflineBackend)


def test_build_backend_openai_with_key():
    backend = build_backend(Settings(elevenlabs_api_key="x", llm_api_key="k"))
    assert isinstance(backend, OpenAICompatibleBackend)


def test_build_backend_elevenlabs_with_agent():
    backend = build_backend(
        Settings(elevenlabs_api_key="x", agents_agent_id="agent-1")
    )
    assert isinstance(backend, ElevenLabsAgentBackend)
    assert backend.name == "elevenlabs"


def test_build_backend_elevenlabs_explicit_requires_agent_id():
    with pytest.raises(ConversationError):
        build_backend(Settings(elevenlabs_api_key="x", llm_provider="elevenlabs"))


# --------------------------------------------------------------------------- #
# ElevenLabs agent backend (network/SDK mocked)
# --------------------------------------------------------------------------- #
class _FakeAgentConversation:
    instances: list["_FakeAgentConversation"] = []

    def __init__(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        self.sent: list[str] = []
        self.ended = False
        self._callback = kwargs.get("callback_agent_response")
        _FakeAgentConversation.instances.append(self)

    def start_session(self):
        # The agent speaks first; this must be discarded.
        self._callback("Hello! How can I help you today?")

    def send_user_message(self, text: str):
        self.sent.append(text)
        self._callback("The capital of France is Paris.")

    def end_session(self):
        self.ended = True


@pytest.fixture
def fake_agent_sdk(monkeypatch):
    _FakeAgentConversation.instances = []
    monkeypatch.setattr("elevenlabs.ElevenLabs", lambda **kwargs: object())
    monkeypatch.setattr(
        "elevenlabs.conversational_ai.conversation.Conversation",
        _FakeAgentConversation,
    )
    return _FakeAgentConversation


def test_elevenlabs_agent_backend_generate(fake_agent_sdk):
    backend = ElevenLabsAgentBackend(agent_id="agent-1", api_key="x", public=True)
    reply = backend.generate(_messages("What is the capital of France?"))
    assert reply == "The capital of France is Paris."
    assert len(fake_agent_sdk.instances) == 1
    conv = fake_agent_sdk.instances[0]
    # Public agents connect without a signed URL and in text-only mode.
    assert conv.kwargs["requires_auth"] is False
    assert conv.kwargs["audio_interface"] is None
    assert conv.sent == ["What is the capital of France?"]


def test_elevenlabs_agent_backend_reuses_session(fake_agent_sdk):
    backend = ElevenLabsAgentBackend(agent_id="agent-1", api_key="x")
    backend.generate(_messages("first"))
    backend.generate(_messages("second"))
    assert len(fake_agent_sdk.instances) == 1
    assert fake_agent_sdk.instances[0].sent == ["first", "second"]


def test_elevenlabs_agent_backend_close(fake_agent_sdk):
    backend = ElevenLabsAgentBackend(agent_id="agent-1", api_key="x")
    backend.generate(_messages("hi"))
    backend.close()
    assert fake_agent_sdk.instances[0].ended is True

