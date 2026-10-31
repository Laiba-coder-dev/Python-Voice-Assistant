"""Response-generation backends.

The conversation engine does not care *how* a reply is produced, only that it
receives one.  Three backends ship out of the box:

* :class:`OpenAICompatibleBackend` - talks to any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, Groq, Together, vLLM, Ollama, ...).
* :class:`ElevenLabsAgentBackend` - forwards each turn to an ElevenLabs Agent,
  which brings its own language model (no separate LLM key required).
* :class:`OfflineBackend` - a deterministic, dependency-free fallback so the
  application still works (and tests still run) without any LLM credentials.

All honour the same :meth:`generate` contract.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import (
    AuthenticationError,
    ConversationError,
    RateLimitError,
    ServiceUnavailableError,
)
from .logging_utils import get_logger

log = get_logger(__name__)


@runtime_checkable
class LLMBackend(Protocol):
    """A source of assistant replies."""

    name: str

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Return a reply given an OpenAI-style message list."""


class OpenAICompatibleBackend:
    """Chat Completions client that works with any compatible provider."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 400,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleBackend":
        assert settings.llm_api_key  # validated by the caller
        return cls(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        except httpx.TimeoutException as exc:
            raise ServiceUnavailableError(
                "The language model took too long to respond."
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceUnavailableError(
                f"Could not reach the language model: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise AuthenticationError(
                "The language model rejected the API key "
                f"({response.status_code}). Check LLM_API_KEY."
            )
        if response.status_code == 429:
            raise RateLimitError(
                "The language model is rate limiting requests. Try again shortly."
            )
        if response.status_code >= 500:
            raise ServiceUnavailableError(
                f"The language model service errored ({response.status_code})."
            )
        if response.status_code >= 400:
            detail = _safe_detail(response)
            raise ConversationError(
                f"The language model rejected the request "
                f"({response.status_code}): {detail}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise ConversationError(
                "The language model returned an unexpected response."
            ) from exc

        text = (content or "").strip()
        if not text:
            raise ConversationError("The language model returned an empty reply.")
        return text


class ElevenLabsAgentBackend:
    """Answer using an ElevenLabs Agent's built-in language model.

    The agent (created in the ElevenLabs dashboard) already bundles a language
    model, speech recognition and a voice.  Here we use only its *text*
    interface: each user turn is forwarded to the agent and its reply is
    returned, so the pipeline gets a real "brain" without any extra LLM key.

    A single text session is reused across turns, so the agent keeps its own
    conversational memory.  The session is opened lazily on the first request,
    which keeps construction offline-friendly for tests.
    """

    name = "elevenlabs"

    def __init__(
        self,
        *,
        agent_id: str,
        api_key: str,
        public: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.agent_id = agent_id
        self.api_key = api_key
        self.public = public
        self.timeout = timeout

        self._conversation = None
        self._lock = threading.Lock()
        self._reply_event = threading.Event()
        self._last_reply: str | None = None

    # -- session management --------------------------------------------- #
    def _on_agent_response(self, text: str) -> None:
        self._last_reply = text
        self._reply_event.set()

    def _ensure_session(self) -> None:
        if self._conversation is not None:
            return

        from elevenlabs import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation

        client = ElevenLabs(api_key=self.api_key)
        conversation = Conversation(
            client=client,
            agent_id=self.agent_id,
            requires_auth=not self.public,
            audio_interface=None,  # text-only (chat) session
            callback_agent_response=self._on_agent_response,
        )
        conversation.start_session()
        self._conversation = conversation

        # The agent greets first; wait for (and discard) that opening message so
        # it is never mistaken for the answer to our question.
        self._reply_event.wait(timeout=min(self.timeout, 15.0))
        self._reply_event.clear()
        self._last_reply = None

    def _reset(self) -> None:
        conversation, self._conversation = self._conversation, None
        if conversation is not None:
            try:
                conversation.end_session()
            except Exception:  # noqa: BLE001 - best effort
                pass

    def close(self) -> None:
        """End the underlying agent session, if one is open."""
        with self._lock:
            self._reset()

    # -- LLMBackend ------------------------------------------------------ #
    def generate(self, messages: list[dict[str, str]]) -> str:
        user_text = _last_user_message(messages).strip()
        if not user_text:
            raise ConversationError("No user message to send to the agent.")

        with self._lock:
            try:
                self._ensure_session()
            except Exception as exc:  # noqa: BLE001
                self._reset()
                raise ServiceUnavailableError(
                    f"Could not start the ElevenLabs agent session: {exc}"
                ) from exc

            self._last_reply = None
            self._reply_event.clear()
            try:
                self._conversation.send_user_message(user_text)
            except Exception as exc:  # noqa: BLE001
                self._reset()
                raise ServiceUnavailableError(
                    f"Could not reach the ElevenLabs agent: {exc}"
                ) from exc

            if not self._reply_event.wait(timeout=self.timeout):
                self._reset()
                raise ServiceUnavailableError(
                    "The ElevenLabs agent did not reply in time."
                )

            reply = (self._last_reply or "").strip()
            if not reply:
                raise ConversationError("The ElevenLabs agent returned an empty reply.")
            return reply


class OfflineBackend:
    """A small rule-based responder used when no LLM key is configured.

    It is deliberately simple.  Its purpose is to keep the full voice loop
    usable and testable without external LLM credentials - not to be clever.
    """

    name = "offline"

    def generate(self, messages: list[dict[str, str]]) -> str:
        user_message = _last_user_message(messages)
        text = user_message.lower().strip()
        last_assistant = _last_assistant_message(messages)

        if not text:
            return "I didn't catch that. Could you say it again?"

        if _matches(text, r"\b(hello|hi|hey|good (morning|afternoon|evening))\b"):
            return "Hello! I'm listening. What would you like to talk about?"

        if _matches(text, r"\b(who are you|what'?s your name|what are you)\b"):
            return (
                "I'm your voice assistant, running on ElevenLabs for speech and "
                "a language model for my answers."
            )

        if _matches(text, r"\b(what time|the time)\b"):
            now = datetime.now()
            hour = now.strftime("%I").lstrip("0") or "12"
            return f"It's {hour}:{now.strftime('%M %p')}."

        if _matches(text, r"\b(what(?:'s| is) the date|what day|today'?s date)\b"):
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

        if _matches(text, r"\b(repeat|say that again|come again)\b") and last_assistant:
            return last_assistant

        if _matches(text, r"\b(thank(s| you)|cheers)\b"):
            return "You're welcome. Anything else?"

        if _matches(text, r"\b(bye|goodbye|see you|stop|exit)\b"):
            return "Goodbye for now. Say the word if you'd like to talk again."

        if _matches(text, r"\b(help|what can you do|capabilities)\b"):
            return (
                "I can hold a spoken conversation: I listen, transcribe your "
                "speech, think of a reply and speak it back. Ask me a question "
                "or just chat. For richer answers, add a language-model API key."
            )

        if _matches(text, r"(tell me a joke)"):
            return (
                "Why did the developer go broke? Because they used up all "
                "their cache."
            )

        # Simple arithmetic: "what is 12 times 4".
        math_answer = _try_arithmetic(text)
        if math_answer is not None:
            return math_answer

        if text.endswith("?"):
            return (
                "That's a good question, but I don't have a language model "
                "connected right now, so I can't answer it properly. Add an "
                "LLM API key to unlock full answers."
            )

        return (
            "I heard you say: "
            f"\u201c{user_message.strip()}\u201d. "
            "I'm in offline mode, so my replies are limited. Configure a "
            "language-model key for genuine conversation."
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if data.get("message"):
                return str(data["message"])
    except Exception:  # noqa: BLE001
        pass
    return response.text[:200]


def _last_user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _last_assistant_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _try_arithmetic(text: str) -> str | None:
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(plus|minus|times|multiplied by|divided by|"
        r"[\+\-\*/x])\s*(-?\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        return None
    left = float(match.group(1))
    op = match.group(2)
    right = float(match.group(3))
    try:
        if op in ("plus", "+"):
            result = left + right
        elif op in ("minus", "-"):
            result = left - right
        elif op in ("times", "multiplied by", "*", "x"):
            result = left * right
        elif op in ("divided by", "/"):
            if right == 0:
                return "I can't divide by zero."
            result = left / right
        else:  # pragma: no cover - exhaustive
            return None
    except Exception:  # noqa: BLE001
        return None
    if result.is_integer():
        return f"That's {int(result)}."
    return f"That's {round(result, 4)}."


def build_backend(settings: Settings) -> LLMBackend:
    """Create the backend selected by ``settings.resolved_llm_provider()``."""
    provider = settings.resolved_llm_provider()
    if provider == "openai":
        if not settings.llm_api_key:
            raise ConversationError(
                "LLM_PROVIDER=openai requires LLM_API_KEY (or OPENAI_API_KEY)."
            )
        log.info(
            "Using OpenAI-compatible backend: %s (%s)",
            settings.llm_model,
            settings.llm_base_url,
        )
        return OpenAICompatibleBackend.from_settings(settings)
    if provider == "elevenlabs":
        if not settings.agents_agent_id:
            raise ConversationError(
                "LLM_PROVIDER=elevenlabs requires ELEVENLABS_AGENT_ID."
            )
        log.info(
            "Using ElevenLabs agent backend: %s (public=%s)",
            settings.agents_agent_id,
            settings.agents_agent_public,
        )
        return ElevenLabsAgentBackend(
            agent_id=settings.agents_agent_id,
            api_key=settings.elevenlabs_api_key or "",
            public=settings.agents_agent_public,
            timeout=settings.llm_timeout,
        )
    log.info("No LLM credentials found; using offline response backend.")
    return OfflineBackend()
