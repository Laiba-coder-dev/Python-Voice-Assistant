"""Console entry point - the real-time, continuous conversation experience.

Usage examples::

    python -m voice_assistant.cli                 # continuous voice loop
    python -m voice_assistant.cli --text "Hi!"    # one typed turn, then exit
    python -m voice_assistant.cli --list-devices  # show microphones/speakers
    python -m voice_assistant.cli --list-voices   # show ElevenLabs voices
    python -m voice_assistant.cli --mode agents   # ElevenLabs Agents (full duplex)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading

from .assistant import SessionState, VoiceAssistant
from .config import Settings
from .logging_utils import configure_logging, get_logger

log = get_logger(__name__)

_USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


class ConsoleReporter:
    """Renders assistant events to the terminal."""

    def __init__(self, assistant_name: str) -> None:
        self.assistant_name = assistant_name

    def on_state(self, state: SessionState) -> None:
        print(_c(f"[{state.value}]", "36"), end=" ", flush=True)

    def on_user(self, text: str) -> None:
        print("\r" + _c("You:  ", "32") + text, flush=True)

    def on_assistant(self, text: str) -> None:
        print(_c(f"{self.assistant_name}: ", "35") + text, flush=True)

    def on_info(self, message: str) -> None:
        print(_c("  · " + message, "90"), flush=True)

    def on_error(self, message: str) -> None:
        print(_c("  ! " + message, "31"), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice_assistant",
        description="A voice-enabled AI assistant powered by ElevenLabs.",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "agents"],
        default="pipeline",
        help="pipeline: Scribe + LLM + TTS. agents: ElevenLabs Conversational AI.",
    )
    parser.add_argument("--text", help="Send one typed message and exit.")
    parser.add_argument(
        "--once", action="store_true", help="Capture a single utterance and exit."
    )
    parser.add_argument("--voice", help="Override the ElevenLabs voice ID.")
    parser.add_argument("--no-speak", action="store_true", help="Do not play audio.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices.")
    parser.add_argument("--list-voices", action="store_true", help="List ElevenLabs voices.")
    parser.add_argument(
        "--log", default=os.getenv("LOG_LEVEL", "WARNING"), help="Log level."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate configuration and exit."
    )
    return parser


def _print_devices() -> int:
    from .audio_handler import AudioError, list_devices

    try:
        devices = list_devices()
    except AudioError as exc:
        print(_c(f"Audio error: {exc}", "31"))
        return 1
    if not devices:
        print("No audio devices found.")
        return 1
    print("Audio devices:")
    for dev in devices:
        kind = []
        if dev.max_input_channels:
            kind.append("input")
        if dev.max_output_channels:
            kind.append("output")
        print(
            f"  {dev.index:>2}: {dev.name} "
            f"({', '.join(kind) or 'none'}; "
            f"in={dev.max_input_channels}, out={dev.max_output_channels})"
        )
    return 0


def _print_voices(settings: Settings) -> int:
    from .text_to_speech import TextToSpeech
    from elevenlabs import ElevenLabs

    try:
        settings.require_elevenlabs()
    except Exception as exc:  # noqa: BLE001
        print(_c(str(exc), "31"))
        return 1
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    try:
        voices = TextToSpeech(client, settings).list_voices()
    except Exception as exc:  # noqa: BLE001
        print(_c(f"Could not list voices: {exc}", "31"))
        return 1
    print("Available voices (id  name):")
    for voice in voices:
        extra = f"  [{voice['labels']}]" if voice.get("labels") else ""
        print(f"  {voice['id']}  {voice['name']}{extra}")
    return 0


def _run_one_shot(assistant: VoiceAssistant, text: str, speak: bool) -> int:
    reporter = ConsoleReporter(assistant.settings.assistant_name)
    try:
        reply = assistant.handle_text(text, speak=speak)
    except Exception as exc:  # noqa: BLE001
        print(_c(f"Error: {exc}", "31"))
        return 1
    if reply is None:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log)

    if args.list_devices:
        return _print_devices()

    settings = Settings.from_env()
    if args.voice:
        settings.tts_voice_id = args.voice

    if args.list_voices:
        return _print_voices(settings)

    problems = settings.validate()
    for problem in problems:
        print(_c(f"  ! {problem}", "33"))

    if args.check:
        if not problems:
            print(_c("Configuration looks good.", "32"))
            return 0
        return 1

    if args.mode == "agents":
        if problems and not settings.elevenlabs_configured:
            print(_c("Cannot start Agents mode without an ElevenLabs API key.", "31"))
            return 1
        return _run_agents(settings)

    # Pipeline mode requires ElevenLabs.
    if not settings.elevenlabs_configured:
        print(
            _c(
                "ELEVENLABS_API_KEY is required to run the assistant. "
                "Copy .env.example to .env and add your key.",
                "31",
            )
        )
        return 1

    reporter = ConsoleReporter(settings.assistant_name)
    assistant = VoiceAssistant(
        settings,
        on_state=reporter.on_state,
        on_user=reporter.on_user,
        on_assistant=reporter.on_assistant,
        on_info=reporter.on_info,
        on_error=reporter.on_error,
    )
    assistant.set_speaking_enabled(not args.no_speak)

    if args.text:
        return _run_one_shot(assistant, args.text, speak=not args.no_speak)

    try:
        if args.once:
            return _run_once(assistant)
        return _run_loop(assistant)
    finally:
        assistant.close()


def _run_once(assistant: VoiceAssistant) -> int:
    """Capture a single utterance, respond, and exit."""
    assistant._ensure_ready()  # noqa: SLF001
    print(_c("Listening for one utterance...", "36"))
    audio = assistant._recorder.record_utterance(timeout=15.0)  # noqa: SLF001
    if audio is None:
        print(_c("No speech detected.", "33"))
        return 1
    assistant.handle_pcm(audio, assistant.settings.sample_rate)
    return 0


def _run_loop(assistant: VoiceAssistant) -> int:
    """Run the continuous pipeline loop, or realtime mode if configured."""
    settings = assistant.settings
    print(_c(f"{settings.assistant_name}: ", "35") + "Starting conversation. " +
          "Press Ctrl+C to stop.", flush=True)

    stop_event = threading.Event()
    try:
        if settings.stt_mode == "realtime":
            from .realtime_stt import run_realtime_assistant

            asyncio.run(run_realtime_assistant(assistant, stop_event))
        else:
            assistant.run(stop_event)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        stop_event.set()
        assistant.stop()
        print("\n" + _c("Stopping...", "36"))
    return 0


def _run_agents(settings: Settings) -> int:
    from .agents_mode import run_agents_conversation

    reporter = ConsoleReporter(settings.assistant_name)
    print(_c("Starting ElevenLabs agent session. Press Ctrl+C to stop.", "36"))
    run_agents_conversation(
        settings,
        on_user=reporter.on_user,
        on_agent=reporter.on_assistant,
        on_error=reporter.on_error,
        on_state=reporter.on_state,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
