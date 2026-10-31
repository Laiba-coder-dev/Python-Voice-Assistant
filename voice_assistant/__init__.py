"""Voice-enabled AI virtual assistant built on ElevenLabs + an LLM.

The package is split into small, single-responsibility modules so that audio
capture, speech recognition, conversation management and speech synthesis can
be reasoned about (and tested) independently:

* :mod:`voice_assistant.config`         - environment/configuration handling
* :mod:`voice_assistant.audio_handler`  - microphone capture + PCM playback
* :mod:`voice_assistant.speech_to_text` - ElevenLabs Scribe transcription
* :mod:`voice_assistant.text_to_speech` - ElevenLabs text-to-speech
* :mod:`voice_assistant.llm`            - pluggable response-generation backends
* :mod:`voice_assistant.assistant`      - conversation + state-machine orchestration
* :mod:`voice_assistant.agents_mode`    - optional ElevenLabs Conversational AI mode
* :mod:`voice_assistant.cli`            - continuous console loop
* :mod:`voice_assistant.app`            - Streamlit graphical interface
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
