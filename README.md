# 🎙️ Aria — AI Voice Assistant

**A real-time, hands-free voice assistant powered by ElevenLabs and an LLM.**

Aria listens to your microphone, transcribes your speech with ElevenLabs
Scribe, generates a context-aware reply with an LLM, and speaks the answer
back through ElevenLabs text-to-speech — in a continuous
listen → think → speak loop.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Scribe%20%2B%20TTS-7c5cff)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen)

---

## ✨ Highlights

- **Full voice loop** — mic → ElevenLabs STT → LLM → ElevenLabs TTS → speakers.
- **Two front ends** — a polished dark-themed **Streamlit dashboard** (animated
  voice orb, live status, chat, settings) and a fast **console app**.
- **Explicit response generation** with conversation history, so follow-ups work.
- **Pluggable LLM backend** — any OpenAI-compatible API, or a built-in offline
  responder that runs with no LLM key.
- **Anti-echo protection** — the mic is idle during playback *and* transcripts
  that echo Aria's own voice are ignored.
- **Graceful errors** — auth, rate-limit, network and audio issues become
  friendly messages instead of crashes.
- **Tested** — 72 tests with mocked APIs (no network or hardware needed).

## 🖥️ Two ways to use it

| Interface | Command | Best for |
| --- | --- | --- |
| **Streamlit dashboard** | `streamlit run app.py` | Everyday use, settings, visual feedback |
| **Console app** | `python -m voice_assistant` | Fast, no-browser conversations |

## 🚀 Quickstart

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd "Virtual Assistant"

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your key
cp .env.example .env             # Windows: copy .env.example .env
# then edit .env and set ELEVENLABS_API_KEY

# 5. Run it
streamlit run app.py             # dashboard
python -m voice_assistant        # or the console app
```

Get an ElevenLabs key at <https://elevenlabs.io/app/settings/api-keys>.
macOS/Linux users may need PortAudio first:
`brew install portaudio` or `sudo apt-get install libportaudio2`.

## 🔄 How it works

```
Idle → Listening → Transcribing → Thinking → Speaking → Listening again
```

1. **Listening** — captures microphone audio with an RMS voice-activity detector.
2. **Transcribing** — sends the clip to ElevenLabs Scribe (`scribe_v2`).
3. **Thinking** — appends the transcript to the conversation history and asks the
   LLM for a reply (it does *not* assume the transcript is the answer).
4. **Speaking** — streams the reply through ElevenLabs TTS to your speakers.

## ⚙️ Configuration

All settings live in `.env` (see [`.env.example`](.env.example)). The essentials:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ELEVENLABS_API_KEY` | – | **Required.** Used for STT and TTS. |
| `STT_MODE` | `batch` | `batch` (reliable) or `realtime` (streaming). |
| `TTS_VOICE_ID` | George | Any ElevenLabs voice ID. |
| `TTS_MODEL_ID` | `eleven_turbo_v2_5` | Text-to-speech model. |
| `INPUT_DEVICE` / `OUTPUT_DEVICE` | system default | Device indexes from `--list-devices`. |
| `LLM_PROVIDER` | `auto` | `auto` / `openai` / `offline`. |
| `LLM_API_KEY` | – | Enables the OpenAI-compatible backend. |
| `ASSISTANT_NAME` | `Aria` | Display name / persona. |
| `ECHO_GUARD` | `true` | Ignore transcripts that echo Aria. |

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q      # -> 72 passed
```

## 📁 Project structure

```
.
├── app.py                     # Streamlit launcher
├── requirements.txt
├── .env.example
├── tests/                     # pytest suite (mocked APIs)
└── voice_assistant/
    ├── config.py              # settings + .env loading
    ├── audio_handler.py       # microphone VAD capture + PCM playback
    ├── speech_to_text.py      # ElevenLabs Scribe (batch)
    ├── text_to_speech.py      # ElevenLabs TTS streaming
    ├── llm.py                 # OpenAI-compatible + offline backends
    ├── assistant.py           # conversation engine + state machine
    ├── realtime_stt.py        # streaming Scribe realtime
    ├── agents_mode.py         # optional ElevenLabs Agents mode
    ├── cli.py                 # console entry point
    └── app.py                 # Streamlit UI
```

## 🛠️ Troubleshooting

- **`ELEVENLABS_API_KEY is not set` / 401** — create `.env` with a valid key and
  restart. Verify with `python -m voice_assistant.cli --check`.
- **No audio devices** — run `python -m voice_assistant.cli --list-devices` and
  set `INPUT_DEVICE` / `OUTPUT_DEVICE`. On Linux install PortAudio and unmute.
- **Aria replies to itself** — use headphones, keep `ECHO_GUARD=true`, and lower
  the speaker volume.
- **Cut off mid-sentence** — increase `SILENCE_DURATION` (e.g. `1.2`).
- **`streamlit: command not found`** — activate the venv or run
  `python -m streamlit run app.py`.

## 🔐 Security

- **Never commit `.env`** — it is git-ignored; only `.env.example` belongs in
  version control.
- API keys are hidden from logs (`repr=False`) and shown redacted in the UI.
- Transcript logging is off by default (`LOG_TRANSCRIPTS=false`).

## 📄 License

No license file is included yet — add one (e.g. MIT) before distributing.
