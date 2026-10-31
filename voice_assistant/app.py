"""Streamlit graphical interface for the voice assistant - "Aria".

Run with::

    streamlit run app.py

This module is the *presentation layer only*.  It talks to the exact same
:class:`~voice_assistant.assistant.VoiceAssistant` backend as before; every
control below is wired to the existing, working functionality:

* **Continuous conversation** - press *Start*; the assistant listens on the
  server's microphone, responds out loud and keeps going until *Stop*.
* **Interactive** - type a message, use a suggestion chip, or push-to-talk
  with ``st.audio_input``.

Audio plays through the machine running the app (this is a local desktop
assistant), so run it on the computer whose microphone/speakers you want to use.
"""

from __future__ import annotations

import html as _html
import queue
import sys
import threading
from pathlib import Path

# Make the package importable no matter where Streamlit sets the CWD from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from voice_assistant.assistant import SessionState, VoiceAssistant  # noqa: E402
from voice_assistant.config import KNOWN_VOICES, Settings  # noqa: E402

# --------------------------------------------------------------------------- #
# Static catalogue options (unchanged from the original UI)
# --------------------------------------------------------------------------- #
TTS_MODELS = [
    "eleven_turbo_v2_5",
    "eleven_flash_v2_5",
    "eleven_multilingual_v2",
    "eleven_v3",
]
STT_MODELS = ["scribe_v2", "scribe_v1"]
LANGUAGES = {
    "Auto-detect": None,
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Portuguese": "por",
    "Dutch": "nld",
    "Polish": "pol",
    "Russian": "rus",
    "Turkish": "tur",
    "Hindi": "hin",
    "Japanese": "jpn",
    "Korean": "kor",
    "Mandarin Chinese": "cmn",
}

# --------------------------------------------------------------------------- #
# State -> presentation maps (driven by the real SessionState, never invented)
# --------------------------------------------------------------------------- #
_ORB_CLASS = {
    SessionState.IDLE: "idle",
    SessionState.LISTENING: "listening",
    SessionState.TRANSCRIBING: "processing",
    SessionState.THINKING: "processing",
    SessionState.SPEAKING: "speaking",
    SessionState.ERROR: "error",
    SessionState.STOPPED: "stopped",
}
_ORB_CAPTION = {
    SessionState.IDLE: "Ready when you are",
    SessionState.LISTENING: "Listening to you",
    SessionState.TRANSCRIBING: "Transcribing your words",
    SessionState.THINKING: "Thinking it through",
    SessionState.SPEAKING: "Aria is speaking",
    SessionState.ERROR: "Something went wrong",
    SessionState.STOPPED: "Session stopped",
}
_STATUS = {
    SessionState.IDLE: ("Ready", "ready"),
    SessionState.LISTENING: ("Listening", "live"),
    SessionState.TRANSCRIBING: ("Processing", "busy"),
    SessionState.THINKING: ("Processing", "busy"),
    SessionState.SPEAKING: ("Speaking", "speak"),
    SessionState.ERROR: ("Error", "error"),
    SessionState.STOPPED: ("Stopped", "muted"),
}

_CHIPS = [
    ("chip_who", "Who are you?"),
    ("chip_interesting", "Tell me something interesting"),
    ("chip_plan", "Help me plan my day"),
]

# --------------------------------------------------------------------------- #
# Icons (inline SVG - no extra dependency, no emoji)
# --------------------------------------------------------------------------- #
_MIC_SVG = (
    '<svg class="orb-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="2" width="6" height="12" rx="3"/>'
    '<path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/>'
    '<line x1="8" y1="22" x2="16" y2="22"/></svg>'
)
_USER_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>'
    '<circle cx="12" cy="7" r="4"/></svg>'
)
_SPARK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3z"/>'
    '<path d="M19 15l.7 1.9L21.5 18l-1.8.7L19 21l-.7-2.3L16.5 18l1.8-1.1L19 15z"/></svg>'
)
_BOLT_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round" width="14" height="14">'
    '<path d="M13 2L4.5 13.5H11l-1 8.5L19.5 10H13l0-8z"/></svg>'
)

# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #
_CSS = """
:root{
  --aria-bg:#070b18; --aria-bg2:#0a1024;
  --aria-panel:rgba(18,24,48,0.62);
  --aria-panel-2:rgba(24,31,61,0.5);
  --aria-border:rgba(140,150,205,0.16);
  --aria-border-strong:rgba(140,150,205,0.30);
  --aria-text:#e9edfb; --aria-muted:#98a2c9;
  --aria-violet:#8b7cff; --aria-violet-2:#6d5bff;
  --aria-blue:#4c8dff; --aria-cyan:#39d3e6;
  --aria-green:#3ddc97; --aria-amber:#ffc36b; --aria-red:#ff6b7f;
  --aria-shadow:0 18px 50px rgba(4,8,22,0.55);
}
#MainMenu, footer{ visibility:hidden; }
.stApp, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 700px at 12% -10%, rgba(109,91,255,0.20), transparent 60%),
    radial-gradient(1000px 650px at 100% 0%, rgba(57,211,230,0.12), transparent 55%),
    linear-gradient(180deg, var(--aria-bg) 0%, var(--aria-bg2) 55%, #060a16 100%);
  color: var(--aria-text);
}
[data-testid="stHeader"]{ background: transparent; }
.block-container{ padding-top: 1.1rem; padding-bottom: 5.0rem; max-width: 1180px; }

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(13,18,38,0.97), rgba(9,13,28,0.97));
  border-right: 1px solid var(--aria-border);
  box-shadow: 24px 0 60px rgba(3,6,16,0.45);
}
[data-testid="stSidebarContent"]{ background: transparent; }
[data-testid="stSidebar"] .block-container{ padding-top: 1.2rem; }

.aria-brand{ display:flex; align-items:center; gap:12px; padding: 2px 2px 14px; }
.aria-logo{ width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
  background: linear-gradient(140deg, var(--aria-violet), var(--aria-blue));
  box-shadow: 0 10px 30px rgba(109,91,255,0.45), inset 0 0 0 1px rgba(255,255,255,0.14); }
.aria-logo svg{ width:24px; height:24px; stroke:#fff; }
.aria-brand-name{ font-size:1.16rem; font-weight:750; line-height:1.1; letter-spacing:.2px; }
.aria-brand-sub{ font-size:.7rem; color:var(--aria-muted); letter-spacing:.16em; text-transform:uppercase; }
.aria-conn{ display:flex; align-items:center; gap:8px; font-size:.79rem; color:var(--aria-muted);
  background:var(--aria-panel); border:1px solid var(--aria-border); border-radius:999px;
  padding:7px 12px; margin-bottom:14px; }
.aria-side-title{ font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase;
  color:var(--aria-muted); margin:6px 0 2px; }

/* ---------------- Connection / status dots ---------------- */
.aria-dot{ width:8px; height:8px; border-radius:50%; flex:0 0 8px; }
.aria-dot--on{ background:var(--aria-green); box-shadow:0 0 0 4px rgba(61,220,151,0.16); }
.aria-dot--off{ background:var(--aria-red); box-shadow:0 0 0 4px rgba(255,107,127,0.16); }
.aria-dot--ready{ background:var(--aria-violet); box-shadow:0 0 0 4px rgba(139,124,255,0.18); }
.aria-dot--live{ background:var(--aria-cyan); animation: dot-pulse 1.3s ease-in-out infinite; }
.aria-dot--busy{ background:var(--aria-amber); animation: dot-pulse 1s ease-in-out infinite; }
.aria-dot--speak{ background:var(--aria-blue); animation: dot-pulse .8s ease-in-out infinite; }
.aria-dot--error{ background:var(--aria-red); box-shadow:0 0 0 4px rgba(255,107,127,0.16); }
.aria-dot--muted{ background:#5b648a; }
@keyframes dot-pulse{
  0%,100%{ box-shadow:0 0 0 0 rgba(57,211,230,0.55); }
  50%{ box-shadow:0 0 0 6px rgba(57,211,230,0); } }

/* ---------------- Widgets ---------------- */
label, .stSelectbox label, .stTextInput label, .stRadio label{
  color: var(--aria-muted) !important; font-size:.8rem !important; font-weight:600;
}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  border-radius:12px; border:1px solid var(--aria-border-strong);
  background: linear-gradient(180deg, rgba(36,45,84,0.78), rgba(24,31,61,0.78));
  color: var(--aria-text); font-weight:600; transition: all .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover{
  border-color: rgba(139,124,255,0.6); transform: translateY(-1px);
  box-shadow: 0 10px 26px rgba(109,91,255,0.28); color:#fff;
}
.stButton > button:focus-visible{ outline:2px solid var(--aria-violet); outline-offset:2px; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background: linear-gradient(135deg, var(--aria-violet), var(--aria-blue));
  border-color: transparent; color:#fff;
  box-shadow: 0 12px 30px rgba(109,91,255,0.42);
}
.stButton > button:disabled{ opacity:.42; transform:none; box-shadow:none; }

.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div{
  background: rgba(12,17,36,0.85) !important;
  border:1px solid var(--aria-border) !important;
  border-radius:12px !important; color:var(--aria-text) !important;
}
.stSelectbox [data-baseweb="select"] svg{ fill: var(--aria-muted); }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"]{
  background:#101733 !important; border:1px solid var(--aria-border) !important;
  border-radius:12px !important;
}
[role="option"]{ color: var(--aria-text) !important; }
[role="option"]:hover{ background: rgba(139,124,255,0.18) !important; }

[data-testid="stExpander"]{
  border:1px solid var(--aria-border) !important; border-radius:14px !important;
  background: var(--aria-panel) !important; overflow:hidden; margin-bottom:8px;
}
[data-testid="stExpander"] summary{ font-weight:650; color:var(--aria-text); }
[data-testid="stExpander"] summary:hover{ color:var(--aria-violet); }

hr{ border-color: var(--aria-border) !important; }

/* ---------------- Header ---------------- */
.aria-header{ display:flex; align-items:center; justify-content:space-between;
  gap:16px; margin: 2px 0 8px; flex-wrap:wrap; }
.aria-hello{ font-size:1.5rem; font-weight:750; letter-spacing:-0.01em; line-height:1.15; }
.aria-hello span{ background: linear-gradient(100deg, #ffffff, #b9c6ff 58%, #7ee9f5);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.aria-subtle{ color:var(--aria-muted); font-size:.84rem; margin-top:3px; }
.aria-status{ display:inline-flex; align-items:center; gap:9px; padding:8px 14px;
  border-radius:999px; border:1px solid var(--aria-border-strong); background:var(--aria-panel);
  font-size:.82rem; font-weight:650; color:var(--aria-text); white-space:nowrap; }
.aria-status .aria-dot{ width:9px; height:9px; }
.aria-status--live{ border-color: rgba(57,211,230,0.45); box-shadow:0 0 22px rgba(57,211,230,0.14); }
.aria-status--speak{ border-color: rgba(76,141,255,0.5); box-shadow:0 0 22px rgba(76,141,255,0.16); }
.aria-status--busy{ border-color: rgba(255,195,107,0.45); }
.aria-status--error{ border-color: rgba(255,107,127,0.5); }

/* ---------------- Voice orb ---------------- */
.aria-stage{ display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:12px; padding: 10px 0 6px; }
.orb{ position:relative; width:172px; height:172px; display:grid; place-items:center; }
.orb-glow{ position:absolute; width:210px; height:210px; border-radius:50%; z-index:1;
  background: radial-gradient(circle, rgba(139,124,255,0.34), transparent 62%);
  filter: blur(10px); animation: orb-glow 6s ease-in-out infinite; }
.orb-ring{ position:absolute; border-radius:50%; border:1px solid rgba(139,124,255,0.34); z-index:2; }
.orb-ring--1{ width:142px; height:142px; animation: orb-ring-pulse 4.6s ease-out infinite; }
.orb-ring--2{ width:172px; height:172px; animation: orb-ring-pulse 4.6s ease-out infinite 1.5s; }
.orb-ring--3{ width:200px; height:200px; border-color: rgba(57,211,230,0.2);
  animation: orb-ring-pulse 4.6s ease-out infinite 3s; }
.orb-core{ position:relative; width:112px; height:112px; border-radius:50%; z-index:3;
  display:grid; place-items:center; animation: orb-breathe 5.5s ease-in-out infinite;
  background: radial-gradient(circle at 34% 30%, rgba(255,255,255,0.95),
    rgba(139,124,255,0.9) 30%, rgba(76,141,255,0.8) 60%, rgba(18,24,52,0.95) 100%);
  box-shadow: inset 0 0 26px rgba(255,255,255,0.32), 0 20px 55px rgba(109,91,255,0.48); }
.orb-core::after{ content:""; position:absolute; inset:8px; border-radius:50%;
  background: radial-gradient(circle at 40% 34%, rgba(255,255,255,0.5), transparent 55%);
  mix-blend-mode: screen; }
.orb-icon{ width:42px; height:42px; stroke: rgba(255,255,255,0.96); fill:none; z-index:4;
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.35)); }
.orb-wave{ display:none; align-items:flex-end; gap:4px; height:26px; }
.orb-wave span{ width:4px; height:6px; border-radius:3px;
  background: linear-gradient(180deg, var(--aria-cyan), var(--aria-violet));
  animation: orb-wave 1s ease-in-out infinite; }
.orb-wave span:nth-child(2){ animation-delay:.12s; }
.orb-wave span:nth-child(3){ animation-delay:.24s; }
.orb-wave span:nth-child(4){ animation-delay:.36s; }
.orb-wave span:nth-child(5){ animation-delay:.48s; }
.orb-wave span:nth-child(6){ animation-delay:.6s; }
.orb-wave span:nth-child(7){ animation-delay:.72s; }
.orb-caption{ font-size:.9rem; color:var(--aria-muted); letter-spacing:.01em; }

@keyframes orb-breathe{ 0%,100%{ transform:scale(1); } 50%{ transform:scale(1.045); } }
@keyframes orb-glow{ 0%,100%{ opacity:.55; transform:scale(1); } 50%{ opacity:.92; transform:scale(1.06); } }
@keyframes orb-ring-pulse{ 0%{ transform:scale(0.84); opacity:0; } 25%{ opacity:.55; } 100%{ transform:scale(1.16); opacity:0; } }
@keyframes orb-wave{ 0%,100%{ height:6px; opacity:.5; } 50%{ height:24px; opacity:1; } }
@keyframes orb-spin{ to{ transform: rotate(360deg); } }

/* State variants */
.orb--idle .orb-core{ animation-duration: 6.5s; }
.orb--listening .orb-ring{ animation-duration:2.3s; border-color: rgba(57,211,230,0.5); }
.orb--listening .orb-glow{ background: radial-gradient(circle, rgba(57,211,230,0.42), transparent 62%); }
.orb--listening .orb-wave{ display:flex; }
.orb--processing .orb-core{ animation: orb-breathe 2.2s ease-in-out infinite; }
.orb--processing .orb-core::before{ content:""; position:absolute; inset:-9px; border-radius:50%;
  background: conic-gradient(from 0deg, transparent 0deg, rgba(139,124,255,0.9) 90deg, transparent 200deg);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 3px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 3px));
  animation: orb-spin 1.5s linear infinite; }
.orb--speaking .orb-core{ animation: orb-breathe 1s ease-in-out infinite; }
.orb--speaking .orb-ring{ animation-duration:1.8s; border-color: rgba(76,141,255,0.5); }
.orb--speaking .orb-wave{ display:flex; }
.orb--speaking .orb-wave span{ animation-duration:.58s; }
.orb--error .orb-glow{ background: radial-gradient(circle, rgba(255,107,127,0.4), transparent 62%); }
.orb--error .orb-core{ background: radial-gradient(circle at 34% 30%, rgba(255,255,255,0.9),
  rgba(255,107,127,0.85) 40%, rgba(90,30,50,0.95) 100%); }
.orb--stopped .orb-core, .orb--stopped .orb-glow, .orb--stopped .orb-ring{ animation:none; opacity:.55; }
.orb--stopped .orb-core{ filter: grayscale(0.6); }
.orb--stopped .orb-glow{ opacity:.2; }

/* ---------------- Conversation ---------------- */
.aria-chat{ display:flex; flex-direction:column; gap:14px; padding:8px 2px 4px;
  max-height: min(50vh, 470px); overflow-y:auto; scrollbar-width:thin; }
.aria-chat::-webkit-scrollbar{ width:8px; }
.aria-chat::-webkit-scrollbar-thumb{ background: rgba(139,124,255,0.32); border-radius:8px; }
.aria-chat--live{ flex-direction: column-reverse; }
.msg{ display:flex; gap:10px; align-items:flex-end; animation: msg-in .3s ease; }
.aria-chat--static .msg{ animation:none; }
.msg--user{ flex-direction:row-reverse; }
.msg-avatar{ width:34px; height:34px; border-radius:11px; flex:0 0 34px; display:grid;
  place-items:center; border:1px solid var(--aria-border); background: var(--aria-panel); }
.msg-avatar svg{ width:17px; height:17px; stroke: var(--aria-text); }
.msg--assistant .msg-avatar{ background: linear-gradient(140deg, rgba(139,124,255,0.92), rgba(76,141,255,0.92));
  border-color: transparent; }
.msg-bubble{ max-width:min(680px, 78%); padding:12px 15px; border-radius:16px;
  font-size:.95rem; line-height:1.55; color:var(--aria-text);
  border:1px solid var(--aria-border); background: var(--aria-panel);
  box-shadow: 0 8px 22px rgba(4,8,22,0.35); overflow-wrap:anywhere; }
.msg--user .msg-bubble{ background: linear-gradient(135deg, rgba(109,91,255,0.94), rgba(59,130,246,0.94));
  border-color: transparent; color:#fff; border-bottom-right-radius:5px; }
.msg--assistant .msg-bubble{ border-bottom-left-radius:5px; }
@keyframes msg-in{ from{ opacity:0; transform: translateY(6px); } to{ opacity:1; transform:none; } }

.aria-empty{ text-align:center; padding: 18px 10px 8px; }
.aria-empty h3{ margin:0 0 6px; font-size:1.2rem; font-weight:700; }
.aria-empty p{ margin:0 auto; max-width:520px; color:var(--aria-muted); font-size:.9rem; line-height:1.6; }

/* suggestion chips */
.st-key-chips button{ width:100%; border-radius:999px; padding:9px 0; font-weight:600; }
[data-testid="stHorizontalBlock"]{ gap: .6rem; }

/* ---------------- Composer / footer ---------------- */
[data-testid="stChatInput"]{
  background: rgba(12,17,36,0.92) !important;
  border:1px solid var(--aria-border-strong) !important; border-radius:16px !important;
  box-shadow: var(--aria-shadow); transition: border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stChatInput"] textarea{ color: var(--aria-text) !important; }
[data-testid="stChatInput"]:focus-within{ border-color: var(--aria-violet) !important;
  box-shadow: 0 0 0 3px rgba(139,124,255,0.22), var(--aria-shadow); }
[data-testid="stChatInput"] button{ color: var(--aria-violet); }
[data-testid="stAudioInput"]{ background: transparent !important; }
.aria-hint{ color:var(--aria-muted); font-size:.76rem; padding-top:14px; }
.aria-powered{ display:flex; align-items:center; justify-content:center; gap:8px;
  color:var(--aria-muted); font-size:.76rem; padding:10px 0 2px; }
.aria-powered b{ color:#c9d2f5; font-weight:650; }
.aria-powered svg{ stroke: var(--aria-violet); }

/* ---------------- Responsive ---------------- */
@media (max-width: 900px){
  .aria-header{ flex-direction:column; align-items:flex-start; }
  .msg-bubble{ max-width:88%; }
  .block-container{ padding-left:1rem; padding-right:1rem; }
  .aria-chat{ max-height: 44vh; }
}
"""


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("events", queue.Queue())
    st.session_state.setdefault("assistant", None)
    st.session_state.setdefault("thread", None)
    st.session_state.setdefault("stop_event", None)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("state", SessionState.IDLE)
    st.session_state.setdefault("error", "")
    st.session_state.setdefault("info", "")
    st.session_state.setdefault("voices", [])
    st.session_state.setdefault("voices_error", "")
    st.session_state.setdefault("last_audio_key", None)
    st.session_state.setdefault("settings_key", None)


def _make_callbacks(events: queue.Queue | None = None) -> dict:
    """Build UI callbacks bound to a concrete event queue.

    The callbacks fire on the assistant worker thread, where Streamlit's
    ``session_state`` proxy is not safe to touch.  Binding the queue directly
    keeps event delivery reliable even after the browser reconnects or reruns.
    """
    target = events if events is not None else st.session_state.events

    def emit(kind: str, payload) -> None:  # noqa: ANN001
        target.put((kind, payload))

    return {
        "on_state": lambda state: emit("state", state),
        "on_user": lambda text: emit("user", text),
        "on_assistant": lambda text: emit("assistant", text),
        "on_info": lambda text: emit("info", text),
        "on_error": lambda text: emit("error", str(text)),
    }


def _drain_events() -> None:
    events: queue.Queue = st.session_state.events
    while not events.empty():
        kind, payload = events.get()
        if kind == "user":
            st.session_state.chat.append({"role": "user", "content": payload})
        elif kind == "assistant":
            st.session_state.chat.append({"role": "assistant", "content": payload})
        elif kind == "state":
            st.session_state.state = payload
            if payload == SessionState.STOPPED:
                st.session_state.running = False
        elif kind == "info":
            st.session_state.info = payload
        elif kind == "error":
            st.session_state.error = payload


# --------------------------------------------------------------------------- #
# Settings / assistant construction
# --------------------------------------------------------------------------- #
def _base_settings() -> Settings:
    return Settings.from_env()


def _build_settings_from_ui() -> Settings:
    settings = _base_settings()
    s = st.session_state
    if s.get("voice_choice_id"):
        settings.tts_voice_id = s.voice_choice_id
    if s.get("custom_voice_id"):
        settings.tts_voice_id = s.custom_voice_id.strip()
    settings.tts_model_id = s.get("tts_model", settings.tts_model_id)
    settings.stt_model_id = s.get("stt_model", settings.stt_model_id)
    settings.stt_mode = s.get("stt_mode", settings.stt_mode)
    settings.stt_language_code = s.get("language_code")
    settings.assistant_name = s.get("assistant_name", settings.assistant_name)
    return settings


def _settings_signature(settings: Settings) -> tuple:
    return (
        settings.tts_voice_id,
        settings.tts_model_id,
        settings.stt_model_id,
        settings.stt_mode,
        settings.stt_language_code,
        settings.assistant_name,
        settings.llm_provider,
        settings.llm_model,
    )


def _ensure_interactive_assistant(settings: Settings) -> VoiceAssistant | None:
    """Return a persistent (history-preserving) assistant for text/push-to-talk."""
    problems = settings.validate()
    hard = [p for p in problems if "ELEVENLABS_API_KEY" in p]
    if hard:
        st.session_state.error = hard[0]
        return None

    signature = _settings_signature(settings)
    assistant: VoiceAssistant | None = st.session_state.assistant
    if assistant is None or st.session_state.settings_key != signature:
        if assistant is not None:
            assistant.close()
        assistant = VoiceAssistant(settings, **_make_callbacks())
        st.session_state.assistant = assistant
        st.session_state.settings_key = signature
    return assistant


# --------------------------------------------------------------------------- #
# Submission helpers (reuse the existing backend methods)
# --------------------------------------------------------------------------- #
def _submit_text(text: str, settings: Settings) -> None:
    text = (text or "").strip()
    if not text:
        return
    assistant = _ensure_interactive_assistant(settings)
    if assistant is None:
        return
    with st.spinner("Thinking…"):
        assistant.handle_text(text, speak=True)
    _drain_events()


def _submit_audio(audio, settings: Settings) -> None:  # noqa: ANN001
    assistant = _ensure_interactive_assistant(settings)
    if assistant is None:
        return
    with st.spinner("Transcribing and responding…"):
        assistant.handle_audio_bytes(audio.getvalue())
    _drain_events()


def _esc(text: str) -> str:
    return _html.escape(text or "").replace("\n", "<br/>")


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _render_header() -> None:
    state: SessionState = st.session_state.state
    name = st.session_state.get("assistant_name") or "Aria"
    label, dot = _STATUS.get(state, ("Ready", "ready"))
    turns = len(st.session_state.chat) // 2
    plural = "" if turns == 1 else "s"
    st.markdown(
        f"""
        <div class="aria-header">
          <div>
            <div class="aria-hello">Good to see you, <span>I'm {_esc(name)}</span></div>
            <div class="aria-subtle">Your AI voice companion · {turns} exchange{plural}</div>
          </div>
          <div class="aria-status aria-status--{dot}">
            <span class="aria-dot aria-dot--{dot}"></span>{_esc(label)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_orb() -> None:
    state: SessionState = st.session_state.state
    cls = _ORB_CLASS.get(state, "idle")
    caption = _ORB_CAPTION.get(state, "Ready when you are")
    st.markdown(
        f"""
        <div class="aria-stage">
          <div class="orb orb--{cls}">
            <span class="orb-glow"></span>
            <span class="orb-ring orb-ring--1"></span>
            <span class="orb-ring orb-ring--2"></span>
            <span class="orb-ring orb-ring--3"></span>
            <span class="orb-core">{_MIC_SVG}</span>
          </div>
          <div class="orb-wave" aria-hidden="true">
            <span></span><span></span><span></span><span></span>
            <span></span><span></span><span></span>
          </div>
          <div class="orb-caption">{_esc(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat(live: bool = False) -> None:
    chat: list[dict] = st.session_state.chat
    if not chat:
        st.markdown(
            '<div class="aria-chat aria-chat--static">'
            '<div class="aria-empty"><p>Waiting for your first words…'
            "</p></div></div>",
            unsafe_allow_html=True,
        )
        return

    messages = list(reversed(chat)) if live else chat
    if live:
        messages = messages[:40]
    rows: list[str] = []
    for message in messages:
        role = "user" if message["role"] == "user" else "assistant"
        avatar = _USER_SVG if role == "user" else _SPARK_SVG
        rows.append(
            f'<div class="msg msg--{role}">'
            f'<div class="msg-avatar">{avatar}</div>'
            f'<div class="msg-bubble">{_esc(message["content"])}</div>'
            f"</div>"
        )
    classes = "aria-chat aria-chat--live" if live else "aria-chat"
    st.markdown(
        f'<div class="{classes}">' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def _render_empty_state(settings: Settings) -> None:
    st.markdown(
        '<div class="aria-empty">'
        "<h3>Welcome to Aria</h3>"
        "<p>I'm your AI voice companion. Press <b>Start session</b> to talk, "
        "or type a message below. Not sure where to begin? Try one of these.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="chips"):
        columns = st.columns(3, gap="small")
        for column, (key, label) in zip(columns, _CHIPS):
            with column:
                if st.button(label, key=key, use_container_width=True):
                    _submit_text(label, settings)
                    st.rerun()


def _render_composer(settings: Settings) -> None:
    text = st.chat_input("Message Aria…")
    if text:
        _submit_text(text, settings)
        st.rerun()

    col_mic, col_start, col_hint = st.columns([1.15, 1.2, 3.0], gap="small")
    with col_mic:
        with st.popover("Voice", icon=":material/mic:", use_container_width=True):
            st.caption("Push to talk")
            audio = st.audio_input(
                "Record a question", key="ptt", label_visibility="collapsed"
            )
            if audio is not None:
                key = (audio.name, audio.size, getattr(audio, "file_id", None))
                if key != st.session_state.last_audio_key:
                    st.session_state.last_audio_key = key
                    _submit_audio(audio, settings)
                    st.rerun()
    with col_start:
        if st.button(
            "Start session",
            icon=":material/graphic_eq:",
            type="primary",
            use_container_width=True,
        ):
            _start_session(settings)
    with col_hint:
        st.markdown(
            '<div class="aria-hint">Enter to send · Shift+Enter for a new line</div>',
            unsafe_allow_html=True,
        )


def _render_footer() -> None:
    st.markdown(
        f'<div class="aria-powered">{_BOLT_SVG}'
        "Powered by <b>ElevenLabs</b> · Speech &amp; Voice</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _render_sidebar() -> None:
    settings = _base_settings()
    connected = settings.elevenlabs_configured
    running = st.session_state.running

    with st.sidebar:
        # Brand ---------------------------------------------------------- #
        st.markdown(
            f"""
            <div class="aria-brand">
              <div class="aria-logo">{_SPARK_SVG}</div>
              <div>
                <div class="aria-brand-name">Aria</div>
                <div class="aria-brand-sub">AI Voice Companion</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Connection indicator ------------------------------------------ #
        if running:
            dot, conn = "live", "Session active"
        elif connected:
            dot, conn = "on", "Connected to ElevenLabs"
        else:
            dot, conn = "off", "ElevenLabs key not set"
        st.markdown(
            f'<div class="aria-conn"><span class="aria-dot aria-dot--{dot}"></span>'
            f"{_esc(conn)}</div>",
            unsafe_allow_html=True,
        )

        # Voice ---------------------------------------------------------- #
        st.markdown('<div class="aria-side-title">Voice</div>', unsafe_allow_html=True)
        voice_options: dict[str, str] = dict(KNOWN_VOICES)
        for voice in st.session_state.voices:
            label = voice["name"]
            if voice.get("labels"):
                label += f" ({voice['labels']})"
            voice_options.setdefault(label, voice["id"])

        labels = list(voice_options.keys())
        default_label = next(
            (lbl for lbl, vid in voice_options.items() if vid == settings.tts_voice_id),
            labels[0],
        )
        selected = st.selectbox(
            "Voice",
            options=labels,
            index=labels.index(default_label) if default_label in labels else 0,
            key="voice_label",
        )
        st.session_state.voice_choice_id = voice_options.get(selected, "")
        st.text_input(
            "Custom voice ID",
            value="",
            key="custom_voice_id",
            help="Paste any ElevenLabs voice ID to override the selection.",
        )
        if st.button("Refresh voice list", icon=":material/refresh:", use_container_width=True):
            _refresh_voices(settings)
        if st.session_state.voices_error:
            st.caption(f"⚠️ {st.session_state.voices_error}")
        elif st.session_state.voices:
            st.caption(f"Loaded {len(st.session_state.voices)} voices from your account.")

        # Collapsible settings ------------------------------------------ #
        with st.expander("Speech & model", expanded=False):
            st.selectbox("Speech model", STT_MODELS, key="stt_model")
            st.selectbox(
                "Speech mode",
                ["batch", "realtime"],
                key="stt_mode",
                help="realtime streams audio for lower latency (requires scribe_v2_realtime).",
            )
            st.selectbox("Language", list(LANGUAGES.keys()), key="language_label")
            st.session_state.language_code = LANGUAGES[st.session_state.language_label]
            st.selectbox("Voice model", TTS_MODELS, key="tts_model")

        with st.expander("Assistant", expanded=False):
            st.text_input(
                "Assistant name", value=settings.assistant_name, key="assistant_name"
            )
            if st.button("Reset conversation", icon=":material/restart_alt:", use_container_width=True):
                if st.session_state.assistant is not None:
                    st.session_state.assistant.reset_conversation()
                st.session_state.chat = []
                st.session_state.error = ""
                st.session_state.info = ""
                if not running:
                    st.rerun()

        # Session controls ---------------------------------------------- #
        st.markdown('<div class="aria-side-title">Session</div>', unsafe_allow_html=True)
        col_start, col_stop = st.columns(2)
        if col_start.button(
            "Start",
            icon=":material/play_arrow:",
            disabled=running,
            type="primary",
            use_container_width=True,
        ):
            _start_session(_build_settings_from_ui())
        if col_stop.button(
            "Stop",
            icon=":material/stop:",
            disabled=not running,
            use_container_width=True,
        ):
            _stop_session()

        if st.session_state.error:
            st.error(st.session_state.error)
            if st.button("Dismiss", use_container_width=True):
                st.session_state.error = ""
                st.rerun()

        # Configuration summary ----------------------------------------- #
        with st.expander("Configuration", expanded=False):
            st.json(settings.public_summary())


def _refresh_voices(settings: Settings) -> None:
    st.session_state.voices_error = ""
    if not settings.elevenlabs_configured:
        st.session_state.voices_error = "Set ELEVENLABS_API_KEY to load voices."
        return
    try:
        from elevenlabs import ElevenLabs

        from voice_assistant.text_to_speech import TextToSpeech

        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        st.session_state.voices = TextToSpeech(client, settings).list_voices()
    except Exception as exc:  # noqa: BLE001
        st.session_state.voices_error = f"Could not load voices: {exc}"


# --------------------------------------------------------------------------- #
# Continuous mode
# --------------------------------------------------------------------------- #
def _run_target(
    assistant: VoiceAssistant,
    stop_event: threading.Event,
    events: queue.Queue,
) -> None:
    from voice_assistant.realtime_stt import run_realtime_assistant

    try:
        if assistant.settings.stt_mode == "realtime":
            import asyncio

            asyncio.run(run_realtime_assistant(assistant, stop_event))
        else:
            assistant.run(stop_event)
    except Exception as exc:  # noqa: BLE001
        events.put(("error", f"Session ended unexpectedly: {exc}"))
        events.put(("state", SessionState.STOPPED))


def _start_session(settings: Settings) -> None:
    problems = settings.validate()
    if problems:
        st.session_state.error = " · ".join(problems)
        st.rerun()
        return

    _drain_events()  # clear any stale events
    stop_event = threading.Event()
    events: queue.Queue = queue.Queue()
    st.session_state.events = events

    previous: VoiceAssistant | None = st.session_state.assistant
    if previous is not None:
        previous.close()

    assistant = VoiceAssistant(settings, **_make_callbacks(events))
    st.session_state.assistant = assistant
    st.session_state.settings_key = _settings_signature(settings)
    st.session_state.stop_event = stop_event
    st.session_state.error = ""
    st.session_state.info = ""
    st.session_state.state = SessionState.LISTENING
    st.session_state.running = True

    thread = threading.Thread(
        target=_run_target,
        args=(assistant, stop_event, events),
        name="assistant",
        daemon=True,
    )
    st.session_state.thread = thread
    thread.start()
    st.rerun()


def _stop_session() -> None:
    stop_event: threading.Event | None = st.session_state.stop_event
    assistant: VoiceAssistant | None = st.session_state.assistant
    if stop_event is not None:
        stop_event.set()
    if assistant is not None:
        assistant.stop()
    thread: threading.Thread | None = st.session_state.thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=5.0)
    st.session_state.running = False
    st.session_state.state = SessionState.STOPPED
    # Force the interactive path to build a fresh (history-cleared) assistant.
    st.session_state.assistant = None
    st.session_state.settings_key = None
    st.rerun()


@st.fragment(run_every=0.6)
def _render_continuous() -> None:
    _drain_events()
    _render_header()
    _render_orb()
    _render_chat(live=True)
    st.caption("Live session in progress — press **Stop** in the sidebar to end.")


# --------------------------------------------------------------------------- #
# Workspace + entry point
# --------------------------------------------------------------------------- #
def _render_workspace(settings: Settings) -> None:
    _render_header()
    _render_orb()
    if st.session_state.chat:
        _render_chat()
    else:
        _render_empty_state(settings)
    _render_composer(settings)


def main() -> None:
    st.set_page_config(
        page_title="Aria · AI Voice Companion",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
    _init_state()
    _render_sidebar()

    base = _base_settings()
    if not base.elevenlabs_configured and not st.session_state.running:
        st.markdown(
            '<div class="aria-conn" style="border-color:rgba(255,107,127,0.4)">'
            '<span class="aria-dot aria-dot--error"></span>'
            "ELEVENLABS_API_KEY is not set — copy .env.example to .env, add your "
            "key, and restart the app.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.running:
        _render_continuous()
    else:
        _render_workspace(base)

    _render_footer()


if __name__ == "__main__":
    main()
