"""Convenience launcher for the Streamlit UI.

Run the graphical interface from the project root with::

    streamlit run app.py

This simply forwards to :func:`voice_assistant.app.main`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_assistant.app import main  # noqa: E402

main()
