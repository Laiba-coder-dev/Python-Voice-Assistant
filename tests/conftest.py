"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def settings():
    from voice_assistant.config import Settings

    return Settings(elevenlabs_api_key="test-key")


@pytest.fixture
def loud_block():
    """A short, loud int16 block that easily clears the VAD threshold."""
    return (np.ones(480, dtype=np.int16) * 20000)


@pytest.fixture
def silent_block():
    return np.zeros(480, dtype=np.int16)
