"""Minimal logging helpers.

Conversation content is privacy-sensitive, so transcript text is only logged
when the user explicitly opts in via ``LOG_TRANSCRIPTS=1``.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once, using ``LOG_LEVEL`` if not provided."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
