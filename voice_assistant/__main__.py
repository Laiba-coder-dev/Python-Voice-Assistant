"""Allow ``python -m voice_assistant`` to launch the console assistant."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
