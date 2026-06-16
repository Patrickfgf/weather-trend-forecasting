"""Logging setup + Windows-safe UTF-8 stdout.

Python 3.14 on Windows defaults stdout/stderr to the locale codec (cp1252), so
printing unicode (degree sign, em-dash, >=) raises UnicodeEncodeError even though
the source file is valid UTF-8. Reconfigure the streams once at program start.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def enable_utf8_stdout() -> None:
    """Force stdout/stderr to UTF-8 where the platform allows (no-op on failure)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # pragma: no cover - platform dependent
                pass


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Idempotent logging setup; returns the project logger."""
    global _CONFIGURED
    enable_utf8_stdout()
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger("weather_forecast")
