"""Timestamped console status lines shared by the email workflow, report memory, and legacy mode."""

from __future__ import annotations

import time

from .log import get_logger

logger = get_logger(__name__)


def log_status(message: str, start_time: float | None = None) -> None:
    """Write a ``[YYYY-MM-DD HH:MM:SS] message`` status line to the console (finance.log handler)."""
    suffix = ""
    if start_time is not None:
        suffix = f" | elapsed={time.perf_counter() - start_time:.1f}s"
    logger.info(f"{message}{suffix}")
