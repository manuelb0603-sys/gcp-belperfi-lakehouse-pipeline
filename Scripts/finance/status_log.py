"""Timestamped console status lines shared by the email workflow, report memory, and legacy mode."""

from __future__ import annotations

import time
from datetime import datetime


def log_status(message: str, start_time: float | None = None) -> None:
    """Write a timestamped status message to stdout for debugging and auditability."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if start_time is not None:
        suffix = f" | elapsed={time.perf_counter() - start_time:.1f}s"
    print(f"[{timestamp}] {message}{suffix}", flush=True)
