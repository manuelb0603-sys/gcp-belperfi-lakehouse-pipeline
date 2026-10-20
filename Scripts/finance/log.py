"""Console logging for the finance package: one handler, the ``[YYYY-MM-DD HH:MM:SS] message`` format.

``get_logger(name)`` returns a standard ``logging`` logger under the ``finance`` namespace. The
first call attaches a single console handler to the ``finance`` logger (level INFO, no propagation
to the root logger, so an app that configures root logging does not print lines twice). The
handler writes to whatever ``sys.stdout`` is at emit time, so ``contextlib.redirect_stdout`` and
redirected scheduled-task output still capture it, flushes after every line, and falls back to
replacement characters when the console encoding cannot show a character (emoji on a cp1252
Windows console) instead of raising.

``status_log.log_status`` and ``run_archive.ProgressRecorder.log`` print through it, so progress
lines look exactly as before. A record logged with ``extra={"preformatted": True}`` is written
as-is (``ProgressRecorder`` stamps its own line once for both progress.log and the console).
"""

from __future__ import annotations

import logging
import sys

ROOT_LOGGER_NAME = "finance"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
LINE_FORMAT = "[%(asctime)s] %(message)s"


class ConsoleFormatter(logging.Formatter):
    """``[timestamp] message`` (plus any traceback); preformatted records pass through unchanged."""

    def __init__(self) -> None:
        super().__init__(LINE_FORMAT, datefmt=TIMESTAMP_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "preformatted", False):
            return record.getMessage()
        return super().format(record)


class ConsoleHandler(logging.Handler):
    """Writes each record to the current ``sys.stdout`` and flushes; never fails on encoding."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = sys.stdout
            if stream is None:  # pythonw / detached processes have no console
                return
            text = self.format(record) + "\n"
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
            stream.flush()
        except Exception:
            self.handleError(record)


def _configure_root() -> logging.Logger:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not any(isinstance(handler, ConsoleHandler) for handler in root.handlers):
        handler = ConsoleHandler()
        handler.setFormatter(ConsoleFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """A logger under the ``finance`` namespace that prints through the shared console handler."""
    _configure_root()
    if name != ROOT_LOGGER_NAME and not name.startswith(ROOT_LOGGER_NAME + "."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
