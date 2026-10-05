"""Harness stop conditions and model-call failure classification."""

from __future__ import annotations


class EmptyModelResponse(ValueError):
    """Ollama answered without any text; treated like an unavailable model call."""


class ModelUnavailableError(RuntimeError):
    """A model call failed twice with a transport/server error and the harness stopped calling it."""


class AgentRoundsExhaustedError(RuntimeError):
    """The author used every allowed round of a phase without returning a report."""


class RunDeadlineExceededError(RuntimeError):
    """The run's max_run_seconds budget ran out before (or during) a model call."""


# Errors that end a run early with a fail decision (fallback or last sanitized report, never sent).
RUN_STOP_ERRORS = (ModelUnavailableError, AgentRoundsExhaustedError, RunDeadlineExceededError)


def _is_timeout(exc: Exception) -> bool:
    """True when a model call failed because its HTTP timeout elapsed."""
    if isinstance(exc, TimeoutError):
        return True
    try:
        import requests
    except ImportError:
        return False
    exceptions = getattr(requests, "exceptions", None)
    return exceptions is not None and isinstance(exc, exceptions.Timeout)


def _is_model_call_failure(exc: Exception) -> bool:
    """True for timeouts, connection failures, HTTP 5xx, and empty responses (worth one retry)."""
    if isinstance(exc, (TimeoutError, ConnectionError, EmptyModelResponse)):
        return True
    try:
        import requests
    except ImportError:
        return False
    exceptions = getattr(requests, "exceptions", None)
    if exceptions is None:
        return False
    if isinstance(exc, (exceptions.Timeout, exceptions.ConnectionError)):
        return True
    if isinstance(exc, exceptions.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status is None or status >= 500
    return False
