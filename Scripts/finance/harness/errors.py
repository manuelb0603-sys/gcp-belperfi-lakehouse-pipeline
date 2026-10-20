"""Harness stop conditions and model-call failure classification."""

from __future__ import annotations

from ..budget import BudgetExceededError, LedgerUnavailableError  # noqa: F401
from ..model_clients import EmptyModelResponse, is_anthropic_timeout, is_anthropic_transient  # noqa: F401


class ModelUnavailableError(RuntimeError):
    """A model call failed twice with a transport/server error and the harness stopped calling it."""


class AgentRoundsExhaustedError(RuntimeError):
    """The author used every allowed round of a phase without returning a report."""


class RunDeadlineExceededError(RuntimeError):
    """The run's max_run_seconds budget ran out before (or during) a model call."""


# Errors that end a run early with a fail decision (fallback or last sanitized report, never sent).
# BudgetExceededError (a Claude spend cap or an unavailable cost ledger) is never retried.
RUN_STOP_ERRORS = (ModelUnavailableError, AgentRoundsExhaustedError, RunDeadlineExceededError, BudgetExceededError)


def _is_timeout(exc: Exception) -> bool:
    """True when a model call failed because its HTTP timeout elapsed (requests or anthropic SDK)."""
    if isinstance(exc, TimeoutError) or is_anthropic_timeout(exc):
        return True
    try:
        import requests
    except ImportError:
        return False
    exceptions = getattr(requests, "exceptions", None)
    return exceptions is not None and isinstance(exc, exceptions.Timeout)


def _is_model_call_failure(exc: Exception) -> bool:
    """True for timeouts, connection failures, HTTP 5xx, and empty responses (worth one retry).

    Anthropic SDK errors are matched by class name (the SDK is never imported here): timeouts,
    connection errors, 429 rate limits and 5xx count; 4xx such as BadRequestError do not.
    """
    if isinstance(exc, (TimeoutError, ConnectionError, EmptyModelResponse)) or is_anthropic_transient(exc):
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
