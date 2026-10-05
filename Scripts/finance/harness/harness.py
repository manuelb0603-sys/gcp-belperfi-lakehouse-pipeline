"""Bounded author/validator harness for the weekly financial report."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from ..report_models import ReportSnapshot, ValidationDecision, ValidationFinding
from .errors import (
    RUN_STOP_ERRORS,
    AgentRoundsExhaustedError,
    BudgetExceededError,
    ModelUnavailableError,
    RunDeadlineExceededError,
    _is_model_call_failure,
    _is_timeout,
)
from .fallback import build_fallback_report
from .feedback import (
    CycleProgress,
    _finding_key,
    _finding_label,
    _update_resolved_findings,
    focused_feedback,
    measure_progress,
)
from .html_safety import sanitize_html
from ..model_clients import ModelClient, OllamaClient
from ..settings import DEFAULTS
from .prompts import (
    MISSING_REPORT_HTML,
    PromptParts,
    as_prompt_parts,
    author_base_prompt,
    author_format_retry_text,
    author_length_limit_error,
    budget_exhausted_notice,
    history_label_requirement,
    new_evidence_section,
    revision_block,
    sanitization_revision_instruction,
    validator_format_retry_suffix,
    validator_length_limit_error,
    validator_prompt_parts,
)
from .protocol import VALIDATOR_RESPONSE_SCHEMA, parse_author_response, parse_json_response
from .trace import HarnessLimits, RunTrace, ToolSpec
from .transport import QueryFunctionClient, _response_meta


# Lower bound for a deadline-capped model call timeout, so a call near the deadline is not cut off instantly.
MIN_CALL_TIMEOUT_SECONDS = 30

RunResult = tuple[dict[str, Any], ValidationDecision, RunTrace]


@dataclass
class _AuthorState:
    # Author context is rebuilt for each call from fixed parts instead of being appended
    # to forever: each tool result appears once, and only the latest revision request is kept.
    base_prompt: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_sections: list[str] = field(default_factory=list)
    pending_tool_errors: list[dict[str, Any]] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    executed_tool_calls: int = 0
    budget_exhausted: bool = False
    revision_block: str = ""
    retry_suffix: str = ""
    # Error findings from earlier validation cycles that are absent from the latest cycle's errors,
    # keyed like the stall detection (category + message prefix) and mapped to a short label.
    resolved_findings: dict[tuple[str, str], str] = field(default_factory=dict)

    def resolved_labels(self) -> list[str]:
        """Short labels of findings already fixed in earlier cycles, oldest first."""
        return list(self.resolved_findings.values())

    def history_used(self) -> bool:
        """True when prior-report history was successfully retrieved as evidence."""
        return any(item.get("tool") == "get_report_history" and "result" in item for item in self.evidence)

    def prompt(self) -> str:
        """Compose the current author prompt from the base, evidence, notes, and revision request."""
        return self.prompt_parts().text

    def prompt_parts(self) -> PromptParts:
        """The author prompt split for caching: the base prompt (system) and everything after it (user)."""
        parts = list(self.evidence_sections)
        if self.history_used():
            parts.append(history_label_requirement())
        if self.budget_exhausted:
            parts.append(budget_exhausted_notice())
        if self.revision_block:
            parts.append(self.revision_block)
        if self.retry_suffix:
            parts.append(self.retry_suffix)
        return PromptParts(self.base_prompt, "".join(parts))


@dataclass
class _ValidationState:
    # Per-run validation bookkeeping carried from one review cycle to the next.
    report: dict[str, Any]
    # The current cycle's author HTML, its sanitized form (None when the HTML gate rejected it),
    # and the HTML shown to the author in the next revision request.
    html: str = ""
    safe_html: str | None = None
    revision_html: str = ""
    # Author rebuttals from the latest revision, passed to the next validator prompt.
    rebuttals: list[dict[str, str]] = field(default_factory=list)
    # The newest author report that passed the HTML gate (returned, never sent, if the run stops early).
    last_safe_report: dict[str, Any] | None = None
    # The latest completed review (validator or HTML gate) and what the next validator prompt sees of it.
    decision: ValidationDecision | None = None
    previous_validation: dict[str, Any] | None = None
    previous_error_counts: dict[str, int] | None = None
    previous_error_keys: set[tuple[str, str]] | None = None
    # Every error finding seen in earlier cycles (key -> short label), for the resolved-findings list.
    earlier_errors: dict[tuple[str, str], str] = field(default_factory=dict)


class ReportHarness:
    def __init__(
        self,
        ollama_url: str | None = None,
        author_model: str | None = None,
        validator_model: str | None = None,
        limits: HarnessLimits | None = None,
        timeout: int = DEFAULTS["ollama_timeout"],
        query: Callable[..., str] | None = None,
        model_options: dict[str, dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        author_client: ModelClient | None = None,
        validator_client: ModelClient | None = None,
    ) -> None:
        """Create a bounded agent harness with a separate author and validator model.

        author_client/validator_client are finance.model_clients ModelClients (Ollama or
        Anthropic; see build_model_client). Without them, the legacy arguments build Ollama
        clients: ollama_url, author_model, validator_model (default: the author model), and
        model_options, optional per-role Ollama settings, e.g. {"author": {"temperature": 0.3,
        "num_predict": 12000, "think": False}, "validator": {...}}. "think" is sent as Ollama's
        top-level think flag; every other key goes into the request's options. query, when given,
        replaces the Ollama transport with a ``query(prompt, url, model, timeout=, response_format=,
        **role_options)`` callable (tests inject fakes this way). timeout is the per-call maximum;
        with limits.max_run_seconds set, each call is also capped at the time left in the run.
        clock supplies the run-deadline time (injectable for tests).
        """
        self.ollama_url = ollama_url
        self.limits = limits or HarnessLimits()
        self.timeout = timeout
        self.query = query
        self.model_options = model_options or {}
        self.clock = clock
        self.author_client: ModelClient = author_client or self._legacy_client("author", author_model)
        if validator_client is None:
            if author_client is not None and validator_model is None:
                validator_client = author_client
            else:
                validator_client = self._legacy_client("validator", validator_model or self.author_client.model)
        self.validator_client: ModelClient = validator_client
        self.author_model = self.author_client.model
        self.validator_model = self.validator_client.model
        self.trace = RunTrace()
        self._validator_unavailable = False
        self._run_started_at: float | None = None
        self._deadline: float | None = None

    # ----- model calls, deadline, and retries -----

    def _deadline_seconds(self) -> float | None:
        """The configured run budget in seconds, or None when the deadline is disabled."""
        limit = self.limits.max_run_seconds
        return float(limit) if limit is not None and float(limit) > 0 else None

    def _deadline_exceeded(self, role: str, **context: Any) -> RunDeadlineExceededError:
        """Record the deadline stop and build the exception that ends the run with a fail decision."""
        elapsed = self.clock() - (self._run_started_at if self._run_started_at is not None else self.clock())
        limit = self._deadline_seconds()
        self.trace.add(
            "run_deadline_exceeded",
            role=role,
            elapsed_seconds=round(elapsed, 3),
            limit_seconds=limit,
            **context,
        )
        return RunDeadlineExceededError(
            f"Run time limit of {limit:g} seconds reached after {elapsed:.0f} seconds (before or during a {role} model call)"
        )

    def _call_timeout(self, role: str, **context: Any) -> tuple[float, bool]:
        """Per-call HTTP timeout, capped at the time left in the run; raises once the deadline has passed.

        Returns (timeout, capped) where capped is True when the deadline, not ollama_timeout, set the timeout.
        """
        if self._deadline is None:
            return self.timeout, False
        remaining = self._deadline - self.clock()
        if remaining <= 0:
            raise self._deadline_exceeded(role, **context)
        capped_timeout = max(remaining, MIN_CALL_TIMEOUT_SECONDS)
        if capped_timeout < self.timeout:
            return capped_timeout, True
        return self.timeout, False

    def _legacy_client(self, role: str, model: str | None) -> ModelClient:
        """Client for the legacy constructor arguments: the injected query callable, else Ollama."""
        if not model:
            raise ValueError(f"ReportHarness needs {role}_client or a model name for the {role}")
        kwargs = self._role_kwargs(role)
        if self.query is not None:
            return QueryFunctionClient(self.query, self.ollama_url, model, kwargs)
        return OllamaClient(
            self.ollama_url, model, options=kwargs.get("options"), think=kwargs.get("think"), default_timeout=self.timeout
        )

    def _client(self, role: str) -> ModelClient:
        return self.author_client if role == "author" else self.validator_client

    @staticmethod
    def _provider(client: Any) -> str:
        return str(getattr(client, "provider", "unknown"))

    def _role_kwargs(self, role: str) -> dict[str, Any]:
        """Per-role Ollama request extras; empty when nothing is configured, so defaults apply."""
        section = self.model_options.get(role) or {}
        options = {key: value for key, value in section.items() if key != "think" and value is not None}
        kwargs: dict[str, Any] = {}
        if options:
            kwargs["options"] = options
        if section.get("think") is not None:
            kwargs["think"] = bool(section["think"])
        return kwargs

    def _query_with_retry(self, role: str, prompt: PromptParts, json_schema: Any, **context: Any) -> str:
        """Call the role's model client, retrying once on a timeout/connection/5xx/empty-response failure.

        Raises RunDeadlineExceededError before any attempt once max_run_seconds has passed, and when
        a call times out because its timeout was capped at the time left in the run.
        """
        client = self._client(role)
        model = client.model
        for attempt in (1, 2):
            timeout, capped = self._call_timeout(role, model=model, attempt=attempt, **context)
            started = time.perf_counter()
            try:
                # Ollama and query fakes get system + user concatenated (the historical single prompt).
                return client.generate(prompt.user, system=prompt.system or None, json_schema=json_schema, timeout=timeout)
            except BudgetExceededError as exc:
                # The Claude budget guard refused the call before it was sent: stop, never retry.
                self.trace.add("budget_exceeded", role=role, model=model, attempt=attempt, error=str(exc), **{
                    key: value for key, value in exc.details.items() if key not in context
                }, **context)
                raise
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 3)
                self._record_exception(f"{role}_model_call", exc, elapsed_seconds=elapsed, call_attempt=attempt, **context)
                if capped and _is_timeout(exc):
                    # The run deadline, not ollama_timeout, cut this call short: stop instead of retrying.
                    raise self._deadline_exceeded(
                        role, model=model, attempt=attempt, call_timeout_seconds=round(timeout, 3), **context
                    ) from exc
                if not _is_model_call_failure(exc):
                    raise
                self.trace.add(
                    "model_call_failed",
                    role=role,
                    model=model,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    elapsed_seconds=elapsed,
                    # A billed Claude call that still failed (empty answer, timeout) carries its cost.
                    **({"cost_usd": exc.cost_usd} if isinstance(getattr(exc, "cost_usd", None), (int, float)) else {}),
                    **context,
                )
                if attempt == 2:
                    raise ModelUnavailableError(
                        f"{role} model {model} failed twice: {type(exc).__name__}: {exc}"
                    ) from exc
                self.trace.add("model_call_retry", role=role, model=model, attempt=attempt + 1, **context)
        raise AssertionError("unreachable")

    def _record_exception(self, phase: str, exc: Exception, **context: Any) -> None:
        """Record an exception before the harness propagates or handles it."""
        self.trace.add(
            "exception",
            phase=phase,
            exception_type=type(exc).__name__,
            error=str(exc),
            **context,
        )

    # ----- tools -----

    def _execute_tool(
        self,
        name: str,
        tool: ToolSpec,
        arguments: dict[str, Any],
        phase: str,
    ) -> dict[str, Any]:
        """Run one tool and return failures as agent-visible evidence for recovery."""
        try:
            result = tool.function(arguments)
        except Exception as exc:
            self._record_exception(f"{phase}_tool_call", exc, tool=name, arguments=arguments)
            self.trace.add(
                "tool_error",
                name=name,
                arguments=arguments,
                error_type=type(exc).__name__,
                error=str(exc),
                phase=phase,
            )
            return {
                "tool": name,
                "arguments": arguments,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

        self.trace.add("tool_call", name=name, arguments=arguments, result=result, phase=phase)
        return {"tool": name, "arguments": arguments, "result": result}

    def _trace_author_action(
        self,
        author_result: dict[str, Any],
        phase: str,
        round_number: int,
        pending_tool_errors: list[dict[str, Any]],
    ) -> None:
        """Make tool use, skips, and recovery after tool errors explicit in the trace."""
        calls = author_result.get("tool_calls") or []
        report_html = author_result.get("report_html")
        report_ready = isinstance(report_html, str) and bool(report_html.strip())
        action = "request_tools" if calls else "report_ready" if report_ready else "no_usable_action"
        tool_names = [str(call.get("name", "")) for call in calls if isinstance(call, dict)]
        self.trace.add(
            "agent_action",
            phase=phase,
            round=round_number,
            action=action,
            requested_tools=tool_names,
            response_keys=sorted(author_result.keys()),
        )

        if not pending_tool_errors:
            return
        failed_tool_names = [str(item.get("tool", "")) for item in pending_tool_errors]
        retried_tools = sorted(set(failed_tool_names).intersection(tool_names))
        if retried_tools:
            disposition = "retry_same_tool"
        elif calls:
            disposition = "continued_with_other_tool"
        elif report_ready:
            disposition = "continued_without_retry"
        else:
            disposition = "no_usable_recovery_action"
        self.trace.add(
            "tool_error_recovery",
            phase=phase,
            round=round_number,
            disposition=disposition,
            failed_tools=self._failed_tool_summaries(pending_tool_errors),
            requested_tools=tool_names,
        )

    @staticmethod
    def _failed_tool_summaries(pending_tool_errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "tool": item.get("tool"),
                "arguments": item.get("arguments"),
                "error": item.get("error"),
            }
            for item in pending_tool_errors
        ]

    def _trace_unresolved_tool_errors(self, pending_tool_errors: list[dict[str, Any]], reason: str) -> None:
        """Mark failed tools unresolved if no later author response could handle them."""
        if not pending_tool_errors:
            return
        self.trace.add(
            "tool_errors_unresolved",
            reason=reason,
            failed_tools=self._failed_tool_summaries(pending_tool_errors),
        )

    def _author_output_retry_text(
        self,
        author_result: dict[str, Any],
        phase: str,
        round_number: int,
    ) -> str:
        """Ask the author to resend malformed or truncated output (tool-call JSON or marker HTML) without losing context."""
        error = str(author_result.get("error", "invalid author response"))
        self.trace.add(
            "author_format_retry",
            phase=phase,
            round=round_number,
            error=error,
            response_chars=author_result.get("response_chars", 0),
            raw_excerpt=author_result.get("raw_excerpt", ""),
        )
        return author_format_retry_text({
            "parse_error": error,
            "response_chars": author_result.get("response_chars", 0),
            "excerpt_near_error": author_result.get("raw_excerpt", ""),
        })

    @staticmethod
    def _tool_prompt(tools: dict[str, ToolSpec]) -> str:
        # Expose only the allowlisted tool metadata to the author model. The list is the
        # contract: the model can request these names and argument objects, nothing else.
        return json.dumps({
            name: {"description": spec.description, "arguments": "JSON object"}
            for name, spec in tools.items()
        }, indent=2)

    # ----- author and validator turns -----

    def _author_call(self, prompt: str | PromptParts) -> dict[str, Any]:
        """Send the current prompt to the author model and parse marker HTML or tool-call JSON.

        Raises ModelUnavailableError when the model fails twice with a transport/server error.
        """
        parts = as_prompt_parts(prompt)
        full_prompt = parts.text
        self.trace.add(
            "model_call",
            role="author",
            model=self.author_model,
            provider=self._provider(self.author_client),
            prompt=full_prompt,
            prompt_chars=len(full_prompt),
        )
        started = time.perf_counter()
        # No format schema: the report comes back as raw HTML between markers (see AUTHOR_PROTOCOL).
        response = self._query_with_retry("author", parts, None)
        elapsed = round(time.perf_counter() - started, 3)
        meta = _response_meta(response)
        try:
            result = parse_author_response(response)
        except Exception as exc:
            error = author_length_limit_error(exc) if meta["done_reason"] == "length" else str(exc)
            excerpt = response if len(response) <= 600 else f"{response[:300]} … {response[-300:]}"
            self.trace.add(
                "model_response",
                role="author",
                parse_status="error",
                error=error,
                raw_response=response,
                response_chars=len(response),
                elapsed_seconds=elapsed,
                **meta,
            )
            return {
                "report_html": "",
                "plain_text": "",
                "error": error,
                "raw_excerpt": excerpt,
                "response_chars": len(response),
            }
        self.trace.add(
            "model_response",
            role="author",
            parse_status="ok",
            keys=list(result),
            raw_response=response,
            response_chars=len(response),
            elapsed_seconds=elapsed,
            **meta,
        )
        return result

    def _validator_call(self, prompt: str | PromptParts, cycle: int | None = None) -> ValidationDecision:
        """Run the validation model and convert its structured result into a typed verdict."""
        self._validator_unavailable = False
        reply = self._validator_reply(prompt, cycle)
        if isinstance(reply, ValidationDecision):
            return reply
        return self._decision_from_payload(cycle, *reply)

    def _validator_reply(
        self, prompt: str | PromptParts, cycle: int | None
    ) -> ValidationDecision | tuple[dict[str, Any], str, dict[str, Any], float]:
        """Query the validator with one format retry: (payload, response, meta, elapsed) or a fail decision."""
        prompt = as_prompt_parts(prompt)
        retry_prompt = prompt
        for attempt in (1, 2):
            self.trace.add(
                "model_call",
                role="validator",
                model=self.validator_model,
                provider=self._provider(self.validator_client),
                prompt=retry_prompt.text,
                prompt_chars=len(retry_prompt.text),
                validation_cycle=cycle,
                attempt=attempt,
            )
            started = time.perf_counter()
            try:
                response = self._query_with_retry(
                    "validator",
                    retry_prompt,
                    VALIDATOR_RESPONSE_SCHEMA,
                    validation_cycle=cycle,
                    format_attempt=attempt,
                )
            except ModelUnavailableError as exc:
                self._validator_unavailable = True
                return ValidationDecision(
                    "fail", [], f"Validator unavailable after a retry ({exc}); the report was not accepted."
                )
            elapsed = round(time.perf_counter() - started, 3)
            meta = _response_meta(response)
            try:
                return parse_json_response(response), response, meta, elapsed
            except Exception as exc:
                length_limited = meta["done_reason"] == "length"
                error = validator_length_limit_error(exc) if length_limited else str(exc)
                self.trace.add(
                    "model_response",
                    role="validator",
                    parse_status="error",
                    error=error,
                    raw_response=response,
                    response_chars=len(response),
                    validation_cycle=cycle,
                    attempt=attempt,
                    elapsed_seconds=elapsed,
                    **meta,
                )
                if attempt == 2:
                    return ValidationDecision("fail", [], "Validator returned malformed JSON twice; the report was not accepted.")
                # The retry request follows the per-cycle content, so the cached system part is unchanged.
                retry_prompt = PromptParts(prompt.system, prompt.user + validator_format_retry_suffix(error, length_limited))
                self.trace.add(
                    "validator_format_retry",
                    validation_cycle=cycle,
                    attempt=attempt,
                    error=error,
                    response_chars=len(response),
                )
        return ValidationDecision("fail", [], "Validator did not return a usable JSON object; the report was not accepted.")

    def _decision_from_payload(
        self, cycle: int | None, payload: dict[str, Any], response: str, meta: dict[str, Any], elapsed: float
    ) -> ValidationDecision:
        """Turn a parsed validator payload into a decision, failing closed on invalid content."""
        decision = str(payload.get("decision", "")).lower()
        if decision not in {"pass", "fail"}:
            self.trace.add(
                "model_response",
                role="validator",
                parse_status="invalid_decision",
                validation_cycle=cycle,
                payload=payload,
                raw_response=response,
                response_chars=len(response),
                elapsed_seconds=elapsed,
                **meta,
            )
            return ValidationDecision("fail", [], "Validator decision was missing or invalid; report was not accepted.")
        try:
            findings = [ValidationFinding(**finding) for finding in payload.get("findings", [])]
        except Exception as exc:
            self._record_exception("validator_payload", exc, validation_cycle=cycle, payload=payload)
            return ValidationDecision("fail", [], "Validator findings were malformed; the report was not accepted.")
        result = ValidationDecision(decision, findings, str(payload.get("summary", "")))
        self.trace.add(
            "model_response",
            role="validator",
            parse_status="ok",
            validation_cycle=cycle,
            payload=payload,
            raw_response=response,
            response_chars=len(response),
            elapsed_seconds=elapsed,
            **meta,
        )
        return result

    # ----- run entry point -----

    def run(
        self,
        objective: str,
        snapshot: ReportSnapshot,
        prompt_context: str,
        tools: dict[str, ToolSpec],
        validator_context: str = "",
        report_requirements: list[str] | None = None,
    ) -> RunResult:
        """Run the harness and guarantee a terminal trace event for success or failure.

        report_requirements is an optional snapshot-derived checklist added once to the author
        base prompt (so revisions inherit it) and to every validator prompt.
        """
        deadline_seconds = self._deadline_seconds()
        self._run_started_at = self.clock()
        self._deadline = self._run_started_at + deadline_seconds if deadline_seconds is not None else None
        self.trace.add(
            "run_started",
            author_model=self.author_model,
            validator_model=self.validator_model,
            author_provider=self._provider(self.author_client),
            validator_provider=self._provider(self.validator_client),
            limits={
                "max_agent_rounds": self.limits.max_agent_rounds,
                "max_tool_calls": self.limits.max_tool_calls,
                "max_calls_per_tool": self.limits.max_calls_per_tool,
                "max_validation_cycles": self.limits.max_validation_cycles,
                "max_report_bytes": self.limits.max_report_bytes,
                "max_run_seconds": self.limits.max_run_seconds,
            },
            deadline_seconds=deadline_seconds,
            tool_names=sorted(tools),
        )
        try:
            result = self._run_impl(objective, snapshot, prompt_context, tools, validator_context, report_requirements)
        except Exception as exc:
            last_event = self.trace.events[-1] if self.trace.events else {}
            self._record_exception(
                "harness_unhandled",
                exc,
                last_event=last_event.get("event"),
                last_phase=last_event.get("phase"),
                last_round=last_event.get("round"),
                last_validation_cycle=last_event.get("cycle", last_event.get("validation_cycle")),
                traceback=traceback.format_exc(),
            )
            self.trace.add(
                "run_aborted",
                exception_type=type(exc).__name__,
                error=str(exc),
                last_event=last_event.get("event"),
            )
            raise

        report, decision, trace = result
        self.trace.add(
            "run_completed",
            decision=decision.decision,
            passed=decision.passed,
            validation_cycles=sum(event.get("event") == "validation_cycle_result" for event in self.trace.events),
            tool_calls=sum(event.get("event") == "tool_call" for event in self.trace.events),
            report_html_bytes=len(str(report.get("report_html", "")).encode("utf-8")),
        )
        return report, decision, trace

    def _reject_tool_call(
        self,
        name: Any,
        arguments: Any,
        error_type: str,
        message: str,
        phase: str,
    ) -> dict[str, Any]:
        """Return a refused tool call as agent-visible evidence instead of aborting the run."""
        self.trace.add(
            "tool_error",
            name=name,
            arguments=arguments,
            error_type=error_type,
            error=message,
            phase=phase,
        )
        return {
            "tool": name,
            "arguments": arguments,
            "error": {"type": error_type, "message": message},
        }

    def _run_tool_calls(
        self,
        calls: list[Any],
        tools: dict[str, ToolSpec],
        state: _AuthorState,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Execute allowlisted calls within budget; refusals become error evidence for the author."""
        results: list[dict[str, Any]] = []
        for call in calls:
            call = call if isinstance(call, dict) else {}
            name = call.get("name")
            arguments = call.get("arguments") or {}
            if name not in tools:
                result = self._reject_tool_call(
                    name,
                    arguments,
                    "ToolNotAllowlisted",
                    f"Tool is not allowlisted: {name}. Available tools: {', '.join(sorted(tools)) or 'none'}. "
                    "Do not request it again; continue with existing evidence.",
                    phase,
                )
            elif state.executed_tool_calls >= self.limits.max_tool_calls:
                state.budget_exhausted = True
                result = self._reject_tool_call(
                    name,
                    arguments,
                    "ToolBudgetExceeded",
                    f"Maximum total tool calls ({self.limits.max_tool_calls}) reached; this call was not executed. "
                    "Proceed with the existing evidence and return the final report.",
                    phase,
                )
            elif state.tool_counts.get(name, 0) >= self.limits.max_calls_per_tool:
                result = self._reject_tool_call(
                    name,
                    arguments,
                    "ToolBudgetExceeded",
                    f"Maximum calls ({self.limits.max_calls_per_tool}) reached for tool {name}; this call was not executed. "
                    "Do not request this tool again; proceed with the existing evidence.",
                    phase,
                )
            else:
                state.tool_counts[name] = state.tool_counts.get(name, 0) + 1
                state.executed_tool_calls += 1
                if state.executed_tool_calls >= self.limits.max_tool_calls:
                    state.budget_exhausted = True
                result = self._execute_tool(name, tools[name], arguments, phase)
            results.append(result)
            if "error" in result:
                state.pending_tool_errors.append(result)
        state.evidence.extend(results)
        return results

    def _author_loop(self, phase: str, tools: dict[str, ToolSpec], state: _AuthorState) -> dict[str, Any]:
        """Let the author gather evidence for a bounded number of rounds until it returns a report.

        Raises AgentRoundsExhaustedError when max_agent_rounds is exhausted; the caller ends the
        run with a fail decision, like an unavailable author.
        """
        # The round loop lets the model gather structured evidence, but it must do so under
        # strict round and tool budgets so it cannot spiral into unbounded exploration.
        for round_number in range(1, self.limits.max_agent_rounds + 1):
            if phase == "initial":
                self.trace.add("agent_round", round=round_number)
            author_result = self._author_call(state.prompt_parts())
            self._trace_author_action(author_result, phase, round_number, state.pending_tool_errors)
            state.pending_tool_errors.clear()
            if author_result.get("error"):
                state.retry_suffix = self._author_output_retry_text(author_result, phase, round_number)
                continue
            state.retry_suffix = ""
            calls = author_result.get("tool_calls") or []
            if not calls:
                return author_result
            if phase == "revision":
                self.trace.add("agent_revision_round", round=round_number)
            new_results = self._run_tool_calls(calls, tools, state, phase)
            state.evidence_sections.append(new_evidence_section(phase, round_number, new_results))
        reason = "Maximum agent rounds exceeded" if phase == "initial" else "Maximum revision agent rounds exceeded"
        self._trace_unresolved_tool_errors(state.pending_tool_errors, reason)
        state.pending_tool_errors.clear()
        self.trace.add("agent_rounds_exhausted", phase=phase, max_agent_rounds=self.limits.max_agent_rounds, error=reason)
        raise AgentRoundsExhaustedError(reason)

    def _take_rebuttals(self, report: Any, responding_to_cycle: int | None) -> list[dict[str, str]]:
        """Detach parsed author rebuttals from the report dict and trace them.

        Rebuttals are passed to the validator verbatim; the harness never checks them or the
        findings they dispute (that judgement belongs to the validator prompt). Malformed blocks
        and rebuttals without prior findings to dispute are ignored with a trace event.
        """
        if not isinstance(report, dict):
            return []
        rebuttals = report.pop("rebuttals", None) or []
        error = report.pop("rebuttal_error", None)
        if responding_to_cycle is None:
            if rebuttals or error:
                self.trace.add(
                    "author_rebuttals_ignored",
                    reason="Rebuttals are only used during a revision.",
                    rebuttal_count=len(rebuttals),
                    error=error,
                )
            return []
        if error:
            self.trace.add(
                "author_rebuttals_ignored",
                responding_to_cycle=responding_to_cycle,
                reason="Malformed or partly invalid rebuttal block.",
                error=error,
                rebuttal_count=len(rebuttals),
            )
        if rebuttals:
            self.trace.add(
                "author_rebuttals",
                responding_to_cycle=responding_to_cycle,
                rebuttal_count=len(rebuttals),
                rebuttals=rebuttals,
            )
        return rebuttals

    def _sanitization_decision(self, exc: Exception, html: str) -> ValidationDecision:
        """Convert a mechanical HTML-gate failure into validator-style feedback for revision."""
        category = getattr(exc, "category", "html")
        size = len(html.encode("utf-8"))
        finding = ValidationFinding(
            severity="error",
            category=category,
            message=str(exc),
            evidence=f"Mechanical HTML gate rejected the report ({size} bytes) before validator review.",
            revision_instruction=sanitization_revision_instruction(category, size, self.limits.max_report_bytes),
        )
        return ValidationDecision("fail", [finding], f"Report rejected by the HTML safety gate: {exc}")

    # ----- run phases -----

    def _run_impl(
        self,
        objective: str,
        snapshot: ReportSnapshot,
        prompt_context: str,
        tools: dict[str, ToolSpec],
        validator_context: str = "",
        report_requirements: list[str] | None = None,
    ) -> RunResult:
        """Run the author loop, collect evidence, and validate the final email HTML.

        Validation is intentionally fail-closed: the model can revise the report only if the
        validator or the HTML gate identifies a concrete problem, and only sanitized HTML (or
        the deterministic fallback) is ever returned.
        """
        state = _AuthorState(base_prompt=author_base_prompt(
            prompt_context, objective, json.dumps(snapshot.to_dict(), indent=2), report_requirements, self._tool_prompt(tools)
        ))
        try:
            report = self._initial_draft(tools, state)
        except RUN_STOP_ERRORS as exc:
            return self._stop_with_fail(exc, "initial", snapshot)
        vstate = _ValidationState(report=report)
        # max_validation_cycles is the total number of reviews, so there are at most max_cycles - 1 revisions.
        max_cycles = max(1, int(self.limits.max_validation_cycles))
        for cycle in range(1, max_cycles + 1):
            result = self._validate_cycle(cycle, snapshot, state, vstate, validator_context, report_requirements)
            if result is not None:
                return result
            if cycle >= max_cycles:
                break
            result = self._request_revision(cycle, snapshot, tools, state, vstate)
            if result is not None:
                return result
        return self._cycles_exhausted(snapshot, vstate)

    def _initial_draft(self, tools: dict[str, ToolSpec], state: _AuthorState) -> dict[str, Any]:
        """Run the initial author phase; raises a RUN_STOP_ERRORS error when the run must stop."""
        report = self._author_loop("initial", tools, state)
        # The initial report has no prior findings to dispute, so any rebuttal block is just dropped.
        self._take_rebuttals(report, responding_to_cycle=None)
        return report

    def _stop_with_fail(
        self,
        exc: Exception,
        phase: str,
        snapshot: ReportSnapshot,
        vstate: _ValidationState | None = None,
        cycle: int | None = None,
    ) -> RunResult:
        """End the run early with a fail decision after an unavailable model, exhausted rounds, the deadline, or a Claude spend cap.

        The returned report is the newest sanitized author report or the deterministic fallback; it is never sent.
        """
        unavailable = isinstance(exc, ModelUnavailableError)
        if unavailable:
            role = "validator" if phase == "validation" else "author"
            context = {"cycle": cycle} if cycle is not None else {}
            self.trace.add("model_unavailable", role=role, phase=phase, **context, error=str(exc))
        if phase == "validation":
            return self._stop_during_validation(exc, snapshot, vstate, cycle)
        reason, summary = self._stop_text(exc, phase, vstate)
        if phase == "initial":
            # Without an author report there is nothing to validate; return the deterministic fallback.
            return build_fallback_report(snapshot, reason), ValidationDecision("fail", [], summary), self.trace
        # Revision: keep the last sanitized report and the latest review's findings, and stop revising.
        report = vstate.last_safe_report or build_fallback_report(snapshot, reason)
        return report, ValidationDecision("fail", list(vstate.decision.findings), summary), self.trace

    def _stop_text(self, exc: Exception, phase: str, vstate: _ValidationState | None) -> tuple[str, str]:
        """(fallback reason, decision summary) for an author-phase stop."""
        rounds = self.limits.max_agent_rounds
        if phase == "initial":
            if isinstance(exc, ModelUnavailableError):
                return (
                    "The author model was unavailable after a retry.",
                    f"Author model unavailable after a retry ({exc}); the fallback report was not validated.",
                )
            if isinstance(exc, RunDeadlineExceededError):
                return "The run time limit was reached before the author returned a report.", f"{exc}; the fallback report was not validated."
            if isinstance(exc, BudgetExceededError):
                return "The Claude spend cap was reached before the author returned a report.", f"{exc}; the fallback report was not validated."
            reason = f"The author used all {rounds} agent rounds without returning a report."
            return reason, f"{reason} ({exc}); the fallback report was not validated."
        last = vstate.decision.summary
        if isinstance(exc, ModelUnavailableError):
            return (
                "The author model was unavailable during revision.",
                f"Author model unavailable during revision after a retry ({exc}); last review: {last}",
            )
        if isinstance(exc, RunDeadlineExceededError):
            return "The run time limit was reached during revision.", f"{exc} during revision; last review: {last}"
        if isinstance(exc, BudgetExceededError):
            return "The Claude spend cap was reached during revision.", f"{exc} during revision; last review: {last}"
        return (
            "The author used all revision agent rounds without returning a report.",
            f"Author used all {rounds} revision agent rounds without returning a report ({exc}); last review: {last}",
        )

    def _stop_during_validation(
        self, exc: Exception, snapshot: ReportSnapshot, vstate: _ValidationState, cycle: int
    ) -> RunResult:
        """Stop when the validator is unavailable, or the deadline or a Claude spend cap hits during a review."""
        unavailable = isinstance(exc, ModelUnavailableError)
        over_budget = isinstance(exc, BudgetExceededError)
        self.trace.add(
            "validation_cycle_result",
            cycle=cycle,
            decision="fail",
            stage="validator_unavailable" if unavailable else "budget_exceeded" if over_budget else "run_deadline",
            summary=str(exc),
            findings=[],
            report_html_bytes=len(vstate.safe_html.encode("utf-8")),
        )
        report = vstate.report
        if vstate.html.strip():
            report["report_html"] = vstate.safe_html
        elif unavailable:
            report = build_fallback_report(snapshot, "The validator was unavailable and the author returned no report.")
        else:
            stop_reason = "The Claude spend cap was reached" if over_budget else "The run time limit was reached"
            report = vstate.last_safe_report or build_fallback_report(
                snapshot, f"{stop_reason} and the author returned no report."
            )
        if unavailable:
            # No review is possible; stop instead of revising blind.
            return report, ValidationDecision("fail", [], str(exc)), self.trace
        # Out of time before this review finished: carry the last completed review's findings.
        last_review = vstate.decision
        findings = list(last_review.findings) if last_review is not None else []
        last = f"; last review: {last_review.summary}" if last_review is not None else "; the report was not reviewed"
        return report, ValidationDecision("fail", findings, f"{exc} during validation{last}"), self.trace

    def _validate_cycle(
        self,
        cycle: int,
        snapshot: ReportSnapshot,
        state: _AuthorState,
        vstate: _ValidationState,
        validator_context: str,
        report_requirements: list[str] | None,
    ) -> RunResult | None:
        """Gate and review the current report once; returns the run result when the run ends here."""
        self._prepare_report(cycle, state, vstate)
        decision = self._gate_report(cycle, state.history_used(), vstate)
        if decision is None:
            prompt = self._build_validator_prompt(snapshot, state, vstate, validator_context, report_requirements)
            try:
                decision = self._validator_call(prompt, cycle=cycle)
            except (RunDeadlineExceededError, BudgetExceededError) as exc:
                return self._stop_with_fail(exc, "validation", snapshot, vstate, cycle)
            except Exception as exc:
                self.trace.add(
                    "validation_cycle_result",
                    cycle=cycle,
                    decision="error",
                    stage="validator_call",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            if self._validator_unavailable:
                return self._stop_with_fail(ModelUnavailableError(decision.summary), "validation", snapshot, vstate, cycle)
        progress = self._record_progress(cycle, decision, state, vstate)
        self._record_review(cycle, decision, progress, state, vstate)
        if decision.passed:
            return vstate.report, decision, self.trace
        return None

    def _prepare_report(self, cycle: int, state: _AuthorState, vstate: _ValidationState) -> None:
        """Trace the cycle start and normalize a missing author report to empty HTML."""
        report = vstate.report
        html = report.get("report_html") if isinstance(report, dict) else None
        self.trace.add(
            "validation_cycle_start",
            cycle=cycle,
            report_html_present=isinstance(html, str) and bool(html.strip()),
            report_html_bytes=len(html.encode("utf-8")) if isinstance(html, str) else 0,
            evidence_count=len(state.evidence),
            history_used=state.history_used(),
        )
        if not isinstance(html, str) or not html.strip():
            self.trace.add(
                "report_missing",
                validation_cycle=cycle,
                author_error=report.get("error") if isinstance(report, dict) else None,
                response_keys=sorted(report.keys()) if isinstance(report, dict) else [],
            )
            html = ""
            report = report if isinstance(report, dict) else {}
            report.setdefault("plain_text", "")
            report["report_html"] = html
            report["status"] = "missing_report_html"
        vstate.report = report
        vstate.html = html

    def _gate_report(self, cycle: int, history_used: bool, vstate: _ValidationState) -> ValidationDecision | None:
        """Run the mechanical HTML gate; returns a fail decision when it rejects the report."""
        vstate.safe_html = None
        if not vstate.html.strip():
            vstate.safe_html = MISSING_REPORT_HTML
            return None
        try:
            vstate.safe_html = sanitize_html(vstate.html, self.limits.max_report_bytes, require_history_label=history_used)
        except ValueError as exc:
            # The exception event keeps the zero-based cycle index used before the harness split.
            self._record_exception("report_sanitization", exc, cycle=cycle - 1)
            decision = self._sanitization_decision(exc, vstate.html)
            self.trace.add(
                "report_sanitization_failed",
                cycle=cycle,
                category=getattr(exc, "category", "html"),
                error=str(exc),
                report_html_bytes=len(vstate.html.encode("utf-8")),
            )
            return decision
        return None

    def _build_validator_prompt(
        self,
        snapshot: ReportSnapshot,
        state: _AuthorState,
        vstate: _ValidationState,
        validator_context: str,
        report_requirements: list[str] | None,
    ) -> PromptParts:
        return validator_prompt_parts(
            validator_context,
            vstate.safe_html,
            json.dumps(snapshot.to_dict(), indent=2),
            report_requirements,
            state.evidence,
            vstate.previous_validation,
            state.resolved_labels(),
            vstate.rebuttals,
        )

    def _record_progress(
        self, cycle: int, decision: ValidationDecision, state: _AuthorState, vstate: _ValidationState
    ) -> CycleProgress:
        """Compare this review with the previous one, update resolved findings, and trace progress/stalls."""
        progress = measure_progress(decision, vstate.previous_error_counts, vstate.previous_error_keys)
        if decision.passed or progress.error_findings:
            # A failed review without itemized errors (for example malformed validator JSON) says nothing
            # about which findings were fixed, so it leaves the resolved list unchanged.
            _update_resolved_findings(state.resolved_findings, vstate.earlier_errors, set(progress.error_keys))
            for finding in progress.error_findings:
                vstate.earlier_errors.setdefault(_finding_key(finding), _finding_label(finding))
        self.trace.add(
            "validation_progress", cycle=cycle, **progress.progress, resolved_findings=state.resolved_labels()
        )
        if progress.stalled(vstate.previous_error_keys):
            # Informational only: the same findings came back unchanged, so the revision made no
            # visible progress (the author did not fix them, or the validator repeated false findings).
            self.trace.add(
                "validation_stalled",
                cycle=cycle,
                repeated_error_count=progress.repeated_error_count,
                rebuttal_count=len(vstate.rebuttals),
                findings=[{"category": category, "message": message} for category, message in sorted(set(progress.error_keys))],
            )
        return progress

    def _record_review(
        self,
        cycle: int,
        decision: ValidationDecision,
        progress: CycleProgress,
        state: _AuthorState,
        vstate: _ValidationState,
    ) -> None:
        """Trace the review result, keep only sanitized HTML in the report, and remember the review."""
        html, safe_html, report = vstate.html, vstate.safe_html, vstate.report
        findings = decision.to_dict()["findings"]
        self.trace.add("validation", cycle=cycle, decision=decision.decision, summary=decision.summary, findings=findings)
        self.trace.add(
            "validation_cycle_result",
            cycle=cycle,
            decision=decision.decision,
            stage="validator" if safe_html is not None else "sanitization",
            summary=decision.summary,
            findings=findings,
            report_html_bytes=len((safe_html if safe_html is not None else html).encode("utf-8")),
            error_count=progress.error_count,
            warning_count=progress.warning_count,
            error_count_delta=progress.progress["error_count_delta"],
            resolved_error_count_by_category=progress.progress["resolved_error_count_by_category"],
            new_error_count_by_category=progress.progress["new_error_count_by_category"],
            repeated_error_count=progress.repeated_error_count,
            resolved_finding_count=len(state.resolved_findings),
        )
        if safe_html is not None:
            vstate.revision_html = safe_html
            report["report_html"] = safe_html
            if html.strip():
                vstate.last_safe_report = dict(report)
        else:
            # Unsanitized HTML may go back to the author for repair, but never out of the harness.
            size = len(html.encode("utf-8"))
            vstate.revision_html = (
                html if size <= self.limits.max_report_bytes
                else f"[omitted: {size} bytes exceeds the {self.limits.max_report_bytes}-byte limit]"
            )
            report["report_html"] = ""
            report["status"] = "sanitization_failed"
        vstate.previous_validation = {
            "cycle": cycle,
            "decision": decision.decision,
            "summary": decision.summary,
            "progress": progress.progress,
            "findings": [finding.__dict__ for finding in decision.findings],
        }
        vstate.previous_error_counts = progress.error_counts
        vstate.previous_error_keys = set(progress.error_keys)
        vstate.decision = decision

    def _request_revision(
        self,
        cycle: int,
        snapshot: ReportSnapshot,
        tools: dict[str, ToolSpec],
        state: _AuthorState,
        vstate: _ValidationState,
    ) -> RunResult | None:
        """Ask the author to fix the top findings; returns the run result when the run must stop."""
        # Replace (not append) the revision request so only the latest attempt is in context.
        state.revision_block = revision_block(
            vstate.revision_html, focused_feedback(vstate.decision, snapshot), state.resolved_labels()
        )
        try:
            vstate.report = self._author_loop("revision", tools, state)
            vstate.rebuttals = self._take_rebuttals(vstate.report, responding_to_cycle=cycle)
        except RUN_STOP_ERRORS as exc:
            return self._stop_with_fail(exc, "revision", snapshot, vstate, cycle)
        return None

    def _cycles_exhausted(self, snapshot: ReportSnapshot, vstate: _ValidationState) -> RunResult:
        """Every allowed review failed: return the last sanitized report (or the fallback) with a fail decision."""
        report = vstate.report
        if not isinstance(report.get("report_html"), str) or not str(report.get("report_html", "")).strip():
            report = build_fallback_report(snapshot, "Validation cycles were exhausted and the author never returned a usable final report payload.")
        decision = vstate.decision
        last_summary = decision.summary if decision is not None else ""
        return report, ValidationDecision(
            "fail",
            list(decision.findings) if decision is not None else [],
            f"Validation cycles were exhausted without a passing report. Last review: {last_summary}".strip(),
        ), self.trace
