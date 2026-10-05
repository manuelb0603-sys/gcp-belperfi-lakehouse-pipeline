"""Bounded author/validator harness for the weekly financial report."""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable

from .report_models import ReportSnapshot, ValidationDecision, ValidationFinding


ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ToolSpec:
    # A tool is a small, explicit, allowlisted action the model may request. The model
    # does not execute Python directly; it emits a structured tool call, and the harness
    # runs the bound function here.
    name: str
    description: str
    function: ToolFunction


@dataclass
class HarnessLimits:
    # These limits keep the agent bounded: it can investigate enough to produce a good
    # report, but it cannot loop forever or consume unbounded tokens or data.
    max_agent_rounds: int = 6
    max_tool_calls: int = 12
    max_calls_per_tool: int = 3
    max_validation_cycles: int = 2
    max_report_bytes: int = 250_000


@dataclass
class RunTrace:
    # Keep a full execution timeline so the run can be inspected later for debugging,
    # auditing, and artifact comparison across runs.
    events: list[dict[str, Any]] = field(default_factory=list)
    # Optional observers (for example a live progress recorder) notified of each event.
    # Listener failures are reported once and never interrupt the run.
    listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list, repr=False, compare=False)
    _listener_failed: bool = field(default=False, repr=False, compare=False)

    def add(self, event: str, **values: Any) -> None:
        """Record a single event in the agent timeline and notify any listeners."""
        item = {"event": event, "timestamp": time.time(), **values}
        self.events.append(item)
        for listener in self.listeners:
            try:
                listener(item)
            except Exception as exc:
                if not self._listener_failed:
                    self._listener_failed = True
                    print(f"Run trace listener failed; continuing without it: {exc}", flush=True)


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

    def history_used(self) -> bool:
        """True when prior-report history was successfully retrieved as evidence."""
        return any(item.get("tool") == "get_report_history" and "result" in item for item in self.evidence)

    def prompt(self) -> str:
        """Compose the current author prompt from the base, evidence, notes, and revision request."""
        parts = [self.base_prompt, *self.evidence_sections]
        if self.history_used():
            parts.append(
                "\n\nHISTORY LABEL REQUIREMENT: get_report_history evidence was used, so the report HTML "
                f"must contain the exact text \"{HISTORY_LABEL}\" introducing any prior-report content "
                "(for example, a short clearly labeled historical note). Reports without this exact label are rejected. "
                "Never present historical figures as current."
            )
        if self.budget_exhausted:
            parts.append(
                "\n\nTOOL BUDGET EXHAUSTED: do not request more tools. Return the final report JSON now "
                "using the snapshot and the evidence above."
            )
        if self.revision_block:
            parts.append(self.revision_block)
        if self.retry_suffix:
            parts.append(self.retry_suffix)
        return "".join(parts)


class _UnsafeMarkupParser(HTMLParser):
    # This parser is a lightweight blocklist: if the model tries to inject HTML that can
    # execute or affect the email client, we reject the run instead of sending it.
    unsafe_tags = {"script", "iframe", "object", "embed", "form", "input", "link", "meta"}

    def __init__(self) -> None:
        """Initialize the parser state for unsafe-tag detection."""
        super().__init__()
        self.unsafe = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Mark the HTML as unsafe when a blocked tag appears."""
        if tag in self.unsafe_tags:
            self.unsafe = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing blocked tags the same as normal opening tags."""
        self.handle_starttag(tag, attrs)


class _ReportTextExtractor(HTMLParser):
    """Extract a readable plain-text alternative from a valid HTML report."""

    block_tags = {"address", "article", "br", "div", "h1", "h2", "h3", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "style":
            self.in_style = True
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style":
            self.in_style = False
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.in_style:
            self.parts.append(data)


def _plain_text_from_html(report_html: str) -> str:
    parser = _ReportTextExtractor()
    parser.feed(report_html)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


HISTORY_LABEL = "Previous report for reference:"


class ReportSanitizationError(ValueError):
    """A report failed the mechanical HTML gate; category maps to a validator finding category."""

    def __init__(self, message: str, category: str = "html") -> None:
        super().__init__(message)
        self.category = category


def sanitize_html(html: str, max_bytes: int = 250_000, require_history_label: bool = False) -> str:
    """Strip unsafe markup while preserving the report layout the model intended."""
    # This is the final mechanical gate before validation: if the model produces HTML that
    # could execute or is too large, the report is rejected before any email is sent.
    if not html or len(html.encode("utf-8")) > max_bytes:
        raise ReportSanitizationError("Report HTML is empty or exceeds the configured size limit")
    cleaned = re.sub(r"<\s*(script|iframe|object|embed|form|input|link|meta)\b[^>]*>.*?<\s*/\s*\1\s*>", "", html, flags=re.I | re.S)
    cleaned = re.sub(r"<\s*(script|iframe|object|embed|form|input|link|meta)\b[^>]*?/?>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+on[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:href|src)\s*=\s*(['\"])\s*javascript:[^'\"]*\1", "", cleaned, flags=re.I)
    parser = _UnsafeMarkupParser()
    parser.feed(cleaned)
    if parser.unsafe:
        raise ReportSanitizationError("Report contains unsafe HTML elements")
    if require_history_label and HISTORY_LABEL not in cleaned:
        raise ReportSanitizationError("Report must explicitly label prior report excerpts", category="history")
    return cleaned


def parse_json_response(text: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    """Parse a direct JSON object returned by the model."""
    candidate = text.strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    if required_keys:
        missing = sorted(required_keys - set(value.keys()))
        if missing:
            raise ValueError(f"Model response is missing required keys: {', '.join(missing)}")
    return value


def build_fallback_report(snapshot: ReportSnapshot, reason: str = "Model response missing required report_html output.") -> dict[str, Any]:
    """Create a deterministic fallback report if the model never emits a valid final document."""
    latest = float(snapshot.latest_total_net_expense)
    previous = float(snapshot.previous_total_net_expense or latest)
    delta = ((latest - previous) / previous * 100) if previous else 0.0
    trend = "up" if delta >= 0 else "down"
    plain_text = (
        f"Weekly Financial Progress Report\n"
        f"As of {snapshot.as_of}. Latest month: {snapshot.latest_month}. "
        f"Net expenses were ${latest:,.2f} versus ${previous:,.2f} last month ({delta:+.1f}% {trend}). "
        f"Disposable cash is ${snapshot.disposable_cash:,.2f}."
    )
    html = (
        "<html><body><h1>Weekly Financial Progress</h1>"
        f"<p><strong>Fallback report:</strong> {reason}</p>"
        f"<p>Latest month: {snapshot.latest_month}. Net expenses: ${latest:,.2f}</p>"
        f"<p>Previous month: {snapshot.previous_month or 'n/a'}. Net expenses: ${previous:,.2f}</p>"
        f"<p>Month-over-month change: {delta:+.1f}% ({trend})</p>"
        f"<p>Disposable cash: ${snapshot.disposable_cash:,.2f}</p>"
        "</body></html>"
    )
    return {"report_html": html, "plain_text": plain_text}


def query_ollama(
    prompt: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
    response_format: str | dict[str, Any] = "json",
) -> str:
    """Call Ollama and return the text response for one model turn."""
    import requests

    response = requests.post(
        ollama_url,
        json={"model": model_name, "prompt": prompt, "stream": False, "format": response_format},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("response")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Ollama response did not contain text")
    return result


AUTHOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        },
        "report_html": {"type": "string"},
        "plain_text": {"type": "string"},
    },
    "additionalProperties": False,
}

VALIDATOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["pass", "fail"]},
        "summary": {"type": "string"},
        "audit": {
            "type": "object",
            "properties": {
                key: {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["pass", "fail"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["status", "evidence"],
                    "additionalProperties": False,
                }
                for key in (
                    "financial_math",
                    "goal_math",
                    "required_content",
                    "visual_design",
                    "readability_accessibility",
                    "html_safety",
                )
            },
            "required": [
                "financial_math",
                "goal_math",
                "required_content",
                "visual_design",
                "readability_accessibility",
                "html_safety",
            ],
            "additionalProperties": False,
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["error", "warning", "info"]},
                    "category": {
                        "type": "string",
                        "enum": ["arithmetic", "evidence", "history", "policy", "html", "completeness", "usability", "visual_design"],
                    },
                    "message": {"type": "string"},
                    "evidence": {"type": "string"},
                    "revision_instruction": {"type": ["string", "null"]},
                },
                "required": ["severity", "category", "message", "evidence", "revision_instruction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decision", "summary", "audit", "findings"],
    "additionalProperties": False,
}


class ReportHarness:
    def __init__(
        self,
        ollama_url: str,
        author_model: str,
        validator_model: str | None = None,
        limits: HarnessLimits | None = None,
        timeout: int = 600,
        query: Callable[..., str] = query_ollama,
    ) -> None:
        """Create a bounded agent harness with a separate author and validator model."""
        self.ollama_url = ollama_url
        self.author_model = author_model
        self.validator_model = validator_model or author_model
        self.limits = limits or HarnessLimits()
        self.timeout = timeout
        self.query = query
        self.trace = RunTrace()

    def _record_exception(self, phase: str, exc: Exception, **context: Any) -> None:
        """Record an exception before the harness propagates or handles it."""
        self.trace.add(
            "exception",
            phase=phase,
            exception_type=type(exc).__name__,
            error=str(exc),
            **context,
        )

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
            failed_tools=[
                {
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments"),
                    "error": item.get("error"),
                }
                for item in pending_tool_errors
            ],
            requested_tools=tool_names,
        )

    def _trace_unresolved_tool_errors(self, pending_tool_errors: list[dict[str, Any]], reason: str) -> None:
        """Mark failed tools unresolved if no later author response could handle them."""
        if not pending_tool_errors:
            return
        self.trace.add(
            "tool_errors_unresolved",
            reason=reason,
            failed_tools=[
                {
                    "tool": item.get("tool"),
                    "arguments": item.get("arguments"),
                    "error": item.get("error"),
                }
                for item in pending_tool_errors
            ],
        )

    def _author_output_retry_text(
        self,
        author_result: dict[str, Any],
        phase: str,
        round_number: int,
    ) -> str:
        """Give the author a bounded chance to repair serialization without losing context."""
        error = str(author_result.get("error", "invalid author response"))
        self.trace.add(
            "author_format_retry",
            phase=phase,
            round=round_number,
            error=error,
            response_chars=author_result.get("response_chars", 0),
            raw_excerpt=author_result.get("raw_excerpt", ""),
        )
        details = {
            "parse_error": error,
            "response_chars": author_result.get("response_chars", 0),
            "excerpt_near_error": author_result.get("raw_excerpt", ""),
        }
        return (
            "\n\nAUTHOR OUTPUT FORMAT ERROR — RETRY REQUIRED:\n"
            f"{json.dumps(details, ensure_ascii=False)}\n"
            "Recreate the response from the original objective, snapshot, tool evidence, and any validator feedback above. "
            "Return exactly one valid JSON object. For a final report include both non-empty string fields "
            "report_html and plain_text; escape every quote and newline inside JSON strings. Do not return an explanation."
        )

    @staticmethod
    def _tool_prompt(tools: dict[str, ToolSpec]) -> str:
        # Expose only the allowlisted tool metadata to the author model. The list is the
        # contract: the model can request these names and argument objects, nothing else.
        return json.dumps({
            name: {"description": spec.description, "arguments": "JSON object"}
            for name, spec in tools.items()
        }, indent=2)

    def _author_call(self, prompt: str) -> dict[str, Any]:
        """Send the current prompt to the author model and parse its JSON response."""
        self.trace.add(
            "model_call",
            role="author",
            model=self.author_model,
            prompt=prompt,
            prompt_chars=len(prompt),
        )
        started = time.perf_counter()
        try:
            response = self.query(
                prompt,
                self.ollama_url,
                self.author_model,
                timeout=self.timeout,
                response_format=AUTHOR_RESPONSE_SCHEMA,
            )
        except Exception as exc:
            self._record_exception("author_model_call", exc, elapsed_seconds=round(time.perf_counter() - started, 3))
            raise
        elapsed = round(time.perf_counter() - started, 3)
        try:
            result = parse_json_response(response)
            if "tool_calls" in result:
                if not isinstance(result["tool_calls"], list):
                    raise ValueError("Author tool_calls must be a JSON array")
            else:
                if not isinstance(result.get("report_html"), str) or not result["report_html"].strip():
                    raise ValueError("Author response is missing non-empty report_html")
                if not isinstance(result.get("plain_text"), str) or not result["plain_text"].strip():
                    result["plain_text"] = _plain_text_from_html(result["report_html"])
                if not result["plain_text"]:
                    raise ValueError("Could not derive plain_text from author report_html")
        except Exception as exc:
            position = getattr(exc, "pos", None)
            excerpt = ""
            if isinstance(position, int):
                excerpt = response[max(0, position - 300): min(len(response), position + 300)]
            self.trace.add(
                "model_response",
                role="author",
                parse_status="error",
                error=str(exc),
                raw_response=response,
                response_chars=len(response),
                elapsed_seconds=elapsed,
            )
            return {
                "report_html": "",
                "plain_text": "",
                "error": str(exc),
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
        )
        return result

    def _validator_call(self, prompt: str, cycle: int | None = None) -> ValidationDecision:
        """Run the validation model and convert its structured result into a typed verdict."""
        retry_prompt = prompt
        payload: dict[str, Any] | None = None
        for attempt in range(2):
            self.trace.add(
                "model_call",
                role="validator",
                model=self.validator_model,
                prompt=retry_prompt,
                prompt_chars=len(retry_prompt),
                validation_cycle=cycle,
                attempt=attempt + 1,
            )
            started = time.perf_counter()
            try:
                response = self.query(
                    retry_prompt,
                    self.ollama_url,
                    self.validator_model,
                    timeout=self.timeout,
                    response_format=VALIDATOR_RESPONSE_SCHEMA,
                )
            except Exception as exc:
                self._record_exception(
                    "validator_model_call",
                    exc,
                    validation_cycle=cycle,
                    attempt=attempt + 1,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                )
                raise
            elapsed = round(time.perf_counter() - started, 3)
            try:
                payload = parse_json_response(response)
                break
            except Exception as exc:
                self.trace.add(
                    "model_response",
                    role="validator",
                    parse_status="error",
                    error=str(exc),
                    raw_response=response,
                    response_chars=len(response),
                    validation_cycle=cycle,
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed,
                )
                if attempt == 0:
                    retry_prompt = (
                        f"{prompt}\n\nFORMAT RETRY: Your previous validator response was not valid JSON "
                        f"({exc}). Re-evaluate the same report and return one concise, schema-conforming JSON object only. "
                        "Do not include unescaped quotation marks, markdown fences, or text outside the JSON object."
                    )
                    self.trace.add(
                        "validator_format_retry",
                        validation_cycle=cycle,
                        attempt=attempt + 1,
                        error=str(exc),
                        response_chars=len(response),
                    )
                    continue
                return ValidationDecision("fail", [], "Validator returned malformed JSON twice; the report was not accepted.")

        if payload is None:
            return ValidationDecision("fail", [], "Validator did not return a usable JSON object; the report was not accepted.")
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
        )
        return result

    def run(
        self,
        objective: str,
        snapshot: ReportSnapshot,
        prompt_context: str,
        tools: dict[str, ToolSpec],
        validator_context: str = "",
    ) -> tuple[dict[str, Any], ValidationDecision, RunTrace]:
        """Run the harness and guarantee a terminal trace event for success or failure."""
        self.trace.add(
            "run_started",
            author_model=self.author_model,
            validator_model=self.validator_model,
            limits={
                "max_agent_rounds": self.limits.max_agent_rounds,
                "max_tool_calls": self.limits.max_tool_calls,
                "max_calls_per_tool": self.limits.max_calls_per_tool,
                "max_validation_cycles": self.limits.max_validation_cycles,
                "max_report_bytes": self.limits.max_report_bytes,
            },
            tool_names=sorted(tools),
        )
        try:
            result = self._run_impl(objective, snapshot, prompt_context, tools, validator_context)
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
        """Let the author gather evidence for a bounded number of rounds until it returns a report."""
        # The round loop lets the model gather structured evidence, but it must do so under
        # strict round and tool budgets so it cannot spiral into unbounded exploration.
        for round_number in range(1, self.limits.max_agent_rounds + 1):
            if phase == "initial":
                self.trace.add("agent_round", round=round_number)
            author_result = self._author_call(state.prompt())
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
            state.evidence_sections.append(
                f"\n\nNEW TOOL EVIDENCE ({phase} round {round_number}; earlier evidence above still applies):\n"
                f"{json.dumps(new_results, indent=2, default=str)}"
            )
        exc = RuntimeError("Maximum agent rounds exceeded" if phase == "initial" else "Maximum revision agent rounds exceeded")
        self._record_exception(f"{phase}_agent_rounds", exc)
        self._trace_unresolved_tool_errors(state.pending_tool_errors, str(exc))
        raise exc

    def _sanitization_decision(self, exc: Exception, html: str) -> ValidationDecision:
        """Convert a mechanical HTML-gate failure into validator-style feedback for revision."""
        category = getattr(exc, "category", "html")
        size = len(html.encode("utf-8"))
        if category == "history":
            instruction = (
                f"Because get_report_history evidence was used, include the exact text \"{HISTORY_LABEL}\" "
                "before any prior-report content (a short clearly labeled historical note) and do not present "
                "historical figures as current."
            )
        elif size > self.limits.max_report_bytes:
            instruction = (
                f"Return a complete report under {self.limits.max_report_bytes} bytes: remove redundant inline "
                "styles and markup while keeping every required section."
            )
        else:
            instruction = (
                "Remove script, iframe, object, embed, form, input, link, and meta elements, event-handler "
                "attributes, and javascript: URLs; use only static email-safe HTML with inline styles."
            )
        finding = ValidationFinding(
            severity="error",
            category=category,
            message=str(exc),
            evidence=f"Mechanical HTML gate rejected the report ({size} bytes) before validator review.",
            revision_instruction=instruction,
        )
        return ValidationDecision("fail", [finding], f"Report rejected by the HTML safety gate: {exc}")

    def _run_impl(
        self,
        objective: str,
        snapshot: ReportSnapshot,
        prompt_context: str,
        tools: dict[str, ToolSpec],
        validator_context: str = "",
    ) -> tuple[dict[str, Any], ValidationDecision, RunTrace]:
        """Run the author loop, collect evidence, and validate the final email HTML."""
        state = _AuthorState(base_prompt=(
            f"{prompt_context}\n\nOBJECTIVE:\n{objective}\n\n"
            f"AUTHORITATIVE SNAPSHOT:\n{json.dumps(snapshot.to_dict(), indent=2)}\n\n"
            f"AVAILABLE TOOLS:\n{self._tool_prompt(tools)}\n\n"
            "Return JSON. To request evidence, return {\"tool_calls\":[{\"name\":\"...\",\"arguments\":{}}]}. "
            "When ready, return {\"report_html\":\"...\",\"plain_text\":\"...\"}. "
            "Return the JSON object directly: do not use markdown fences, outer quotes, or a Python repr. "
            "Escape newline characters inside string values as JSON \\n escapes. "
            "If tool evidence contains an error, use its message to correct the arguments and retry if useful; "
            "the complete conversation and earlier evidence remain in context. Do not repeat the identical failing call."
        ))
        report = self._author_loop("initial", tools, state)

        # Validation is intentionally fail-closed: the model can revise the report only if
        # the validator or the HTML gate identifies a concrete problem, and only sanitized
        # HTML (or the deterministic fallback) is ever returned.
        previous_validation: dict[str, Any] | None = None
        previous_error_counts: dict[str, int] | None = None
        decision: ValidationDecision | None = None
        for cycle in range(self.limits.max_validation_cycles + 1):
            html = report.get("report_html") if isinstance(report, dict) else None
            history_used = state.history_used()
            self.trace.add(
                "validation_cycle_start",
                cycle=cycle + 1,
                report_html_present=isinstance(html, str) and bool(html.strip()),
                report_html_bytes=len(html.encode("utf-8")) if isinstance(html, str) else 0,
                evidence_count=len(state.evidence),
                history_used=history_used,
            )
            if not isinstance(html, str) or not html.strip():
                self.trace.add(
                    "report_missing",
                    validation_cycle=cycle + 1,
                    author_error=report.get("error") if isinstance(report, dict) else None,
                    response_keys=sorted(report.keys()) if isinstance(report, dict) else [],
                )
                html = ""
                report = report if isinstance(report, dict) else {}
                report.setdefault("plain_text", "")
                report["report_html"] = html
                report["status"] = "missing_report_html"
            decision = None
            safe_html: str | None = None
            if not html.strip():
                safe_html = "<html><body><p>Missing report_html in author output.</p></body></html>"
            else:
                try:
                    safe_html = sanitize_html(html, self.limits.max_report_bytes, require_history_label=history_used)
                except ValueError as exc:
                    self._record_exception("report_sanitization", exc, cycle=cycle)
                    decision = self._sanitization_decision(exc, html)
                    self.trace.add(
                        "report_sanitization_failed",
                        cycle=cycle + 1,
                        category=getattr(exc, "category", "html"),
                        error=str(exc),
                        report_html_bytes=len(html.encode("utf-8")),
                    )
            if decision is None:
                prior_validation_text = (
                    "PREVIOUS VALIDATION CYCLE (context for tracking revisions only; independently re-check the current report and disagree with prior findings when the evidence warrants it):\n"
                    f"{json.dumps(previous_validation, indent=2, ensure_ascii=False)}\n\n"
                    if previous_validation is not None else ""
                )
                validator_prompt = (
                    f"{validator_context}\n\nVALIDATE THIS REPORT:\n{safe_html}\n\n"
                    f"AUTHORITATIVE SNAPSHOT:\n{json.dumps(snapshot.to_dict(), indent=2)}\n\n"
                    f"TOOL EVIDENCE:\n{json.dumps(state.evidence, indent=2, default=str)}\n\n"
                    f"{prior_validation_text}"
                    "If the report is missing report_html or plain_text, return fail with a revision instruction "
                    "telling the author to return a JSON object that includes both keys and a valid HTML report."
                )
                try:
                    decision = self._validator_call(validator_prompt, cycle=cycle + 1)
                except Exception as exc:
                    self.trace.add(
                        "validation_cycle_result",
                        cycle=cycle + 1,
                        decision="error",
                        stage="validator_call",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
            error_findings = [finding for finding in decision.findings if finding.severity.lower() == "error"]
            warning_count = sum(finding.severity.lower() == "warning" for finding in decision.findings)
            error_counts: dict[str, int] = {}
            for finding in error_findings:
                category = finding.category or "uncategorized"
                error_counts[category] = error_counts.get(category, 0) + 1
            if not decision.passed and not error_findings:
                error_counts["unclassified"] = 1
            error_count = sum(error_counts.values())
            progress = {
                "error_count": error_count,
                "warning_count": warning_count,
                "errors_by_category": error_counts,
                "prior_error_count": sum(previous_error_counts.values()) if previous_error_counts is not None else None,
                "error_count_delta": (
                    sum(previous_error_counts.values()) - error_count
                    if previous_error_counts is not None else None
                ),
                "resolved_error_count_by_category": {},
                "new_error_count_by_category": {},
            }
            if previous_error_counts is not None:
                categories = set(previous_error_counts) | set(error_counts)
                progress["resolved_error_count_by_category"] = {
                    category: previous_error_counts.get(category, 0) - error_counts.get(category, 0)
                    for category in sorted(categories)
                    if previous_error_counts.get(category, 0) > error_counts.get(category, 0)
                }
                progress["new_error_count_by_category"] = {
                    category: error_counts.get(category, 0) - previous_error_counts.get(category, 0)
                    for category in sorted(categories)
                    if error_counts.get(category, 0) > previous_error_counts.get(category, 0)
                }
            self.trace.add("validation_progress", cycle=cycle + 1, **progress)
            self.trace.add(
                "validation",
                cycle=cycle + 1,
                decision=decision.decision,
                summary=decision.summary,
                findings=decision.to_dict()["findings"],
            )
            self.trace.add(
                "validation_cycle_result",
                cycle=cycle + 1,
                decision=decision.decision,
                stage="validator" if safe_html is not None else "sanitization",
                summary=decision.summary,
                findings=decision.to_dict()["findings"],
                report_html_bytes=len((safe_html if safe_html is not None else html).encode("utf-8")),
                error_count=error_count,
                warning_count=warning_count,
                error_count_delta=progress["error_count_delta"],
                resolved_error_count_by_category=progress["resolved_error_count_by_category"],
                new_error_count_by_category=progress["new_error_count_by_category"],
            )
            if safe_html is not None:
                revision_html = safe_html
                report["report_html"] = safe_html
            else:
                # Unsanitized HTML may go back to the author for repair, but never out of the harness.
                revision_html = (
                    html if len(html.encode("utf-8")) <= self.limits.max_report_bytes
                    else f"[omitted: {len(html.encode('utf-8'))} bytes exceeds the {self.limits.max_report_bytes}-byte limit]"
                )
                report["report_html"] = ""
                report["status"] = "sanitization_failed"
            previous_validation = {
                "cycle": cycle + 1,
                "decision": decision.decision,
                "summary": decision.summary,
                "progress": progress,
                "findings": [finding.__dict__ for finding in decision.findings],
            }
            previous_error_counts = error_counts
            if decision.passed:
                return report, decision, self.trace
            if cycle >= self.limits.max_validation_cycles:
                break
            feedback = json.dumps(
                {"validation": decision.to_dict(), "progress": progress},
                indent=2,
                ensure_ascii=False,
            )
            failed_report = json.dumps(
                {
                    "report_html": revision_html,
                    "plain_text": report.get("plain_text", ""),
                },
                indent=2,
            )
            # Replace (not append) the revision request so only the latest attempt is in context.
            state.revision_block = (
                f"\n\nFAILED REPORT FOR REVISION:\n{failed_report}\n\n"
                f"VALIDATOR FEEDBACK FOR REVISION:\n{feedback}\n"
                "Revise the report and return the complete report JSON with both 'report_html' and 'plain_text' keys. "
                "Resolve every remaining error, review resolved/new error counts, and do not reintroduce a fixed issue in another section. "
                "You may request additional evidence with tool_calls before returning the complete report."
            )
            report = self._author_loop("revision", tools, state)

        if not isinstance(report.get("report_html"), str) or not str(report.get("report_html", "")).strip():
            report = build_fallback_report(snapshot, "Validation cycles were exhausted and the author never returned a usable final report payload.")
        last_summary = decision.summary if decision is not None else ""
        return report, ValidationDecision(
            "fail",
            list(decision.findings) if decision is not None else [],
            f"Validation cycles were exhausted without a passing report. Last review: {last_summary}".strip(),
        ), self.trace
