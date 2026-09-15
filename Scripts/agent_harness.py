"""Bounded author/validator harness for the weekly financial report."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable

from report_models import ReportSnapshot, ValidationDecision, ValidationFinding


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

    def add(self, event: str, **values: Any) -> None:
        """Record a single event in the agent timeline."""
        self.events.append({"event": event, "timestamp": time.time(), **values})


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


def sanitize_html(html: str, max_bytes: int = 250_000, require_history_label: bool = False) -> str:
    """Strip unsafe markup while preserving the report layout the model intended."""
    # This is the final mechanical gate before validation: if the model produces HTML that
    # could execute or is too large, the report is rejected before any email is sent.
    if not html or len(html.encode("utf-8")) > max_bytes:
        raise ValueError("Report HTML is empty or exceeds the configured size limit")
    cleaned = re.sub(r"<\s*(script|iframe|object|embed|form|input|link|meta)\b[^>]*>.*?<\s*/\s*\1\s*>", "", html, flags=re.I | re.S)
    cleaned = re.sub(r"<\s*(script|iframe|object|embed|form|input|link|meta)\b[^>]*?/?>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+on[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:href|src)\s*=\s*(['\"])\s*javascript:[^'\"]*\1", "", cleaned, flags=re.I)
    parser = _UnsafeMarkupParser()
    parser.feed(cleaned)
    if parser.unsafe:
        raise ValueError("Report contains unsafe HTML elements")
    if require_history_label and "Previous report for reference:" not in cleaned:
        raise ValueError("Report must explicitly label prior report excerpts")
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
    response_format: str = "json",
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
        try:
            response = self.query(prompt, self.ollama_url, self.author_model, timeout=self.timeout)
        except Exception as exc:
            self._record_exception("author_model_call", exc)
            raise
        try:
            result = parse_json_response(response)
            if "tool_calls" in result:
                if not isinstance(result["tool_calls"], list):
                    raise ValueError("Author tool_calls must be a JSON array")
            else:
                missing = {"report_html", "plain_text"} - set(result)
                if missing:
                    raise ValueError(f"Author response is missing required keys: {', '.join(sorted(missing))}")
        except Exception as exc:
            self.trace.add(
                "model_response",
                role="author",
                parse_status="error",
                error=str(exc),
                raw_response=response,
                response_chars=len(response),
            )
            return {"report_html": "", "plain_text": "", "error": str(exc)}
        self.trace.add(
            "model_response",
            role="author",
            parse_status="ok",
            keys=list(result),
            raw_response=response,
            response_chars=len(response),
        )
        return result

    def _validator_call(self, prompt: str) -> ValidationDecision:
        """Run the validation model and convert its structured result into a typed verdict."""
        self.trace.add(
            "model_call",
            role="validator",
            model=self.validator_model,
            prompt=prompt,
            prompt_chars=len(prompt),
        )
        try:
            response = self.query(prompt, self.ollama_url, self.validator_model, timeout=self.timeout)
        except Exception as exc:
            self._record_exception("validator_model_call", exc)
            raise
        try:
            payload = parse_json_response(response)
        except Exception as exc:
            self.trace.add(
                "model_response",
                role="validator",
                parse_status="error",
                error=str(exc),
                raw_response=response,
                response_chars=len(response),
            )
            return ValidationDecision("fail", [], "Validator returned malformed JSON, so the report was not accepted.")
        decision = str(payload.get("decision", "")).lower()
        if decision not in {"pass", "fail"}:
            self.trace.add(
                "model_response",
                role="validator",
                parse_status="invalid_decision",
                payload=payload,
                raw_response=response,
                response_chars=len(response),
            )
            return ValidationDecision("fail", [], "Validator decision was missing or invalid; report was not accepted.")
        try:
            findings = [ValidationFinding(**finding) for finding in payload.get("findings", [])]
        except Exception as exc:
            self._record_exception("validator_payload", exc, payload=payload)
            return ValidationDecision("fail", [], "Validator findings were malformed; the report was not accepted.")
        result = ValidationDecision(decision, findings, str(payload.get("summary", "")))
        self.trace.add(
            "model_response",
            role="validator",
            parse_status="ok",
            payload=payload,
            raw_response=response,
            response_chars=len(response),
        )
        self.trace.add("validation", decision=result.decision, findings=result.to_dict()["findings"])
        return result

    def run(
        self,
        objective: str,
        snapshot: ReportSnapshot,
        prompt_context: str,
        tools: dict[str, ToolSpec],
        validator_context: str = "",
    ) -> tuple[dict[str, Any], ValidationDecision, RunTrace]:
        """Run the author loop, collect evidence, and validate the final email HTML."""
        evidence: list[dict[str, Any]] = []
        tool_counts: dict[str, int] = {}
        author_prompt = (
            f"{prompt_context}\n\nOBJECTIVE:\n{objective}\n\n"
            f"AUTHORITATIVE SNAPSHOT:\n{json.dumps(snapshot.to_dict(), indent=2)}\n\n"
            f"AVAILABLE TOOLS:\n{self._tool_prompt(tools)}\n\n"
            "Return JSON. To request evidence, return {\"tool_calls\":[{\"name\":\"...\",\"arguments\":{}}]}. "
            "When ready, return {\"report_html\":\"...\",\"plain_text\":\"...\"}. "
            "Return the JSON object directly: do not use markdown fences, outer quotes, or a Python repr. "
            "Escape newline characters inside string values as JSON \\n escapes."
        )
        # The agent round loop lets the model gather structured evidence, but it must do so
        # under strict tool budgets so it cannot spiral into unbounded exploration.
        for round_number in range(self.limits.max_agent_rounds):
            self.trace.add("agent_round", round=round_number + 1)
            author_result = self._author_call(author_prompt)
            calls = author_result.get("tool_calls") or []
            if not calls:
                report = author_result
                break
            for call in calls:
                if len(evidence) >= self.limits.max_tool_calls:
                    exc = RuntimeError("Maximum total tool calls exceeded")
                    self._record_exception("initial_tool_budget", exc)
                    raise exc
                name = call.get("name")
                if name not in tools:
                    exc = ValueError(f"Tool is not allowlisted: {name}")
                    self._record_exception("initial_tool_allowlist", exc, tool=name)
                    raise exc
                tool_counts[name] = tool_counts.get(name, 0) + 1
                if tool_counts[name] > self.limits.max_calls_per_tool:
                    exc = RuntimeError(f"Maximum calls exceeded for tool: {name}")
                    self._record_exception("initial_tool_budget", exc, tool=name)
                    raise exc
                arguments = call.get("arguments") or {}
                try:
                    result = tools[name].function(arguments)
                except Exception as exc:
                    self._record_exception("initial_tool_call", exc, tool=name, arguments=arguments)
                    raise
                evidence.append({"tool": name, "arguments": arguments, "result": result})
                self.trace.add("tool_call", name=name, arguments=arguments, result=result)
            author_prompt = f"{author_prompt}\n\nNEW TOOL EVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}"
        else:
            exc = RuntimeError("Maximum agent rounds exceeded")
            self._record_exception("initial_agent_rounds", exc)
            raise exc

        # Validation is intentionally fail-closed: the model can revise the report only if
        # the validator identifies a concrete problem, and even then the report must remain
        # safe and within the configured limits.
        for cycle in range(self.limits.max_validation_cycles + 1):
            html = report.get("report_html") if isinstance(report, dict) else None
            if not isinstance(html, str) or not html.strip():
                html = ""
                report = report if isinstance(report, dict) else {}
                report.setdefault("plain_text", "")
                report["report_html"] = html
                report["status"] = "missing_report_html"
            history_used = any(item["tool"] == "get_report_history" for item in evidence)
            try:
                safe_html = sanitize_html(html, self.limits.max_report_bytes, require_history_label=history_used) if html.strip() else "<html><body><p>Missing report_html in author output.</p></body></html>"
            except Exception as exc:
                self._record_exception("report_sanitization", exc, cycle=cycle)
                raise
            validator_prompt = (
                f"{validator_context}\n\nVALIDATE THIS REPORT:\n{safe_html}\n\n"
                f"AUTHORITATIVE SNAPSHOT:\n{json.dumps(snapshot.to_dict(), indent=2)}\n\n"
                f"TOOL EVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}\n\n"
                "If the report is missing report_html or plain_text, return fail with a revision instruction "
                "telling the author to return a JSON object that includes both keys and a valid HTML report."
            )
            decision = self._validator_call(validator_prompt)
            report["report_html"] = safe_html
            if decision.passed:
                return report, decision, self.trace
            if cycle >= self.limits.max_validation_cycles:
                break
            feedback = json.dumps(decision.to_dict(), indent=2)
            failed_report = json.dumps(
                {
                    "report_html": safe_html,
                    "plain_text": report.get("plain_text", ""),
                },
                indent=2,
            )
            author_prompt = (
                f"{author_prompt}\n\nFAILED REPORT FOR REVISION:\n{failed_report}\n\n"
                f"VALIDATOR FEEDBACK FOR REVISION:\n{feedback}\n"
                "Revise the report and return the complete report JSON with both 'report_html' and 'plain_text' keys. "
                "You may request additional evidence with tool_calls before returning the complete report."
            )
            report = self._author_call(author_prompt)
            for revision_round in range(self.limits.max_agent_rounds):
                calls = report.get("tool_calls") or []
                if not calls:
                    break
                self.trace.add("agent_revision_round", round=revision_round + 1)
                for call in calls:
                    if len(evidence) >= self.limits.max_tool_calls:
                        exc = RuntimeError("Maximum total tool calls exceeded")
                        self._record_exception("revision_tool_budget", exc)
                        raise exc
                    name = call.get("name")
                    if name not in tools:
                        exc = ValueError(f"Tool is not allowlisted: {name}")
                        self._record_exception("revision_tool_allowlist", exc, tool=name)
                        raise exc
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    if tool_counts[name] > self.limits.max_calls_per_tool:
                        exc = RuntimeError(f"Maximum calls exceeded for tool: {name}")
                        self._record_exception("revision_tool_budget", exc, tool=name)
                        raise exc
                    arguments = call.get("arguments") or {}
                    try:
                        result = tools[name].function(arguments)
                    except Exception as exc:
                        self._record_exception("revision_tool_call", exc, tool=name, arguments=arguments)
                        raise
                    evidence.append({"tool": name, "arguments": arguments, "result": result})
                    self.trace.add("tool_call", name=name, arguments=arguments, result=result, phase="revision")
                author_prompt = f"{author_prompt}\n\nNEW TOOL EVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}"
                report = self._author_call(author_prompt)
            else:
                exc = RuntimeError("Maximum revision agent rounds exceeded")
                self._record_exception("revision_agent_rounds", exc)
                raise exc

        if not isinstance(report.get("report_html"), str) or not str(report.get("report_html", "")).strip():
            report = build_fallback_report(snapshot, "Validation cycles were exhausted and the author never returned a usable final report payload.")
        return report, ValidationDecision("fail", [], "Validation cycles were exhausted without a usable final report payload."), self.trace

        raise RuntimeError("Validation loop ended unexpectedly")