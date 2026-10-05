"""Bounded author/validator harness for the weekly financial report."""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable

from .ollama import api_url
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
                "\n\nTOOL BUDGET EXHAUSTED: do not request more tools. Return the final report now between "
                f"{REPORT_START_MARKER} and {REPORT_END_MARKER}, using the snapshot and the evidence above."
            )
        if self.revision_block:
            parts.append(self.revision_block)
        if self.retry_suffix:
            parts.append(self.retry_suffix)
        return "".join(parts)


class _UnsafeMarkupParser(HTMLParser):
    # This parser is a lightweight blocklist: if the model tries to inject HTML that can
    # execute or affect the email client, the report is rejected (and sent back to the
    # author as revision feedback) instead of being sent.
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
    # could execute, is too large, or lacks a required history label, ReportSanitizationError
    # is raised and the harness returns it to the author as feedback; nothing unsanitized is sent.
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


FEEDBACK_PRIORITY = [
    "arithmetic", "completeness", "evidence", "history", "policy", "html", "usability", "visual_design",
]
MAX_REVISION_FINDINGS = 3

# Phrases in a finding that point at a snapshot field the author must copy exactly.
_SNAPSHOT_FIELD_HINTS = [
    (("disposable", "projected savings"), "disposable_cash"),
    (("remaining gap", "remaining_gap"), "remaining_gap_after_cuts"),
    (("shortfall",), "savings_shortfall"),
    (("mom", "month-over-month", "month over month"), "mom_change_percent"),
    (("previous month total", "previous total", "previous_total", "prior month total"), "previous_total_net_expense"),
    (("latest total", "current total", "total net expense", "latest_total", "current-month total"), "latest_total_net_expense"),
]


def _authoritative_values(text: str, snapshot: ReportSnapshot) -> dict[str, Any]:
    """Snapshot values a finding refers to, so the author copies them instead of recomputing."""
    lowered = text.lower()

    def mentions(phrase: str, haystack: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", haystack) is not None

    values: dict[str, Any] = {}
    for phrases, field_name in _SNAPSHOT_FIELD_HINTS:
        if any(mentions(phrase, lowered) for phrase in phrases) and hasattr(snapshot, field_name):
            values[field_name] = getattr(snapshot, field_name)
    for metric in snapshot.categories:
        # Category names are matched as written (case-sensitive) so words like "other" in prose don't match.
        if mentions(metric.category, text):
            values[f"category:{metric.category} ({metric.parent_group})"] = {
                key: getattr(metric, key, None)
                for key in ("current_spend", "previous_spend", "suggested_cap", "percent_of_cap", "bar_width_percent")
            }
    return values


def focused_feedback(decision: ValidationDecision, snapshot: ReportSnapshot) -> dict[str, Any]:
    """Top-priority error findings for the next revision, with the exact snapshot values they concern."""
    errors = [finding for finding in decision.findings if finding.severity.lower() == "error"]
    rank = {category: index for index, category in enumerate(FEEDBACK_PRIORITY)}
    ordered = sorted(enumerate(errors), key=lambda item: (rank.get(item[1].category, len(rank)), item[0]))
    selected = [finding for _, finding in ordered[:MAX_REVISION_FINDINGS]]
    fixes = []
    for finding in selected:
        item: dict[str, Any] = {
            "category": finding.category,
            "message": finding.message,
            "evidence": finding.evidence,
            "revision_instruction": finding.revision_instruction,
        }
        values = _authoritative_values(
            " ".join(str(part or "") for part in (finding.message, finding.evidence, finding.revision_instruction)),
            snapshot,
        )
        if values:
            item["authoritative_values"] = values
        fixes.append(item)
    feedback: dict[str, Any] = {"review_summary": decision.summary, "fix_these": fixes}
    remaining = len(errors) - len(selected)
    if remaining > 0:
        feedback["remaining_lower_priority_errors"] = remaining
    if not errors:
        feedback["note"] = "The review failed without itemized errors; re-check the report against the required sections and the snapshot."
    return feedback


STALL_MESSAGE_PREFIX_CHARS = 80


def _finding_key(finding: ValidationFinding) -> tuple[str, str]:
    """Normalized category plus message prefix, used only to notice findings repeated across cycles."""
    message = re.sub(r"[^a-z0-9]+", " ", str(finding.message or "").lower()).strip()
    return (str(finding.category or "").lower(), message[:STALL_MESSAGE_PREFIX_CHARS])


REPORT_START_MARKER = "<<<REPORT_HTML>>>"
REPORT_END_MARKER = "<<<END_REPORT_HTML>>>"
REBUTTALS_START_MARKER = "<<<REBUTTALS>>>"
REBUTTALS_END_MARKER = "<<<END_REBUTTALS>>>"
MAX_REBUTTALS = 5
MAX_REBUTTAL_FIELD_CHARS = 400

# During a revision the author may dispute a finding the report already satisfies by adding,
# after the report end marker, a JSON array between the rebuttal markers. Rebuttals never
# become part of the report; the validator checks each quote against the report itself.
REBUTTAL_PROTOCOL = (
    "DISPUTING A FINDING: fix every genuine finding. Only if you are sure the previous report ALREADY satisfies a "
    "finding (the validator missed existing text), you may dispute it instead of editing: after "
    f"{REPORT_END_MARKER}, add\n"
    f"{REBUTTALS_START_MARKER}\n"
    "[{\"finding\": \"<short finding message>\", \"quote\": \"<exact text copied from the report that satisfies it>\"}]\n"
    f"{REBUTTALS_END_MARKER}\n"
    "The quote must be copied verbatim from the report you return. Rebuttals are optional and are never part of the report HTML."
)

# The author returns tool requests as small JSON objects, but the final report as raw HTML
# between markers: small models cannot reliably escape a ~20 KB HTML document inside a JSON
# string, so the report never passes through JSON.
AUTHOR_PROTOCOL = (
    "RESPONSE FORMAT: reply in exactly one of two forms.\n"
    "1. To request evidence, reply with only a JSON object, for example: "
    "{\"tool_calls\":[{\"name\":\"get_category_detail\",\"arguments\":{\"category\":\"Shopping\"}}]}\n"
    f"2. When the report is ready, reply with the complete HTML document between the markers, for example:\n"
    f"{REPORT_START_MARKER}\n<html><body>...complete report...</body></html>\n{REPORT_END_MARKER}\n"
    "Write the HTML as-is between the markers: do not wrap it in JSON, quotes, or markdown fences, and do not "
    "escape its quotes. Do not add a plain-text version; it is derived from the HTML."
)


class ModelReply(str):
    """Model text plus Ollama generation metadata; behaves as a plain string for parsing."""

    done_reason: str | None
    eval_count: int | None
    prompt_eval_count: int | None

    def __new__(
        cls,
        text: str,
        done_reason: str | None = None,
        eval_count: int | None = None,
        prompt_eval_count: int | None = None,
    ) -> "ModelReply":
        reply = super().__new__(cls, text)
        reply.done_reason = done_reason
        reply.eval_count = eval_count
        reply.prompt_eval_count = prompt_eval_count
        return reply


class EmptyModelResponse(ValueError):
    """Ollama answered without any text; treated like an unavailable model call."""


class ModelUnavailableError(RuntimeError):
    """A model call failed twice with a transport/server error and the harness stopped calling it."""


class AgentRoundsExhaustedError(RuntimeError):
    """The author used every allowed round of a phase without returning a report."""


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


def _response_meta(response: str) -> dict[str, Any]:
    """Generation metadata for trace events; absent for plain-string test doubles."""
    return {
        "done_reason": getattr(response, "done_reason", None),
        "eval_count": getattr(response, "eval_count", None),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
    }


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first decodable JSON object in text, ignoring fences and surrounding prose."""
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_author_response(text: str) -> dict[str, Any]:
    """Parse an author turn: marker-delimited report HTML, or a JSON tool-call object.

    Markers take precedence. A whole HTML document without markers (starting with <!doctype
    or <html and closed by </html>) is also accepted as the report, and a legacy JSON object
    with a report_html string still parses. plain_text is always derived from the HTML.
    """
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    body, rebuttals, rebuttal_error = _extract_rebuttals(body)
    result = _parse_author_body(body)
    if "report_html" in result:
        if rebuttals:
            result["rebuttals"] = rebuttals
        if rebuttal_error:
            result["rebuttal_error"] = rebuttal_error
    return result


def _extract_rebuttals(body: str) -> tuple[str, list[dict[str, str]], str | None]:
    """Remove an optional rebuttal block from an author reply and parse it leniently.

    Returns (body without the block, valid rebuttals, error or None). The block is removed
    wherever it appears (also inside the report or an HTML comment), so it never reaches the
    report HTML. A malformed block yields an error description, never an exception.
    """
    start = body.find(REBUTTALS_START_MARKER)
    if start == -1:
        return body, [], None
    content_start = start + len(REBUTTALS_START_MARKER)
    end = body.find(REBUTTALS_END_MARKER, content_start)
    if end != -1:
        raw = body[content_start:end]
        cut_end = end + len(REBUTTALS_END_MARKER)
    else:
        # Unclosed block: drop it up to the next report marker (so the report survives) or the end.
        stops = [pos for pos in (body.find(REPORT_START_MARKER, content_start), body.find(REPORT_END_MARKER, content_start)) if pos != -1]
        cut_end = min(stops) if stops else len(body)
        raw = body[content_start:cut_end]
    cut_start = start
    # Drop a wrapping HTML comment (<!-- <<<REBUTTALS>>> ... <<<END_REBUTTALS>>> -->) along with the block.
    prefix = re.search(r"<!--\s*$", body[:cut_start])
    if prefix:
        suffix = re.match(r"\s*-->", body[cut_end:])
        if suffix:
            cut_start = prefix.start()
            cut_end += suffix.end()
    remaining = (body[:cut_start] + body[cut_end:]).strip()
    if end == -1:
        return remaining, [], f"{REBUTTALS_END_MARKER} is missing"
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.I).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return remaining, [], f"Rebuttals are not valid JSON: {exc}"
    if isinstance(value, dict):
        value = value["rebuttals"] if isinstance(value.get("rebuttals"), list) else [value]
    if not isinstance(value, list):
        return remaining, [], "Rebuttals must be a JSON array of {finding, quote} objects"
    rebuttals: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        finding, quote = item.get("finding"), item.get("quote")
        if isinstance(finding, str) and isinstance(quote, str) and finding.strip() and quote.strip():
            rebuttals.append({
                "finding": finding.strip()[:MAX_REBUTTAL_FIELD_CHARS],
                "quote": quote.strip()[:MAX_REBUTTAL_FIELD_CHARS],
            })
    if not rebuttals:
        return remaining, [], "Rebuttals contained no entries with non-empty finding and quote strings"
    error = None
    if len(rebuttals) > MAX_REBUTTALS:
        error = f"Only the first {MAX_REBUTTALS} of {len(rebuttals)} rebuttals were kept"
        rebuttals = rebuttals[:MAX_REBUTTALS]
    return remaining, rebuttals, error


def _parse_author_body(body: str) -> dict[str, Any]:
    start = body.find(REPORT_START_MARKER)
    if start != -1:
        end = body.find(REPORT_END_MARKER, start + len(REPORT_START_MARKER))
        if end == -1:
            raise ValueError(
                f"Report start marker {REPORT_START_MARKER} was found but {REPORT_END_MARKER} is missing; "
                "the output was likely truncated. Return the complete report and close it with the end marker."
            )
        html = body[start + len(REPORT_START_MARKER):end].strip()
        html = re.sub(r"^```(?:html)?\s*|\s*```$", "", html, flags=re.I).strip()
        return _report_result(html)

    value = _first_json_object(body)
    if value is not None:
        if "tool_calls" in value:
            if not isinstance(value["tool_calls"], list):
                raise ValueError("Author tool_calls must be a JSON array")
            return {"tool_calls": value["tool_calls"]}
        if isinstance(value.get("report_html"), str) and value["report_html"].strip():
            return _report_result(value["report_html"])

    unfenced = re.sub(r"^```(?:html)?\s*|\s*```$", "", body, flags=re.I).strip()
    if re.match(r"<!doctype html|<html\b", unfenced, flags=re.I) and re.search(r"</html>\s*$", unfenced, flags=re.I):
        return _report_result(unfenced)

    raise ValueError(
        f"Response contained neither a report between {REPORT_START_MARKER} and {REPORT_END_MARKER} "
        "nor a JSON object with tool_calls"
    )


def _report_result(html: str) -> dict[str, Any]:
    if not html.strip():
        raise ValueError("Author report between the markers is empty")
    plain_text = _plain_text_from_html(html)
    if not plain_text:
        raise ValueError("Could not derive plain_text from the author report HTML")
    return {"report_html": html, "plain_text": plain_text}


def query_ollama(
    prompt: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
    response_format: str | dict[str, Any] | None = "json",
    options: dict[str, Any] | None = None,
    think: bool | None = None,
) -> ModelReply:
    """Call Ollama and return the text response for one model turn with generation metadata.

    ``ollama_url`` may be the server base URL or any Ollama API endpoint; it is normalized to /api/generate.
    """
    import requests

    payload: dict[str, Any] = {"model": model_name, "prompt": prompt, "stream": False}
    if response_format is not None:
        payload["format"] = response_format
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    response = requests.post(api_url(ollama_url, "generate"), json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    result = body.get("response")
    if not isinstance(result, str) or not result.strip():
        if body.get("done_reason") == "length":
            # The token limit was spent before any answer (often on hidden thinking); retrying the
            # same call would fail the same way, so let the format/length handling report it.
            thinking_chars = len(body.get("thinking") or "")
            print(
                f"Ollama hit num_predict with no answer text ({thinking_chars} thinking chars); "
                "raise num_predict or set think=false for this role.",
                flush=True,
            )
            result = ""
        else:
            raise EmptyModelResponse("Ollama response did not contain text")
    return ModelReply(
        result,
        done_reason=body.get("done_reason"),
        eval_count=body.get("eval_count"),
        prompt_eval_count=body.get("prompt_eval_count"),
    )

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
        model_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Create a bounded agent harness with a separate author and validator model.

        model_options holds optional per-role Ollama settings, e.g. {"author": {"temperature": 0.3,
        "num_predict": 12000, "think": False}, "validator": {...}}. "think" is sent as Ollama's
        top-level think flag; every other key goes into the request's options.
        """
        self.ollama_url = ollama_url
        self.author_model = author_model
        self.validator_model = validator_model or author_model
        self.limits = limits or HarnessLimits()
        self.timeout = timeout
        self.query = query
        self.model_options = model_options or {}
        self.trace = RunTrace()
        self._validator_unavailable = False

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

    def _query_with_retry(self, role: str, prompt: str, model: str, response_format: Any, **context: Any) -> str:
        """Call the model, retrying once on a timeout/connection/5xx/empty-response failure."""
        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                return self.query(
                    prompt,
                    self.ollama_url,
                    model,
                    timeout=self.timeout,
                    response_format=response_format,
                    **self._role_kwargs(role),
                )
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 3)
                self._record_exception(f"{role}_model_call", exc, elapsed_seconds=elapsed, call_attempt=attempt, **context)
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
        details = {
            "parse_error": error,
            "response_chars": author_result.get("response_chars", 0),
            "excerpt_near_error": author_result.get("raw_excerpt", ""),
        }
        return (
            "\n\nAUTHOR OUTPUT FORMAT ERROR — RETRY REQUIRED:\n"
            f"{json.dumps(details, ensure_ascii=False)}\n"
            "Recreate the response from the original objective, snapshot, tool evidence, and any validator feedback above. "
            "Either return only a JSON object with tool_calls, or return the complete report HTML between "
            f"{REPORT_START_MARKER} and {REPORT_END_MARKER} (end marker required). Do not return an explanation."
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
        """Send the current prompt to the author model and parse marker HTML or tool-call JSON.

        Raises ModelUnavailableError when the model fails twice with a transport/server error.
        """
        self.trace.add(
            "model_call",
            role="author",
            model=self.author_model,
            prompt=prompt,
            prompt_chars=len(prompt),
        )
        started = time.perf_counter()
        # No format schema: the report comes back as raw HTML between markers (see AUTHOR_PROTOCOL).
        response = self._query_with_retry("author", prompt, self.author_model, None)
        elapsed = round(time.perf_counter() - started, 3)
        meta = _response_meta(response)
        try:
            result = parse_author_response(response)
        except Exception as exc:
            error = str(exc)
            if meta["done_reason"] == "length":
                error = (
                    "Output hit the token limit before the response was complete; be more concise "
                    f"(trim repeated inline styles and prose) but keep every required section. Parser detail: {exc}"
                )
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

    def _validator_call(self, prompt: str, cycle: int | None = None) -> ValidationDecision:
        """Run the validation model and convert its structured result into a typed verdict."""
        self._validator_unavailable = False
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
                response = self._query_with_retry(
                    "validator",
                    retry_prompt,
                    self.validator_model,
                    VALIDATOR_RESPONSE_SCHEMA,
                    validation_cycle=cycle,
                    format_attempt=attempt + 1,
                )
            except ModelUnavailableError as exc:
                self._validator_unavailable = True
                return ValidationDecision(
                    "fail", [], f"Validator unavailable after a retry ({exc}); the report was not accepted."
                )
            elapsed = round(time.perf_counter() - started, 3)
            meta = _response_meta(response)
            try:
                payload = parse_json_response(response)
                break
            except Exception as exc:
                error = str(exc)
                if meta["done_reason"] == "length":
                    error = f"Validator output hit the token limit before the JSON was complete ({exc})"
                self.trace.add(
                    "model_response",
                    role="validator",
                    parse_status="error",
                    error=error,
                    raw_response=response,
                    response_chars=len(response),
                    validation_cycle=cycle,
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed,
                    **meta,
                )
                if attempt == 0:
                    length_hint = (
                        " Your previous answer ran out of tokens: keep the audit evidence and findings short."
                        if meta["done_reason"] == "length" else ""
                    )
                    retry_prompt = (
                        f"{prompt}\n\nFORMAT RETRY: Your previous validator response was not valid JSON "
                        f"({error}). Re-evaluate the same report and return one concise, schema-conforming JSON object only. "
                        f"Do not include unescaped quotation marks, markdown fences, or text outside the JSON object.{length_hint}"
                    )
                    self.trace.add(
                        "validator_format_retry",
                        validation_cycle=cycle,
                        attempt=attempt + 1,
                        error=error,
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
        """Let the author gather evidence for a bounded number of rounds until it returns a report.

        Raises AgentRoundsExhaustedError when max_agent_rounds is exhausted; the caller ends the
        run with a fail decision, like an unavailable author.
        """
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
            f"{AUTHOR_PROTOCOL}\n"
            "If tool evidence contains an error, use its message to correct the arguments and retry if useful; "
            "the complete conversation and earlier evidence remain in context. Do not repeat the identical failing call."
        ))
        try:
            report = self._author_loop("initial", tools, state)
        except (ModelUnavailableError, AgentRoundsExhaustedError) as exc:
            # Without an author report there is nothing to validate; return the deterministic fallback, never sent.
            if isinstance(exc, ModelUnavailableError):
                self.trace.add("model_unavailable", role="author", phase="initial", error=str(exc))
                reason = "The author model was unavailable after a retry."
                summary = f"Author model unavailable after a retry ({exc}); the fallback report was not validated."
            else:
                reason = f"The author used all {self.limits.max_agent_rounds} agent rounds without returning a report."
                summary = f"{reason} ({exc}); the fallback report was not validated."
            return build_fallback_report(snapshot, reason), ValidationDecision("fail", [], summary), self.trace
        # The initial report has no prior findings to dispute, so any rebuttal block is just dropped.
        self._take_rebuttals(report, responding_to_cycle=None)
        rebuttals: list[dict[str, str]] = []
        last_safe_report: dict[str, Any] | None = None

        # Validation is intentionally fail-closed: the model can revise the report only if
        # the validator or the HTML gate identifies a concrete problem, and only sanitized
        # HTML (or the deterministic fallback) is ever returned.
        previous_validation: dict[str, Any] | None = None
        previous_error_counts: dict[str, int] | None = None
        previous_error_keys: set[tuple[str, str]] | None = None
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
                if previous_validation is not None and rebuttals:
                    prior_validation_text += (
                        "AUTHOR REBUTTALS (verify each quote against the report): the author says the report above "
                        "already satisfies these prior findings. If a quote is present in the report and satisfies the "
                        "requirement, do not repeat that finding; otherwise keep it and explain why the quote is absent or insufficient.\n"
                        f"{json.dumps(rebuttals, indent=2, ensure_ascii=False)}\n\n"
                    )
                validator_prompt = (
                    f"{validator_context}\n\nVALIDATE THIS REPORT:\n{safe_html}\n\n"
                    f"AUTHORITATIVE SNAPSHOT:\n{json.dumps(snapshot.to_dict(), indent=2)}\n\n"
                    f"TOOL EVIDENCE:\n{json.dumps(state.evidence, indent=2, default=str)}\n\n"
                    f"{prior_validation_text}"
                    "If the report HTML is missing, return fail with a revision instruction telling the author to return "
                    f"the complete HTML report between {REPORT_START_MARKER} and {REPORT_END_MARKER}."
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
                if self._validator_unavailable:
                    # No review is possible; stop instead of revising blind. The report is kept but never sent.
                    self.trace.add("model_unavailable", role="validator", phase="validation", cycle=cycle + 1, error=decision.summary)
                    self.trace.add(
                        "validation_cycle_result",
                        cycle=cycle + 1,
                        decision="fail",
                        stage="validator_unavailable",
                        summary=decision.summary,
                        findings=[],
                        report_html_bytes=len(safe_html.encode("utf-8")),
                    )
                    if html.strip():
                        report["report_html"] = safe_html
                    else:
                        report = build_fallback_report(snapshot, "The validator was unavailable and the author returned no report.")
                    return report, decision, self.trace
            error_findings = [finding for finding in decision.findings if finding.severity.lower() == "error"]
            warning_count = sum(finding.severity.lower() == "warning" for finding in decision.findings)
            error_counts: dict[str, int] = {}
            for finding in error_findings:
                category = finding.category or "uncategorized"
                error_counts[category] = error_counts.get(category, 0) + 1
            if not decision.passed and not error_findings:
                error_counts["unclassified"] = 1
            error_count = sum(error_counts.values())
            error_keys = [_finding_key(finding) for finding in error_findings]
            repeated_error_count = (
                sum(key in previous_error_keys for key in error_keys) if previous_error_keys is not None else None
            )
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
                # Error findings whose normalized category + message prefix also appeared in the previous cycle.
                "repeated_error_count": repeated_error_count,
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
            if previous_error_keys and error_keys and set(error_keys) == previous_error_keys:
                # Informational only: the same findings came back unchanged, so the revision made no
                # visible progress (the author did not fix them, or the validator repeated false findings).
                self.trace.add(
                    "validation_stalled",
                    cycle=cycle + 1,
                    repeated_error_count=repeated_error_count,
                    rebuttal_count=len(rebuttals),
                    findings=[{"category": category, "message": message} for category, message in sorted(set(error_keys))],
                )
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
                repeated_error_count=repeated_error_count,
            )
            if safe_html is not None:
                revision_html = safe_html
                report["report_html"] = safe_html
                if html.strip():
                    last_safe_report = dict(report)
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
            previous_error_keys = set(error_keys)
            if decision.passed:
                return report, decision, self.trace
            if cycle >= self.limits.max_validation_cycles:
                break
            feedback = json.dumps(focused_feedback(decision, snapshot), indent=2, ensure_ascii=False)
            # Replace (not append) the revision request so only the latest attempt is in context.
            state.revision_block = (
                "\n\nPREVIOUS REPORT HTML (edit this document; it failed review):\n"
                f"----- BEGIN PREVIOUS REPORT -----\n{revision_html}\n----- END PREVIOUS REPORT -----\n\n"
                f"PRIORITY FIXES FROM THE REVIEW:\n{feedback}\n"
                "Fix only the listed issues with minimal edits to the previous HTML and keep everything else unchanged; "
                "use the exact authoritative values given and do not reintroduce a fixed issue in another section. "
                "Then return the complete revised report between "
                f"{REPORT_START_MARKER} and {REPORT_END_MARKER}. "
                "You may request additional evidence with tool_calls before returning the report.\n\n"
                f"{REBUTTAL_PROTOCOL}"
            )
            try:
                report = self._author_loop("revision", tools, state)
                rebuttals = self._take_rebuttals(report, responding_to_cycle=cycle + 1)
            except (ModelUnavailableError, AgentRoundsExhaustedError) as exc:
                # Keep the last sanitized report (never sent: the decision stays fail) and stop revising.
                if isinstance(exc, ModelUnavailableError):
                    self.trace.add("model_unavailable", role="author", phase="revision", cycle=cycle + 1, error=str(exc))
                    reason = "The author model was unavailable during revision."
                    summary = f"Author model unavailable during revision after a retry ({exc}); last review: {decision.summary}"
                else:
                    reason = "The author used all revision agent rounds without returning a report."
                    summary = (
                        f"Author used all {self.limits.max_agent_rounds} revision agent rounds without returning a report "
                        f"({exc}); last review: {decision.summary}"
                    )
                report = last_safe_report or build_fallback_report(snapshot, reason)
                return report, ValidationDecision("fail", list(decision.findings), summary), self.trace

        if not isinstance(report.get("report_html"), str) or not str(report.get("report_html", "")).strip():
            report = build_fallback_report(snapshot, "Validation cycles were exhausted and the author never returned a usable final report payload.")
        last_summary = decision.summary if decision is not None else ""
        return report, ValidationDecision(
            "fail",
            list(decision.findings) if decision is not None else [],
            f"Validation cycles were exhausted without a passing report. Last review: {last_summary}".strip(),
        ), self.trace
