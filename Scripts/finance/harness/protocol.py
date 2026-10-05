"""Wire protocol between the harness and the models: markers, reply parsing, validator schema."""

from __future__ import annotations

import json
import re
from typing import Any

from .html_safety import _plain_text_from_html


REPORT_START_MARKER = "<<<REPORT_HTML>>>"
REPORT_END_MARKER = "<<<END_REPORT_HTML>>>"
REBUTTALS_START_MARKER = "<<<REBUTTALS>>>"
REBUTTALS_END_MARKER = "<<<END_REBUTTALS>>>"
MAX_REBUTTALS = 5
MAX_REBUTTAL_FIELD_CHARS = 400


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
