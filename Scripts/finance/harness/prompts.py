"""Instruction text the harness sends to the author and validator models."""

from __future__ import annotations

import json
from typing import Any, NamedTuple

from .html_safety import HISTORY_LABEL
from .protocol import REBUTTALS_END_MARKER, REBUTTALS_START_MARKER, REPORT_END_MARKER, REPORT_START_MARKER


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

# Snapshot-derived, month-specific checklist (finance/report_requirements.py) shown to both models.
REQUIREMENTS_HEADER = (
    "REPORT REQUIREMENTS CHECKLIST (derived from the authoritative snapshot; every item is mandatory)"
)
VALIDATOR_CHECKLIST_INSTRUCTION = (
    "Check every checklist item above explicitly against the report. List each failed item (for example a renamed "
    "or missing required heading, a wrong value, or a missing disclosure) as an error finding whose evidence quotes "
    "the report verbatim (or names the section searched when it is missing) and cites the checklist item."
)

class PromptParts(NamedTuple):
    """A model prompt split for prompt caching: stable instructions (system) and per-call content (user).

    Ollama receives ``text`` (system + user concatenated exactly, the historical single prompt);
    Anthropic sends system as a cached system block and user as the user message.
    """

    system: str
    user: str

    @property
    def text(self) -> str:
        return f"{self.system}{self.user}"


def as_prompt_parts(prompt: "str | PromptParts") -> PromptParts:
    """Accept a plain prompt string (all user content) or PromptParts."""
    return prompt if isinstance(prompt, PromptParts) else PromptParts("", str(prompt))


# Stand-in shown to the validator when the author returned no report HTML at all.
MISSING_REPORT_HTML = "<html><body><p>Missing report_html in author output.</p></body></html>"


def requirements_block(requirements: list[str] | None, for_validator: bool = False) -> str:
    """Render the checklist prompt block, or an empty string when no requirements were supplied."""
    if not requirements:
        return ""
    block = f"{REQUIREMENTS_HEADER}:\n" + "\n".join(f"- {line}" for line in requirements) + "\n\n"
    if for_validator:
        block += f"{VALIDATOR_CHECKLIST_INSTRUCTION}\n\n"
    return block


def author_base_prompt(
    prompt_context: str, objective: str, snapshot_json: str, requirements: list[str] | None, tool_prompt: str
) -> str:
    """The fixed opening of every author prompt: context, objective, snapshot, checklist, tools, protocol."""
    return (
        f"{prompt_context}\n\nOBJECTIVE:\n{objective}\n\n"
        f"AUTHORITATIVE SNAPSHOT:\n{snapshot_json}\n\n"
        f"{requirements_block(requirements)}"
        f"AVAILABLE TOOLS:\n{tool_prompt}\n\n"
        f"{AUTHOR_PROTOCOL}\n"
        "If tool evidence contains an error, use its message to correct the arguments and retry if useful; "
        "the complete conversation and earlier evidence remain in context. Do not repeat the identical failing call."
    )


def new_evidence_section(phase: str, round_number: int, results: list[dict[str, Any]]) -> str:
    """Author prompt section with the tool results from one round."""
    return (
        f"\n\nNEW TOOL EVIDENCE ({phase} round {round_number}; earlier evidence above still applies):\n"
        f"{json.dumps(results, indent=2, default=str)}"
    )


def history_label_requirement() -> str:
    """Author notice that prior-report evidence must be introduced with HISTORY_LABEL."""
    return (
        "\n\nHISTORY LABEL REQUIREMENT: get_report_history evidence was used, so the report HTML "
        f"must contain the exact text \"{HISTORY_LABEL}\" introducing any prior-report content "
        "(for example, a short clearly labeled historical note). Reports without this exact label are rejected. "
        "Never present historical figures as current."
    )


def budget_exhausted_notice() -> str:
    """Author notice that no more tools may be requested."""
    return (
        "\n\nTOOL BUDGET EXHAUSTED: do not request more tools. Return the final report now between "
        f"{REPORT_START_MARKER} and {REPORT_END_MARKER}, using the snapshot and the evidence above."
    )


def author_length_limit_error(exc: Exception) -> str:
    """Parse error shown to the author when its output was cut off by the token limit."""
    return (
        "Output hit the token limit before the response was complete; be more concise "
        f"(trim repeated inline styles and prose) but keep every required section. Parser detail: {exc}"
    )


def author_format_retry_text(details: dict[str, Any]) -> str:
    """Author prompt suffix asking it to resend malformed or truncated output."""
    return (
        "\n\nAUTHOR OUTPUT FORMAT ERROR — RETRY REQUIRED:\n"
        f"{json.dumps(details, ensure_ascii=False)}\n"
        "Recreate the response from the original objective, snapshot, tool evidence, and any validator feedback above. "
        "Either return only a JSON object with tool_calls, or return the complete report HTML between "
        f"{REPORT_START_MARKER} and {REPORT_END_MARKER} (end marker required). Do not return an explanation."
    )


def validator_length_limit_error(exc: Exception) -> str:
    """Parse error recorded when the validator's JSON was cut off by the token limit."""
    return f"Validator output hit the token limit before the JSON was complete ({exc})"


def validator_format_retry_prompt(prompt: str, error: str, length_limited: bool) -> str:
    """Validator prompt repeated with a request for one schema-conforming JSON object."""
    return f"{prompt}{validator_format_retry_suffix(error, length_limited)}"


def validator_format_retry_suffix(error: str, length_limited: bool) -> str:
    """The text validator_format_retry_prompt appends to the validator prompt (added to the user part)."""
    length_hint = (
        " Your previous answer ran out of tokens: keep the audit evidence and findings short."
        if length_limited else ""
    )
    return (
        f"\n\nFORMAT RETRY:Your previous validator response was not valid JSON "
        f"({error}). Re-evaluate the same report and return one concise, schema-conforming JSON object only. "
        f"Do not include unescaped quotation marks, markdown fences, or text outside the JSON object.{length_hint}"
    )


def sanitization_revision_instruction(category: str, size: int, max_report_bytes: int) -> str:
    """Revision instruction for a report rejected by the mechanical HTML gate."""
    if category == "history":
        return (
            f"Because get_report_history evidence was used, include the exact text \"{HISTORY_LABEL}\" "
            "before any prior-report content (a short clearly labeled historical note) and do not present "
            "historical figures as current."
        )
    if size > max_report_bytes:
        return (
            f"Return a complete report under {max_report_bytes} bytes: remove redundant inline "
            "styles and markup while keeping every required section."
        )
    return (
        "Remove script, iframe, object, embed, form, input, link, and meta elements, event-handler "
        "attributes, and javascript: URLs; use only static email-safe HTML with inline styles."
    )


def _prior_validation_text(
    previous_validation: dict[str, Any] | None, resolved_labels: list[str], rebuttals: list[dict[str, str]]
) -> str:
    """Previous cycle, resolved findings, and author rebuttals for the validator (empty on the first cycle)."""
    if previous_validation is None:
        return ""
    text = (
        "PREVIOUS VALIDATION CYCLE (context for tracking revisions only; independently re-check the current report and disagree with prior findings when the evidence warrants it):\n"
        f"{json.dumps(previous_validation, indent=2, ensure_ascii=False)}\n\n"
    )
    if resolved_labels:
        text += (
            "FINDINGS RESOLVED IN EARLIER CYCLES (you accepted these as fixed; re-flag only with new "
            "verbatim evidence from the current report):\n"
            + "\n".join(f"- {label}" for label in resolved_labels)
            + "\n\n"
        )
    if rebuttals:
        text += (
            "AUTHOR REBUTTALS (verify each quote against the report): the author says the report above "
            "already satisfies these prior findings. If a quote is present in the report and satisfies the "
            "requirement, do not repeat that finding; otherwise keep it and explain why the quote is absent or insufficient.\n"
            f"{json.dumps(rebuttals, indent=2, ensure_ascii=False)}\n\n"
        )
    return text


def validator_prompt(
    validator_context: str,
    safe_html: str,
    snapshot_json: str,
    requirements: list[str] | None,
    evidence: list[dict[str, Any]],
    previous_validation: dict[str, Any] | None,
    resolved_labels: list[str],
    rebuttals: list[dict[str, str]],
) -> str:
    """The full validator prompt for one review cycle."""
    return validator_prompt_parts(
        validator_context, safe_html, snapshot_json, requirements, evidence, previous_validation, resolved_labels, rebuttals
    ).text


def validator_prompt_parts(
    validator_context: str,
    safe_html: str,
    snapshot_json: str,
    requirements: list[str] | None,
    evidence: list[dict[str, Any]],
    previous_validation: dict[str, Any] | None,
    resolved_labels: list[str],
    rebuttals: list[dict[str, str]],
) -> PromptParts:
    """Validator prompt split into the stable validator instructions (system) and this cycle's review content."""
    return PromptParts(validator_context, (
        f"\n\nVALIDATE THIS REPORT:\n{safe_html}\n\n"
        f"AUTHORITATIVE SNAPSHOT:\n{snapshot_json}\n\n"
        f"{requirements_block(requirements, for_validator=True)}"
        f"TOOL EVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}\n\n"
        f"{_prior_validation_text(previous_validation, resolved_labels, rebuttals)}"
        "If the report HTML is missing, return fail with a revision instruction telling the author to return "
        f"the complete HTML report between {REPORT_START_MARKER} and {REPORT_END_MARKER}."
    ))


def revision_block(revision_html: str, feedback: dict[str, Any], resolved_labels: list[str]) -> str:
    """The revision request that replaces (never appends to) the previous one in the author prompt."""
    already_fixed = (
        "ALREADY FIXED IN EARLIER CYCLES (do not reintroduce):\n"
        + "\n".join(f"- {label}" for label in resolved_labels)
        + "\n"
        if resolved_labels else ""
    )
    return (
        "\n\nPREVIOUS REPORT HTML (edit this document; it failed review):\n"
        f"----- BEGIN PREVIOUS REPORT -----\n{revision_html}\n----- END PREVIOUS REPORT -----\n\n"
        f"PRIORITY FIXES FROM THE REVIEW:\n{json.dumps(feedback, indent=2, ensure_ascii=False)}\n"
        f"{already_fixed}"
        "Fix only the listed issues with minimal edits to the previous HTML and keep everything else unchanged; "
        "use the exact authoritative values given and do not reintroduce a fixed issue in another section. "
        "Then return the complete revised report between "
        f"{REPORT_START_MARKER} and {REPORT_END_MARKER}. "
        "You may request additional evidence with tool_calls before returning the report.\n\n"
        f"{REBUTTAL_PROTOCOL}"
    )
