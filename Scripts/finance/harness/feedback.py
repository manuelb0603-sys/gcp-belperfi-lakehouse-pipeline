"""Revision feedback, finding keys, resolved-findings and per-cycle progress bookkeeping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..report_models import ReportSnapshot, ValidationDecision, ValidationFinding


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


def focused_feedback(
    decision: ValidationDecision, snapshot: ReportSnapshot, max_findings: int | None = MAX_REVISION_FINDINGS
) -> dict[str, Any]:
    """Prioritized error findings for the next revision, with the exact snapshot values they concern.

    Strict guidance sends the top MAX_REVISION_FINDINGS; max_findings=None (guided) sends every error.
    """
    errors = [finding for finding in decision.findings if finding.severity.lower() == "error"]
    rank = {category: index for index, category in enumerate(FEEDBACK_PRIORITY)}
    ordered = sorted(enumerate(errors), key=lambda item: (rank.get(item[1].category, len(rank)), item[0]))
    selected = [finding for _, finding in (ordered if max_findings is None else ordered[:max_findings])]
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


MAX_RESOLVED_FINDINGS = 10
RESOLVED_LABEL_MESSAGE_CHARS = 100


def _finding_label(finding: ValidationFinding) -> str:
    """Short human-readable label for a finding: category plus a truncated message."""
    message = re.sub(r"\s+", " ", str(finding.message or "")).strip()
    if len(message) > RESOLVED_LABEL_MESSAGE_CHARS:
        message = message[:RESOLVED_LABEL_MESSAGE_CHARS].rstrip() + "…"
    return f"{finding.category or 'uncategorized'}: {message}"


def _update_resolved_findings(
    resolved: dict[tuple[str, str], str],
    earlier_errors: dict[tuple[str, str], str],
    current_keys: set[tuple[str, str]],
) -> None:
    """Bookkeeping only: track earlier error findings that are absent from the latest cycle's errors.

    A finding that comes back is removed again; the list keeps the most recent MAX_RESOLVED_FINDINGS.
    """
    for key in [key for key in resolved if key in current_keys]:
        del resolved[key]
    for key, label in earlier_errors.items():
        if key not in current_keys and key not in resolved:
            resolved[key] = label
    while len(resolved) > MAX_RESOLVED_FINDINGS:
        del resolved[next(iter(resolved))]


@dataclass
class CycleProgress:
    """Error/warning counts of one review compared with the previous cycle."""

    error_findings: list[ValidationFinding]
    warning_count: int
    error_counts: dict[str, int]
    error_keys: list[tuple[str, str]]
    # Trace/prompt payload: counts, deltas, and per-category resolved/new errors.
    progress: dict[str, Any]

    @property
    def error_count(self) -> int:
        return self.progress["error_count"]

    @property
    def repeated_error_count(self) -> int | None:
        return self.progress["repeated_error_count"]

    def stalled(self, previous_error_keys: set[tuple[str, str]] | None) -> bool:
        """True when the same error findings came back unchanged from the previous cycle."""
        return bool(previous_error_keys and self.error_keys and set(self.error_keys) == previous_error_keys)


def measure_progress(
    decision: ValidationDecision,
    previous_error_counts: dict[str, int] | None,
    previous_error_keys: set[tuple[str, str]] | None,
) -> CycleProgress:
    """Count a review's errors and warnings and compare them with the previous cycle's."""
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
    return CycleProgress(error_findings, warning_count, error_counts, error_keys, progress)
