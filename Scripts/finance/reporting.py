"""Shared finance snapshot and agentic HTML report generation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from .harness import HarnessLimits, ReportHarness, ToolSpec
from .bigquery_tools import (
    get_category_detail,
    get_report_period_spending,
    get_transaction_detail,
)
from .report_metrics import calculate_report_snapshot
from .report_requirements import build_report_requirements
from .html_text import html_to_text
from .memory_search import search_long_facts
from .model_clients import build_model_client
from .settings import PROJECT_ROOT, ReportSettings


# Stored reports are excerpted without their CSS/script/head boilerplate.
HISTORY_SKIP_TAGS = ("style", "script", "head", "title")


def _history_text(entry: dict[str, Any]) -> str:
    """Prefer stored plain text; otherwise strip markup so excerpts are not CSS boilerplate."""
    text = str(entry.get("response_text") or "")
    if not text.strip() or re.search(r"<\s*(html|body|style|div|table)\b", text, re.I):
        text = html_to_text(
            text if text.strip() else str(entry.get("response_html") or ""),
            skip_tags=HISTORY_SKIP_TAGS,
            separator=" ",
        )
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class FinanceReportResult:
    """Report-generation result shared by the email and interactive UI callers."""

    report_html: str
    plain_text: str
    snapshot: Any
    decision: Any
    trace: Any
    metrics: dict[str, Any]
    rows: list[Any]

    @property
    def passed(self) -> bool:
        return bool(self.decision and self.decision.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_html": self.report_html,
            "plain_text": self.plain_text,
            "snapshot": self.snapshot.to_dict(),
            "decision": self.decision.to_dict() if self.decision else None,
            "trace": list(self.trace.events),
            "metrics": self.metrics,
            "rows": self.rows,
        }


def _report_rows(config: Any, project_name: str, year: int | None, month: int | None) -> tuple[date | None, list[Any]]:
    period = get_report_period_spending(project_name, year=year, month=month)
    report_month = period["report_month"]
    rows = period["rows"]
    if report_month is not None and rows:
        report_month_key = report_month.isoformat()
        has_selected_month = any(
            (row.get("transaction_month") if isinstance(row, dict) else row.transaction_month) == report_month_key
            for row in rows
        )
        if not has_selected_month:
            return report_month, []
    return report_month, rows


def _build_tools(
    config: dict[str, Any] | ReportSettings,
    project_name: str,
    snapshot: Any,
    rows: list[Any],
    memory_module: Any,
    thread_key: str,
    ollama_url: str,
    model_name: str,
) -> dict[str, ToolSpec]:
    """The author's tools; memory_module is a MemoryStore (or any object with its methods) or None."""
    settings = ReportSettings.coerce(config)
    selected_month = snapshot.latest_month
    previous_month = snapshot.previous_month

    def spending_summary(_: dict[str, Any]) -> dict[str, Any]:
        return {"source": "selected_report_period", "rows": rows}

    def category_detail(arguments: dict[str, Any]) -> dict[str, Any]:
        category = str(arguments.get("category", "")).strip()
        requested_month = str(arguments.get("month", selected_month))[:10]
        if requested_month not in {selected_month, previous_month}:
            raise ValueError("Category detail is restricted to the selected report month and its predecessor")
        previous = previous_month if requested_month == selected_month else None
        return get_category_detail(project_name, category, requested_month, previous)

    def transaction_detail(arguments: dict[str, Any]) -> dict[str, Any]:
        requested_month = str(arguments.get("month", selected_month))[:10]
        if requested_month not in {selected_month, previous_month}:
            raise ValueError("Transaction detail is restricted to the selected report month and its predecessor")
        category = arguments.get("category")
        return get_transaction_detail(
            project_name,
            requested_month,
            str(category) if category else None,
            int(arguments.get("limit", 25)),
        )

    def semantic_memory(arguments: dict[str, Any]) -> dict[str, Any]:
        if memory_module is None:
            return {"retrieval": "unavailable", "facts": []}
        query_text = str(arguments.get("query") or "").strip()
        if not query_text:
            raise ValueError(
                "get_semantic_memory requires a non-empty 'query'. Retry with a focused "
                "natural-language query, or skip memory retrieval if none is needed."
            )
        embedding_model = settings.embedding_model_name
        timeout = settings.ollama_timeout
        limit = min(10, max(1, int(arguments.get("limit", 5))))
        facts, retrieval = search_long_facts(
            query_text, memory_module, ollama_url, embedding_model, timeout, limit
        )
        return {"query": query_text, "retrieval": retrieval, "facts": facts}

    def episodic_memory(_: dict[str, Any]) -> dict[str, Any]:
        if memory_module is None:
            return {"entries": []}
        thread_id = memory_module.get_or_create_thread("email", thread_key)
        return {"entries": memory_module.get_short_term(thread_id, limit=3)}

    def report_history(_: dict[str, Any]) -> dict[str, Any]:
        entries = episodic_memory({}).get("entries", [])
        reports = []
        for entry in entries:
            text = _history_text(entry)
            if text:
                reports.append({
                    "subject": entry.get("subject") or "Prior weekly report",
                    "excerpt": f"Previous report for reference: {text[:1000]} (historical; not authoritative)",
                })
        return {"reports": reports}

    return {
        "get_spending_summary": ToolSpec(
            "get_spending_summary",
            "Return spending rows for the selected report month and predecessor only.",
            spending_summary,
        ),
        "get_category_detail": ToolSpec(
            "get_category_detail",
            "Get category evidence from the selected report month and its predecessor. Arguments: category, optional month.",
            category_detail,
        ),
        "get_transaction_detail": ToolSpec(
            "get_transaction_detail",
            "Get up to 100 transaction rows from the selected report month or its predecessor. Arguments: optional month/category/limit.",
            transaction_detail,
        ),
        "get_semantic_memory": ToolSpec(
            "get_semantic_memory",
            "Hybrid-search long-term historical facts for relevant preferences, goals, or habits. These facts are not current financial evidence. Requires a focused non-empty query; if no context is needed, skip this tool.",
            semantic_memory,
        ),
        "get_episodic_memory": ToolSpec(
            "get_episodic_memory",
            "Get recent report thread context, which is historical only.",
            episodic_memory,
        ),
        "get_report_history": ToolSpec(
            "get_report_history",
            "Get labeled prior report excerpts for historical style/context only.",
            report_history,
        ),
    }


def role_config_for(config: dict[str, Any] | ReportSettings, ollama_url: str | None, model_name: str | None) -> ReportSettings:
    """Settings for build_model_client where the caller's Ollama URL and author model fill the legacy keys."""
    return ReportSettings.coerce(config).with_model_overrides(ollama_url, model_name)


def _required_amount(value: float | None, raw: Any, key: str) -> float:
    if value is None:
        raise ValueError(f"{key} is required for agent reports and must be an amount; got {raw!r}")
    return value


def generate_finance_report(
    config: dict[str, Any] | ReportSettings,
    project_name: str,
    ollama_url: str,
    model_name: str,
    memory_module: Any = None,
    thread_key: str = "finance:default",
    year: int | None = None,
    month: int | None = None,
    rows: list[Any] | None = None,
    selected_report_month: date | None = None,
    trace_listeners: list[Callable[[dict[str, Any]], None]] | None = None,
) -> FinanceReportResult:
    """Build and validate an HTML report for latest or explicitly selected finance month.

    config is a config dict or ReportSettings; memory_module is a MemoryStore (or None).
    trace_listeners receive each harness trace event as it happens (for live progress).
    """
    # The caller's model_name and ollama_url stand in for the legacy keys; model_roles (or
    # model_name/validator_model_name/ollama_options) then pick each role's provider.
    settings = role_config_for(config, ollama_url, model_name)
    if rows is None:
        selected_report_month, rows = _report_rows(settings, project_name, year, month)
    if not rows:
        raise ValueError("No finance data is available for the requested report month")
    snapshot = calculate_report_snapshot(
        rows,
        monthly_deposit=_required_amount(settings.monthly_budget, settings.monthly_budget_raw, "Monthly_budget"),
        fixed_costs=_required_amount(settings.fixed_costs, settings.fixed_costs_raw, "Fixed_costs"),
        savings_goal=settings.savings_goal_target,
        category_goals=settings.category_goals,
    )
    if selected_report_month is not None and snapshot.latest_month != selected_report_month.isoformat():
        raise ValueError("Selected report month is missing from the fetched spending rows")

    limits = HarnessLimits(**settings.limits())
    author_client = build_model_client(settings, "author")
    validator_client = build_model_client(settings, "validator")
    harness = ReportHarness(
        ollama_url=ollama_url,
        author_model=author_client.model,
        validator_model=validator_client.model,
        limits=limits,
        timeout=settings.ollama_timeout,
        author_client=author_client,
        validator_client=validator_client,
    )
    if trace_listeners:
        harness.trace.listeners.extend(trace_listeners)
    prompt_dir = PROJECT_ROOT / "agent" / "prompts"
    author_files = ["personality.system.md", "ui.system.md", "financial-settings.md"]
    prompt_context = "\n\n".join((prompt_dir / filename).read_text(encoding="utf-8") for filename in author_files)
    validator_context = (prompt_dir / "validator.system.md").read_text(encoding="utf-8")
    tools = _build_tools(settings, project_name, snapshot, rows, memory_module, thread_key, ollama_url, model_name)
    selected_label = date.fromisoformat(snapshot.latest_month).strftime("%B %Y")
    started = time.perf_counter()
    trace = harness.trace
    try:
        report, decision, trace = harness.run(
            objective=f"Create a polished financial progress HTML report for {selected_label} using only current selected-period evidence and explicitly labeled historical context.",
            snapshot=snapshot,
            prompt_context=prompt_context,
            validator_context=validator_context,
            tools=tools,
            report_requirements=build_report_requirements(snapshot),
        )
    except Exception as exc:
        trace.add("exception", phase="finance_report_generation", exception_type=type(exc).__name__, error=str(exc))
        try:
            setattr(exc, "run_trace", trace)
            setattr(exc, "report_snapshot", snapshot)
        except Exception:
            pass
        print(f"Finance report generation failed: {exc}")
        raise

    report_html = str(report.get("report_html", ""))
    plain_text = str(report.get("plain_text", ""))
    progress_events = [
        event for event in trace.events if event.get("event") == "validation_progress"
    ]
    initial_validation_errors = progress_events[0]["error_count"] if progress_events else None
    final_validation_errors = progress_events[-1]["error_count"] if progress_events else None
    metrics = {
        "model": author_client.model,
        "validator_model": validator_client.model,
        "author_provider": author_client.provider,
        "validator_provider": validator_client.provider,
        "mode": "agent",
        "status": "validated" if decision and decision.passed else "validation_failed",
        "validation": decision.decision if decision else "error",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "agent_rounds": sum(event.get("event") == "agent_round" for event in trace.events),
        "tool_calls": sum(event.get("event") == "tool_call" for event in trace.events),
        "validation_cycles": sum(event.get("event") == "validation_cycle_result" for event in trace.events),
        "validator_calls": sum(
            event.get("event") == "model_call" and event.get("role") == "validator"
            for event in trace.events
        ),
        "author_format_retries": sum(event.get("event") == "author_format_retry" for event in trace.events),
        "validator_format_retries": sum(event.get("event") == "validator_format_retry" for event in trace.events),
        "initial_validation_errors": initial_validation_errors,
        "final_validation_errors": final_validation_errors,
        "validation_error_reduction": (
            initial_validation_errors - final_validation_errors
            if initial_validation_errors is not None and final_validation_errors is not None
            else None
        ),
        "validation_cycles_improved": sum(
            (event.get("error_count_delta") or 0) > 0 for event in progress_events
        ),
        "validation_cycles_regressed": sum(
            (event.get("error_count_delta") or 0) < 0 for event in progress_events
        ),
        "report_bytes": len(report_html.encode("utf-8")),
    }
    return FinanceReportResult(report_html, plain_text, snapshot, decision, trace, metrics, rows)
