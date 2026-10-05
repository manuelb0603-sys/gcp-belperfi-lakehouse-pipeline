"""LLM-written comparison of archived agent runs (the ``eval`` model role).

gather_run() reads one run folder (run.json, metrics.json, progress.json, trace.jsonl,
final_report.txt) into a bounded summary; build_prompt() turns several summaries into a
system prompt (stable instructions) plus a user message (the run data as JSON);
compare_runs() calls the eval ModelClient and saves agent/runs/comparisons/<stamp>_<n>runs.html
with a sibling .json metadata file, then refreshes the run index. ``Scripts/compare_runs.py``
is the CLI. Only the standard library is used; the Markdown answer is converted to escaped HTML.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .budget import BudgetExceededError
from .model_clients import ModelClient, ModelReply, build_model_client
from .pricing import price_for
from .run_archive import COMPARISONS_DIR, INDEX_STYLE, NON_RUN_DIRS, format_local, run_times, update_index
from .settings import PROJECT_ROOT, ReportSettings, load_config

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "agent" / "runs"
MAX_RUNS = 10
# Total characters of run data in the user message; with the instructions this keeps a
# five-run prompt under about 60k characters.
DATA_BUDGET_CHARS = 54000
MAX_RUN_CHARS = 24000
REPORT_EXCERPT_CHARS = 1500

NOTABLE_EVENTS = {
    "validation_stalled",
    "author_rebuttals",
    "author_rebuttals_ignored",
    "run_deadline_exceeded",
    "agent_rounds_exhausted",
    "model_call_failed",
    "model_call_retry",
    "model_unavailable",
    "report_sanitization_failed",
    "author_format_retry",
    "validator_format_retry",
    "tool_error",
    "exception",
    "run_aborted",
}
# Fields kept for notable events (everything else, e.g. raw excerpts and tracebacks, is dropped).
EVENT_FIELDS = (
    "role", "phase", "cycle", "responding_to_cycle", "attempt", "error_type", "exception_type", "error",
    "reason", "repeated_error_count", "rebuttal_count", "elapsed_seconds", "limit_seconds",
    "max_agent_rounds", "category", "name", "response_chars",
)
OPTION_KEYS = ("author_options", "validator_options", "ollama_options", "model_roles", "role_options", "author_role", "validator_role")

SYSTEM_PROMPT = """You are an evaluation analyst for a personal-finance report pipeline. Each run is one attempt by an AUTHOR model to write an HTML finance report from a deterministic snapshot, reviewed by a VALIDATOR model in up to max_validation_cycles cycles. A run passes when the validator returns decision "pass"; on a fail the author revises using the top validator errors. Runs can also stop on model failures, timeouts, the run deadline (max_run_seconds), or exhausted author rounds.

You receive JSON describing several runs: timings, code version (git commit and whether the working tree had uncommitted changes), models and options, outcome metrics, per-cycle validator decisions with each error finding (category, message, evidence), notable events (stalls, rebuttals, retries, deadline stops, exceptions), per-role model-call statistics (tokens, seconds, done_reason "length" = output token limit hit), and a short excerpt of the final report text (head and tail).

Write a clear comparison in GitHub-flavored Markdown with these sections:
1. ## Summary: which run/setup performed best and why, in 3-5 sentences.
2. ## Comparison table: one Markdown table with a row per run (run id, author, validator, outcome, cycles, duration, final errors, notable issues).
3. ## Per-run diagnosis: for each run, why it passed or failed, its slowest phase, token-limit hits, retries, and stalls.
4. ## Validator findings: whether the error findings look genuine or false positives. Judge from the finding's evidence text versus the report excerpt and the other findings; say when the excerpt is too short to decide. Note findings repeated unchanged across cycles.
5. ## Author vs validator models: assess each model in its role across the runs.
6. ## Recommendations: concrete, prioritized changes (model choice per role, options such as think / num_predict / num_ctx / timeouts, prompt or validator-instruction changes).

Before attributing a difference in outcome to the models, compare the runs' code_version: runs on different commits, or with dirty=true, may differ because of code or prompt changes rather than the models, so say so and weigh model conclusions accordingly (a missing code_version is "not recorded").

Rules: cite run ids and the numbers exactly as given in the data; never invent metrics, models, or events that are not in the data; say "not recorded" when something is missing. Keep it concise (about 600-1200 words). Do not reproduce the report text beyond short quotes. Output only the Markdown."""


class CompareError(Exception):
    """A user-facing error from the comparison command (message is printed as-is)."""


# ----- reading runs -----


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _read_trace(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        pass
    return events


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _report_excerpt(text: str, limit: int) -> str:
    """Head and tail of the report text (findings often concern the Top 3 table near the end)."""
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head - 7
    return text[:head].rstrip() + "\n[...]\n" + text[-tail:].lstrip()


def load_run(run_dir: Path) -> dict[str, Any]:
    """Everything the summary needs from one run folder, unbounded (bounded later by render_run)."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run.json") or {}
    metrics = _read_json(run_dir / "metrics.json") or {}
    progress = _read_json(run_dir / "progress.json") or {}
    events = _read_trace(run_dir / "trace.jsonl")
    try:
        report_text = (run_dir / "final_report.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        report_text = ""
    times = run_times(run_dir, progress or None)
    started = next((event for event in events if event.get("event") == "run_started"), {})

    def pick(*keys: str) -> Any:
        for source in (metrics, progress, started):
            for key in keys:
                if source.get(key) not in (None, ""):
                    return source[key]
        return None

    models = {
        "author": {"provider": pick("author_provider") or "ollama", "model": pick("model", "author_model")},
        "validator": {"provider": pick("validator_provider") or "ollama", "model": pick("validator_model")},
    }
    options = {key: source[key] for source in (manifest, metrics, progress) for key in OPTION_KEYS if key in source}

    outcome_keys = (
        "status", "validation", "runtime_seconds", "agent_rounds", "tool_calls", "validation_cycles",
        "validator_calls", "author_format_retries", "validator_format_retries", "initial_validation_errors",
        "final_validation_errors", "validation_error_reduction", "validation_cycles_improved",
        "validation_cycles_regressed", "report_bytes", "exception_type",
    )
    outcome = {key: metrics[key] for key in outcome_keys if key in metrics}
    if not metrics:
        outcome = {"status": progress.get("status") or manifest.get("status"), "tool_calls": progress.get("tool_calls")}
    for key in ("delivery", "note", "error"):
        if progress.get(key):
            outcome[f"progress_{key}"] = progress[key]
    if manifest.get("error") and "error" not in outcome:
        outcome["error"] = manifest["error"]

    # Validation cycles: decision, counts, and error findings.
    cycles: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "validation_cycle_result":
            continue
        findings = event.get("findings") if isinstance(event.get("findings"), list) else []
        errors = [
            {"category": item.get("category"), "message": item.get("message"), "evidence": item.get("evidence")}
            for item in findings
            if isinstance(item, dict) and item.get("severity", "error") == "error"
        ]
        cycles.append({
            "cycle": event.get("cycle"),
            "decision": event.get("decision"),
            "stage": event.get("stage"),
            "error_count": event.get("error_count"),
            "warning_count": event.get("warning_count"),
            "repeated_error_count": event.get("repeated_error_count"),
            "resolved_finding_count": event.get("resolved_finding_count"),
            "summary": event.get("summary"),
            "errors": errors,
        })

    # Model calls: pair model_call (prompt size) with the following model_response of the same role.
    calls: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for event in events:
        name = event.get("event")
        role = str(event.get("role") or "")
        if name == "model_call":
            pending[role] = {"role": role, "prompt_chars": event.get("prompt_chars"), "cycle": event.get("validation_cycle")}
        elif name in {"model_response", "model_call_failed"}:
            call = pending.pop(role, {"role": role})
            call.update({
                "seconds": event.get("elapsed_seconds"),
                "in_tokens": event.get("prompt_eval_count"),
                "out_tokens": event.get("eval_count"),
                "done_reason": event.get("done_reason") if name == "model_response" else "failed",
                "response_chars": event.get("response_chars"),
            })
            if name == "model_response" and event.get("parse_status") not in (None, "ok"):
                call["parse_status"] = event.get("parse_status")
            usage = event.get("usage")
            if isinstance(usage, dict) and usage.get("cache_read_input_tokens"):
                call["cache_read_tokens"] = usage.get("cache_read_input_tokens")
            calls.append({key: value for key, value in call.items() if value is not None})
    for call in pending.values():  # a call that never answered (killed or timed out without an event)
        calls.append({**{key: value for key, value in call.items() if value is not None}, "done_reason": "no_response"})

    notable = []
    for event in events:
        if event.get("event") in NOTABLE_EVENTS:
            item = {"event": event.get("event")}
            item.update({key: event[key] for key in EVENT_FIELDS if event.get(key) not in (None, "", [], {})})
            if event.get("event") == "author_rebuttals":
                item["findings"] = [str(rebuttal.get("finding", "")) for rebuttal in event.get("rebuttals") or [] if isinstance(rebuttal, dict)]
            notable.append(item)
    resolved_total = 0
    for event in events:
        if event.get("event") == "validation_progress" and isinstance(event.get("resolved_findings"), list):
            resolved_total = len(event["resolved_findings"])
    tools: dict[str, int] = {}
    for event in events:
        if event.get("event") == "tool_call":
            tools[str(event.get("name"))] = tools.get(str(event.get("name")), 0) + 1

    code_version = next(
        (source["code_version"] for source in (manifest, progress) if isinstance(source.get("code_version"), dict)),
        None,
    )
    limits = started.get("limits") or manifest.get("harness_limits") or progress.get("harness_limits")

    return {
        "run_id": run_dir.name,
        "start": format_local(times["start"]),
        "end": "running" if times["running"] else format_local(times["end"]),
        "duration_seconds": times["duration_seconds"],
        "code_version": code_version,
        "models": models,
        "options": options,
        "limits": limits,
        "deadline_seconds": started.get("deadline_seconds"),
        "outcome": outcome,
        "cycles": cycles,
        "model_calls": calls,
        "events": notable,
        "resolved_findings_at_end": resolved_total,
        "tools_used": tools,
        "report_chars": len(report_text),
        "report_text": report_text,
    }


def _call_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for call in calls:
        role = call.get("role") or "unknown"
        entry = totals.setdefault(role, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0, "in_tokens": 0, "out_tokens": 0, "max_out_tokens": 0, "length_stops": 0, "failed": 0})
        entry["calls"] += 1
        seconds = _number(call.get("seconds")) or 0.0
        entry["seconds"] = round(entry["seconds"] + seconds, 1)
        entry["max_seconds"] = round(max(entry["max_seconds"], seconds), 1)
        entry["in_tokens"] += int(_number(call.get("in_tokens")) or 0)
        out_tokens = int(_number(call.get("out_tokens")) or 0)
        entry["out_tokens"] += out_tokens
        entry["max_out_tokens"] = max(entry["max_out_tokens"], out_tokens)
        entry["length_stops"] += call.get("done_reason") == "length"
        entry["failed"] += call.get("done_reason") in {"failed", "no_response"}
    return totals


# Progressively tighter limits; render_run uses the first level whose JSON fits the budget.
_LEVELS = [
    {"message": 300, "evidence": 300, "summary": 300, "errors": 6, "calls": 24, "events": 15, "excerpt": REPORT_EXCERPT_CHARS},
    {"message": 220, "evidence": 200, "summary": 200, "errors": 4, "calls": 12, "events": 10, "excerpt": 1200},
    {"message": 160, "evidence": 140, "summary": 120, "errors": 3, "calls": 6, "events": 8, "excerpt": 900},
    {"message": 120, "evidence": 90, "summary": 0, "errors": 2, "calls": 0, "events": 5, "excerpt": 600},
    {"message": 90, "evidence": 0, "summary": 0, "errors": 1, "calls": 0, "events": 3, "excerpt": 300},
]


def render_run(raw: dict[str, Any], level: int = 0) -> dict[str, Any]:
    """Bounded, JSON-ready summary of a loaded run at one truncation level."""
    limits = _LEVELS[min(level, len(_LEVELS) - 1)]
    cycles = []
    previous_keys: list[tuple[str, str]] | None = None
    previous_cycle = None
    previous_summary = None
    for cycle in raw["cycles"]:
        keys = [(str(item.get("category")), str(item.get("message"))) for item in cycle["errors"]]
        item = {key: cycle[key] for key in ("cycle", "decision", "stage", "error_count", "warning_count", "repeated_error_count", "resolved_finding_count") if cycle.get(key) is not None}
        if limits["summary"] and cycle.get("summary"):
            if cycle["summary"] == previous_summary:
                item["summary_same_as_cycle"] = previous_cycle
            else:
                item["summary"] = _clip(cycle["summary"], limits["summary"])
        previous_summary = cycle.get("summary")
        if keys and keys == previous_keys:
            item["errors_same_as_cycle"] = previous_cycle
        else:
            errors = []
            for finding in cycle["errors"][: limits["errors"]]:
                entry = {"category": finding.get("category"), "message": _clip(finding.get("message"), limits["message"])}
                if limits["evidence"]:
                    entry["evidence"] = _clip(finding.get("evidence"), limits["evidence"])
                errors.append(entry)
            if errors:
                item["errors"] = errors
            if len(cycle["errors"]) > limits["errors"]:
                item["errors_omitted"] = len(cycle["errors"]) - limits["errors"]
        previous_keys, previous_cycle = keys, cycle.get("cycle")
        cycles.append(item)
    calls = raw["model_calls"]
    summary: dict[str, Any] = {
        "run_id": raw["run_id"],
        "start": raw["start"],
        "end": raw["end"],
        "duration_seconds": raw["duration_seconds"],
        "code_version": raw.get("code_version") or "not recorded",
        "models": raw["models"],
    }
    if raw["options"]:
        summary["options"] = raw["options"]
    if raw["limits"]:
        summary["limits"] = raw["limits"]
    summary["outcome"] = {key: (_clip(value, 300) if isinstance(value, str) else value) for key, value in raw["outcome"].items()}
    summary["validation_cycles"] = cycles
    summary["model_call_totals"] = _call_totals(calls)
    if limits["calls"] and calls:
        if len(calls) > limits["calls"]:
            half = limits["calls"] // 2
            summary["model_calls"] = calls[:half] + [{"omitted_calls": len(calls) - 2 * half}] + calls[-half:]
        else:
            summary["model_calls"] = calls
    events = [{key: (_clip(value, 200) if isinstance(value, str) else value) for key, value in event.items()} for event in raw["events"]]
    for event in events:
        if "findings" in event:
            event["findings"] = [_clip(finding, 100) for finding in event["findings"][:3]]
    if events:
        summary["events"] = events[: limits["events"]]
        if len(events) > limits["events"]:
            summary["events_omitted"] = len(events) - limits["events"]
    if raw["resolved_findings_at_end"]:
        summary["resolved_findings_at_end"] = raw["resolved_findings_at_end"]
    if raw["tools_used"]:
        summary["tools_used"] = raw["tools_used"]
    summary["report_chars"] = raw["report_chars"]
    summary["report_excerpt"] = _report_excerpt(raw["report_text"], limits["excerpt"])
    return summary


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, indent=1, default=str))


def gather_run(run_dir: Path, budget: int = MAX_RUN_CHARS) -> dict[str, Any]:
    """Bounded summary of one run whose JSON stays within budget characters when possible."""
    raw = load_run(run_dir)
    for level in range(len(_LEVELS)):
        summary = render_run(raw, level)
        if _size(summary) <= budget:
            return summary
    return summary


def resolve_runs(run_ids: list[str], runs_root: Path) -> list[Path]:
    """Run folders for the given IDs (folder names or paths); raises CompareError listing unknown IDs."""
    if len(run_ids) < 2:
        raise CompareError("Give at least two run IDs to compare.")
    if len(run_ids) > MAX_RUNS:
        raise CompareError(f"Compare at most {MAX_RUNS} runs at a time (got {len(run_ids)}).")
    folders: list[Path] = []
    unknown: list[str] = []
    for run_id in run_ids:
        name = Path(run_id.strip().rstrip("\\/")).name
        folder = Path(runs_root) / name
        if not name or name in NON_RUN_DIRS or not folder.is_dir():
            unknown.append(run_id)
        elif folder not in folders:
            folders.append(folder)
    if unknown:
        raise CompareError(f"Unknown run ID(s) in {runs_root}: " + ", ".join(unknown))
    if len(folders) < 2:
        raise CompareError("Give at least two different run IDs to compare.")
    return folders


def build_prompt(runs: list[dict[str, Any]]) -> tuple[str, str]:
    """(system, user) for the eval model; the user part is the run data as JSON."""
    data = {"run_count": len(runs), "runs": runs}
    user = (
        "Compare these finance-report agent runs. Times are local; durations and model-call seconds "
        "are wall-clock seconds; tokens are provider counts (in = prompt, out = output).\n\nRUN DATA (JSON):\n"
        + json.dumps(data, ensure_ascii=False, indent=1, default=str)
    )
    return SYSTEM_PROMPT, user


def gather_runs(folders: list[Path]) -> list[dict[str, Any]]:
    budget = max(3000, min(MAX_RUN_CHARS, DATA_BUDGET_CHARS // max(1, len(folders))))
    return [gather_run(folder, budget) for folder in folders]


# ----- Markdown -> safe HTML -----

_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\*\w])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _inline(text: str) -> str:
    """Escape text and apply code spans, bold, and italic. Links keep only their text."""
    parts = []
    position = 0
    for match in _INLINE_CODE.finditer(text):
        parts.append(_emphasis(text[position:match.start()]))
        parts.append(f"<code>{html.escape(match.group(2).strip(), quote=True)}</code>")
        position = match.end()
    parts.append(_emphasis(text[position:]))
    return "".join(parts)


def _emphasis(text: str) -> str:
    text = _LINK.sub(lambda match: match.group(1), text)
    escaped = html.escape(text, quote=True)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    return _ITALIC.sub(r"<em>\1</em>", escaped)


def _table_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def markdown_to_html(markdown: str) -> str:
    """Convert a Markdown subset (headings, lists, tables, code, bold/italic, quotes, rules) to HTML.

    Every piece of source text is HTML-escaped; the only tags in the output are the ones this
    function writes, without attributes, so raw HTML, scripts, and links never pass through.
    """
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    list_stack: list[tuple[str, int]] = []  # (tag, indent)
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(_inline(line) for line in paragraph) + "</p>")
            paragraph.clear()

    def close_lists(indent: int = -1) -> None:
        while list_stack and list_stack[-1][1] > indent:
            out.append(f"</li></{list_stack.pop()[0]}>")

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_paragraph()
            close_lists()
            fence = stripped[:3]
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(lines[index])
                index += 1
            out.append("<pre><code>" + html.escape("\n".join(code), quote=True) + "</code></pre>")
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            # A blank line inside a list keeps the list open only if the next line continues it.
            next_line = next((item for item in lines[index + 1:] if item.strip()), "")
            if list_stack and not _LIST_ITEM.match(next_line):
                close_lists()
            index += 1
            continue
        heading = _HEADING.match(stripped)
        if heading:
            flush_paragraph()
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue
        if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", stripped):
            flush_paragraph()
            close_lists()
            out.append("<hr>")
            index += 1
            continue
        if "|" in stripped and index + 1 < len(lines) and _TABLE_RULE.match(lines[index + 1]):
            flush_paragraph()
            close_lists()
            header = _table_cells(stripped)
            rows = []
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_table_cells(lines[index]))
                index += 1
            parts = ["<table><thead><tr>" + "".join(f"<th>{_inline(cell)}</th>" for cell in header) + "</tr></thead><tbody>"]
            for row in rows:
                row = (row + [""] * len(header))[: max(len(header), 1)]
                parts.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>")
            parts.append("</tbody></table>")
            out.append("".join(parts))
            continue
        item = _LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            indent = len(item.group(1).replace("\t", "    "))
            tag = "ol" if item.group(2)[0].isdigit() else "ul"
            if list_stack and indent > list_stack[-1][1]:
                out.append(f"<{tag}><li>")
                list_stack.append((tag, indent))
            else:
                close_lists(indent)
                if list_stack and list_stack[-1][1] == indent and list_stack[-1][0] == tag:
                    out.append("</li><li>")
                else:
                    if list_stack and list_stack[-1][1] == indent:
                        out.append(f"</li></{list_stack.pop()[0]}>")
                    out.append(f"<{tag}><li>")
                    list_stack.append((tag, indent))
            out.append(_inline(item.group(3)))
            index += 1
            continue
        if list_stack and line[:1] in (" ", "\t"):
            out.append(" " + _inline(stripped))  # continuation of a list item
            index += 1
            continue
        close_lists()
        if stripped.startswith(">"):
            flush_paragraph()
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            out.append("<blockquote>" + "<br>".join(_inline(text) for text in quote) + "</blockquote>")
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    close_lists()
    return "\n".join(out)


# ----- cost, pages, metadata -----


def estimate_cost(provider: str | None, model: str | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """USD estimate for an Anthropic call (all input tokens at the base input price; finance.pricing table)."""
    if provider != "anthropic":
        return None
    price = price_for(model)
    if input_tokens is None or output_tokens is None:
        return None
    return round(input_tokens / 1_000_000 * price[0] + output_tokens / 1_000_000 * price[1], 4)


PAGE_STYLE = INDEX_STYLE + (
    " main{max-width:1100px} .meta{margin:.2rem 0} pre{background:#fff1f5;padding:.8rem;overflow:auto}"
    " code{font:13px ui-monospace,Consolas,monospace} blockquote{border-left:4px solid #e8ccd8;margin:0;padding:.2rem 1rem}"
    " .warning{background:#fff4e5;border:1px solid #f0c27b;padding:.6rem}"
)


def render_page(meta: dict[str, Any], body_html: str) -> str:
    created = format_local(datetime.fromisoformat(meta["created_at"])) if meta.get("created_at") else ""
    run_links = ", ".join(
        f'<a href="../{html.escape(run_id, quote=True)}/final_report.html">{html.escape(run_id)}</a>' for run_id in meta["run_ids"]
    )
    usage = meta.get("usage") or {}
    tokens = ""
    if meta.get("input_tokens") is not None or meta.get("output_tokens") is not None:
        tokens = f" Tokens in/out: {meta.get('input_tokens', '?')}/{meta.get('output_tokens', '?')}."
    if usage.get("cache_read_input_tokens"):
        tokens += f" Cache read: {usage['cache_read_input_tokens']}."
    cost = f" Estimated cost: ${meta['estimated_cost_usd']:.4f}." if meta.get("estimated_cost_usd") is not None else ""
    warning = ""
    if meta.get("done_reason") == "length":
        warning = '<p class="warning">The eval model hit its output token limit; this comparison may be cut off.</p>'
    title = f"Run comparison ({len(meta['run_ids'])} runs)"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{PAGE_STYLE}</style></head>
<body><main><p><a href="../index.html">&larr; All runs</a></p>
<h1>{html.escape(title)}</h1>
<p class="meta">Created {html.escape(created)} (local) by {html.escape(str(meta.get('eval_provider')))}:{html.escape(str(meta.get('eval_model')))}.</p>
<p class="meta">Runs: {run_links}</p>
<p class="meta muted">Prompt: {meta.get('prompt_chars', 0):,} characters.{html.escape(tokens)}{html.escape(cost)} Stop: {html.escape(str(meta.get('done_reason')))}. Written by a model from run metrics and traces; verify important claims against the runs.</p>
{warning}
<div class="panel">
{body_html}
</div></main></body></html>"""


def _output_paths(folder: Path, created: datetime, run_count: int) -> tuple[Path, Path]:
    stamp = created.astimezone().strftime("%Y%m%dT%H%M%S")
    base = f"{stamp}_{run_count}runs"
    suffix = 1
    while (folder / f"{base}.html").exists() or (folder / f"{base}.json").exists():
        suffix += 1
        base = f"{stamp}_{run_count}runs_{suffix}"
    return folder / f"{base}.html", folder / f"{base}.json"


def compare_runs(
    run_ids: list[str],
    *,
    config: dict[str, Any] | ReportSettings,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    client: ModelClient | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    echo: Callable[[str], None] = print,
) -> Path:
    """Gather the runs, ask the eval model for a comparison, and save the HTML + JSON pages.

    Raises CompareError (nothing saved) when a run is unknown or the model fails or refuses.
    """
    settings = ReportSettings.coerce(config)
    folders = resolve_runs(run_ids, runs_root)
    runs = gather_runs(folders)
    system, user = build_prompt(runs)
    if client is None:
        client = build_model_client(settings, "eval", run_id="compare")
    spec = settings.role("eval")
    provider = getattr(client, "provider", spec.get("provider"))
    model = getattr(client, "model", spec.get("model"))
    echo(f"Comparing {len(folders)} runs with {provider}:{model} ({len(system) + len(user):,} prompt characters)...")
    try:
        reply = client.generate(user, system=system, timeout=spec.get("timeout"))
    except BudgetExceededError as exc:
        raise CompareError(f"{exc}. Nothing was saved.") from exc
    except Exception as exc:  # any provider/network failure: report it and save nothing
        raise CompareError(f"Eval model call failed ({provider}:{model}): {type(exc).__name__}: {exc}") from exc
    done_reason = getattr(reply, "done_reason", None)
    if done_reason == "refusal":
        details = (getattr(reply, "metadata", None) or {}).get("stop_details")
        raise CompareError(f"The eval model ({provider}:{model}) declined to answer; nothing was saved." + (f" Details: {details}" if details else ""))
    markdown = str(reply or "").strip()
    if not markdown:
        reason = f" (stop reason: {done_reason})" if done_reason else ""
        raise CompareError(f"The eval model ({provider}:{model}) returned no text{reason}; nothing was saved.")
    usage = getattr(reply, "usage", None)
    input_tokens = getattr(reply, "prompt_eval_count", None)
    output_tokens = getattr(reply, "eval_count", None)
    if isinstance(usage, dict) and usage.get("input_tokens") is not None:
        input_tokens = sum(int(usage.get(key) or 0) for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    created = now()
    folder = Path(runs_root) / COMPARISONS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    html_path, json_path = _output_paths(folder, created, len(folders))
    meta = {
        "created_at": created.isoformat(),
        "run_ids": [path.name for path in folders],
        "eval_provider": provider,
        "eval_model": model,
        "prompt_chars": len(system) + len(user),
        "system_chars": len(system),
        "user_chars": len(user),
        "done_reason": done_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usage": usage,
        # The cost the guarded Claude client recorded in the ledger, else an estimate from the token counts.
        "estimated_cost_usd": (
            round(recorded_cost, 4) if isinstance(recorded_cost := (getattr(reply, "metadata", None) or {}).get("cost_usd"), (int, float))
            else estimate_cost(provider, model, input_tokens, output_tokens)
        ),
        "html_file": html_path.name,
        "markdown": markdown,
    }
    html_path.write_text(render_page(meta, markdown_to_html(markdown)), encoding="utf-8")
    json_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    update_index(Path(runs_root))
    if done_reason == "length":
        echo("Warning: the eval model hit its output token limit; the comparison may be cut off.")
    if meta["estimated_cost_usd"] is not None:
        echo(f"Tokens in/out: {input_tokens}/{output_tokens}; estimated cost ${meta['estimated_cost_usd']:.4f}")
    return html_path


# ----- CLI -----


def _resolve_config_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() or path.exists() else PROJECT_ROOT / path


def main(argv: list[str] | None = None, *, client: ModelClient | None = None, runs_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_runs.py",
        description="Ask the eval model (model_roles.eval, else the validator) to compare archived agent runs.",
    )
    parser.add_argument("run_ids", nargs="+", metavar="RUN_ID", help="run folder names under agent/runs (at least two)")
    parser.add_argument("--config", help="config file (default: config.json in the project root)")
    parser.add_argument("--dry-run-prompt", action="store_true", help="print the prompt and exit without calling the model")
    parser.add_argument("--runs-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.runs_dir) if args.runs_dir else (runs_root or DEFAULT_RUNS_ROOT)
    try:
        folders = resolve_runs(args.run_ids, root)
        if args.dry_run_prompt:
            system, user = build_prompt(gather_runs(folders))
            print("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user + "\n")
            total = len(system) + len(user)
            line = f"Prompt: {total:,} characters (system {len(system):,}, user {len(user):,}), about {total // 4:,} tokens."
            try:
                spec = ReportSettings.from_config(load_config(_resolve_config_path(args.config))).role("eval")
                line += f" Eval model: {spec['provider']}:{spec['model']}."
                if spec["provider"] == "anthropic":
                    max_cost = estimate_cost("anthropic", spec["model"], total // 4, spec.get("max_tokens"))
                    if max_cost is not None:
                        line += f" Estimated cost at most ${max_cost:.4f} (input estimate plus max_tokens output)."
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                line += f" (Eval model unknown: {exc})"
            print(line)
            print("No model call was made.")
            return 0
        settings = ReportSettings.from_config(load_config(_resolve_config_path(args.config)))
        if settings.role("eval")["provider"] == "anthropic":
            print("Note: run summaries (findings, metrics, report excerpts) are sent to the Claude API.")
        output = compare_runs([folder.name for folder in folders], config=settings, runs_root=root, client=client)
    except CompareError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Comparison saved: {output}")
    return 0
