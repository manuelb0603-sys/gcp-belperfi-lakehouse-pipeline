"""Persist agent run artifacts and maintain a browser-openable run index."""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .budget import COST_EVENTS, BudgetExceededError, CostLedger, status_line as budget_status_line
from .log import get_logger
from .settings import DEFAULTS, PROJECT_ROOT

logger = get_logger(__name__)

# Where report runs are archived (the email workflow, replays, and comparisons default to it).
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "agent" / "runs"


def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10, check=True
    )
    return completed.stdout


def git_code_version(cwd: Path | None = None, runner: Callable[[list[str], Path], str] = _run_git) -> dict[str, Any]:
    """Short commit hash and dirty flag of the working tree; tolerates git being missing or failing.

    Returns {"commit": "abc1234", "dirty": False}, or {"commit": None, "dirty": None, "error": ...}.
    """
    cwd = Path(cwd) if cwd is not None else PROJECT_ROOT
    try:
        commit = runner(["rev-parse", "--short", "HEAD"], cwd).strip() or None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"commit": None, "dirty": None, "error": f"git unavailable: {type(exc).__name__}"}
    try:
        dirty: bool | None = bool(runner(["status", "--porcelain"], cwd).strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


def collect_run_metadata(settings: Any, code_version: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run-start metadata for run.json/progress.json: code version, guidance, author/validator role settings, limits.

    settings is a finance.settings.ReportSettings. Role settings hold provider, model, options,
    think, and timeouts only; server URLs and credentials are never included. guidance is the
    effective mode; harness_limits are the configured values (guided runs double the tool budgets,
    recorded as effective_harness_limits).
    """
    from .harness.trace import HarnessLimits, effective_limits, limits_dict  # lazy: keeps run_archive imports light

    metadata = {
        "code_version": code_version if code_version is not None else git_code_version(),
        "guidance": settings.guidance,
        "author_role": settings.public_role("author"),
        "validator_role": settings.public_role("validator"),
        "harness_limits": settings.limits(),
    }
    effective = limits_dict(effective_limits(HarnessLimits(**settings.limits()), settings.guidance))
    if effective != metadata["harness_limits"]:
        metadata["effective_harness_limits"] = effective
    return metadata


def metrics_from_trace(events: list[dict[str, Any]]) -> dict[str, int]:
    """Run counts derived from trace events: agent rounds, tool calls, validation cycles, validator calls."""
    return {
        "agent_rounds": sum(event.get("event") == "agent_round" for event in events),
        "tool_calls": sum(event.get("event") == "tool_call" for event in events),
        "validation_cycles": sum(event.get("event") == "validation_cycle_result" for event in events),
        "validator_calls": sum(
            event.get("event") == "model_call" and event.get("role") == "validator"
            for event in events
        ),
    }


def save_run(
    root: Path,
    run_id: str,
    manifest: dict[str, Any],
    trace: list[dict[str, Any]],
    report_html: str,
    report_text: str,
    metrics: dict[str, Any],
    ledger_path: Path | None = None,
) -> Path:
    """Write the run's files and refresh the index (ledger_path: see update_index)."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (run_dir / "trace.jsonl").write_text("".join(json.dumps(item, default=str) + "\n" for item in trace), encoding="utf-8")
    (run_dir / "final_report.html").write_text(report_html, encoding="utf-8")
    (run_dir / "final_report.txt").write_text(report_text, encoding="utf-8")
    update_index(root, ledger_path=ledger_path)
    return run_dir


COMPARISONS_DIR = "comparisons"
# Validator replays (finance/replay.py) save one JSON result per call here.
REPLAYS_DIR = "replays"
# Folders under agent/runs that hold tool output rather than runs.
NON_RUN_DIRS = frozenset({COMPARISONS_DIR, REPLAYS_DIR})
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_TIMESTAMP_PATTERN = re.compile(rb'"timestamp":\s*"?(-?[0-9]+(?:\.[0-9]+)?)')
_LINE_HEAD_BYTES = 4096
_BACKSCAN_LIMIT_BYTES = 8 * 1024 * 1024


def _line_timestamp(head: bytes) -> float | None:
    """Epoch seconds from the start of a trace line (json.dumps writes "timestamp" second)."""
    match = _TIMESTAMP_PATTERN.search(head)
    return float(match.group(1)) if match else None


def trace_first_last_timestamps(trace_path: Path) -> tuple[float | None, float | None]:
    """Timestamps of the first and last trace.jsonl lines, reading only the heads of those lines."""
    try:
        with Path(trace_path).open("rb") as handle:
            first = _line_timestamp(handle.readline(_LINE_HEAD_BYTES))
            size = handle.seek(0, os.SEEK_END)
            # Skip trailing newlines, then scan backwards for the newline before the last line.
            end = size
            while end > 0:
                handle.seek(end - 1)
                if handle.read(1) not in (b"\n", b"\r"):
                    break
                end -= 1
            position = end
            line_start = 0
            while position > 0 and end - position <= _BACKSCAN_LIMIT_BYTES:
                read_from = max(0, position - 65536)
                handle.seek(read_from)
                newline = handle.read(position - read_from).rfind(b"\n")
                if newline >= 0:
                    line_start = read_from + newline + 1
                    break
                position = read_from
            handle.seek(line_start)
            last = _line_timestamp(handle.read(min(_LINE_HEAD_BYTES, max(0, end - line_start))))
    except OSError:
        return None, None
    return first, last


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _from_epoch(value: float | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def run_times(run_dir: Path, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start/end of a run (aware datetimes), whether it is still running, and its duration.

    progress.json supplies started_at, and updated_at once the status is final. Older runs
    without progress.json use the first and last trace.jsonl timestamps (epoch seconds).
    """
    run_dir = Path(run_dir)
    if progress is None and (run_dir / "progress.json").exists():
        try:
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = None
    start = end = None
    running = False
    if isinstance(progress, dict):
        start = _parse_iso(progress.get("started_at"))
        if progress.get("status") != "running":
            end = _parse_iso(progress.get("updated_at"))
        elif not (run_dir / "metrics.json").exists():
            running = True
    if start is None or (end is None and not running):
        first, last = trace_first_last_timestamps(run_dir / "trace.jsonl")
        start = start or _from_epoch(first)
        if end is None and not running:
            end = _from_epoch(last)
    duration = round((end - start).total_seconds(), 1) if start and end else None
    return {"start": start, "end": end, "running": running, "duration_seconds": duration}


def format_local(value: datetime | None) -> str:
    """Local-time display string (YYYY-MM-DD HH:MM:SS), or "" when unknown."""
    return value.astimezone().strftime(TIME_FORMAT) if value else ""


def list_comparisons(root: Path) -> list[dict[str, Any]]:
    """Comparison metadata from <root>/comparisons/*.json, newest first (adds _html and _sort)."""
    items: list[dict[str, Any]] = []
    folder = Path(root) / COMPARISONS_DIR
    if not folder.is_dir():
        return items
    for path in folder.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        meta["_html"] = Path(str(meta.get("html_file") or path.with_suffix(".html").name)).name
        meta["_sort"] = str(meta.get("created_at") or path.stem)
        items.append(meta)
    items.sort(key=lambda item: item["_sort"], reverse=True)
    return items


def list_replays(root: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Validator replay results from <root>/replays/*.json, newest first (at most limit; adds _file)."""
    folder = Path(root) / REPLAYS_DIR
    if not folder.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json"), reverse=True)[:limit]:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict):
            meta["_file"] = path.name
            items.append(meta)
    return items


INDEX_STYLE = (
    "body{font:14px system-ui,sans-serif;margin:2rem;background:#fff8fb;color:#33252b}"
    " table{border-collapse:collapse;width:100%;background:white}"
    " th,td{border:1px solid #e8ccd8;padding:.55rem;text-align:left;vertical-align:top}"
    " th{background:#f7dce8} a{color:#8d315b} button{margin:1rem 0;padding:.5rem .8rem}"
    " .panel{background:white;border:1px solid #e8ccd8;padding:1rem;margin-bottom:1rem}"
    " .muted{color:#7a6570} .nowrap{white-space:nowrap} tr.diff td{background:#fff1f5}"
    " .table-scroll{overflow-x:auto;max-width:100%} .table-scroll table{width:auto;min-width:100%}"
    " #comparison{overflow-x:auto} #comparison td,#comparison th{white-space:nowrap}"
    " @media (max-width:700px){body{margin:1rem}}"
    " #compare-command{width:100%;font:13px ui-monospace,Consolas,monospace;padding:.4rem;box-sizing:border-box}"
)

INDEX_SCRIPT = r"""
const keys = ['start','end','author model','validator model','guidance','mode','status','validation','cost','runtime_seconds','agent_rounds','tool_calls','validation_cycles','validator_calls','author_format_retries','validator_format_retries','initial_validation_errors','final_validation_errors','validation_error_reduction','validation_cycles_improved','validation_cycles_regressed','report_bytes'];
function cell(tag, text) { const el = document.createElement(tag); el.textContent = text; return el; }
function quoteArg(id) { return /^[A-Za-z0-9_.\-]+$/.test(id) ? id : '"' + id.replace(/"/g, '') + '"'; }
function compareSelected() {
    const target = document.getElementById('comparison');
    const panel = document.getElementById('command-panel');
    const ids = [...document.querySelectorAll('.run-select:checked')].map(item => item.value);
    target.replaceChildren();
    if (ids.length < 2) { target.textContent = 'Select at least two runs.'; panel.hidden = true; return; }
    const table = document.createElement('table');
    const head = document.createElement('tr');
    head.appendChild(cell('th', 'metric'));
    ids.forEach(id => head.appendChild(cell('th', id)));
    table.appendChild(document.createElement('thead')).appendChild(head);
    const body = table.appendChild(document.createElement('tbody'));
    keys.forEach(key => {
        const values = ids.map(id => { const value = (metrics[id] || {})[key]; return value === null || value === undefined ? '' : String(value); });
        if (values.every(value => value === '')) return;
        const row = document.createElement('tr');
        if (key !== 'start' && key !== 'end' && new Set(values).size > 1) row.className = 'diff';
        row.appendChild(cell('th', key));
        values.forEach(value => row.appendChild(cell('td', value)));
        body.appendChild(row);
    });
    target.appendChild(table);
    target.appendChild(cell('p', 'Shaded rows differ between the selected runs.')).className = 'muted';
    document.getElementById('compare-command').value = '..\\.venv\\Scripts\\python.exe Scripts\\reports\\compare_runs.py ' + ids.map(quoteArg).join(' ');
    document.getElementById('copy-status').textContent = '';
    panel.hidden = false;
}
function copyCommand() {
    const input = document.getElementById('compare-command');
    const status = document.getElementById('copy-status');
    const fallback = () => { input.focus(); input.select(); status.textContent = 'Copy is unavailable here; the command is selected, press Ctrl+C.'; };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(input.value).then(() => { status.textContent = 'Copied.'; }, fallback);
    } else {
        fallback();
    }
}
"""


# Display columns derived per run by update_index (also compared side by side in the page).
AUTHOR_MODEL_COLUMN = "author model"
VALIDATOR_MODEL_COLUMN = "validator model"
GUIDANCE_COLUMN = "guidance"
_TRACE_HEAD_LINES = 5
_TRACE_HEAD_LINE_BYTES = 65536
INDEX_HEADERS = [
    "run_id", "start", "end", AUTHOR_MODEL_COLUMN, VALIDATOR_MODEL_COLUMN, GUIDANCE_COLUMN, "mode", "status",
    "validation", "cost", "runtime_seconds", "agent_rounds", "tool_calls", "validation_cycles", "validator_calls",
    "author_format_retries", "validator_format_retries", "initial_validation_errors", "final_validation_errors",
    "validation_error_reduction", "validation_cycles_improved", "validation_cycles_regressed", "report_bytes",
]
_NOWRAP_COLUMNS = frozenset({"start", "end", AUTHOR_MODEL_COLUMN, VALIDATOR_MODEL_COLUMN, GUIDANCE_COLUMN})


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def trace_run_started(trace_path: Path) -> dict[str, Any] | None:
    """The run_started event from the first few trace.jsonl lines (bounded read), or None."""
    try:
        with Path(trace_path).open("rb") as handle:
            for _ in range(_TRACE_HEAD_LINES):
                line = handle.readline(_TRACE_HEAD_LINE_BYTES)
                if not line:
                    break
                if b"run_started" not in line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("event") == "run_started":
                    return event
    except OSError:
        return None
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def run_role_models(
    run_dir: Path,
    run_meta: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Provider and model of the author and validator roles of one run ("" when unknown).

    Sources in order: author_role/validator_role recorded in run.json/progress.json; the flat
    model/validator_model plus author_provider/validator_provider fields of metrics.json
    (then progress.json); the run_started trace event of older runs (first lines only).
    A missing provider is filled from a later source that names the same model.
    """
    flat_keys = {"author": ("model", "author_provider"), "validator": ("validator_model", "validator_provider")}
    candidates: dict[str, list[tuple[str, str]]] = {"author": [], "validator": []}
    for role in candidates:
        for source in (run_meta, progress):
            meta = source.get(f"{role}_role") if isinstance(source, dict) else None
            if isinstance(meta, dict):
                candidates[role].append((_text(meta.get("provider")), _text(meta.get("model"))))
        model_key, provider_key = flat_keys[role]
        for source in (metrics, progress):
            if isinstance(source, dict):
                candidates[role].append((_text(source.get(provider_key)), _text(source.get(model_key))))

    def resolve(options: list[tuple[str, str]]) -> dict[str, str] | None:
        for provider, model in options:
            if model:
                if not provider:
                    provider = next((p for p, m in options if p and m == model), "")
                return {"provider": provider, "model": model}
        return None

    resolved = {role: resolve(options) for role, options in candidates.items()}
    if any(value is None or not value["provider"] for value in resolved.values()):
        started = trace_run_started(Path(run_dir) / "trace.jsonl") or {}
        for role in candidates:
            candidates[role].append((_text(started.get(f"{role}_provider")), _text(started.get(f"{role}_model"))))
            resolved[role] = resolve(candidates[role])
    return {role: value or {"provider": "", "model": ""} for role, value in resolved.items()}


def model_label(provider: str, model: str) -> str:
    """"provider:model" (just the model when the provider was not recorded), or ""."""
    if not model:
        return ""
    return f"{provider}:{model}" if provider else model


def _guidance_label(*sources: dict[str, Any] | None) -> str:
    """The run's guidance setting from run.json/progress.json/metrics.json, or "" when absent."""
    for source in sources:
        if not isinstance(source, dict) or source.get("guidance") is None:
            continue
        value = source["guidance"]
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, dict):
            for key in ("mode", "name", "level", "value"):
                if value.get(key) not in (None, ""):
                    return str(value[key])
            return json.dumps(value, sort_keys=True, default=str)
        if isinstance(value, (list, tuple)):
            return ", ".join(str(part) for part in value)
        return str(value)
    return ""


FAILED_SUFFIX = "_failed"


def _ledger_costs(ledger_path: Path) -> dict[str, float]:
    """Claude spend per ledger run id (one read); {} when the ledger is missing or unreadable."""
    try:
        return CostLedger(ledger_path).spend_by_run()
    except BudgetExceededError:  # LedgerUnavailableError: the index just leaves those costs blank
        return {}


def update_index(root: Path, ledger_path: Path | None = None) -> Path:
    """Rewrite root/index.html from the run folders, comparisons, and replays.

    A run whose metrics/progress has no anthropic_cost_usd shows its spend from the Claude cost
    ledger when the ledger has entries for that run id (a "_failed" folder also matches the id
    without the suffix). ledger_path is the configured ledger; by default the ledger file named
    in settings.DEFAULTS inside root. The ledger is read at most once per index build.
    """
    root.mkdir(parents=True, exist_ok=True)
    if ledger_path is None:
        ledger_path = root / Path(DEFAULTS["anthropic_cost_ledger_path"]).name
    ledger_costs: dict[str, float] | None = None
    rows: list[dict[str, Any]] = []
    titles: dict[tuple[int, str], str] = {}  # (row number, column) -> model cell tooltip
    run_dirs =(path for path in root.iterdir() if path.is_dir() and path.name not in NON_RUN_DIRS)
    for run_dir in sorted(run_dirs, reverse=True):
        # Finished runs have metrics.json; in-progress runs only have progress.json.
        try:
            progress = None
            metrics = None
            if (run_dir / "progress.json").exists():
                progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            if (run_dir / "metrics.json").exists():
                item = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                metrics = item if isinstance(item, dict) else None
            elif isinstance(progress, dict):
                item = {
                    "model": progress.get("model"),
                    "mode": "agent",
                    "status": progress.get("status"),
                    "runtime_seconds": progress.get("elapsed_seconds"),
                    "tool_calls": progress.get("tool_calls"),
                    "anthropic_cost_usd": progress.get("anthropic_cost_usd"),
                }
            else:
                continue
            times = run_times(run_dir, progress if isinstance(progress, dict) else None)
            item["start"] = format_local(times["start"])
            item["end"] = "running" if times["running"] else format_local(times["end"])
            item["run_id"] = run_dir.name
            cost = item.get("anthropic_cost_usd")
            if not isinstance(cost, (int, float)) and isinstance(progress, dict):
                cost = progress.get("anthropic_cost_usd")
            if not isinstance(cost, (int, float)):
                # Older runs did not record their Claude cost; the ledger did.
                if ledger_costs is None:
                    ledger_costs = _ledger_costs(Path(ledger_path))
                ledger_ids = [run_dir.name]
                if run_dir.name.endswith(FAILED_SUFFIX):
                    ledger_ids.append(run_dir.name[: -len(FAILED_SUFFIX)])
                cost = next((ledger_costs[name] for name in ledger_ids if name in ledger_costs), None)
                if cost is not None:
                    item["anthropic_cost_usd"] = cost
            # Blank for runs without Claude calls.
            item["cost"] = f"${cost:.4f}" if isinstance(cost, (int, float)) else None
            run_meta = _read_json_dict(run_dir / "run.json")
            progress_dict = progress if isinstance(progress, dict) else None
            roles = run_role_models(run_dir, run_meta, progress_dict, metrics)
            for role, column in (("author", AUTHOR_MODEL_COLUMN), ("validator", VALIDATOR_MODEL_COLUMN)):
                item[column] = model_label(**roles[role])
                if item[column]:
                    # Tooltip: role plus full provider:model, or a note when the provider is unknown.
                    suffix = "" if roles[role]["provider"] else " (provider not recorded)"
                    titles[(len(rows) + 1, column)] = f"{role}: {item[column]}{suffix}"
            item[GUIDANCE_COLUMN] = _guidance_label(run_meta, progress_dict, metrics)
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    headers = INDEX_HEADERS
    table_rows = []
    for number, item in enumerate(rows, start=1):
        run_id = html.escape(str(item.get("run_id", "")))
        cells = [f"<td>{number}</td>", f'<td><input type="checkbox" class="run-select" value="{run_id}"></td>']
        for header in headers:
            raw = item.get(header)
            value = html.escape("" if raw is None else str(raw))
            if header == "run_id":
                value = f'<a href="{run_id}/final_report.html">{value}</a>'
            css = ' class="nowrap"' if header in _NOWRAP_COLUMNS else ""
            if header in (AUTHOR_MODEL_COLUMN, VALIDATOR_MODEL_COLUMN) and value:
                css += f' title="{html.escape(titles.get((number, header), ""), quote=True)}"'
            cells.append(f"<td{css}>{value}</td>")
        cells.append(f'<td><a href="{run_id}/trace.jsonl">trace</a></td>')
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    comparison_rows = []
    for meta in list_comparisons(root):
        created = format_local(_parse_iso(meta.get("created_at"))) or meta["_sort"]
        run_ids = ", ".join(str(run_id) for run_id in meta.get("run_ids") or [])
        model = ":".join(str(part) for part in (meta.get("eval_provider"), meta.get("eval_model")) if part)
        link = html.escape(f"{COMPARISONS_DIR}/{meta['_html']}", quote=True)
        comparison_rows.append(
            f'<tr><td class="nowrap">{html.escape(created)}</td><td>{html.escape(run_ids)}</td>'
            f'<td>{html.escape(model)}</td><td><a href="{link}">open</a></td></tr>'
        )
    if comparison_rows:
        comparisons_html = (
            "<table><thead><tr><th>date</th><th>runs</th><th>eval model</th><th>page</th></tr></thead>"
            f"<tbody>{''.join(comparison_rows)}</tbody></table>"
        )
    else:
        comparisons_html = '<p class="muted">No comparisons yet. Select runs above and run the command shown under Comparison.</p>'
    replay_rows = []
    for meta in list_replays(root):
        created = format_local(_parse_iso(meta.get("created_at"))) or meta["_file"]
        model = ":".join(str(part) for part in (meta.get("provider"), meta.get("model")) if part)
        cost = meta.get("estimated_cost_usd")
        cells = [
            created, str(meta.get("run_id") or ""), str(meta.get("cycle") or ""), model,
            str(meta.get("decision") or meta.get("status") or ""), "" if cost is None else f"${cost:.4f}",
        ]
        link = html.escape(f"{REPLAYS_DIR}/{meta['_file']}", quote=True)
        replay_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + f'<td><a href="{link}">json</a></td></tr>'
        )
    if replay_rows:
        replays_html = (
            "<table><thead><tr><th>date</th><th>run</th><th>cycle</th><th>validator model</th><th>decision</th>"
            f"<th>est. cost</th><th>result</th></tr></thead><tbody>{''.join(replay_rows)}</tbody></table>"
        )
    else:
        replays_html = '<p class="muted">No validator replays yet (Scripts/reports/replay_validator.py).</p>'
    generated = datetime.now(timezone.utc).isoformat()
    # "</" is escaped so run data can never close the script element.
    metrics_json = json.dumps({item.get("run_id"): item for item in rows}, default=str).replace("</", "<\\/")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Finance Agent Runs</title>
<style>{INDEX_STYLE}</style></head>
<body><h1>Finance Agent Runs</h1><p>Generated {html.escape(generated)}. Start and end are local times. Select reports to inspect or compare objective run metrics.</p>
<button onclick="compareSelected()">Compare selected runs</button>
<div class="table-scroll"><table><thead><tr><th>No.</th><th>compare</th>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}<th>trace</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Comparison</h2>
<div id="comparison" class="panel">Select two or more runs and choose Compare selected runs.</div>
<div id="command-panel" class="panel" hidden>
<p>For an LLM-written comparison of these runs (eval model role), run this from the project root. Add <code>--dry-run-prompt</code> to inspect the prompt without calling the model.</p>
<input id="compare-command" type="text" readonly onclick="this.select()" aria-label="compare_runs command">
<button onclick="copyCommand()">Copy</button> <span id="copy-status" class="muted"></span>
</div>
<h2>Comparisons</h2>
{comparisons_html}
<h2>Validator replays</h2>
{replays_html}
<script>
const metrics = {metrics_json};
{INDEX_SCRIPT}
</script></body></html>"""
    index = root / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


class ProgressRecorder:
    """Stream run trace events into the run directory so an in-progress run can be watched.

    Writes trace.jsonl (one event per line, appended as it happens), progress.json (a
    small status snapshot), and progress.log (one readable line per meaningful event).
    save_run rewrites trace.jsonl once at the end with the complete trace.
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        model: str,
        validator_model: str,
        echo: bool = True,
        author_provider: str = "ollama",
        validator_provider: str = "ollama",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """metadata (collect_run_metadata: code version, role settings, limits) is stored in progress.json."""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._started = time.perf_counter()
        now = datetime.now(timezone.utc).isoformat()
        self.state: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "model": model,
            "validator_model": validator_model,
            "author_provider": author_provider,
            "validator_provider": validator_provider,
            "started_at": now,
            "updated_at": now,
            "elapsed_seconds": 0.0,
            "phase": "starting",
            "agent_round": None,
            "validation_cycle": None,
            "last_event": None,
            "tool_calls": 0,
            "last_validation_error_count": None,
            "last_repeated_error_count": None,
            "stalled_validation_cycles": 0,
            "author_rebuttals": 0,
        }
        if metadata:
            self.state.update(metadata)
        self._write_progress()
        self.log(
            f"Run {run_id} started (author={model}, validator={validator_model})"
            if author_provider == validator_provider == "ollama"
            else f"Run {run_id} started (author={author_provider}:{model}, validator={validator_provider}:{validator_model})"
        )

    def __call__(self, event: dict[str, Any]) -> None:
        """RunTrace listener: persist the event and refresh the live status files."""
        with (self.run_dir / "trace.jsonl").open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, default=str) + "\n")
        self._update_state(event)
        line = _describe_event(event)
        if line:
            self.log(line)
        self._write_progress()

    def log(self, message: str) -> None:
        """Append one timestamped line to progress.log and echo it to the console."""
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        try:
            with (self.run_dir / "progress.log").open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError as exc:
            logger.warning(f"Could not write progress.log: {exc}")
        if self.echo:
            logger.info(line, extra={"preformatted": True})

    def finish(self, status: str, **details: Any) -> None:
        """Record the final run status; never raises so archiving and delivery can continue."""
        self.state.update(details)
        self.state["status"] = status
        self.state["phase"] = "finished"
        summary = ", ".join(f"{key}={value}" for key, value in details.items())
        self.log(f"Run finished: status={status}" + (f" ({summary})" if summary else ""))
        self._write_progress()

    def move_to(self, run_dir: Path) -> bool:
        """Rename the run directory (for example to add a _failed suffix) when it is safe."""
        target = Path(run_dir)
        if target.exists():
            return False
        try:
            os.replace(self.run_dir, target)
        except OSError:
            return False
        self.run_dir = target
        self.state["run_id"] = target.name
        self._write_progress()
        return True

    def _update_state(self, event: dict[str, Any]) -> None:
        name = event.get("event")
        self.state["last_event"] = name
        if name == "agent_round":
            self.state["agent_round"] = event.get("round")
            self.state["phase"] = "author"
        elif name == "model_call":
            self.state["phase"] = f"{event.get('role')}_model_call"
        elif name in {"tool_call", "tool_error"}:
            self.state["phase"] = "tools"
            if name == "tool_call":
                self.state["tool_calls"] += 1
        elif name == "validation_cycle_start":
            self.state["validation_cycle"] = event.get("cycle")
            self.state["phase"] = "validation"
        elif name == "validation_cycle_result":
            if event.get("error_count") is not None:
                self.state["last_validation_error_count"] = event.get("error_count")
            self.state["last_repeated_error_count"] = event.get("repeated_error_count")
        elif name == "validation_stalled":
            self.state["stalled_validation_cycles"] += 1
        elif name == "author_rebuttals":
            self.state["author_rebuttals"] += event.get("rebuttal_count") or 0
        elif name == "budget_exceeded":
            self.state["budget_exceeded"] = event.get("error")
        if name in COST_EVENTS and isinstance(event.get("cost_usd"), (int, float)):
            total = (self.state.get("anthropic_cost_usd") or 0) + event["cost_usd"]
            self.state["anthropic_cost_usd"] = round(total, 6)
        elif name == "run_completed":
            self.state["phase"] = "completed"
        elif name == "run_aborted":
            self.state["phase"] = "aborted"

    def _write_progress(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["elapsed_seconds"] = round(time.perf_counter() - self._started, 1)
        path = self.run_dir / "progress.json"
        temp_path = self.run_dir / "progress.json.tmp"
        payload = json.dumps(self.state, indent=2, default=str)
        try:
            temp_path.write_text(payload, encoding="utf-8")
            os.replace(temp_path, path)
        except OSError:
            # A reader holding progress.json open on Windows can block the atomic swap.
            try:
                path.write_text(payload, encoding="utf-8")
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Could not write progress.json: {exc}")


def _describe_event(event: dict[str, Any]) -> str | None:
    """Turn a trace event into a short progress.log line, or None for low-signal events."""
    name = event.get("event")
    if name == "run_started":
        deadline = event.get("deadline_seconds")
        limit = f"; time limit {deadline:g}s" if deadline else "; no time limit"
        return f"Harness started; tools: {', '.join(event.get('tool_names') or []) or 'none'}{limit}"
    if name == "agent_round":
        return f"Author round {event.get('round')}"
    if name == "model_call":
        where = ""
        if event.get("validation_cycle") is not None:
            where = f", validation cycle {event.get('validation_cycle')} attempt {event.get('attempt')}"
        return f"{event.get('role')} model call started ({event.get('model')}, {event.get('prompt_chars', 0):,} prompt chars{where})"
    if name == "model_response":
        tokens = ""
        if event.get("eval_count") is not None or event.get("prompt_eval_count") is not None:
            tokens = f", tokens in/out {event.get('prompt_eval_count', '?')}/{event.get('eval_count', '?')}"
        stop = f", stopped: {event.get('done_reason')}" if event.get("done_reason") not in (None, "stop") else ""
        cost = f", cost ${event['cost_usd']:.4f}" if isinstance(event.get("cost_usd"), (int, float)) else ""
        return (
            f"{event.get('role')} response: parse {event.get('parse_status')}, "
            f"{event.get('response_chars', 0):,} chars in {event.get('elapsed_seconds', '?')}s{tokens}{cost}{stop}"
        )
    if name == "anthropic_budget":
        if event.get("ledger_error"):
            return f"Claude budget: cost ledger unavailable, so Claude calls will be refused ({event.get('ledger_error')})"
        return budget_status_line(event)
    if name == "budget_exceeded":
        return f"Claude budget guard refused the {event.get('role')} call: {event.get('error')}; stopping with a fail decision"
    if name == "model_call_failed":
        return (
            f"{event.get('role')} model call failed (attempt {event.get('attempt')}, "
            f"{event.get('elapsed_seconds', '?')}s): {event.get('error_type')}: {event.get('error')}"
        )
    if name == "model_call_retry":
        return f"Retrying {event.get('role')} model call (attempt {event.get('attempt')})"
    if name == "model_unavailable":
        return f"{event.get('role')} model unavailable during {event.get('phase')}; stopping with a fail decision"
    if name == "run_deadline_exceeded":
        return (
            f"Run time limit reached ({event.get('elapsed_seconds')}s elapsed of {event.get('limit_seconds')}s) "
            f"at a {event.get('role')} model call; stopping with a fail decision"
        )
    if name == "agent_rounds_exhausted":
        return (
            f"Author used all {event.get('max_agent_rounds')} rounds in {event.get('phase')} without a report; "
            "stopping with a fail decision"
        )
    if name == "tool_call":
        return f"Tool {event.get('name')} ok ({event.get('phase')})"
    if name == "tool_error":
        return f"Tool {event.get('name')} error: {event.get('error_type')}: {event.get('error')}"
    if name in {"author_format_retry", "validator_format_retry"}:
        return f"{name.replace('_', ' ')}: {event.get('error')}"
    if name == "validation_cycle_start":
        return f"Validation cycle {event.get('cycle')} started ({event.get('report_html_bytes', 0):,} report bytes)"
    if name == "report_sanitization_failed":
        return f"HTML gate rejected report ({event.get('category')}): {event.get('error')}"
    if name == "validation_cycle_result":
        return (
            f"Validation cycle {event.get('cycle')} result ({event.get('stage', 'validator')}): "
            f"{event.get('decision')}, {event.get('error_count', '?')} errors, {event.get('warning_count', '?')} warnings"
            + (
                f", {event.get('repeated_error_count')} repeated from the previous cycle"
                if event.get("repeated_error_count") else ""
            )
            + (
                f", {event.get('resolved_finding_count')} earlier finding(s) resolved"
                if event.get("resolved_finding_count") else ""
            )
        )
    if name == "author_rebuttals":
        findings = "; ".join(str(item.get("finding", ""))[:80] for item in event.get("rebuttals") or [])
        return (
            f"Author disputed {event.get('rebuttal_count', 0)} finding(s) from validation cycle "
            f"{event.get('responding_to_cycle')}: {findings}"
        )
    if name == "author_rebuttals_ignored":
        return f"Author rebuttals ignored: {event.get('error') or event.get('reason')}"
    if name == "validation_stalled":
        return (
            f"Validation stalled: cycle {event.get('cycle')} repeated {event.get('repeated_error_count')} error "
            f"finding(s) from the previous cycle unchanged ({event.get('rebuttal_count', 0)} author rebuttal(s))"
        )
    if name == "exception":
        return f"Exception in {event.get('phase')}: {event.get('exception_type')}: {event.get('error')}"
    if name == "run_completed":
        return f"Harness completed: decision={event.get('decision')}, tool calls={event.get('tool_calls')}"
    if name == "run_aborted":
        reason = event.get("exception_type") or event.get("status") or "aborted"
        error = event.get("error")
        return f"Harness aborted: {reason}" + (f": {error}" if error else "")
    return None
