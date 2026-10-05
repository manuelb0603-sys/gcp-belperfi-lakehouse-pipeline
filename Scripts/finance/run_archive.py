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

from .settings import PROJECT_ROOT


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
    """Run-start metadata for run.json/progress.json: code version, author/validator role settings, limits.

    settings is a finance.settings.ReportSettings. Role settings hold provider, model, options,
    think, and timeouts only; server URLs and credentials are never included.
    """
    return {
        "code_version": code_version if code_version is not None else git_code_version(),
        "author_role": settings.public_role("author"),
        "validator_role": settings.public_role("validator"),
        "harness_limits": settings.limits(),
    }


def save_run(root: Path, run_id: str, manifest: dict[str, Any], trace: list[dict[str, Any]], report_html: str, report_text: str, metrics: dict[str, Any]) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (run_dir / "trace.jsonl").write_text("".join(json.dumps(item, default=str) + "\n" for item in trace), encoding="utf-8")
    (run_dir / "final_report.html").write_text(report_html, encoding="utf-8")
    (run_dir / "final_report.txt").write_text(report_text, encoding="utf-8")
    update_index(root)
    return run_dir


COMPARISONS_DIR = "comparisons"
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


INDEX_STYLE = (
    "body{font:14px system-ui,sans-serif;margin:2rem;background:#fff8fb;color:#33252b}"
    " table{border-collapse:collapse;width:100%;background:white}"
    " th,td{border:1px solid #e8ccd8;padding:.55rem;text-align:left;vertical-align:top}"
    " th{background:#f7dce8} a{color:#8d315b} button{margin:1rem 0;padding:.5rem .8rem}"
    " .panel{background:white;border:1px solid #e8ccd8;padding:1rem;margin-bottom:1rem}"
    " .muted{color:#7a6570} .nowrap{white-space:nowrap} tr.diff td{background:#fff1f5}"
    " #compare-command{width:100%;font:13px ui-monospace,Consolas,monospace;padding:.4rem;box-sizing:border-box}"
)

INDEX_SCRIPT = r"""
const keys = ['start','end','model','author_provider','validator_model','validator_provider','mode','status','validation','runtime_seconds','agent_rounds','tool_calls','validation_cycles','validator_calls','author_format_retries','validator_format_retries','initial_validation_errors','final_validation_errors','validation_error_reduction','validation_cycles_improved','validation_cycles_regressed','report_bytes'];
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
    document.getElementById('compare-command').value = '..\\.venv\\Scripts\\python.exe Scripts\\compare_runs.py ' + ids.map(quoteArg).join(' ');
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


def update_index(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_dirs = (path for path in root.iterdir() if path.is_dir() and path.name != COMPARISONS_DIR)
    for run_dir in sorted(run_dirs, reverse=True):
        # Finished runs have metrics.json; in-progress runs only have progress.json.
        try:
            progress = None
            if (run_dir / "progress.json").exists():
                progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            if (run_dir / "metrics.json").exists():
                item = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            elif isinstance(progress, dict):
                item = {
                    "model": progress.get("model"),
                    "mode": "agent",
                    "status": progress.get("status"),
                    "runtime_seconds": progress.get("elapsed_seconds"),
                    "tool_calls": progress.get("tool_calls"),
                }
            else:
                continue
            times = run_times(run_dir, progress if isinstance(progress, dict) else None)
            item["start"] = format_local(times["start"])
            item["end"] = "running" if times["running"] else format_local(times["end"])
            item["run_id"] = run_dir.name
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    headers = ["run_id", "start", "end", "model", "mode", "status", "validation", "runtime_seconds", "agent_rounds", "tool_calls", "validation_cycles", "validator_calls", "author_format_retries", "validator_format_retries", "initial_validation_errors", "final_validation_errors", "validation_error_reduction", "validation_cycles_improved", "validation_cycles_regressed", "report_bytes"]
    table_rows = []
    for number, item in enumerate(rows, start=1):
        run_id = html.escape(str(item.get("run_id", "")))
        cells = [f"<td>{number}</td>", f'<td><input type="checkbox" class="run-select" value="{run_id}"></td>']
        for header in headers:
            raw = item.get(header)
            value = html.escape("" if raw is None else str(raw))
            if header == "run_id":
                value = f'<a href="{run_id}/final_report.html">{value}</a>'
            css = ' class="nowrap"' if header in {"start", "end"} else ""
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
    generated = datetime.now(timezone.utc).isoformat()
    # "</" is escaped so run data can never close the script element.
    metrics_json = json.dumps({item.get("run_id"): item for item in rows}, default=str).replace("</", "<\\/")
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Finance Agent Runs</title>
<style>{INDEX_STYLE}</style></head>
<body><h1>Finance Agent Runs</h1><p>Generated {html.escape(generated)}. Start and end are local times. Select reports to inspect or compare objective run metrics.</p>
<button onclick="compareSelected()">Compare selected runs</button>
<table><thead><tr><th>No.</th><th>compare</th>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}<th>trace</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Comparison</h2>
<div id="comparison" class="panel">Select two or more runs and choose Compare selected runs.</div>
<div id="command-panel" class="panel" hidden>
<p>For an LLM-written comparison of these runs (eval model role), run this from the project root. Add <code>--dry-run-prompt</code> to inspect the prompt without calling the model.</p>
<input id="compare-command" type="text" readonly onclick="this.select()" aria-label="compare_runs command">
<button onclick="copyCommand()">Copy</button> <span id="copy-status" class="muted"></span>
</div>
<h2>Comparisons</h2>
{comparisons_html}
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
            print(f"Could not write progress.log: {exc}", flush=True)
        if self.echo:
            print(line, flush=True)

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
                print(f"Could not write progress.json: {exc}", flush=True)


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
        return (
            f"{event.get('role')} response: parse {event.get('parse_status')}, "
            f"{event.get('response_chars', 0):,} chars in {event.get('elapsed_seconds', '?')}s{tokens}{stop}"
        )
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
