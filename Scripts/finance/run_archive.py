"""Persist agent run artifacts and maintain a browser-openable run index."""

from __future__ import annotations

import html
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def update_index(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        # Finished runs have metrics.json; in-progress runs only have progress.json.
        try:
            if (run_dir / "metrics.json").exists():
                item = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            elif (run_dir / "progress.json").exists():
                progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
                item = {
                    "model": progress.get("model"),
                    "mode": "agent",
                    "status": progress.get("status"),
                    "runtime_seconds": progress.get("elapsed_seconds"),
                    "tool_calls": progress.get("tool_calls"),
                }
            else:
                continue
            item["run_id"] = run_dir.name
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    headers = ["run_id", "model", "mode", "status", "validation", "runtime_seconds", "agent_rounds", "tool_calls", "validation_cycles", "validator_calls", "author_format_retries", "validator_format_retries", "initial_validation_errors", "final_validation_errors", "validation_error_reduction", "validation_cycles_improved", "validation_cycles_regressed", "report_bytes"]
    table_rows = []
    for item in rows:
        run_id = html.escape(str(item.get("run_id", "")))
        cells = []
        cells.append(f'<td><input type="checkbox" class="run-select" value="{run_id}"></td>')
        for header in headers:
            value = html.escape(str(item.get(header, "")))
            if header == "run_id":
                value = f'<a href="{run_id}/final_report.html">{value}</a>'
            cells.append(f"<td>{value}</td>")
        cells.append(f'<td><a href="{run_id}/trace.jsonl">trace</a></td>')
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    generated = datetime.now(timezone.utc).isoformat()
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Finance Agent Runs</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;background:#fff8fb;color:#33252b}} table{{border-collapse:collapse;width:100%;background:white}} th,td{{border:1px solid #e8ccd8;padding:.55rem;text-align:left}} th{{background:#f7dce8}} a{{color:#8d315b}} button{{margin:1rem 0;padding:.5rem .8rem}} #comparison{{background:white;border:1px solid #e8ccd8;padding:1rem;white-space:pre-wrap}}</style></head>
<body><h1>Finance Agent Runs</h1><p>Generated {html.escape(generated)}. Select reports to inspect or compare objective run metrics.</p>
<button onclick="compareSelected()">Compare selected runs</button>
<table><thead><tr><th>compare</th>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}<th>trace</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Comparison</h2><div id="comparison">Select two or more runs and choose Compare selected runs.</div>
<script>
const metrics = {json.dumps({item.get("run_id"): item for item in rows}, default=str)};
function compareSelected() {{
    const ids = [...document.querySelectorAll('.run-select:checked')].map(item => item.value);
    if (ids.length < 2) {{ document.getElementById('comparison').textContent = 'Select at least two runs.'; return; }}
    const keys = ['model','mode','status','validation','runtime_seconds','agent_rounds','tool_calls','validation_cycles','validator_calls','author_format_retries','validator_format_retries','initial_validation_errors','final_validation_errors','validation_error_reduction','validation_cycles_improved','validation_cycles_regressed','report_bytes'];
    document.getElementById('comparison').textContent = keys.map(key => key + ': ' + ids.map(id => id + '=' + (metrics[id][key] ?? '')).join(' | ')).join('\\n');
}}
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

    def __init__(self, run_dir: Path, run_id: str, model: str, validator_model: str, echo: bool = True) -> None:
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
        self._write_progress()
        self.log(f"Run {run_id} started (author={model}, validator={validator_model})")

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
