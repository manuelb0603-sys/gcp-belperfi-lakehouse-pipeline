"""Persist agent run artifacts and maintain a browser-openable run index."""

from __future__ import annotations

import html
import json
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
    for metrics_path in sorted(root.glob("*/metrics.json"), reverse=True):
        try:
            item = json.loads(metrics_path.read_text(encoding="utf-8"))
            item["run_id"] = metrics_path.parent.name
            rows.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    headers = ["run_id", "model", "mode", "status", "validation", "runtime_seconds", "agent_rounds", "tool_calls", "report_bytes"]
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
    const keys = ['model','mode','status','validation','runtime_seconds','agent_rounds','tool_calls','report_bytes'];
    document.getElementById('comparison').textContent = keys.map(key => key + ': ' + ids.map(id => id + '=' + (metrics[id][key] ?? '')).join(' | ')).join('\\n');
}}
</script></body></html>"""
    index = root / "index.html"
    index.write_text(page, encoding="utf-8")
    return index