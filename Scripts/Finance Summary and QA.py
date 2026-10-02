"""Gradio finance summary and Q&A backed by the gold monthly spending table."""

from __future__ import annotations

import calendar
from decimal import Decimal
from typing import Any

import gradio as gr

from finance.bigquery_tools import get_report_period_spending, get_spending_years
from finance.qa import answer_finance_question as shared_answer_finance_question
from finance.reporting import generate_finance_report
from finance.settings import load_config


MONTHS = [(calendar.month_abbr[number], number) for number in range(1, 13)]
CONFIG = load_config()
PROJECT_NAME = CONFIG.get("project_name")
OLLAMA_URL = CONFIG.get("model_server_url", "http://localhost:11434/api/generate")
OLLAMA_MODEL = CONFIG.get("model_name", "gemma4:e4b")


def _serialize_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Convert BigQuery row values to plain, prompt-safe Python types."""
    serialized = []
    for row in rows:
        item = dict(row)
        serialized.append({
            key: value.isoformat() if hasattr(value, "isoformat") else float(value) if isinstance(value, Decimal) else value
            for key, value in item.items()
        })
    return serialized


def available_years() -> list[int]:
    """Return years present in the gold monthly spending table."""
    if not PROJECT_NAME:
        raise ValueError("project_name is missing from config.json")
    return get_spending_years(PROJECT_NAME)


def _month_rows(year: int, month: int) -> list[dict[str, Any]]:
    """Fetch the selected month and its predecessor for deterministic comparisons."""
    if not PROJECT_NAME:
        raise ValueError("project_name is missing from config.json")
    return get_report_period_spending(PROJECT_NAME, year=year, month=month)["rows"]


def load_finance_summary(year: int | str | None, month: int) -> tuple[str, dict[str, Any] | None, str]:
    """Generate the validated shared finance report for the selected month."""
    if year is None:
        return "<p>Select a year first.</p>", None, "Choose a year, then click a month."
    if not PROJECT_NAME:
        return "<p>Project is not configured.</p>", None, "project_name is missing from config.json."
    try:
        selected_year = int(year)
        period = get_report_period_spending(PROJECT_NAME, year=selected_year, month=month)
        rows = period["rows"]
        selected_month = period["report_month"]
        selected_key = selected_month.isoformat() if selected_month else None
        current_month_rows = [
            row for row in rows
            if str(row.get("transaction_month", ""))[:10] == selected_key
        ]
        if not current_month_rows:
            label = f"{calendar.month_name[month]} {selected_year}"
            return f"<div class='empty-state'><h3>No data for {label}</h3><p>Choose a different month.</p></div>", None, f"No gold-layer spending data found for {label}."

        result = generate_finance_report(
            config=CONFIG,
            project_name=PROJECT_NAME,
            ollama_url=OLLAMA_URL,
            model_name=OLLAMA_MODEL,
            year=selected_year,
            month=month,
            rows=rows,
            selected_report_month=selected_month,
            thread_key=f"finance-ui:{CONFIG.get('memory_schema', 'memory')}",
        )
        report_context = {
            "selected_month": result.snapshot.latest_month,
            "snapshot": result.snapshot.to_dict(),
            "rows": _serialize_rows(rows),
            "plain_text": result.plain_text,
            "report_html": result.report_html,
        }
        if not result.report_html:
            message = result.decision.summary if result.decision else "The report generator did not return validated HTML."
            return "<div class='empty-state'><h3>Finance report generation failed</h3></div>", report_context, message
        status = "Validated finance report ready." if result.passed else "Report generated but did not pass validation. Review before relying on it."
        return f"<div class='finance-scroll'>{result.report_html}</div>", report_context, status
    except Exception as exc:
        return "<div class='empty-state'><h3>Could not generate the finance summary</h3><p>Check the status message below.</p></div>", None, f"Finance report request failed: {exc}"


def answer_finance_question(question: str, report_context: dict[str, Any] | None) -> str:
    """Answer against retrieved report evidence and long-term finance facts."""
    if not report_context:
        return "Load a year and month summary before asking a question."
    return shared_answer_finance_question(
        question=question,
        report_context={
            key: value for key, value in report_context.items()
            if key != "report_html" and key != "plain_text"
        },
        report_html=report_context.get("report_html", ""),
        config=CONFIG,
        ollama_url=OLLAMA_URL,
        model_name=OLLAMA_MODEL,
    )


def refresh_year_choices() -> tuple[Any, str]:
    try:
        years = available_years()
        choices = [str(year) for year in years]
        selected = choices[0] if choices else None
        return gr.update(choices=choices, value=selected), f"Found {len(choices)} year(s) in the finance table."
    except Exception as exc:
        return gr.update(choices=[], value=None), f"Could not load years from BigQuery: {exc}"


try:
    INITIAL_YEARS = [str(year) for year in available_years()]
    INITIAL_STATUS = f"Found {len(INITIAL_YEARS)} year(s). Choose a year, then click a month."
except Exception as initial_error:
    INITIAL_YEARS = []
    INITIAL_STATUS = f"Could not load years from BigQuery: {initial_error}"


APP_CSS = """
.finance-scroll { max-height: 610px; overflow-y: auto; padding: 4px 12px 12px 4px; }
.finance-report { max-width: 980px; margin: 8px auto; padding: 28px; border: 1px solid #f0d7d2; border-radius: 20px; background: linear-gradient(145deg,#fffdfb,#fff5f2); color: #302522; box-shadow: 0 12px 32px rgba(88,45,34,.08); }
.finance-report .report-kicker { color:#a34c43; letter-spacing:.14em; font-size:.75rem; font-weight:800; }
.finance-report h2 { margin:.35rem 0; font-size:2rem; }
.finance-report .report-subtitle,.finance-report .data-note { color:#786a65; }
.finance-report .metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:22px 0; }
.finance-report .metric { padding:16px; border:1px solid #f1deda; border-radius:14px; background:#fff; }
.finance-report .metric span,.finance-report .metric small { display:block; color:#786a65; }
.finance-report .metric strong { display:block; margin:8px 0; font-size:1.2rem; }
.finance-report .metric small { font-size:.78rem; line-height:1.4; }
.finance-report h3 { margin-top:28px; }
.finance-report table { width:100%; border-collapse:collapse; background:#fff; }
.finance-report th,.finance-report td { padding:11px 12px; border-bottom:1px solid #f1e7e4; text-align:left; }
.finance-report th { color:#76544d; background:#fff8f6; }
.finance-report .amount { text-align:right; font-variant-numeric:tabular-nums; }
.table-wrap { overflow-x:auto; border:1px solid #f1e7e4; border-radius:12px; }
.empty-state { padding:30px; border-radius:16px; background:#fff8f6; color:#57433d; }
.month-button { min-width:64px; }
"""


with gr.Blocks(title="Finance Summary & QA") as app:
    gr.Markdown("# Finance Summary & QA\nChoose a year and month to view an authoritative spending summary, then ask questions about that selection.")
    with gr.Row():
        year_dropdown = gr.Dropdown(
            label="Year",
            choices=INITIAL_YEARS,
            value=INITIAL_YEARS[0] if INITIAL_YEARS else None,
            interactive=True,
            scale=3,
        )
        refresh_years_button = gr.Button("Refresh years", scale=1)
    gr.Markdown("### Select a month")
    month_buttons = []
    for offset in (0, 6):
        with gr.Row():
            for month_label, month_number in MONTHS[offset : offset + 6]:
                month_button = gr.Button(month_label, elem_classes=["month-button"])
                month_buttons.append((month_button, month_number))

    gr.Markdown("### Financial summary")
    summary_panel = gr.HTML(value="<div class='empty-state'>Select a year and month to load your summary.</div>")
    selected_report = gr.State(value=None)
    status_message = gr.Markdown(INITIAL_STATUS)
    for month_button, month_number in month_buttons:
        month_button.click(
            fn=lambda year, selected_month=month_number: load_finance_summary(year, selected_month),
            inputs=[year_dropdown],
            outputs=[summary_panel, selected_report, status_message],
        )
    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a question about the Financial Summary",
            placeholder="For example: Which categories changed most compared with the prior month?",
            lines=2,
            scale=5,
        )
        question_button = gr.Button("Ask", variant="primary", scale=1)
    answer_output = gr.Markdown(label="Answer")
    question_button.click(
        answer_finance_question,
        inputs=[question_input, selected_report],
        outputs=answer_output,
    )
    refresh_years_button.click(
        refresh_year_choices,
        outputs=[year_dropdown, status_message],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, css=APP_CSS)
