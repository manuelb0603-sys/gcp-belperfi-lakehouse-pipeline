"""Gradio finance app: the monthly report and Q&A tab, and the finance chat tab.

The chat keeps its transaction frame in the finance.chat.data module cache (never in gr.State)
and one small ChatSession per browser session.
"""

from __future__ import annotations

import calendar
import threading
from collections import OrderedDict
from datetime import date
from decimal import Decimal
from typing import Any, Iterator

import gradio as gr

from finance.bigquery_tools import get_report_period_spending, get_spending_years
from finance.budget import BudgetExceededError
from finance.chat import data as chat_data
from finance.chat import prompts as chat_prompts
from finance.chat.engine import ChatEvent, run_turn
from finance.chat.profile import suggest_examples
from finance.chat.session import ChatSession
from finance.model_clients import build_model_client
from finance.memory import open_memory
from finance.qa import answer_finance_question as shared_answer_finance_question
from finance.report_runner import budget_stop_message
from finance.reporting import generate_finance_report
from finance.settings import (
    APP_FALLBACK_MODEL_NAME,
    APP_FALLBACK_MODEL_SERVER_URL,
    ReportSettings,
    load_config,
    report_thread_key,
)


MONTHS = [(calendar.month_abbr[number], number) for number in range(1, 13)]
SETTINGS = ReportSettings.from_config(load_config())
PROJECT_NAME = SETTINGS.project_name
OLLAMA_URL = SETTINGS.model_server_url or APP_FALLBACK_MODEL_SERVER_URL
OLLAMA_MODEL = SETTINGS.model_name or APP_FALLBACK_MODEL_NAME
# Same thread key as the weekly email so the report's history tool sees the email reports.
REPORT_THREAD_KEY = report_thread_key(SETTINGS)


def _open_app_memory() -> Any:
    """Open the shared MemoryStore; the app keeps working without memory if this fails."""
    try:
        return open_memory(SETTINGS.memory_schema)
    except Exception as exc:
        print(f"Memory unavailable; continuing without memory: {exc}")
        return None


MEMORY = _open_app_memory()


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
            config=SETTINGS,
            project_name=PROJECT_NAME,
            ollama_url=OLLAMA_URL,
            model_name=OLLAMA_MODEL,
            year=selected_year,
            month=month,
            rows=rows,
            selected_report_month=selected_month,
            memory_module=MEMORY,
            thread_key=REPORT_THREAD_KEY,
        )
        report_context = {
            "selected_month": result.snapshot.latest_month,
            "snapshot": result.snapshot.to_dict(),
            "rows": _serialize_rows(rows),
            "plain_text": result.plain_text,
            "report_html": result.report_html,
        }
        budget_message = budget_stop_message(result)
        if budget_message:
            # A Claude spend cap stopped the report: say so plainly instead of a validation failure.
            return "<div class='empty-state'><h3>Claude budget reached</h3></div>", report_context, budget_message
        if not result.report_html:
            message = result.decision.summary if result.decision else "The report generator did not return validated HTML."
            return "<div class='empty-state'><h3>Finance report generation failed</h3></div>", report_context, message
        status = "Validated finance report ready." if result.passed else "Report generated but did not pass validation. Review before relying on it."
        return f"<div class='finance-scroll'>{result.report_html}</div>", report_context, status
    except BudgetExceededError as exc:
        return "<div class='empty-state'><h3>Claude budget reached</h3></div>", None, str(exc)
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
        config=SETTINGS,
        ollama_url=OLLAMA_URL,
        model_name=OLLAMA_MODEL,
        memory_module=MEMORY,
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


# ----- finance chat -----

# Safe placeholders; replaced on page load (and on Refresh data) by questions that fit the data profile.
CHAT_EXAMPLES = list(chat_prompts.STATIC_EXAMPLES)
LATEST_MONTH = "Latest month"
MAX_CHAT_SESSIONS = 32
_CHAT_SESSIONS: "OrderedDict[str, ChatSession]" = OrderedDict()
_CHAT_FOCUS: dict[str, date | None] = {}
_CHAT_LOCK = threading.Lock()


def _session_key(request: Any) -> str:
    return str(getattr(request, "session_hash", None) or "default")


def chat_session(request: Any = None) -> ChatSession:
    """The ChatSession for this browser session (bounded LRU; the frame is never stored in it)."""
    key = _session_key(request)
    with _CHAT_LOCK:
        session = _CHAT_SESSIONS.get(key)
        if session is None:
            session = ChatSession(key, history_turns=SETTINGS.chat_history_turns)
            _CHAT_SESSIONS[key] = session
            while len(_CHAT_SESSIONS) > MAX_CHAT_SESSIONS:
                _CHAT_SESSIONS.popitem(last=False)
        _CHAT_SESSIONS.move_to_end(key)
        session.focus_month = _CHAT_FOCUS.get(key, session.focus_month)
        return session


def _parse_focus(value: Any) -> date | None:
    text = str(value or "").strip()
    if len(text) == 7 and text[4] == "-":
        try:
            return date(int(text[:4]), int(text[5:7]), 1)
        except ValueError:
            return None
    return None


def focus_month_choices(frame: Any = None, today: date | None = None) -> list[str]:
    """"Latest month" plus YYYY-MM months, newest first (from the data when loaded, else the last 24)."""
    if frame is not None and len(frame):
        months = sorted({str(period) for period in frame["month"].unique()}, reverse=True)
    else:
        today = today or date.today()
        start = chat_data.default_start(today)
        months = []
        current = date(today.year, today.month, 1)
        while current >= start:
            months.append(f"{current:%Y-%m}")
            current = date(current.year - 1, 12, 1) if current.month == 1 else date(current.year, current.month - 1, 1)
    return [LATEST_MONTH] + months


def set_focus_month(value: Any, request: gr.Request = None) -> None:
    key = _session_key(request)
    with _CHAT_LOCK:
        _CHAT_FOCUS[key] = _parse_focus(value)
        if key in _CHAT_SESSIONS:
            _CHAT_SESSIONS[key].focus_month = _CHAT_FOCUS[key]


def chat_examples(loaded: Any = None) -> list[str]:
    """Example questions generated from the data profile; the static examples if loading or profiling fails."""
    try:
        loaded = loaded or chat_data.load_chat_data(SETTINGS)
        return suggest_examples(loaded.profile)
    except Exception:
        return list(chat_prompts.STATIC_EXAMPLES)


def _examples_update(examples: list[str]) -> Any:
    return gr.update(examples=[{"text": text} for text in examples])


def load_chat_examples() -> Any:
    """Page-load handler: show example chips that fit the loaded data (uses the cached frame)."""
    return _examples_update(chat_examples())


def refresh_chat_data() -> tuple[Any, str, Any]:
    """Drop the cached transactions and reload them (one BigQuery query); refresh the example chips."""
    try:
        chat_data.refresh(PROJECT_NAME)
        loaded = chat_data.load_chat_data(SETTINGS)
    except Exception as exc:
        return gr.update(), chat_prompts.DATA_ERROR_MESSAGE.format(reason=exc), gr.update()
    dates = loaded.catalog["date_range"]
    return (
        gr.update(choices=focus_month_choices(loaded.frame)),
        f"Loaded {loaded.catalog['transactions']} transactions from {dates['min']} to {dates['max']}.",
        _examples_update(chat_examples(loaded)),
    )


def _event_message(event: ChatEvent) -> Any:
    if event.kind == "explanation":
        return gr.ChatMessage(
            role="assistant", content=event.content,
            metadata={"title": "How I computed this", "status": event.status or "done"},
        )
    if event.kind == "figure":
        return gr.ChatMessage(role="assistant", content=gr.Plot(event.content))
    if event.kind == "table":
        return gr.ChatMessage(role="assistant", content=gr.Dataframe(event.content, interactive=False))
    return gr.ChatMessage(role="assistant", content=str(event.content))


def chat_respond(message: str, history: list[Any], request: gr.Request = None) -> Iterator[list[Any]]:
    """ChatInterface generator: yields the assistant messages for this turn as they are ready."""
    session = chat_session(request)
    session.sync_history(history)
    try:
        loaded = chat_data.load_chat_data(SETTINGS)
    except Exception as exc:
        yield [gr.ChatMessage(role="assistant", content=chat_prompts.DATA_ERROR_MESSAGE.format(reason=exc))]
        return
    if session.client is None:
        try:
            # One guarded client per chat session: Claude calls are recorded with run_id "chat".
            session.client = build_model_client(SETTINGS, "chat", run_id="chat")
        except Exception as exc:
            yield [gr.ChatMessage(role="assistant", content=chat_prompts.MODEL_SETUP_MESSAGE.format(reason=exc))]
            return
    messages: list[Any] = []
    for event in run_turn(session, message, client=session.client, df=loaded.frame, catalog=loaded.catalog,
                          settings=SETTINGS, profile=loaded.profile):
        rendered = _event_message(event)
        if event.kind == "explanation" and messages:
            messages[0] = rendered  # update the "How I computed this" message in place
        else:
            messages.append(rendered)
        yield list(messages)


with gr.Blocks(title="Finance Summary & QA") as app:
    with gr.Tabs():
        with gr.Tab("Monthly report"):
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
        with gr.Tab("Chat"):
            gr.Markdown("Ask about your spending. Answers are computed from your transactions; the model only picks the analysis and explains it.")
            with gr.Row():
                focus_dropdown = gr.Dropdown(
                    label="Focus month", choices=focus_month_choices(), value=LATEST_MONTH, interactive=True, scale=3,
                )
                refresh_data_button = gr.Button("Refresh data", scale=1)
            chat_status = gr.Markdown("")
            chatbot = gr.Chatbot(height=620, label="Finance chat")
            gr.ChatInterface(
                fn=chat_respond,
                chatbot=chatbot,
                examples=CHAT_EXAMPLES,
                concurrency_limit=1,
            )
            focus_dropdown.change(set_focus_month, inputs=[focus_dropdown], outputs=None)
            refresh_data_button.click(refresh_chat_data, outputs=[focus_dropdown, chat_status, chatbot])
    app.load(load_chat_examples, outputs=[chatbot])


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, css=APP_CSS)
