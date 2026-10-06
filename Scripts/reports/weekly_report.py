"""Weekly finance email entry point (the scheduled task runs this file).

Loads config.json, then runs either agent mode (finance.report_runner: harness report, run
archive, delivery, memory) or the legacy proof of concept (finance.legacy_report: one Ollama
prompt over the gold-layer summary). The pieces live in finance/; the names below stay
importable from this module for existing callers and tests.
"""

import sys
import time
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])  # Scripts/, so `import finance` works from this subfolder
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from finance import report_runner  # noqa: E402
from finance.email_delivery import build_email_message, send_email
from finance.legacy_report import (  # noqa: F401  (re-exported)
    FAILED_REPORT_TEXT,
    _memory_context_to_prompt,
    build_ollama_prompt,
    fetch_bigquery_summary,
    format_data_payload,
    query_ollama,
)
from finance.report_memory import extract_json_payload, open_report_memory, remember_report
from finance.run_archive import DEFAULT_RUNS_ROOT
from finance.settings import DEFAULT_CONFIG_PATH, ReportSettings, load_config
from finance.status_log import log_status

# Earlier names of the report-memory helpers, kept for existing imports.
_extract_json_payload = extract_json_payload
_init_memory_module = open_report_memory
_remember_report = remember_report

# Agent runs are archived here; tests patch it (and send_email) on this module.
RUN_ARCHIVE_ROOT = DEFAULT_RUNS_ROOT


def run_agent_weekly_report(
    config,
    results: list,
    memory_module,
    sender: str,
    receivers: list[str],
    receiver_header: str,
    password: str,
    project_name: str,
    ollama_url: str,
    model_name: str,
    memory_schema: str,
) -> None:
    """Agent-mode run (finance.report_runner) archived under RUN_ARCHIVE_ROOT and sent with send_email."""
    report_runner.run_agent_weekly_report(
        config, results, memory_module, sender, receivers, receiver_header,
        password, project_name, ollama_url, model_name, memory_schema,
        archive_root=RUN_ARCHIVE_ROOT, send=send_email,
    )


def run_weekly_report(config_path: Path | None = None) -> None:
    """Orchestrate the weekly report generation and email send."""
    start_time = time.perf_counter()
    log_status("Starting weekly report run", start_time)

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        settings = ReportSettings.from_config(load_config(config_path))
        log_status(f"Loaded config from {config_path}")
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return

    receivers = settings.email_listings
    sender = settings.email_sender
    receiver_header = ", ".join(receivers)
    password = settings.gmail_app_credentials
    project_name = settings.project_name
    ollama_url = settings.model_server_url
    model_name = settings.model_name
    days_back = settings.days_back
    ollama_stream = settings.ollama_stream
    ollama_timeout = settings.ollama_timeout
    memory_schema = settings.memory_schema
    report_mode = settings.report_mode

    if not receivers:
        print("No email recipients configured in email_listings.")
        return
    if not sender or not password:
        print("Email sender or credentials are missing.")
        return
    if not project_name:
        print("project_name is missing from config.")
        return
    if not ollama_url or not model_name:
        print("Ollama endpoint or model_name is missing from config.")
        return

    log_status("Initializing memory module")
    memory_module = open_report_memory(memory_schema)
    if report_mode == "agent":
        # The shared report service resolves the latest month directly; do not constrain
        # the email run to the legacy rolling days_back window.
        run_agent_weekly_report(
            settings, [], memory_module, sender, receivers, receiver_header,
            password, project_name, ollama_url, model_name, memory_schema,
        )
        return

    try:
        log_status(f"Fetching BigQuery summary for project '{project_name}' over last {days_back} days")
        results = fetch_bigquery_summary(project_name, days_back=days_back)
        log_status(f"BigQuery fetch complete: {len(results)} rows returned")
    except Exception as exc:
        print(f"Failed to fetch BigQuery summary: {exc}")
        return

    if not results:
        print("BigQuery returned no results. Aborting report generation.")
        return

    log_status("Building memory context and prompt")
    memory_context = _memory_context_to_prompt(memory_module, receiver_header, sender)
    data_payload = format_data_payload(results)
    prompt = build_ollama_prompt(settings, data_payload, memory_context=memory_context)

    try:
        log_status(f"Generating report with Ollama model '{model_name}' using timeout={ollama_timeout}s")
        report_text = query_ollama(prompt, ollama_url, model_name, stream=ollama_stream, timeout=ollama_timeout)
    except Exception as exc:
        print(f"Failed to query Ollama: {exc}")
        return

    if not report_text or report_text == FAILED_REPORT_TEXT:
        print("Ollama returned an empty or failed report. Aborting.")
        return

    log_status(f"Received report from Ollama: {len(report_text)} characters")
    msg_root = build_email_message(sender, receiver_header, report_text)

    try:
        log_status(f"Sending email to {len(receivers)} recipients")
        send_email(sender, receivers, password, msg_root)
        log_status("Email sent successfully")
    except Exception as exc:
        print(f"Failed to send email: {exc}")
        return

    memory_module = open_report_memory(memory_schema)
    if memory_module is not None:
        log_status("Saving report to long/short-term memory")
        remember_report(memory_module, sender, receivers, receiver_header, prompt, report_text, model_name, ollama_url, settings)
        log_status("Memory persistence complete")

    total_elapsed = time.perf_counter() - start_time
    log_status(f"Weekly report run complete: total elapsed={total_elapsed:.1f}s", start_time)


if __name__ == "__main__":
    # Show progress lines immediately even when output is redirected to a file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    run_weekly_report()
