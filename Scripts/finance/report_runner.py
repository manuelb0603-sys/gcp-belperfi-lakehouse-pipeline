"""Agent-mode weekly email run: live progress, report generation, archive, delivery, and memory.

run_agent_weekly_report creates the run folder at start (progress.json/progress.log/trace.jsonl
stream while it runs), generates the report through finance.reporting.generate_finance_report,
archives it (renaming the folder with a "_failed" suffix on an unhandled exception), reports a
budget stop or failed validation as not sent, respects dry_run, sends the email, and only then
saves the report to memory.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import run_archive
from .budget import anthropic_cost_from_events, configured_ledger_path
from .email_delivery import build_email_message, send_email
from .report_memory import open_report_memory, remember_report
from .settings import ReportSettings
from .status_log import log_status

MEMORY_PROMPT = "Create the validated weekly finance report."


def budget_stop_message(result: Any) -> str | None:
    """Why a Claude spend cap stopped the report (result.metrics["budget_exceeded"]), else None."""
    return result.metrics.get("budget_exceeded") or None


def start_progress(
    archive_root: Path,
    run_id: str,
    model_name: str,
    validator_model: str,
    author_provider: str = "ollama",
    validator_provider: str = "ollama",
    metadata: dict[str, Any] | None = None,
    ledger_path: Path | None = None,
) -> Any:
    """Create the run directory and live progress files up front; a failure only disables them."""
    try:
        progress = run_archive.ProgressRecorder(
            archive_root / run_id, run_id, model_name, validator_model,
            author_provider=author_provider, validator_provider=validator_provider,
            metadata=metadata,
        )
        run_archive.update_index(archive_root, ledger_path=ledger_path)
        log_status(f"Live run progress: {progress.run_dir / 'progress.log'}")
        return progress
    except Exception as exc:
        print(f"Live run progress unavailable; continuing without it: {exc}", flush=True)
        return None


def _failure_metrics(exc: Exception, events: list[dict[str, Any]], author_role: dict[str, Any], validator_role: dict[str, Any], started: float) -> dict[str, Any]:
    """metrics.json for a run that ended with an unhandled exception."""
    return {
        "model": str(author_role["model"]),
        "validator_model": str(validator_role["model"]),
        "author_provider": author_role["provider"],
        "validator_provider": validator_role["provider"],
        "mode": "agent",
        "status": "exception",
        "validation": "error",
        "exception_type": type(exc).__name__,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        **run_archive.metrics_from_trace(events),
        "report_bytes": 0,
        "anthropic_cost_usd": anthropic_cost_from_events(events),
    }


def run_agent_weekly_report(
    config: dict | ReportSettings,
    results: list,
    memory_module: Any,
    sender: str,
    receivers: list[str],
    receiver_header: str,
    password: str,
    project_name: str,
    ollama_url: str,
    model_name: str,
    memory_schema: str,
    *,
    archive_root: Path | None = None,
    send: Callable[..., None] | None = None,
) -> None:
    """Generate the latest-month agent report, archive it, and handle email delivery.

    archive_root defaults to agent/runs; send defaults to email_delivery.send_email (the email
    entry point passes its own module attributes so tests can patch them there).
    """
    from . import reporting  # looked up per call so tests can patch generate_finance_report

    archive_root = Path(archive_root) if archive_root is not None else run_archive.DEFAULT_RUNS_ROOT
    send = send or send_email
    started = time.perf_counter()
    thread_key = receiver_header or sender or "email:unknown"
    # Provider and model per role (model_roles, else the legacy model_name/validator_model_name keys).
    settings = reporting.role_config_for(config, ollama_url, model_name)
    author_role = settings.role("author")
    validator_role = settings.role("validator")
    ledger_path = configured_ledger_path(settings)
    # Code version, role settings, and limits recorded at start so run comparisons can tell
    # code changes from model changes (never secrets or server addresses).
    run_metadata = run_archive.collect_run_metadata(settings)
    author_model = str(author_role["model"])
    validator_model = str(validator_role["model"])
    # One run directory is created at start so progress can be watched while the run executes.
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + f"_{author_model.replace(':', '_')}"
    progress = start_progress(
        archive_root, run_id, author_model, validator_model, author_role["provider"], validator_role["provider"],
        metadata=run_metadata, ledger_path=ledger_path,
    )

    def archive(manifest: dict[str, Any], events: list[dict[str, Any]], report_html: str, plain_text: str, metrics: dict[str, Any]) -> None:
        run_archive.save_run(
            archive_root, run_id, {"run_id": run_id, "mode": "agent", **manifest},
            events, report_html, plain_text, metrics, ledger_path=ledger_path,
        )

    try:
        result = reporting.generate_finance_report(
            config=settings,
            project_name=project_name,
            ollama_url=ollama_url,
            model_name=model_name,
            memory_module=memory_module,
            thread_key=thread_key,
            trace_listeners=[progress] if progress is not None else None,
            run_id=run_id,  # the Claude run cap (anthropic_run_budget_usd) is counted per run id
        )
    except Exception as exc:
        print(f"Shared finance report generation failed: {exc}", flush=True)
        failed_trace = getattr(exc, "run_trace", None)
        failed_snapshot = getattr(exc, "report_snapshot", None)
        # Keep the _failed suffix by renaming the live directory; if that is not possible,
        # keep the original directory and record the failure in its status files.
        failed_run_id = f"{run_id}{run_archive.FAILED_SUFFIX}"
        if progress is None or progress.move_to(archive_root / failed_run_id):
            run_id = failed_run_id
        events = failed_trace.events if failed_trace is not None else []
        events.append({
            "event": "exception",
            "timestamp": time.time(),
            "phase": "finance_report_generation_unhandled",
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        events.append({"event": "run_aborted", "timestamp": time.time(), "phase": "finance_report_generation"})
        failed_metrics = _failure_metrics(exc, events, author_role, validator_role, started)
        if progress is not None:
            progress.finish("exception", error=f"{type(exc).__name__}: {exc}")
        archive(
            {
                "status": "exception",
                "exception_type": type(exc).__name__,
                "error": str(exc),
                **run_metadata,
                "snapshot": failed_snapshot.to_dict() if failed_snapshot is not None else None,
            },
            events, "", "", failed_metrics,
        )
        print(f"Failed agent run trace archived at {archive_root / run_id}")
        return

    archive(
        {"status": result.metrics["status"], **run_metadata, "snapshot": result.snapshot.to_dict()},
        result.trace.events, result.report_html, result.plain_text, result.metrics,
    )
    print(f"Agent run archived at {archive_root / run_id}")
    if not result.passed:
        budget_message = budget_stop_message(result)
        if budget_message:
            # A Claude spend cap stopped the run: fail decision, nothing is sent.
            print(f"Report was not sent: {budget_message}")
        else:
            print("Report was not sent because validation did not pass.")
        if progress is not None:
            details = {"delivery": "not_sent"}
            if budget_message:
                details["budget"] = budget_message
            progress.finish(result.metrics["status"], **details)
        return
    if settings.dry_run:
        print("Dry run enabled; report was rendered and archived but not sent.")
        if progress is not None:
            progress.finish(result.metrics["status"], delivery="dry_run")
        return

    msg_root = build_email_message(sender, receiver_header, result.report_html, plain_text=result.plain_text)
    try:
        send(sender, receivers, password, msg_root)
    except Exception as exc:
        result.trace.add(
            "exception",
            phase="email_delivery",
            exception_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        result.trace.add("run_aborted", phase="email_delivery", status="delivery_failed")
        result.metrics["status"] = "delivery_failed"
        result.metrics["delivery_error_type"] = type(exc).__name__
        archive(
            {"status": "delivery_failed", **run_metadata, "snapshot": result.snapshot.to_dict(), "error": str(exc)},
            result.trace.events, result.report_html, result.plain_text, result.metrics,
        )
        print(f"Email delivery failed; updated run trace archived at {archive_root / run_id}: {exc}")
        if progress is not None:
            progress.finish("delivery_failed", delivery="failed", error=f"{type(exc).__name__}: {exc}")
        return
    if progress is not None:
        progress.finish(result.metrics["status"], delivery="sent")
    remember_report(
        open_report_memory(memory_schema),
        sender,
        receivers,
        receiver_header,
        MEMORY_PROMPT,
        result.report_html,
        model_name,
        ollama_url,
        settings,
        report_plain_text=result.plain_text,
    )
    log_status(f"Shared finance report and email workflow complete in {time.perf_counter() - started:.1f}s")
