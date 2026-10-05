import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import json
import importlib.util
import os
import sys
import re
import time
import traceback
from typing import Any

from finance.ollama import embed_text, generate_text
from finance.settings import load_config as _shared_load_config
from finance.settings import parse_money


def _parse_money(value: Any) -> float:
    """Normalize a config value into a float so the deterministic financial snapshot is safe."""
    return parse_money(value)


def log_status(message: str, start_time: float | None = None) -> None:
    """Write a timestamped status message to stdout for debugging and auditability."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if start_time is not None:
        suffix = f" | elapsed={time.perf_counter() - start_time:.1f}s"
    print(f"[{timestamp}] {message}{suffix}", flush=True)


def load_config(config_path: Path) -> dict:
    """Read the active runtime config from disk so the script can run in the right mode."""
    return _shared_load_config(config_path)


def fetch_bigquery_summary(
    project_name: str,
    days_back: int = 60
) -> list:
    """Read the latest gold-layer monthly summary rows for the report snapshot."""
    
    from google.cloud import bigquery

    query = f"""
        SELECT transaction_month, category, parent_group, gross_spending,
               total_refunds_and_credits, net_real_expense, 
               total_transaction_count, average_transaction_amount,
               SUM(net_real_expense) OVER (PARTITION BY transaction_month)
                   AS monthly_total_net_real_expense
        FROM `{project_name}.gold.fact_monthly_spending`
        WHERE transaction_month >= CURRENT_DATE() - INTERVAL {days_back} DAY
        ORDER BY transaction_month DESC, net_real_expense DESC
    """
    
    bq_client = bigquery.Client(project=project_name)
    query_job = bq_client.query(query)
    results = list(query_job.result())
    
    return results


def format_data_payload(results: list) -> str:
    """Turn gold-layer data into a concise, authoritative prompt payload for the model."""
    if not results:
        return "No current spending data was returned."

    # Build the authoritative prompt payload from BigQuery results. The model is told
    # that these values are the only valid source for current-period calculations, which
    # keeps historical memory and prior HTML from being mistaken for current facts.
    months = sorted({row.transaction_month for row in results}, reverse=True)
    latest_month = months[0]
    previous_month = months[1] if len(months) > 1 else None
    totals = {
        row.transaction_month: row.monthly_total_net_real_expense
        for row in results
    }

    data_payload = (
        "AUTHORITATIVE CURRENT DATA - use these values for every calculation. "
        "Do not use financial numbers from prior context or prior reports.\n"
        f"Latest available month: {latest_month.strftime('%B %Y')}\n"
        f"Latest month total net real expense: ${totals[latest_month]:.2f}\n"
    )
    if previous_month is not None:
        data_payload += (
            f"Previous month: {previous_month.strftime('%B %Y')}\n"
            f"Previous month total net real expense: ${totals[previous_month]:.2f}\n"
        )
    data_payload += f"Category detail as of {date.today().strftime('%Y-%m-%d')}:\n"
    for row in results:
        data_payload += (
            f"- {row.transaction_month.strftime('%B %Y')}: {row.category}: "
            f"${row.net_real_expense:.2f} across {row.total_transaction_count} transactions\n"
        )

    return data_payload


def _memory_context_to_prompt(memory_module: Any, receiver_header: str, sender: str, limit: int = 20, recent_limit: int = 1) -> str:
    """Pull recent thread history only; long-term facts are retrieved on demand by agent tools."""
    if memory_module is None:
        return ""

    lines: list[str] = []

    try:
        thread_key = receiver_header or sender or "email:unknown"
        thread_id = memory_module.get_or_create_thread("email", thread_key)
        recent_entries = memory_module.get_short_term(thread_id, limit=recent_limit)
        if recent_entries:
            if not lines:
                lines.append("Relevant prior memory/context:")
            lines.append("Recent weekly thread history:")
            # Keep the previous report only as UI and style continuity. The current report
            # still must be anchored to the fresh gold-layer numbers.
            for entry in recent_entries:
                subject = entry.get("subject") or "Weekly report"
                text = entry.get("response_html") or entry.get("response_text") or ""
                if text:
                    lines.append(f"Previous complete report ({subject}) - use only as a historical reference for layout and goals; do not reuse its financial numbers:\n{text.strip()}")
    except Exception as exc:
        print(f"Recent-thread memory context lookup failed: {exc}")

    return "\n".join(lines)


def build_ollama_prompt(config: dict, data_payload: str, memory_context: str | None = None) -> str:
    """Compose the final prompt so the model knows the objective, constraints, and data sources."""
    memory_context = memory_context or ""
    return f"""System: You are Mogumogu-chan, a warm, lively personal-finance assistant. Open with a friendly greeting addressed to the reader, use natural Japanese food imagery and a few fitting food emoji (such as 🍡, 🍙, 🍵, 🍣, 🌸), and close with kind encouragement. Keep the personality personable, not childish; financial facts must stay clear and accurate.
Create a consistent, readable HTML email using Material 3-inspired hierarchy and spacing without making it a generic gray/blue dashboard. Center one content column (about 640 px wide) on a clearly visible pale blush-pink page background (such as #FFF1F5) with a white or near-white report surface. Use dark warm charcoal or berry text and bright rose/hot-pink/coral as the primary accents; avoid dominant purple/plum tones. Carry pink through section markers, the hero, and all snapshot bars. Use one coordinated family of rose shades for positive-spend bars, never green for Needs or peach for Wants. In every chart row, align category labels left, place a proportional bar in the middle, and put the exact currency amount in a fixed-width right-aligned column sharing one right edge. Scale all positive bars against the same maximum category spend across all groups; handle negative/refund categories separately. Maintain the established 8 px spacing rhythm and readable type sizes. Keep secondary data grouped and scannable, but do not omit any required section or metric for compactness. Avoid clutter, tiny text, repeated heavy shadows, nested cards, and excessive emoji. Reserve green/red for financial direction and label trends in words. Tables should have comfortable padding, subtle separators, and clearly aligned numbers. Keep contrast strong and use email-safe HTML/inline styles.

Instructions:
- Include all available main category groups (for example Needs/Essentials and Wants/Discretionary) in the category overview. Uncontrollable/fixed categories may appear in the overview for transparency but must not be presented as discretionary targets or blamed on the reader.
- Give a comparison of the current month to the previous month, highlighting any significant changes in spending patterns.
- Give a brief overview of the financial health, focusing on net expenses and budget adherence.
- Use a warm, encouraging tone, and provide actionable insights or tips for better financial management.
- Consider the relevant prior memory/context below when interpreting the current data and writing the email.
- The Data section below is the only authoritative source for this report's financial calculations. Treat every number, month, KPI, trend, and recommendation in prior memory as historical; recalculate all current and previous-month values from the Data section.
- In the Data section, use the explicit Latest month total net real expense and Previous month total net real expense values for the KPI cards and month-over-month calculation. Never substitute a total from prior memory or prior report HTML.
- Prior report HTML is included only to preserve the established UI layout, section order, tone, and goal-suggestion style. Do not copy its figures or claim that its values belong to the current snapshot.
- Do not include markdown or code blocks in the output. The output should be a polished HTML email body with clear sections and a warm, encouraging tone.

Context:
- Monthly deposit: {config.get('Monthly_budget')}.
- Monthly Fixed costs: {config.get('Fixed_costs')}.
- Vehicles: {config.get('Vehicle_Info')}.
- Authoritative numeric monthly savings target: {config.get('Savings_goal_target', 'not configured')}.
- This is a follow-up report; the recipient already knows the recurring details.
{memory_context}
Include these sections in this exact order; do not omit them:
1. Friendly Mogumogu greeting with a Japanese-inspired food emoji and brief month-specific welcome.
2. Monthly Snapshot: label the current month and show each available lifestyle category grouped by its main Needs/Essentials or Wants/Discretionary parent group. Include a horizontal pink bar per category plus its exact spend amount; bar lengths share one scale. Do not substitute a table or prose for the bars. Keep values readable independently of bar color/length.
3. Spending Trends & Analysis: explain the important supported month-over-month changes, including the direction and context.
4. Mogumogu's Financial Health Check: a brief, warm summary grounded in the data.
5. The Sushi Stack: a visibly grouped vertical KPI comparison showing latest-month total net expense, previous-month total net expense, MoM percent change, and disposable cash. Use the current and previous month labels and exact values. Disposable cash is monthly deposit - (current net expense + fixed costs); it may be negative. Add a short calculation note. Keep the current total primary and the other three subordinate, but include all four.
6. Top 3 Discretionary Categories: a table with up to the three highest-spend eligible discretionary categories, current spend, previous spend, MoM trend, target/goal, and percentage of goal where supported. Prefer configured category goals; otherwise derive a suggested cap using the numeric savings target and spending trends, and label it as a suggestion, not a configured goal. Do not classify fixed/uncontrollable categories as discretionary or invent values; label unavailable historical/target inputs clearly. Calculate projected savings as monthly deposit - (latest total net expense + fixed costs). If below the numeric savings target, allocate that shortfall proportionally across all current discretionary/Wants spend; each cap is current category spend minus its proportional cut, floored at zero. If projected savings already meet the target, use previous-month spend as a cap for categories that increased, otherwise maintain current spend. Explain the basis, show current spend as a percentage of the cap only when the cap is positive, and note if discretionary cuts alone cannot reach the target.
7. Mogumogu's Tips for the Week: one or two actionable, kind tips tied to the evidence, ending with a supportive sign-off and a small Japanese-food emoji.

Today's date: {date.today().strftime('%Y-%m-%d')}
Data:
{data_payload}

Please write a polished HTML email body with clear sections and a warm, encouraging tone. No Markdown syntax. This means do not include * or ** in responses."""


def query_ollama(prompt: str, ollama_url: str, model_name: str, stream: bool = False, timeout: int = 600) -> str:
    """Send a prompt to Ollama and return the generated text, including a timeout for slow models."""
    request_started = time.perf_counter()
    log_status(f"Querying Ollama model '{model_name}' at {ollama_url} (stream={stream}, timeout={timeout}s)")
    try:
        result = generate_text(prompt, ollama_url, model_name, timeout=timeout, stream=stream)
        elapsed = time.perf_counter() - request_started
        log_status(f"Ollama response received: bytes={len(result)}, elapsed={elapsed:.1f}s")
        return result
    except Exception as exc:
        log_status(f"Ollama request failed with exception: {exc}")
        return "Failed to generate report."


def build_email_message(sender: str, receiver_header: str, report_text: str, plain_text: str | None = None) -> MIMEMultipart:
    """Wrap the HTML report in a multipart email so it works in both text and HTML clients."""
    msg_root = MIMEMultipart('related')
    msg_root['From'] = sender
    msg_root['To'] = receiver_header
    msg_root['Subject'] = f"Weekly Financial Progress Report Card {date.today().strftime('%Y-%m-%d')}"

    msg_alt = MIMEMultipart('alternative')
    msg_root.attach(msg_alt)

    if plain_text and plain_text.strip():
        # Agent mode supplies a validated plain-text rendering of the report.
        plain_body = plain_text.strip()
    else:
        plain_lines = [f"Weekly Financial Progress — {date.today().strftime('%B %d, %Y')}", ""]
        rpt_first = report_text.split('\n\n')[0] if report_text else ''
        if rpt_first:
            plain_lines.extend([rpt_first, ""])
        plain_body = "\n".join(plain_lines)
    msg_alt.attach(MIMEText(plain_body, 'plain'))
    msg_alt.attach(MIMEText(report_text, 'html'))

    return msg_root


def send_email(sender: str, receivers: list[str], password: str, msg_root: MIMEMultipart) -> None:
    """Deliver the final report via the configured Gmail account."""
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receivers, msg_root.as_string())


def _init_memory_module(memory_schema: str = "memory"):
    """Load the SQLite-backed memory module for the selected schema if it exists."""
    try:
        if not re.fullmatch(r"[A-Za-z0-9_]+", memory_schema):
            raise ValueError("memory_schema may contain only letters, numbers, and underscores")

        memory_path = Path(__file__).with_name("memory.py")
        # MEMORY_DB_PATH overrides the schema-derived default database location.
        database_path = (
            Path(os.environ["MEMORY_DB_PATH"])
            if os.environ.get("MEMORY_DB_PATH")
            else memory_path.parent.parent / "agent" / "memory" / f"{memory_schema}.sqlite3"
        )
        spec = importlib.util.spec_from_file_location("memory_module", str(memory_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.init_db(str(database_path))
        log_status(f"Using memory schema '{memory_schema}' at {database_path}")
        return module
    except Exception as exc:
        print(f"Memory initialization failed: {exc}")
        return None


def _extract_json_payload(payload: str) -> list[dict[str, Any]]:
    """Pull a JSON list out of a noisy Ollama response so memory facts can be extracted reliably."""
    try:
        data = json.loads(payload)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[\s*\{.*?\}\s*\]", payload, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def _ollama_embedding(text: str, ollama_url: str, model_name: str, timeout: int = 600) -> list[float]:
    """Request one vector from Ollama's /api/embed endpoint."""
    return embed_text(text, ollama_url, model_name, timeout)


def _remember_report(memory_module: Any, sender: str, receivers: list[str], receiver_header: str, prompt: str, report_text: str, model_name: str, ollama_url: str, config: dict, report_plain_text: str | None = None) -> None:
    """Persist the report in short-term memory and extract reusable long facts for future runs.

    report_text is the report HTML. When report_plain_text is supplied (agent mode), it is
    stored as the readable text and used for fact extraction instead of raw HTML.
    """
    if memory_module is None:
        return

    try:
        log_status("Starting memory persistence for this report")
        thread_id = memory_module.get_or_create_thread("email", receiver_header or sender or "email:unknown")
        # Keep the complete report available to the next run for UI and goal continuity.
        has_plain_text = bool(report_plain_text and report_plain_text.strip())
        plain_text = report_plain_text if has_plain_text else (report_text or "")

        st_id = memory_module.save_short_term(
            thread_internal_id=thread_id,
            subject=f"Weekly Financial Progress Report Card {date.today().strftime('%Y-%m-%d')}",
            sender=sender,
            prompt=prompt,
            response_html=report_text,
            response_text=plain_text,
            tags=["weekly_report"],
            meta={"recipients": receivers},
        )
        log_status(f"Saved short-term memory row with thread id={thread_id}, short_term id={st_id}")

        fact_model = config.get("fact_model_name", model_name)
        embedding_model = config.get("embedding_model_name", "nomic-embed-text")
        embedding_timeout = int(config.get("ollama_timeout", 600))
        fact_prompt = (
            f"Extract a JSON array of concise facts from the following {'plain-text' if has_plain_text else 'HTML'} report. "
            "Each item must be an object with keys 'fact_key', 'fact_text', and optional 'weight'.\n\n"
            f"Report:\n{plain_text}\n\nReturn ONLY valid JSON."
        )
        log_status(f"Extracting facts from completed report using model '{fact_model}'")
        fact_res = query_ollama(fact_prompt, ollama_url, fact_model, stream=False)
        facts = _extract_json_payload(fact_res)
        log_status(f"Fact extraction returned {len(facts)} candidate fact(s)")
        # Each extracted fact is stored as a reusable long-term memory item so future runs
        # can inherit patterns and goals without relying on the previous raw numbers.
        for fact in facts:
            try:
                fk = str(fact.get("fact_key") or fact.get("key") or "").strip()
                ft = str(fact.get("fact_text") or fact.get("text") or "").strip()
                if not fk or not ft:
                    continue
                w = float(fact.get("weight", 1.0)) if fact.get("weight") is not None else 1.0
                try:
                    embedding = _ollama_embedding(ft, ollama_url, embedding_model, embedding_timeout)
                except Exception as embedding_exc:
                    log_status(f"Could not embed long fact '{fk}'; it will be backfilled on retrieval: {embedding_exc}")
                    embedding = None
                memory_module.upsert_long_fact(
                    fk,
                    ft,
                    source_short_term_id=st_id,
                    weight=w,
                    embedding=embedding,
                    embedding_model=embedding_model if embedding is not None else None,
                )
            except Exception as fact_exc:
                log_status(f"Could not persist long fact: {fact_exc}")
                continue
        log_status("Completed memory persistence and fact extraction")
    except Exception as exc:
        print(f"Memory logging failed: {exc}")


RUN_ARCHIVE_ROOT = Path(__file__).resolve().parent.parent / "agent" / "runs"


def _start_progress(archive_root: Path, run_id: str, model_name: str, validator_model: str) -> Any:
    """Create the run directory and live progress files up front; a failure only disables them."""
    from run_archive import ProgressRecorder, update_index

    try:
        progress = ProgressRecorder(archive_root / run_id, run_id, model_name, validator_model)
        update_index(archive_root)
        log_status(f"Live run progress: {progress.run_dir / 'progress.log'}")
        return progress
    except Exception as exc:
        print(f"Live run progress unavailable; continuing without it: {exc}", flush=True)
        return None


def run_agent_weekly_report(
    config: dict,
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
) -> None:
    """Generate the latest-month agent report, archive it, and handle email delivery."""
    from finance.reporting import generate_finance_report
    from run_archive import save_run

    started = time.perf_counter()
    thread_key = receiver_header or sender or "email:unknown"
    validator_model = config.get("validator_model_name", model_name)
    archive_root = RUN_ARCHIVE_ROOT
    # One run directory is created at start so progress can be watched while the run executes.
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + f"_{model_name.replace(':', '_')}"
    progress = _start_progress(archive_root, run_id, model_name, validator_model)
    try:
        result = generate_finance_report(
            config=config,
            project_name=project_name,
            ollama_url=ollama_url,
            model_name=model_name,
            memory_module=memory_module,
            thread_key=thread_key,
            trace_listeners=[progress] if progress is not None else None,
        )
    except Exception as exc:
        print(f"Shared finance report generation failed: {exc}", flush=True)
        failed_trace = getattr(exc, "run_trace", None)
        failed_snapshot = getattr(exc, "report_snapshot", None)
        # Keep the _failed suffix by renaming the live directory; if that is not possible,
        # keep the original directory and record the failure in its status files.
        failed_run_id = f"{run_id}_failed"
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
        failed_metrics = {
            "model": model_name,
            "validator_model": validator_model,
            "mode": "agent",
            "status": "exception",
            "validation": "error",
            "exception_type": type(exc).__name__,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "agent_rounds": sum(event.get("event") == "agent_round" for event in events),
            "tool_calls": sum(event.get("event") == "tool_call" for event in events),
            "validation_cycles": sum(event.get("event") == "validation_cycle_result" for event in events),
            "validator_calls": sum(event.get("event") == "model_call" and event.get("role") == "validator" for event in events),
            "report_bytes": 0,
        }
        if progress is not None:
            progress.finish("exception", error=f"{type(exc).__name__}: {exc}")
        save_run(
            archive_root,
            run_id,
            {
                "run_id": run_id,
                "mode": "agent",
                "status": "exception",
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "snapshot": failed_snapshot.to_dict() if failed_snapshot is not None else None,
            },
            events,
            "",
            "",
            failed_metrics,
        )
        print(f"Failed agent run trace archived at {archive_root / run_id}")
        return

    save_run(
        archive_root,
        run_id,
        {
            "run_id": run_id,
            "mode": "agent",
            "status": result.metrics["status"],
            "snapshot": result.snapshot.to_dict(),
        },
        result.trace.events,
        result.report_html,
        result.plain_text,
        result.metrics,
    )
    print(f"Agent run archived at {archive_root / run_id}")
    if not result.passed:
        print("Report was not sent because validation did not pass.")
        if progress is not None:
            progress.finish(result.metrics["status"], delivery="not_sent")
        return
    if config.get("dry_run", False):
        print("Dry run enabled; report was rendered and archived but not sent.")
        if progress is not None:
            progress.finish(result.metrics["status"], delivery="dry_run")
        return

    msg_root = build_email_message(sender, receiver_header, result.report_html, plain_text=result.plain_text)
    try:
        send_email(sender, receivers, password, msg_root)
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
        save_run(
            archive_root,
            run_id,
            {
                "run_id": run_id,
                "mode": "agent",
                "status": "delivery_failed",
                "snapshot": result.snapshot.to_dict(),
                "error": str(exc),
            },
            result.trace.events,
            result.report_html,
            result.plain_text,
            result.metrics,
        )
        print(f"Email delivery failed; updated run trace archived at {archive_root / run_id}: {exc}")
        if progress is not None:
            progress.finish("delivery_failed", delivery="failed", error=f"{type(exc).__name__}: {exc}")
        return
    if progress is not None:
        progress.finish(result.metrics["status"], delivery="sent")
    memory_module = _init_memory_module(memory_schema)
    _remember_report(
        memory_module,
        sender,
        receivers,
        receiver_header,
        "Create the validated weekly finance report.",
        result.report_html,
        model_name,
        ollama_url,
        config,
        report_plain_text=result.plain_text,
    )
    log_status(f"Shared finance report and email workflow complete in {time.perf_counter() - started:.1f}s")


def run_weekly_report(config_path: Path | None = None) -> None:
    """Orchestrate the weekly report generation and email send."""
    start_time = time.perf_counter()
    log_status("Starting weekly report run", start_time)

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"

    try:
        config = load_config(config_path)
        log_status(f"Loaded config from {config_path}")
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return

    # Memory module is optional; initialize lazily if present
    memory = None

    receivers = config.get("email_listings", [])
    sender = config.get("email_sender")
    receiver_header = ", ".join(receivers)
    password = config.get("Gmail_app_credentials")
    project_name = config.get("project_name")
    ollama_url = config.get("model_server_url")
    model_name = config.get("model_name")
    days_back = config.get("days_back", 60)
    ollama_stream = config.get("ollama_stream", False)
    ollama_timeout = int(config.get("ollama_timeout", 600))
    memory_schema = config.get("memory_schema", "memory")
    report_mode = config.get("report_mode", "legacy").lower()

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
    memory_module = _init_memory_module(memory_schema)
    if report_mode == "agent":
        # The shared report service resolves the latest month directly; do not constrain
        # the email run to the legacy rolling days_back window.
        run_agent_weekly_report(
            config, [], memory_module, sender, receivers, receiver_header,
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
    prompt = build_ollama_prompt(config, data_payload, memory_context=memory_context)

    try:
        log_status(f"Generating report with Ollama model '{model_name}' using timeout={ollama_timeout}s")
        report_text = query_ollama(prompt, ollama_url, model_name, stream=ollama_stream, timeout=ollama_timeout)
    except Exception as exc:
        print(f"Failed to query Ollama: {exc}")
        return

    if not report_text or report_text == "Failed to generate report.":
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

    memory_module = _init_memory_module(memory_schema)
    if memory_module is not None:
        log_status("Saving report to long/short-term memory")
        _remember_report(memory_module, sender, receivers, receiver_header, prompt, report_text, model_name, ollama_url, config)
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
