import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import json
import importlib.util
import re
import time
from typing import Any, Dict


def _parse_money(value: Any) -> float:
    """Normalize a config value into a float so the deterministic financial snapshot is safe."""
    # Keep this logic deterministic and local. The model should not be the source of
    # truth for the financial snapshot; we normalize user config values before they are
    # used in the authoritative calculations.
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?\s*[kKmM]?", str(value or ""))
    if not matches:
        raise ValueError(f"Could not parse a financial amount from: {value!r}")
    total = 0.0
    for match in matches:
        normalized = match.replace(" ", "")
        multiplier = 1
        if normalized[-1:].lower() == "k":
            multiplier = 1_000
            normalized = normalized[:-1]
        elif normalized[-1:].lower() == "m":
            multiplier = 1_000_000
            normalized = normalized[:-1]
        total += float(normalized) * multiplier
    return total


def _load_prompt_files() -> tuple[str, str]:
    """Load author context separately from the validator-only system prompt."""
    prompt_dir = Path(__file__).resolve().parent.parent / "agent" / "prompts"
    author_names = ["personality.system.md", "ui.system.md", "financial-settings.md"]
    author_context = "\n\n".join((prompt_dir / name).read_text(encoding="utf-8") for name in author_names)
    validator_context = (prompt_dir / "validator.system.md").read_text(encoding="utf-8")
    return author_context, validator_context


def log_status(message: str, start_time: float | None = None) -> None:
    """Write a timestamped status message to stdout for debugging and auditability."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if start_time is not None:
        suffix = f" | elapsed={time.perf_counter() - start_time:.1f}s"
    print(f"[{timestamp}] {message}{suffix}")


def load_config(config_path: Path) -> dict:
    """Read the active runtime config from disk so the script can run in the right mode."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    """Pull remembered facts and prior report history into the prompt without treating them as new truth."""
    if memory_module is None:
        return ""

    lines: list[str] = []

    try:
        facts = memory_module.get_long_facts(limit=limit)
        if facts:
            lines.append("Relevant prior memory/context:")
            # Long-term facts are used as background context, not as the current financial
            # truth source. They help preserve the user's habits and goals while the gold
            # data stays authoritative.
            for fact in facts:
                key = fact.get("fact_key") or fact.get("key") or "memory"
                text = fact.get("fact_text") or fact.get("text") or ""
                if text:
                    lines.append(f"- {key}: {text}")
    except Exception as exc:
        print(f"Long-term memory context lookup failed: {exc}")

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
    return f"""System: You are a friendly financial assistant hungry for insights named Mogumogu-chan, writing a short, clear email. Mogumogu is the Japanese onomatopoeia for a cute, playful eating. You consumed the latest financial digest to share. Some of your playful puns are related to japanese foods. This is an automated insights weekly script about this month's expenses progress. Output a nice trendy graphical representation in HTML for the email.
Use a soft pink page background for the whole email body.
Place the content inside a centered white/light-pink panel with rounded corners and a thin light pink border.
The panel should look like a lined container: subtle border, gentle shadow, and enough padding around the sections.
Add thin pink separator lines between sections. Don't use white text.
Percent change should be color-coded: green for positive change, red for negative change, and gray for no change.

Instructions:
- Don't mention fixed costs, insurance, uncontrollable category expenses like Health & Wellness, Payments, Bills, Utilities, Professional & Business Services and Automotive. However they are included in terms of expenses for Net Real Expense.
- Don't mention monthly deposits, the reader already knows about them.
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
- Savings goal: {config.get('Savings_goal')}.
- This is a follow-up report; the recipient already knows the recurring details.
{memory_context}
Include In Order: 
- Fun greeting
- Monthly Snapshot including and the Month for that data and KPI Card of comparison of lifestyle net expenses accross the categories including the number (horizontal pink color  bars).
- Spending Trends & Analysis
- MoguMogu's Financial Health Check
- KPI Cards (The Sushi Stack (Key Metrics)) Vertical layout of:
  Total Net Expense Spend (Net Real Expense from all categories of the latest month)
  Previous Month Total Net Expense Spend (Net Real Expense from all categories of the previous month)
  Mom % Change (Net Real Expense from all categories)
  Disposable Cash which can go negative. You must use the formula for disposable Cash: (Monthly deposit - (net real expense + Monthly fixed costs)) Explain how you got this number so I can troubleshoot your calculations.
- KPI Card of Top 3 Discretionary Categories [Table] (total spent, previous spent, Mom% Trend with color coding, targeted goal suggestion, and percentage of goal spent).
Explain why the goal suggestion is what it is, and how to improve the spending habits for that category.
- Momogu's Tips for the Week.

Today's date: {date.today().strftime('%Y-%m-%d')}
Data:
{data_payload}

Please write a polished HTML email body with clear sections and a warm, encouraging tone. No Markdown syntax. This means do not include * or ** in responses."""


def query_ollama(prompt: str, ollama_url: str, model_name: str, stream: bool = False, timeout: int = 600) -> str:
    """Send a prompt to Ollama and return the generated text, including a timeout for slow models."""
    import requests

    request_started = time.perf_counter()
    log_status(f"Querying Ollama model '{model_name}' at {ollama_url} (stream={stream}, timeout={timeout}s)")
    try:
        response = requests.post(ollama_url, json={
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }, timeout=timeout)
        elapsed = time.perf_counter() - request_started
        log_status(
            f"Ollama response received: status={response.status_code}, bytes={len(response.text or '')}, elapsed={elapsed:.1f}s"
        )
        if response.status_code != 200:
            print(f"Ollama request failed with HTTP {response.status_code}: {response.text[:500]}")
            return "Failed to generate report."
        try:
            payload = response.json()
        except ValueError:
            print(f"Ollama returned invalid JSON: {response.text[:500]}")
            return "Failed to generate report."
        return payload.get("response", "Failed to generate report.")
    except requests.exceptions.Timeout:
        log_status(f"Ollama request timed out after {time.perf_counter() - request_started:.1f}s")
        return "Failed to generate report."
    except Exception as exc:
        log_status(f"Ollama request failed with exception: {exc}")
        return "Failed to generate report."


def build_email_message(sender: str, receiver_header: str, report_text: str) -> MIMEMultipart:
    """Wrap the HTML report in a multipart email so it works in both text and HTML clients."""
    msg_root = MIMEMultipart('related')
    msg_root['From'] = sender
    msg_root['To'] = receiver_header
    msg_root['Subject'] = f"Weekly Financial Progress Report Card {date.today().strftime('%Y-%m-%d')}"

    msg_alt = MIMEMultipart('alternative')
    msg_root.attach(msg_alt)

    plain_lines = [f"Weekly Financial Progress — {date.today().strftime('%B %d, %Y')}", ""]
    rpt_first = report_text.split('\n\n')[0] if report_text else ''
    if rpt_first:
        plain_lines.extend([rpt_first, ""])

    plain_text = "\n".join(plain_lines)
    msg_alt.attach(MIMEText(plain_text, 'plain'))
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
        database_path = memory_path.parent.parent / "agent" / "memory" / f"{memory_schema}.sqlite3"
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


def _remember_report(memory_module: Any, sender: str, receivers: list[str], receiver_header: str, prompt: str, report_text: str, model_name: str, ollama_url: str, config: dict) -> None:
    """Persist the report in short-term memory and extract reusable long facts for future runs."""
    if memory_module is None:
        return

    try:
        log_status("Starting memory persistence for this report")
        thread_id = memory_module.get_or_create_thread("email", receiver_header or sender or "email:unknown")
        # Keep the complete report available to the next run for UI and goal continuity.
        plain_text = report_text or ""

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
        fact_prompt = (
            "Extract a JSON array of concise facts from the following HTML report. "
            "Each item must be an object with keys 'fact_key', 'fact_text', and optional 'weight'.\n\n"
            f"Report:\n{report_text}\n\nReturn ONLY valid JSON."
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
                memory_module.upsert_long_fact(fk, ft, source_short_term_id=st_id, weight=w)
            except Exception:
                continue
        log_status("Completed memory persistence and fact extraction")
    except Exception as exc:
        print(f"Memory logging failed: {exc}")


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
    """Run the bounded author/validator flow and only send when the report passes validation."""
    from agent_harness import HarnessLimits, ReportHarness, ToolSpec
    from bigquery_tools import get_category_detail, get_spending_summary, get_transaction_detail
    from report_metrics import calculate_report_snapshot
    from run_archive import save_run

    monthly_deposit = _parse_money(config.get("Monthly_budget"))
    fixed_costs = _parse_money(config.get("Fixed_costs"))
    savings_goal = _parse_money(config.get("Savings_goal")) if config.get("Savings_goal") else None
    snapshot = calculate_report_snapshot(
        results,
        monthly_deposit=monthly_deposit,
        fixed_costs=fixed_costs,
        savings_goal=savings_goal,
        category_goals=config.get("category_goals", {}),
    )
    thread_key = receiver_header or sender or "email:unknown"

    def semantic_memory(_: dict) -> dict:
        """Return durable memory facts as tool evidence for the author model."""
        if memory_module is None:
            return {"facts": []}
        return {"facts": memory_module.get_long_facts(limit=10)}

    def episodic_memory(_: dict) -> dict:
        """Return recent report thread context so the author can keep style and tone continuity."""
        if memory_module is None:
            return {"entries": []}
        thread_id = memory_module.get_or_create_thread("email", thread_key)
        return {"entries": memory_module.get_short_term(thread_id, limit=3)}

    def report_history(_: dict) -> dict:
        """Return explicit historical excerpts labeled as non-authoritative reference material."""
        entries = episodic_memory({}).get("entries", [])
        excerpts = []
        for entry in entries:
            text = entry.get("response_text") or entry.get("response_html") or ""
            if text:
                excerpts.append({
                    "subject": entry.get("subject") or "Weekly report",
                    "excerpt": f'Previous report for reference: "{text[:1000]}" (historical; not authoritative)',
                })
        return {"reports": excerpts}

    limits = HarnessLimits(
        max_agent_rounds=int(config.get("max_agent_rounds", 6)),
        max_tool_calls=int(config.get("max_tool_calls", 12)),
        max_calls_per_tool=int(config.get("max_calls_per_tool", 3)),
        max_validation_cycles=int(config.get("max_validation_cycles", 2)),
        max_report_bytes=int(config.get("max_report_bytes", 250000)),
    )
    tools = {
        "get_spending_summary": ToolSpec("get_spending_summary", "Get authoritative monthly category spending.", lambda args: get_spending_summary(project_name, int(args.get("days_back", config.get("days_back", 90))))),
        "get_category_detail": ToolSpec("get_category_detail", "Get summary evidence for one category and two months.", lambda args: get_category_detail(project_name, str(args["category"]), str(args["month"]), args.get("previous_month"))),
        "get_transaction_detail": ToolSpec("get_transaction_detail", "Get bounded transaction rows from gold.fact_transactions.", lambda args: get_transaction_detail(project_name, str(args["month"]), args.get("category"), int(args.get("limit", 25)))),
        "get_semantic_memory": ToolSpec("get_semantic_memory", "Get durable weighted financial facts.", semantic_memory),
        "get_episodic_memory": ToolSpec("get_episodic_memory", "Get recent report thread context.", episodic_memory),
        "get_report_history": ToolSpec("get_report_history", "Get explicitly labeled historical report excerpts.", report_history),
    }
    harness = ReportHarness(
        ollama_url=ollama_url,
        author_model=model_name,
        validator_model=config.get("validator_model_name", model_name),
        limits=limits,
        timeout=int(config.get("ollama_timeout", 600)),
    )
    prompt_context, validator_context = _load_prompt_files()
    started = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + f"_{model_name.replace(':', '_')}"
    report = {}
    decision = None
    try:
        report, decision, trace = harness.run(
            objective="Create the weekly financial progress email using current evidence and clearly labeled history.",
            snapshot=snapshot,
            prompt_context=prompt_context,
            validator_context=validator_context,
            tools=tools,
        )
        status = "validated" if decision.passed else "validation_failed"
    except Exception as exc:
        print(f"Agent harness failed: {exc}")
        trace = harness.trace
        trace.add(
            "exception",
            phase="agent_orchestration",
            exception_type=type(exc).__name__,
            error=str(exc),
        )
        status = "failed"

    report_html = str(report.get("report_html", ""))
    report_text = str(report.get("plain_text", ""))
    metrics = {
        "model": model_name,
        "validator_model": config.get("validator_model_name", model_name),
        "mode": "agent",
        "status": status,
        "validation": decision.decision if decision else "error",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "agent_rounds": sum(event.get("event") == "agent_round" for event in trace.events),
        "tool_calls": sum(event.get("event") == "tool_call" for event in trace.events),
        "report_bytes": len(report_html.encode("utf-8")),
    }
    archive_root = Path(__file__).resolve().parent.parent / "agent" / "runs"
    save_run(
        archive_root,
        run_id,
        {"run_id": run_id, "mode": "agent", "status": status, "snapshot": snapshot.to_dict()},
        trace.events,
        report_html,
        report_text,
        metrics,
    )
    print(f"Agent run archived at {archive_root / run_id}")
    if not decision or not decision.passed:
        print("Report was not sent because validator did not pass.")
        return
    if config.get("dry_run", False):
        print("Dry run enabled; report was rendered and archived but not sent.")
        return
    msg_root = build_email_message(sender, receiver_header, report_html)
    send_email(sender, receivers, password, msg_root)
    memory_module = _init_memory_module(memory_schema)
    _remember_report(memory_module, sender, receivers, receiver_header, prompt_context, report_html, model_name, ollama_url, config)


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

    # The agent mode is intentionally different from the legacy path: Python computes the
    # financial snapshot and the agent is only allowed to gather evidence and write the
    # narrative after the deterministic facts are established.
    log_status("Initializing memory module")
    memory_module = _init_memory_module(memory_schema)
    if report_mode == "agent":
        run_agent_weekly_report(
            config, results, memory_module, sender, receivers, receiver_header,
            password, project_name, ollama_url, model_name, memory_schema,
        )
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
    run_weekly_report()
