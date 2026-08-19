import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.cloud import bigquery
import requests
from pathlib import Path
import json
import importlib.util
import re
import time
from typing import Any, Dict


def log_status(message: str, start_time: float | None = None) -> None:
    """Print a timestamped status message with elapsed time for debugging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if start_time is not None:
        suffix = f" | elapsed={time.perf_counter() - start_time:.1f}s"
    print(f"[{timestamp}] {message}{suffix}")


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_bigquery_summary(
    project_name: str,
    days_back: int = 60
) -> list:
    """Fetch summary data from BigQuery Gold Layer.

    Args:
        project_name: GCP project ID to query
        days_back: Number of days to look back for transaction history

    Returns:
        List of rows with spending metrics from gold.fact_monthly_spending table
    """
    
    query = f"""
        SELECT transaction_month, category, parent_group, gross_spending,
               total_refunds_and_credits, net_real_expense, 
               total_transaction_count, average_transaction_amount
        FROM `{project_name}.gold.fact_monthly_spending`
        WHERE transaction_month >= CURRENT_DATE() - INTERVAL {days_back} DAY
        ORDER BY transaction_month DESC, net_real_expense DESC
    """
    
    bq_client = bigquery.Client(project=project_name)
    query_job = bq_client.query(query)
    results = list(query_job.result())
    
    return results


def format_data_payload(results: list) -> str:
    """Format BigQuery results into a clean string for the LLM.

    Args:
        results: List of rows from BigQuery query containing spending data per category/month

    Returns:
        Formatted data payload with readable text representation suitable as input to Ollama prompt template construction process before being sent via REST API endpoint POST request handler method body function implementation pattern style approach usage for automated weekly reporting generation workflows in Python script execution contexts like cron job or scheduled task automation systems configuration management purposes here showing dynamic string formatting using f-string interpolation patterns typical of modern Python codebase conventions
    """
    data_payload = f"Month Summary Data as of {date.today().strftime('%Y-%m-%d')}:\n"
    for row in results:
        data_payload += f"- {row.transaction_month.strftime('%B')}: - {row.category}: ${row.net_real_expense:.2f} across {row.total_transaction_count} transactions\n"

    return data_payload


def _memory_context_to_prompt(memory_module: Any, receiver_header: str, sender: str, limit: int = 20, recent_limit: int = 5) -> str:
    """Return compact long-term facts plus recent thread history formatted for prompt context."""
    if memory_module is None:
        return ""

    lines: list[str] = []

    try:
        facts = memory_module.get_long_facts(limit=limit)
        if facts:
            lines.append("Relevant prior memory/context:")
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
            for entry in recent_entries:
                subject = entry.get("subject") or "Weekly report"
                text = entry.get("response_text") or ""
                if text:
                    summary = text.strip().replace("\n", " ")
                    lines.append(f"- {subject}: {summary[:300]}")
    except Exception as exc:
        print(f"Recent-thread memory context lookup failed: {exc}")

    return "\n".join(lines)


def build_ollama_prompt(config: dict, data_payload: str, memory_context: str | None = None) -> str:
    """Build the full prompt for Ollama from config, payload, and prior memory facts."""
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
  Total Net Expense Spend (Net Real Expense from all categories)
  Previous Month Total Net Expense Spend (Net Real Expense from all categories)
  Mom % Change (Net Real Expense from all categories)
  Disposable Cash which can go negative (Monthly deposit-(net real expense + Monthly fixed costs) Explain how you got this number so I can troubleshoot your calculations).
- KPI Card of Top 3 Discretionary Categories [Table] (total spent, previous spent, Mom% Trend with color coding, targeted goal suggestion, and percentage of goal spent).
Explain why the goal suggestion is what it is, and how to improve the spending habits for that category.
- Momogu's Tips for the Week.

Today's date: {date.today().strftime('%Y-%m-%d')}
Data:
{data_payload}

Please write a polished HTML email body with clear sections and a warm, encouraging tone. No Markdown syntax. This means do not include * or ** in responses."""


def query_ollama(prompt: str, ollama_url: str, model_name: str, stream: bool = False) -> str:
    """Query local Ollama server and return the generated report text."""
    request_started = time.perf_counter()
    log_status(f"Querying Ollama model '{model_name}' at {ollama_url} (stream={stream})")
    try:
        response = requests.post(ollama_url, json={
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }, timeout=600)
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
    """Build the multipart email message for both plain text and HTML."""
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
    """Send email via Gmail SMTP."""
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receivers, msg_root.as_string())


def _init_memory_module():
    """Load the local SQLite memory module if available. Best-effort only."""
    try:
        memory_path = Path(__file__).with_name("memory.py")
        spec = importlib.util.spec_from_file_location("memory_module", str(memory_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.init_db()
        return module
    except Exception as exc:
        print(f"Memory initialization failed: {exc}")
        return None


def _extract_json_payload(payload: str) -> list[dict[str, Any]]:
    """Attempt to parse a JSON list from Ollama responses that may include extra text."""
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
    """Save report prompt/response and extract upsertable facts. Never blocks email sends."""
    if memory_module is None:
        return

    try:
        log_status("Starting memory persistence for this report")
        thread_id = memory_module.get_or_create_thread("email", receiver_header or sender or "email:unknown")
        plain_text = "\n".join([
            f"Weekly Financial Progress — {date.today().strftime('%B %d, %Y')}",
            "",
            *([report_text.split('\n\n')[0]] if report_text else []),
            "",
        ])

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

    log_status("Initializing memory module")
    memory_module = _init_memory_module()
    log_status("Building memory context and prompt")
    memory_context = _memory_context_to_prompt(memory_module, receiver_header, sender)
    data_payload = format_data_payload(results)
    prompt = build_ollama_prompt(config, data_payload, memory_context=memory_context)

    try:
        log_status(f"Generating report with Ollama model '{model_name}'")
        report_text = query_ollama(prompt, ollama_url, model_name, stream=ollama_stream)
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

    memory_module = _init_memory_module()
    if memory_module is not None:
        log_status("Saving report to long/short-term memory")
        _remember_report(memory_module, sender, receivers, receiver_header, prompt, report_text, model_name, ollama_url, config)
        log_status("Memory persistence complete")

    total_elapsed = time.perf_counter() - start_time
    log_status(f"Weekly report run complete: total elapsed={total_elapsed:.1f}s", start_time)



if __name__ == "__main__":
    run_weekly_report()
