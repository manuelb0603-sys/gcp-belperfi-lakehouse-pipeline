"""Legacy report mode (report_mode "legacy"): one Ollama prompt over the gold-layer summary.

The working proof of concept: a rolling days_back BigQuery summary, recent thread history as
context, and a single model call whose HTML is emailed as-is (no harness or validator). Keep its
prompt and behavior unchanged; agent mode lives in finance.reporting and finance.report_runner.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

from .log import get_logger
from .ollama import generate_text
from .settings import ReportSettings
from .status_log import log_status

logger = get_logger(__name__)

# query_ollama's return value when the model call fails (callers compare against it).
FAILED_REPORT_TEXT = "Failed to generate report."


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
        logger.warning(f"Recent-thread memory context lookup failed: {exc}")

    return "\n".join(lines)


def build_ollama_prompt(config: dict | ReportSettings, data_payload: str, memory_context: str | None = None) -> str:
    """Compose the final prompt so the model knows the objective, constraints, and data sources."""
    settings = ReportSettings.coerce(config)
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
- Monthly deposit: {settings.monthly_budget_raw}.
- Monthly Fixed costs: {settings.fixed_costs_raw}.
- Vehicles: {settings.vehicle_info}.
- Authoritative numeric monthly savings target: {settings.savings_goal_target_raw}.
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
        return FAILED_REPORT_TEXT
