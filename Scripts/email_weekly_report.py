import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.cloud import bigquery
import requests
from pathlib import Path
import json


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


def build_ollama_prompt(config: dict, data_payload: str) -> str:
    """Build the full prompt for Ollama from config and payload."""
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
- Do not include markdown or code blocks in the output. The output should be a polished HTML email body with clear sections and a warm, encouraging tone.

Context:
- Monthly deposit: {config.get('Monthly_budget')}.
- Monthly Fixed costs: {config.get('Fixed_costs')}.
- Vehicles: {config.get('Vehicle_Info')}.
- Savings goal: {config.get('Savings_goal')}.
- This is a follow-up report; the recipient already knows the recurring details.
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
    response = requests.post(ollama_url, json={
        "model": model_name,
        "prompt": prompt,
        "stream": stream
    })
    return response.json().get("response", "Failed to generate report.")


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


def run_weekly_report(config_path: Path | None = None) -> None:
    """Orchestrate the weekly report generation and email send."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return

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
        results = fetch_bigquery_summary(project_name, days_back=days_back)
    except Exception as exc:
        print(f"Failed to fetch BigQuery summary: {exc}")
        return

    if not results:
        print("BigQuery returned no results. Aborting report generation.")
        return

    data_payload = format_data_payload(results)
    prompt = build_ollama_prompt(config, data_payload)

    try:
        report_text = query_ollama(prompt, ollama_url, model_name, stream=ollama_stream)
    except Exception as exc:
        print(f"Failed to query Ollama: {exc}")
        return

    if not report_text or report_text == "Failed to generate report.":
        print("Ollama returned an empty or failed report. Aborting.")
        return

    msg_root = build_email_message(sender, receiver_header, report_text)

    try:
        send_email(sender, receivers, password, msg_root)
    except Exception as exc:
        print(f"Failed to send email: {exc}")
        return

    print("Report sent successfully!")


if __name__ == "__main__":
    run_weekly_report()
