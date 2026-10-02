"""Allowlisted BigQuery tools for the report agent."""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any


def _client(project_name: str):
    from google.cloud import bigquery

    return bigquery.Client(project=project_name), bigquery


def _rows(result: Any) -> list[dict[str, Any]]:
    return [{key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in dict(row).items()} for row in result]


def _month_end(value: date) -> date:
    """Convert any date within a month to that month's final calendar day."""
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def get_spending_years(project_name: str) -> list[int]:
    """Return years represented in the gold monthly-spending table."""
    client, _ = _client(project_name)
    query = f"""
        SELECT DISTINCT EXTRACT(YEAR FROM transaction_month) AS year
        FROM `{project_name}.gold.fact_monthly_spending`
        ORDER BY year DESC
    """
    return [int(row.year) for row in client.query(query).result()]


def get_latest_spending_month(project_name: str) -> date | None:
    """Return the newest available month normalized to day one, or None if empty."""
    client, _ = _client(project_name)
    query = f"""
        SELECT MAX(transaction_month) AS latest_month
        FROM `{project_name}.gold.fact_monthly_spending`
    """
    row = next(iter(client.query(query).result()), None)
    latest_month = getattr(row, "latest_month", None) if row is not None else None
    return date(latest_month.year, latest_month.month, 1) if latest_month is not None else None


def get_monthly_spending(project_name: str, report_month: date | str) -> dict[str, Any]:
    """Get a selected month and predecessor, translating month starts to stored month ends."""
    selected_month = date.fromisoformat(report_month) if isinstance(report_month, str) else report_month
    if selected_month.day != 1:
        raise ValueError("report_month must be the first day of a month")
    previous_month = date(
        selected_month.year - (selected_month.month == 1),
        12 if selected_month.month == 1 else selected_month.month - 1,
        1,
    )
    client, bigquery = _client(project_name)
    query = f"""
        SELECT transaction_month, category, parent_group, gross_spending,
               total_refunds_and_credits, net_real_expense,
               total_transaction_count, average_transaction_amount
        FROM `{project_name}.gold.fact_monthly_spending`
        WHERE transaction_month IN UNNEST(@months)
        ORDER BY transaction_month DESC, net_real_expense DESC
    """
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("months", "DATE", [_month_end(selected_month), _month_end(previous_month)]),
    ])
    rows = _rows(client.query(query, job_config=config).result())
    # The gold table stores transaction_month as LAST_DAY(transaction_date, MONTH).
    # Normalize returned month keys to month starts for snapshot calculations and UI state.
    for row in rows:
        month_value = row.get("transaction_month")
        if month_value:
            row["transaction_month"] = date.fromisoformat(str(month_value)[:10]).replace(day=1).isoformat()
    return {
        "source": "gold.fact_monthly_spending",
        "report_month": selected_month,
        "previous_month": previous_month,
        "rows": rows,
    }


def get_report_period_spending(
    project_name: str,
    year: int | None = None,
    month: int | None = None,
) -> dict[str, Any]:
    """Resolve latest or explicit year/month and fetch that month with its predecessor."""
    if (year is None) != (month is None):
        raise ValueError("year and month must either both be provided or both be omitted")
    if year is None:
        report_month = get_latest_spending_month(project_name)
        if report_month is None:
            return {"source": "gold.fact_monthly_spending", "report_month": None, "previous_month": None, "rows": []}
    else:
        if not 1 <= int(month) <= 12:
            raise ValueError("month must be between 1 and 12")
        report_month = date(int(year), int(month), 1)
    return get_monthly_spending(project_name, report_month)


def get_spending_summary(project_name: str, days_back: int = 90) -> dict[str, Any]:
    if not 1 <= int(days_back) <= 730:
        raise ValueError("days_back must be between 1 and 730")
    client, bigquery = _client(project_name)
    query = f"""
        SELECT transaction_month, category, parent_group, gross_spending,
               total_refunds_and_credits, net_real_expense,
               total_transaction_count, average_transaction_amount
        FROM `{project_name}.gold.fact_monthly_spending`
        WHERE transaction_month >= DATE_SUB(CURRENT_DATE(), INTERVAL @days_back DAY)
        ORDER BY transaction_month DESC, net_real_expense DESC
    """
    config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("days_back", "INT64", int(days_back))])
    return {"source": "gold.fact_monthly_spending", "rows": _rows(client.query(query, job_config=config).result())}


def get_category_detail(project_name: str, category: str, month: str, previous_month: str | None = None) -> dict[str, Any]:
    if not category or len(category) > 120:
        raise ValueError("category is required and must be at most 120 characters")
    if len(month) != 10:
        raise ValueError("month must be an ISO date")
    client, bigquery = _client(project_name)
    query = f"""
        SELECT transaction_month, category, parent_group, gross_spending,
               total_refunds_and_credits, net_real_expense,
               total_transaction_count, average_transaction_amount
        FROM `{project_name}.gold.fact_monthly_spending`
        WHERE category = @category AND transaction_month IN UNNEST(@months)
        ORDER BY transaction_month DESC
    """
    months = [month] + ([previous_month] if previous_month else [])
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("category", "STRING", category),
        bigquery.ArrayQueryParameter("months", "DATE", [_month_end(date.fromisoformat(item)) for item in months]),
    ])
    return {"source": "gold.fact_monthly_spending", "category": category, "rows": _rows(client.query(query, job_config=config).result())}


def get_transaction_detail(project_name: str, month: str, category: str | None = None, limit: int = 25) -> dict[str, Any]:
    if len(month) != 10:
        raise ValueError("month must be an ISO date")
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be between 1 and 100")
    client, bigquery = _client(project_name)
    category_filter = "AND category = @category" if category else ""
    query = f"""
        SELECT transaction_id, issuer, transaction_date, transaction_month,
               description, amount, transaction_type, category, parent_group
        FROM `{project_name}.gold.fact_transactions`
        WHERE transaction_month = @month {category_filter}
        ORDER BY transaction_date DESC, ABS(amount) DESC
        LIMIT @limit
    """
    parameters: list[Any] = [
        bigquery.ScalarQueryParameter("month", "DATE", _month_end(date.fromisoformat(month))),
        bigquery.ScalarQueryParameter("limit", "INT64", int(limit)),
    ]
    if category:
        parameters.append(bigquery.ScalarQueryParameter("category", "STRING", category))
    config = bigquery.QueryJobConfig(query_parameters=parameters)
    return {"source": "gold.fact_transactions", "month": month, "category": category, "rows": _rows(client.query(query, job_config=config).result())}