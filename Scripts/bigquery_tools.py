"""Allowlisted BigQuery tools for the report agent."""

from __future__ import annotations

from datetime import date
from typing import Any


def _client(project_name: str):
    from google.cloud import bigquery

    return bigquery.Client(project=project_name), bigquery


def _rows(result: Any) -> list[dict[str, Any]]:
    return [{key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in dict(row).items()} for row in result]


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
        bigquery.ArrayQueryParameter("months", "DATE", [date.fromisoformat(item) for item in months]),
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
        bigquery.ScalarQueryParameter("month", "DATE", date.fromisoformat(month)),
        bigquery.ScalarQueryParameter("limit", "INT64", int(limit)),
    ]
    if category:
        parameters.append(bigquery.ScalarQueryParameter("category", "STRING", category))
    config = bigquery.QueryJobConfig(query_parameters=parameters)
    return {"source": "gold.fact_transactions", "month": month, "category": category, "rows": _rows(client.query(query, job_config=config).result())}