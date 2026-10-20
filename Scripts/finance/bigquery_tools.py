"""Allowlisted BigQuery tools for the report agent."""

from __future__ import annotations

import calendar
import json
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
        SELECT transaction_month, month, year, category, parent_group, gross_spending,
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
        SELECT transaction_month, month, year, category, parent_group, gross_spending,
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


def _period_filters(category: str | None, parent_group: str | None) -> tuple[str | None, str | None, str]:
    """Validate optional category/parent_group filters; return them with their SQL condition."""
    category = str(category).strip() if category else None
    parent_group = str(parent_group).strip() if parent_group else None
    for name, value in (("category", category), ("parent_group", parent_group)):
        if value is not None and len(value) > 120:
            raise ValueError(f"{name} must be at most 120 characters")
    conditions = [f"AND {name} = @{name}" for name, value in (("category", category), ("parent_group", parent_group)) if value]
    return category, parent_group, " ".join(conditions)


def _filter_parameters(bigquery: Any, category: str | None, parent_group: str | None) -> list[Any]:
    return [
        bigquery.ScalarQueryParameter(name, "STRING", value)
        for name, value in (("category", category), ("parent_group", parent_group)) if value
    ]


# Per-period totals summed over the matching category rows (average = net spend per transaction).
_PERIOD_METRICS = """
               ROUND(SUM(gross_spending), 2) AS gross_spending,
               ROUND(SUM(total_refunds_and_credits), 2) AS total_refunds_and_credits,
               ROUND(SUM(net_real_expense), 2) AS net_real_expense,
               SUM(total_transaction_count) AS total_transaction_count,
               ROUND(SAFE_DIVIDE(SUM(net_real_expense), SUM(total_transaction_count)), 2) AS average_transaction_amount"""


def get_daily_spending(
    project_name: str, month: str, category: str | None = None, parent_group: str | None = None
) -> dict[str, Any]:
    """Per-day spending totals for one month from gold.fact_daily_spending (read-only, parameterized).

    Sums every category (or only the optional category/parent_group) into one row per day.
    """
    if len(month) != 10:
        raise ValueError("month must be an ISO date")
    start = date.fromisoformat(month).replace(day=1)
    category, parent_group, conditions = _period_filters(category, parent_group)
    client, bigquery = _client(project_name)
    query = f"""
        SELECT transaction_date, day_of_week, month, year,{_PERIOD_METRICS}
        FROM `{project_name}.gold.fact_daily_spending`
        WHERE transaction_month = @month {conditions}
        GROUP BY transaction_date, day_of_week, month, year
        ORDER BY transaction_date
    """
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("month", "DATE", _month_end(start)),
        *_filter_parameters(bigquery, category, parent_group),
    ])
    return {
        "source": "gold.fact_daily_spending",
        "month": start.isoformat(),
        "category": category,
        "parent_group": parent_group,
        "rows": _rows(client.query(query, job_config=config).result()),
    }


def get_weekly_spending(
    project_name: str, month: str, category: str | None = None, parent_group: str | None = None
) -> dict[str, Any]:
    """Per-week (Monday-Sunday) spending totals for the weeks overlapping one month, from
    gold.fact_weekly_spending (read-only, parameterized).

    Weeks are whole weeks, so the first and last can include days of the neighboring months and
    the weekly totals need not add up to the monthly total.
    """
    if len(month) != 10:
        raise ValueError("month must be an ISO date")
    start = date.fromisoformat(month).replace(day=1)
    category, parent_group, conditions = _period_filters(category, parent_group)
    client, bigquery = _client(project_name)
    query = f"""
        SELECT week_start, week_end, iso_week, year,{_PERIOD_METRICS}
        FROM `{project_name}.gold.fact_weekly_spending`
        WHERE week_end >= @month_start AND week_start <= @month_end {conditions}
        GROUP BY week_start, week_end, iso_week, year
        ORDER BY week_start
    """
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("month_start", "DATE", start),
        bigquery.ScalarQueryParameter("month_end", "DATE", _month_end(start)),
        *_filter_parameters(bigquery, category, parent_group),
    ])
    return {
        "source": "gold.fact_weekly_spending",
        "month": start.isoformat(),
        "category": category,
        "parent_group": parent_group,
        "note": "Weeks run Monday to Sunday; the first and last weeks can include days outside the month, "
        "so weekly totals do not add up to the monthly total.",
        "rows": _rows(client.query(query, job_config=config).result()),
    }


def _previous_month_start(value: date) -> date:
    return date(value.year - (value.month == 1), 12 if value.month == 1 else value.month - 1, 1)


def explain_category_change(
    project_name: str, category: str, month: str, top_n: int = 5, parent_group: str | None = None
) -> dict[str, Any]:
    """Explain one category's change from the previous month to ``month`` (read-only, one parameterized SELECT).

    Reads that category's Sale/Return/Adjustment rows (as gold.fact_monthly_spending does) for the
    month and its predecessor (optionally only one parent_group, matching a snapshot row) from
    gold.fact_transactions and returns the compact summary from
    finance.category_change.explain_category_change (never raw rows).
    """
    from .category_change import ANALYZED_TYPES, MAX_RESULT_BYTES, TOP_N_MAX, explain_category_change as summarize

    category = str(category or "").strip()
    if not category or len(category) > 120:
        raise ValueError("category is required and must be at most 120 characters")
    if len(month) != 10:
        raise ValueError("month must be an ISO date")
    parent_group = str(parent_group).strip() if parent_group else None
    if parent_group is not None and len(parent_group) > 120:
        raise ValueError("parent_group must be at most 120 characters")
    if not 1 <= int(top_n) <= TOP_N_MAX:
        raise ValueError(f"top_n must be between 1 and {TOP_N_MAX}")
    current = date.fromisoformat(month).replace(day=1)
    previous = _previous_month_start(current)
    client, bigquery = _client(project_name)
    group_filter = "AND parent_group = @parent_group" if parent_group else ""
    query = f"""
        SELECT transaction_month, transaction_date, description, amount, transaction_type
        FROM `{project_name}.gold.fact_transactions`
        WHERE category = @category {group_filter}
          AND transaction_month IN UNNEST(@months)
          AND transaction_type IN UNNEST(@types)
    """
    parameters: list[Any] = [
        bigquery.ScalarQueryParameter("category", "STRING", category),
        bigquery.ArrayQueryParameter("months", "DATE", [_month_end(current), _month_end(previous)]),
        bigquery.ArrayQueryParameter("types", "STRING", list(ANALYZED_TYPES)),
    ]
    if parent_group:
        parameters.append(bigquery.ScalarQueryParameter("parent_group", "STRING", parent_group))
    config = bigquery.QueryJobConfig(query_parameters=parameters)
    rows = _rows(client.query(query, job_config=config).result())
    source = {"source": "gold.fact_transactions", **({"parent_group": parent_group} if parent_group else {})}
    # Leave room for the source key so the whole result stays within MAX_RESULT_BYTES.
    budget = MAX_RESULT_BYTES - len(json.dumps(source, separators=(",", ":")))
    return {**source, **summarize(rows, category, current.isoformat(), previous.isoformat(), int(top_n), budget)}
