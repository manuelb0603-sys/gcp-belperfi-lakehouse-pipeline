"""Deterministic financial calculations for the report harness."""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any, Iterable

from report_models import CategoryMetric, ReportSnapshot


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _month_key(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _percent_change(current: float, previous: float | None) -> float | None:
    if previous is None or previous == 0:
        return None
    return round(((current - previous) / abs(previous)) * 100, 2)


def calculate_report_snapshot(
    rows: Iterable[Any],
    monthly_deposit: float,
    fixed_costs: float,
    savings_goal: float | None = None,
    category_goals: dict[str, float] | None = None,
    as_of: date | None = None,
) -> ReportSnapshot:
    """Calculate report facts without involving an LLM."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot calculate a report snapshot from empty data")

    by_month: dict[str, float] = defaultdict(float)
    by_category: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in materialized:
        month = _month_key(_value(row, "transaction_month"))
        category = str(_value(row, "category", "Other"))
        parent_group = str(_value(row, "parent_group", "Other"))
        amount = _number(_value(row, "net_real_expense"))
        by_month[month] += amount
        by_category[(month, category, parent_group)] += amount

    months = sorted(by_month, reverse=True)
    latest_month = months[0]
    previous_month = months[1] if len(months) > 1 else None
    latest_total = round(by_month[latest_month], 2)
    previous_total = round(by_month[previous_month], 2) if previous_month else None
    goals = category_goals or {}

    categories: list[CategoryMetric] = []
    category_names = {
        (category, parent_group)
        for month, category, parent_group in by_category
        if month in {latest_month, previous_month}
    }
    for category, parent_group in sorted(category_names):
        current = round(by_category[(latest_month, category, parent_group)], 2)
        previous = round(by_category[(previous_month, category, parent_group)], 2) if previous_month else 0.0
        goal = goals.get(category)
        categories.append(CategoryMetric(
            category=category,
            parent_group=parent_group,
            current_spend=current,
            previous_spend=previous,
            mom_change_percent=_percent_change(current, previous if previous_month else None),
            goal_amount=goal,
            goal_percent=round((current / goal) * 100, 2) if goal and goal > 0 else None,
        ))

    return ReportSnapshot(
        as_of=(as_of or date.today()).isoformat(),
        latest_month=latest_month,
        previous_month=previous_month,
        latest_total_net_expense=latest_total,
        previous_total_net_expense=previous_total,
        mom_change_percent=_percent_change(latest_total, previous_total),
        monthly_deposit=round(float(monthly_deposit), 2),
        fixed_costs=round(float(fixed_costs), 2),
        disposable_cash=round(float(monthly_deposit) - (latest_total + float(fixed_costs)), 2),
        savings_goal=float(savings_goal) if savings_goal is not None else None,
        categories=categories,
        source_tables=["gold.fact_monthly_spending"],
    )