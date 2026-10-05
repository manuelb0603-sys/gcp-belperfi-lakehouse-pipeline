"""Deterministic financial calculations for the report harness."""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any, Iterable

from .report_models import CategoryMetric, ReportSnapshot


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


DISCRETIONARY_GROUPS = {"wants", "discretionary", "wants/discretionary"}


def _is_discretionary(parent_group: str) -> bool:
    return parent_group.strip().lower() in DISCRETIONARY_GROUPS


def _direction(current: float, previous: float, has_prior: bool) -> str:
    if not has_prior:
        return "no_prior"
    if abs(current - previous) < 0.005:
        return "flat"
    return "increase" if current > previous else "decrease"


def _suggested_cap(
    current: float,
    previous: float,
    has_prior: bool,
    target: float | None,
    shortfall: float | None,
    eligible_total: float,
) -> tuple[float | None, str]:
    """Apply the ui.system.md goal-setting method to one discretionary category."""
    if target is None:
        return None, "No numeric savings target is configured."
    if current <= 0:
        return None, "No positive current discretionary spend to cap."
    if shortfall and shortfall > 0:
        cut = min(current, shortfall * current / eligible_total)
        return round(max(0.0, current - cut), 2), "Proportional share of the savings shortfall."
    if has_prior and current > previous and previous > 0:
        return round(previous, 2), "Spending increased; capped at previous-month spend."
    return round(current, 2), "Savings target met; maintain current spend."


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

    category_names = {
        (category, parent_group)
        for month, category, parent_group in by_category
        if month in {latest_month, previous_month}
    }
    base_rows = []
    for category, parent_group in sorted(category_names):
        current = round(by_category[(latest_month, category, parent_group)], 2)
        previous = round(by_category[(previous_month, category, parent_group)], 2) if previous_month else 0.0
        base_rows.append((category, parent_group, current, previous))

    disposable_cash = round(float(monthly_deposit) - (latest_total + float(fixed_costs)), 2)
    target = float(savings_goal) if savings_goal is not None else None
    discretionary = [row for row in base_rows if _is_discretionary(row[1])]
    eligible_total = round(sum(row[2] for row in discretionary if row[2] > 0), 2)
    max_positive = round(max((row[2] for row in base_rows if row[2] > 0), default=0.0), 2)

    shortfall = round(max(0.0, target - disposable_cash), 2) if target is not None else None
    if target is None:
        method = "no_savings_target"
    elif shortfall > 0:
        method = "shortfall_allocation"
    else:
        method = "trend_maintenance"
    if discretionary and all(goals.get(row[0]) is not None for row in discretionary):
        method = "configured"
    cuts_insufficient = bool(shortfall and shortfall > eligible_total)
    remaining_gap = round(max(0.0, shortfall - eligible_total), 2) if shortfall is not None else None

    categories: list[CategoryMetric] = []
    for category, parent_group, current, previous in base_rows:
        goal = goals.get(category)
        is_discretionary = _is_discretionary(parent_group)
        cap, source, basis = None, None, ""
        if is_discretionary:
            if goal is not None:
                cap, source, basis = round(float(goal), 2), "configured", "Configured category goal."
            else:
                cap, basis = _suggested_cap(
                    current, previous, previous_month is not None, target, shortfall, eligible_total
                )
                source = "suggested" if cap is not None else None
        categories.append(CategoryMetric(
            category=category,
            parent_group=parent_group,
            current_spend=current,
            previous_spend=previous,
            mom_change_percent=_percent_change(current, previous if previous_month else None),
            goal_amount=goal,
            goal_percent=round((current / goal) * 100, 2) if goal and goal > 0 else None,
            is_discretionary=is_discretionary,
            suggested_cap=cap,
            cap_source=source,
            cap_basis=basis,
            percent_of_cap=round((current / cap) * 100, 2) if cap and cap > 0 else None,
            bar_width_percent=round((current / max_positive) * 100, 2) if current > 0 and max_positive > 0 else 0.0,
            is_refund=current < 0,
            mom_direction=_direction(current, previous, previous_month is not None),
        ))

    top_discretionary = [
        row[0] for row in sorted(
            (row for row in discretionary if row[2] > 0), key=lambda row: (-row[2], row[0])
        )[:3]
    ]

    return ReportSnapshot(
        as_of=(as_of or date.today()).isoformat(),
        latest_month=latest_month,
        previous_month=previous_month,
        latest_total_net_expense=latest_total,
        previous_total_net_expense=previous_total,
        mom_change_percent=_percent_change(latest_total, previous_total),
        monthly_deposit=round(float(monthly_deposit), 2),
        fixed_costs=round(float(fixed_costs), 2),
        disposable_cash=disposable_cash,
        savings_goal=target,
        categories=categories,
        source_tables=["gold.fact_monthly_spending"],
        projected_savings=disposable_cash,
        savings_shortfall=shortfall,
        eligible_discretionary_total=eligible_total,
        goal_method=method,
        discretionary_cuts_insufficient=cuts_insufficient,
        remaining_gap_after_cuts=remaining_gap,
        max_positive_spend=max_positive,
        top_discretionary_categories=top_discretionary,
    )
