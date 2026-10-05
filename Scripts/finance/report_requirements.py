"""Month-specific report requirements checklist derived only from the authoritative snapshot.

The checklist is content guidance given verbatim to both the author and the validator; it
restates required headings and the exact snapshot values/disclosures in short imperative
lines. It never inspects report HTML: judging the report against it stays in the validator.
"""

from __future__ import annotations

from datetime import date

from .report_models import CategoryMetric, ReportSnapshot

# Canonical section headings from agent/prompts/ui.system.md, in required order (after the greeting).
REQUIRED_SECTION_HEADINGS = (
    "Monthly Snapshot",
    "Spending Trends & Analysis",
    "Mogumogu's Financial Health Check",
    "The Sushi Stack",
    "Top 3 Discretionary Categories",
    "Mogumogu's Tips for the Week",
)


def format_money(value: float) -> str:
    """Format money as $1,234.56, with negatives as -$1,234.56."""
    amount = round(float(value), 2)
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def month_label(month: str | None) -> str:
    """Format a YYYY-MM or YYYY-MM-DD snapshot month as "August 2026"."""
    if not month:
        return "the previous month"
    try:
        return date(int(month[:4]), int(month[5:7]), 1).strftime("%B %Y")
    except (TypeError, ValueError):
        return str(month)


def _percent_with_direction(percent: float | None, direction: str) -> str:
    if percent is None:
        return "no previous-month comparison" if direction == "no_prior" else "no percentage (previous amount was $0.00)"
    if abs(percent) < 0.005:
        return "0.00% (no change)"
    word = "increase" if percent > 0 else "decrease"
    return f"{abs(percent):.2f}% {word}"


def _top_row(index: int, metric: CategoryMetric, previous_label: str) -> str:
    previous = (
        "no prior-month data (say so)"
        if metric.mom_direction == "no_prior"
        else format_money(metric.previous_spend)
    )
    parts = [
        f"{index}. {metric.category}: current {format_money(metric.current_spend)}",
        f"previous ({previous_label}) {previous}",
        f"MoM {_percent_with_direction(metric.mom_change_percent, metric.mom_direction)}",
    ]
    if metric.suggested_cap is None:
        reason = metric.cap_basis or "no cap can be calculated"
        parts.append(f"no cap and no percent; state why: {reason}")
    else:
        label = "configured goal" if metric.cap_source == "configured" else "advisor-suggested cap"
        parts.append(f"{label} {format_money(metric.suggested_cap)}")
        if metric.percent_of_cap is None:
            parts.append(
                f"no percent of cap; state that it does not apply because the cap is {format_money(metric.suggested_cap)}"
            )
        else:
            parts.append(f"percent of cap {metric.percent_of_cap:.2f}%")
    return "; ".join(parts) + "."


def build_report_requirements(snapshot: ReportSnapshot) -> list[str]:
    """Return short, verifiable, month-specific requirement lines derived only from the snapshot."""
    latest_label = month_label(snapshot.latest_month)
    previous_label = month_label(snapshot.previous_month) if snapshot.previous_month else "previous month"
    headings = " -> ".join(f'"{heading}"' for heading in REQUIRED_SECTION_HEADINGS)
    lines = [
        "Start with a friendly Mogumogu greeting (no fixed heading), then use exactly these section headings in this "
        f"order: {headings}. Keep the wording exactly as given (one small food emoji before or after a heading is "
        "allowed); do not rename, reword, singularize, or add month names to them.",
    ]

    # Monthly Snapshot bars: every category per parent group with its exact amount.
    groups: dict[str, list[CategoryMetric]] = {}
    for metric in snapshot.categories:
        groups.setdefault(metric.parent_group, []).append(metric)
    lines.append(
        f"Monthly Snapshot ({latest_label}): show a bar row for every category below, grouped under its parent group, "
        "with this exact current-month amount."
    )
    # Needs/essential groups first, then Wants/discretionary, keeping snapshot order otherwise.
    for group, metrics in sorted(groups.items(), key=lambda item: any(m.is_discretionary for m in item[1])):
        items = ", ".join(
            f"{m.category} {format_money(m.current_spend)}" + (" (label as refund/credit)" if m.is_refund else "")
            for m in metrics
        )
        lines.append(f"{group}: {items}.")

    # Sushi Stack.
    previous_total = (
        format_money(snapshot.previous_total_net_expense)
        if snapshot.previous_total_net_expense is not None
        else "not available (say so)"
    )
    lines.append(
        f"The Sushi Stack: {latest_label} total net expense {format_money(snapshot.latest_total_net_expense)}; "
        f"{previous_label} total net expense {previous_total}; month-over-month change "
        f"{_percent_with_direction(snapshot.mom_change_percent, 'has_prior' if snapshot.previous_month else 'no_prior')}; "
        f"disposable cash {format_money(snapshot.disposable_cash)} = monthly deposit {format_money(snapshot.monthly_deposit)} "
        f"- ({format_money(snapshot.latest_total_net_expense)} net expense + {format_money(snapshot.fixed_costs)} fixed costs)"
        + (", keep the minus sign." if snapshot.disposable_cash < 0 else ".")
    )

    # Goal-method-specific disclosures.
    if snapshot.savings_goal is None or snapshot.goal_method == "no_savings_target":
        lines.append(
            "State that no numeric savings target is configured; do not invent a dollar savings goal or suggested caps."
        )
    else:
        target = format_money(snapshot.savings_goal)
        projected = format_money(snapshot.projected_savings if snapshot.projected_savings is not None else snapshot.disposable_cash)
        shortfall = snapshot.savings_shortfall or 0.0
        if shortfall > 0:
            lines.append(
                f"Savings target {target}; projected savings (disposable cash) {projected}; shortfall "
                f"{format_money(shortfall)}; eligible discretionary spend {format_money(snapshot.eligible_discretionary_total)}."
            )
        else:
            lines.append(f"Projected savings {projected} meet the {target} savings target (shortfall $0.00).")
        if snapshot.discretionary_cuts_insufficient:
            gap = format_money(snapshot.remaining_gap_after_cuts or 0.0)
            lines.append(
                f"State explicitly that discretionary cuts alone cannot reach the {target} savings target, "
                f"and that the remaining gap after cutting all discretionary spend is {gap}."
            )
        if snapshot.goal_method == "configured":
            lines.append(
                "Label every cap as a configured goal (from your category goals), not an advisor-suggested cap."
            )
        elif snapshot.goal_method == "trend_maintenance":
            lines.append(
                "Explain that the advisor-suggested caps are trend-based maintenance caps (previous-month spend where "
                "spending increased, otherwise current spend), and that they are planning suggestions, not guarantees."
            )
        elif snapshot.goal_method == "shortfall_allocation" and not snapshot.discretionary_cuts_insufficient:
            lines.append(
                f"Explain that the advisor-suggested caps share the {format_money(shortfall)} shortfall proportionally "
                "across all discretionary categories (not only the top three), as planning suggestions, not guarantees."
            )

    # Top 3 Discretionary Categories rows.
    metrics_by_name = {m.category: m for m in snapshot.categories if m.is_discretionary}
    top = [metrics_by_name[name] for name in snapshot.top_discretionary_categories if name in metrics_by_name]
    if top:
        lines.append(
            "Top 3 Discretionary Categories table, exactly these rows in this order "
            "(columns: category, current spend, previous spend, MoM change, goal/cap, percent of goal):"
        )
        lines.extend(_top_row(index, metric, previous_label) for index, metric in enumerate(top, start=1))
    else:
        lines.append(
            "Top 3 Discretionary Categories: there are no eligible Wants/Discretionary categories with positive spend; "
            "keep the section and say so instead of inventing rows."
        )

    lines.append("Write every amount exactly as listed here ($1,234.56 style; negatives as -$1,234.56).")
    return lines
