"""Deterministic explanation of one category's month-over-month change (explain_category_change tool).

``explain_category_change(rows, ...)`` turns the category's fact_transactions rows for the report
month and its predecessor into a compact JSON-safe summary (at most ``MAX_RESULT_BYTES``): per-month
count, net spend, average and largest transaction; a count-versus-average breakdown of the change;
the largest increases and decreases grouped by description; descriptions new this month (names); and the
largest single transactions this month. It never returns raw rows. The math mirrors
gold.fact_monthly_spending: only Sale/Return/Adjustment rows count and ``spend = -amount``.

Change breakdown (exact, sums to the change): with counts C0, C1 and averages A0, A1,
``from_transaction_count = (C1 - C0) * A0`` and ``from_average_amount = (A1 - A0) * C1``. When either
month has no transactions the whole change is attributed to the transaction count.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from typing import Any, Iterable

ANALYZED_TYPES = ("Sale", "Return", "Adjustment")  # mirrors gold.fact_monthly_spending
MAX_RESULT_BYTES = 2048
TOP_N_DEFAULT = 5
TOP_N_MAX = 10
LARGEST_TRANSACTIONS_MAX = 3
DESCRIPTION_CHARS = 40
# Per-description breakdown is flagged as limited when at least this share of the distinct
# descriptions occurs only once across both months (and there are enough transactions to judge).
UNIQUE_DESCRIPTION_SHARE = 0.9
UNIQUE_DESCRIPTION_MIN_TRANSACTIONS = 5

LIMITED_DESCRIPTIONS_NOTE = (
    "Descriptions are nearly all unique, so the per-description breakdown is limited; the transaction "
    "count, average transaction, and largest transactions are the reliable drivers."
)


def _value(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def _month_start(value: Any) -> str:
    text = value.isoformat() if isinstance(value, date) else str(value or "")
    return f"{text[:7]}-01" if len(text) >= 7 else text


def _number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def normalize_description(value: Any) -> str:
    """Trimmed, upper-cased description with inner whitespace collapsed (grouping key)."""
    text = " ".join(str(value or "").split()).upper()
    return text[:DESCRIPTION_CHARS].rstrip() or "(NO DESCRIPTION)"


def _month_stats(spends: list[float]) -> dict[str, Any]:
    count = len(spends)
    net = round(sum(spends), 2)
    return {
        "transaction_count": count,
        "net_spend": net,
        "average_transaction": round(net / count, 2) if count else None,
        "largest_transaction": round(max(spends), 2) if count else None,
    }


def change_breakdown(previous: list[float], current: list[float]) -> dict[str, Any]:
    """Split the net-spend change into a transaction-count part and an average-amount part (sums exactly)."""
    c0, c1 = len(previous), len(current)
    total0, total1 = sum(previous), sum(current)
    change = round(total1 - total0, 2)
    if c0 == 0 or c1 == 0:
        from_count = change
    else:
        a0 = total0 / c0
        from_count = round((c1 - c0) * a0, 2)
    from_average = round(change - from_count, 2)
    return {
        "change_usd": change,
        "from_transaction_count": from_count,
        "from_average_amount": from_average,
        "transaction_count_change": c1 - c0,
    }


def _fit(result: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Trim the longest list (then descriptions) until the compact JSON fits max_bytes; marks "truncated"."""
    lists = ("top_increases", "top_decreases", "new_descriptions", "largest_transactions")

    def size() -> int:
        return len(json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def longest() -> str:
        return max(lists, key=lambda key: len(result.get(key) or []))

    if size() <= max_bytes:
        return result
    result["truncated"] = True
    while size() > max_bytes and len(result.get(longest()) or []) > 1:
        result[longest()].pop()
    if size() > max_bytes:
        result["new_descriptions"] = [text[:16].rstrip() for text in result.get("new_descriptions") or []]
        for key in lists[:2] + lists[3:]:
            for item in result.get(key) or []:
                item["description"] = item["description"][:16].rstrip()
    while size() > max_bytes and any(result.get(key) for key in lists):
        result[longest()].pop()
    return result


def explain_category_change(
    rows: Iterable[Any],
    category: str,
    current_month: str,
    previous_month: str,
    top_n: int = TOP_N_DEFAULT,
    max_bytes: int = MAX_RESULT_BYTES,
) -> dict[str, Any]:
    """Compact, deterministic explanation of what drove a category's change between two months.

    rows are fact_transactions rows (dicts or objects with transaction_month, transaction_date,
    description, amount, and optionally transaction_type) for this category; other months and
    non-analyzed transaction types are ignored. Raises ValueError when neither month has a row.
    """
    top_n = max(1, min(TOP_N_MAX, int(top_n)))
    current_key, previous_key = _month_start(current_month), _month_start(previous_month)
    spends: dict[str, list[float]] = {current_key: [], previous_key: []}
    by_description: dict[str, dict[str, float]] = defaultdict(lambda: {current_key: 0.0, previous_key: 0.0})
    occurrences: dict[str, int] = defaultdict(int)
    month_counts: dict[str, dict[str, int]] = defaultdict(lambda: {current_key: 0, previous_key: 0})
    current_rows: list[tuple[str, str, float]] = []
    for row in rows:
        kind = _value(row, "transaction_type")
        if kind is not None and str(kind) not in ANALYZED_TYPES:
            continue
        month = _month_start(_value(row, "transaction_month") or _value(row, "transaction_date"))
        if month not in spends:
            continue
        spend = -_number(_value(row, "amount"))
        description = normalize_description(_value(row, "description"))
        spends[month].append(spend)
        by_description[description][month] += spend
        occurrences[description] += 1
        month_counts[description][month] += 1
        if month == current_key:
            day = _value(row, "transaction_date")
            day = day.isoformat() if isinstance(day, date) else str(day or "")[:10]
            current_rows.append((day, description, spend))
    if not spends[current_key] and not spends[previous_key]:
        raise ValueError(
            f"No Sale/Return/Adjustment transactions for category {category!r} in {current_key} or {previous_key}; "
            "use a category name exactly as shown in the snapshot."
        )

    changes = [
        {
            "description": description,
            "current": round(amounts[current_key], 2),
            "previous": round(amounts[previous_key], 2),
            "change": round(amounts[current_key] - amounts[previous_key], 2),
        }
        for description, amounts in by_description.items()
    ]
    increases = sorted((item for item in changes if item["change"] >= 0.005), key=lambda item: (-item["change"], item["description"]))
    decreases = sorted((item for item in changes if item["change"] <= -0.005), key=lambda item: (item["change"], item["description"]))
    # Names only (largest current spend first); their amounts are in top_increases.
    new = [
        description for description, _ in sorted(
            (
                (description, by_description[description][current_key])
                for description, counts in month_counts.items()
                if counts[current_key] and not counts[previous_key]
            ),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    largest = sorted(current_rows, key=lambda item: (-item[2], item[0], item[1]))[: min(top_n, LARGEST_TRANSACTIONS_MAX)]

    total_transactions = len(spends[current_key]) + len(spends[previous_key])
    single_use = sum(1 for count in occurrences.values() if count == 1)
    unique_share = single_use / len(occurrences) if occurrences else 0.0

    result: dict[str, Any] = {
        "category": category,
        "current_month": current_key,
        "previous_month": previous_key,
        "current": _month_stats(spends[current_key]),
        "previous": _month_stats(spends[previous_key]),
        "change_breakdown": change_breakdown(spends[previous_key], spends[current_key]),
        "top_increases": increases[:top_n],
        "top_decreases": decreases[:top_n],
        "new_descriptions": new[:top_n],
        "largest_transactions": [
            {"date": day, "description": description, "amount": round(spend, 2)} for day, description, spend in largest
        ],
        "distinct_descriptions": len(occurrences),
    }
    if total_transactions >= UNIQUE_DESCRIPTION_MIN_TRANSACTIONS and unique_share >= UNIQUE_DESCRIPTION_SHARE:
        result["limitations"] = LIMITED_DESCRIPTIONS_NOTE
    return _fit(result, max_bytes)
