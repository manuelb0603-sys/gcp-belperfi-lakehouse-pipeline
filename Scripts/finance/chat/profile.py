"""Data profile for the finance chat: what the loaded transactions can meaningfully support.

``build_data_profile(df, today=...)`` runs once per data load (``data.load_chat_data`` caches it
next to the catalog). It records the schema used, date coverage (complete vs partial months),
the transaction types present, and, per dimension, how well the data supports grouping by it,
as ``usable``, ``limited``, or ``unusable`` with a short plain-language reason. The profile
drives the DATA PROFILE prompt block, ``validate_and_repair`` (limited dimensions are swapped
for usable ones), the executor's auto chart, and the example questions in the Chat tab.

Thresholds (all counts are over analyzed rows: Sale, Return, Adjustment):

* category, parent_group, issuer, merchant: ``unusable`` with fewer than 2 values; ``limited``
  when the median value has fewer than ``MIN_MEDIAN_PER_VALUE`` (3) transactions, or when there
  are more than 10 values and the top 10 cover less than ``MIN_TOP10_SHARE`` (40%) of them.
* weekday: ``unusable`` with fewer than 2 weekdays seen; ``limited`` when the median weekday has
  fewer than ``MIN_MEDIAN_PER_VALUE`` transactions.
* day_of_month: ``limited`` when the median day has fewer than ``MIN_DAY_OF_MONTH_MEDIAN`` (2).
* month: ``usable`` with at least ``MIN_COMPLETE_MONTHS`` (2) complete months; ``limited`` with
  one complete month or several partial ones; ``unusable`` with a single partial month.
  A month is complete when the data covers it to within ``COMPLETE_EDGE_DAYS`` at both ends and
  the month is over (``analysis_spec.month_coverage``).
* quarter: ``usable`` with at least 2 quarters with data; week: ``usable`` with at least 4 weeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .analysis_spec import month_coverage
from .data import ANALYSIS_TYPES  # Sale/Return/Adjustment, mirrors gold fact_monthly_spending

USABLE, LIMITED, UNUSABLE = "usable", "limited", "unusable"
MIN_MEDIAN_PER_VALUE = 3
MIN_TOP10_SHARE = 0.40
MIN_DAY_OF_MONTH_MEDIAN = 2
MIN_COMPLETE_MONTHS = 2
MIN_QUARTERS = 2
MIN_WEEKS = 4
TOP_VALUES = 5
PROFILED_DIMENSIONS = (
    "category", "parent_group", "issuer", "merchant", "weekday", "day_of_month", "month", "quarter", "week",
)
CATEGORICAL = ("category", "parent_group", "issuer", "merchant")

# Columns the chat reads from gold.fact_transactions (see definitions/gold/fact_transactions.sqlx).
SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("issuer", "STRING", "Card issuer (for example Amex, Chase); the 'card' dimension."),
    ("transaction_date", "DATE", "Purchase date; drives month, quarter, week, weekday, and day_of_month."),
    ("description", "STRING", "Raw card descriptor; normalized into the 'merchant' dimension (never shown to the model)."),
    ("amount", "NUMERIC", "Signed amount: purchases negative, refunds and credits positive; spend = -amount."),
    ("transaction_type", "STRING", "Sale, Return, Adjustment, Payment, ...; only Sale/Return/Adjustment are analyzed."),
    ("category", "STRING", "Standardized category from gold dim_category ('Other' when unmapped)."),
    ("parent_group", "STRING", "Budget group from dim_category: Needs, Wants, or Payments & Others."),
)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    gold_type: str
    dtype: str
    meaning: str


@dataclass(frozen=True)
class DimensionProfile:
    name: str
    distinct: int
    median_per_value: float
    min_per_value: int
    top10_transaction_share: float  # 0..1
    top10_spend_share: float  # 0..1, by gross spend (purchases)
    status: str
    reason: str
    top_values: tuple[str, ...] = ()  # most frequent values (not sent to the model for merchant)

    @property
    def usable(self) -> bool:
        return self.status == USABLE


@dataclass(frozen=True)
class DataProfile:
    today: date
    schema: tuple[ColumnInfo, ...]
    transactions: int  # analyzed rows
    data_min: date | None
    data_max: date | None
    complete_months: tuple[str, ...]  # YYYY-MM
    partial_months: tuple[str, ...]
    latest_complete_month: str | None
    data_end_lag_days: int | None
    transaction_types: dict[str, int]
    analysis_types: tuple[str, ...]
    dimensions: dict[str, DimensionProfile] = field(default_factory=dict)

    def status(self, dimension: str) -> str:
        info = self.dimensions.get(dimension)
        return info.status if info is not None else USABLE

    def is_usable(self, dimension: str) -> bool:
        return self.status(dimension) == USABLE

    @property
    def anchored_to_data_end(self) -> bool:
        """True when the data stops before the current month, so relative periods use the data's end."""
        return self.data_max is not None and self.data_max < self.today.replace(day=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "today": self.today.isoformat(),
            "transactions": self.transactions,
            "date_coverage": {
                "min": self.data_min.isoformat() if self.data_min else None,
                "max": self.data_max.isoformat() if self.data_max else None,
                "complete_months": list(self.complete_months),
                "partial_months": list(self.partial_months),
                "latest_complete_month": self.latest_complete_month,
                "data_end_lag_days": self.data_end_lag_days,
            },
            "transaction_types": dict(self.transaction_types),
            "analysis_types": list(self.analysis_types),
            "dimensions": {
                name: {
                    "status": info.status, "reason": info.reason, "distinct": info.distinct,
                    "median_per_value": info.median_per_value, "min_per_value": info.min_per_value,
                    "top10_transaction_share": round(info.top10_transaction_share, 3),
                    "top10_spend_share": round(info.top10_spend_share, 3),
                }
                for name, info in self.dimensions.items()
            },
        }


# ----- building -----


def _plural(count: float, word: str) -> str:
    number = f"{count:g}"
    return f"{number} {word}" if count == 1 else f"{number} {word}s"


def _value_stats(labels: pd.Series, spend: pd.Series) -> tuple[int, float, int, float, float, tuple[str, ...]]:
    counts = labels.value_counts(sort=True)
    if counts.empty:
        return 0, 0.0, 0, 0.0, 0.0, ()
    total = float(counts.sum())
    gross = spend.clip(lower=0)
    gross_by = gross.groupby(labels, observed=True).sum().sort_values(ascending=False)
    gross_total = float(gross_by.sum())
    top10_spend = float(gross_by.head(10).sum()) / gross_total if gross_total > 0 else 1.0
    return (
        int(counts.size),
        float(counts.median()),
        int(counts.min()),
        float(counts.head(10).sum()) / total,
        top10_spend,
        tuple(str(value) for value in counts.head(TOP_VALUES).index),
    )


def _categorical_status(name: str, distinct: int, median: float, top10: float) -> tuple[str, str]:
    noun = {"issuer": "card", "parent_group": "group", "category": "category", "merchant": "merchant"}[name]
    plural = "categories" if noun == "category" else f"{noun}s"
    if distinct < 2:
        return UNUSABLE, f"only {distinct} {noun if distinct == 1 else plural} in the data"
    reasons = []
    if median < MIN_MEDIAN_PER_VALUE:
        reasons.append(f"{distinct} {plural} with a median of {_plural(median, 'transaction')} each")
    if distinct > 10 and top10 < MIN_TOP10_SHARE:
        reasons.append(f"the top 10 {plural} cover only a small share of transactions")
    if reasons:
        return LIMITED, "; ".join(reasons)
    return USABLE, f"{distinct} {plural}, a median of {_plural(median, 'transaction')} each"


def _dimension(name: str, labels: pd.Series, spend: pd.Series, status: tuple[str, str] | None = None) -> DimensionProfile:
    distinct, median, minimum, top10, top10_spend, top_values = _value_stats(labels, spend)
    if status is None:
        status = _categorical_status(name, distinct, median, top10)
    return DimensionProfile(name, distinct, median, minimum, top10, top10_spend, status[0], status[1], top_values)


def _month_status(complete: int, touched: int) -> tuple[str, str]:
    if complete >= MIN_COMPLETE_MONTHS:
        return USABLE, f"{_plural(complete, 'complete month')} of data"
    if touched <= 1:
        return UNUSABLE, "the data covers a single month"
    if complete == 1:
        return LIMITED, "only 1 complete month of data; the others are partial"
    return LIMITED, f"no complete month yet ({_plural(touched, 'partial month')})"


def _month_text(value: date) -> str:
    return f"{value:%Y-%m}"


def build_data_profile(df: pd.DataFrame, *, today: date) -> DataProfile:
    """Profile the chat's transaction frame (``data.frame_from_rows`` shape)."""
    schema = tuple(
        ColumnInfo(name, gold_type, str(df[name].dtype) if name in df.columns else "missing", meaning)
        for name, gold_type, meaning in SCHEMA
    )
    types = df["transaction_type"].value_counts().to_dict() if "transaction_type" in df.columns else {}
    rows = df[df["transaction_type"].isin(ANALYSIS_TYPES)] if len(df) else df
    if rows.empty:
        empty = {name: DimensionProfile(name, 0, 0.0, 0, 0.0, 0.0, UNUSABLE, "there is no spending data yet")
                 for name in PROFILED_DIMENSIONS}
        return DataProfile(today, schema, 0, None, None, (), (), None, None,
                           {str(k): int(v) for k, v in types.items()}, ANALYSIS_TYPES, empty)

    dates = rows["transaction_date"]
    data_min, data_max = dates.min().date(), dates.max().date()
    complete, partial = month_coverage(data_min, data_max, today)
    spend = rows["spend"].astype(float)

    dimensions: dict[str, DimensionProfile] = {}
    for name in CATEGORICAL:
        dimensions[name] = _dimension(name, rows[name].astype(str), spend)

    weekday = rows["weekday"].astype(str)
    stats = _value_stats(weekday, spend)
    if stats[0] < 2:
        status = (UNUSABLE, f"transactions fall on only {_plural(stats[0], 'weekday')}")
    elif stats[1] < MIN_MEDIAN_PER_VALUE:
        status = (LIMITED, f"a median of only {_plural(stats[1], 'transaction')} per weekday")
    else:
        status = (USABLE, f"a median of {_plural(stats[1], 'transaction')} per weekday")
    dimensions["weekday"] = _dimension("weekday", weekday, spend, status)

    day = dates.dt.day.astype(str)
    stats = _value_stats(day, spend)
    status = (
        (LIMITED, f"a median of only {_plural(stats[1], 'transaction')} per day of the month")
        if stats[1] < MIN_DAY_OF_MONTH_MEDIAN else
        (USABLE, f"a median of {_plural(stats[1], 'transaction')} per day of the month")
    )
    dimensions["day_of_month"] = _dimension("day_of_month", day, spend, status)

    month = dates.dt.strftime("%Y-%m")
    touched = len(complete) + len(partial)
    dimensions["month"] = _dimension("month", month, spend, _month_status(len(complete), touched))

    quarter = dates.dt.year.astype(str) + "-Q" + dates.dt.quarter.astype(str)
    quarters = int(quarter.nunique())
    status = (USABLE, f"{_plural(quarters, 'quarter')} with data") if quarters >= MIN_QUARTERS else (
        UNUSABLE, "the data covers a single quarter")
    dimensions["quarter"] = _dimension("quarter", quarter, spend, status)

    week = (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    weeks = int(week.nunique())
    if weeks >= MIN_WEEKS:
        status = (USABLE, f"{_plural(weeks, 'week')} with data")
    elif weeks >= 2:
        status = (LIMITED, f"only {_plural(weeks, 'week')} with data")
    else:
        status = (UNUSABLE, "the data covers a single week")
    dimensions["week"] = _dimension("week", week, spend, status)

    return DataProfile(
        today=today,
        schema=schema,
        transactions=int(len(rows)),
        data_min=data_min,
        data_max=data_max,
        complete_months=tuple(_month_text(m) for m in complete),
        partial_months=tuple(_month_text(m) for m in partial),
        latest_complete_month=_month_text(complete[-1]) if complete else None,
        data_end_lag_days=(today - data_max).days,
        transaction_types={str(k): int(v) for k, v in types.items()},
        analysis_types=ANALYSIS_TYPES,
        dimensions=dimensions,
    )


# ----- example questions -----


def _month_name(text: str) -> str:
    return date(int(text[:4]), int(text[5:7]), 1).strftime("%b %Y")


def suggest_examples(profile: DataProfile | None, *, minimum: int = 4, maximum: int = 6) -> list[str]:
    """4-6 example questions that fit the data: only usable dimensions and the actual date range."""
    from . import prompts

    if profile is None or profile.transactions == 0:
        return list(prompts.STATIC_EXAMPLES)
    examples: list[str] = []
    usable = profile.is_usable
    months = len(profile.complete_months)
    if usable("month"):
        count = min(6, months)
        template = prompts.EXAMPLE_MONTHLY_BY_CATEGORY if usable("category") else prompts.EXAMPLE_MONTHLY
        examples.append(template.format(months=count))
    issuers = profile.dimensions["issuer"].top_values
    if usable("issuer") and len(issuers) >= 2:
        first, second = sorted(issuers[:2])
        template = prompts.EXAMPLE_COMPARE_CARDS_BY_MONTH if usable("month") else prompts.EXAMPLE_COMPARE_CARDS
        examples.append(template.format(first=first, second=second))
    if usable("weekday"):
        examples.append(prompts.EXAMPLE_WEEKDAY)
    groups = set(profile.dimensions["parent_group"].top_values)
    if usable("category") and usable("parent_group") and {"Wants", "Needs"} <= groups:
        examples.append(prompts.EXAMPLE_WANTS_VS_NEEDS)
    elif usable("category"):
        examples.append(prompts.EXAMPLE_TOP_CATEGORIES)
    if usable("merchant"):
        if profile.latest_complete_month:
            examples.append(prompts.EXAMPLE_TOP_MERCHANTS_MONTH.format(month=_month_name(profile.latest_complete_month)))
        else:
            examples.append(prompts.EXAMPLE_TOP_MERCHANTS)
    if profile.latest_complete_month and len(examples) < maximum:
        examples.append(prompts.EXAMPLE_MONTH_VS_PREVIOUS.format(month=_month_name(profile.latest_complete_month)))
    for fallback in prompts.STATIC_EXAMPLES:
        if len(examples) >= minimum:
            break
        if fallback not in examples:
            examples.append(fallback)
    return examples[:maximum]


def safe_profile(df: pd.DataFrame, *, today: date) -> DataProfile | None:
    """build_data_profile that never raises (the chat works without a profile)."""
    try:
        return build_data_profile(df, today=today)
    except Exception:  # pragma: no cover - defensive; profiling must not break the chat
        import logging

        logging.getLogger(__name__).exception("Could not profile the chat data")
        return None
