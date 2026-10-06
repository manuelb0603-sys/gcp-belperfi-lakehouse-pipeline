"""Declarative analysis spec for the finance chat.

The model only fills in an ``AnalysisSpec`` (exported as JSON schema for tool definitions);
deterministic Python resolves the period (``resolve_period``), repairs or rejects the spec
against the data catalog (``validate_and_repair``), and executes it (``executor.run_analysis``).
The model never writes code and never sees transaction rows.
"""

from __future__ import annotations

import calendar
import difflib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .merchants import clean_label, normalize_merchant

Metric = Literal["net_spend", "gross_spend", "refunds", "transaction_count", "avg_transaction", "share_of_total"]
Dimension = Literal[
    "month", "quarter", "week", "weekday", "day_of_month", "category", "parent_group", "issuer", "merchant"
]
Period = Literal[
    "all", "this_month", "last_month", "last_3_months", "last_6_months", "last_12_months",
    "year_to_date", "last_year", "this_quarter", "last_quarter", "focus_month", "custom",
]
Sort = Literal["auto", "desc", "asc", "label"]
Transform = Literal["none", "cumulative", "rolling_3", "pct_change"]
CompareTo = Literal["none", "previous_period", "same_period_last_year"]
Chart = Literal["auto", "line", "bar", "stacked_bar", "grouped_bar", "heatmap", "pie", "treemap", "table", "none"]

METRICS: tuple[str, ...] = Metric.__args__  # type: ignore[attr-defined]
DIMENSIONS: tuple[str, ...] = Dimension.__args__  # type: ignore[attr-defined]
PERIODS: tuple[str, ...] = Period.__args__  # type: ignore[attr-defined]
TIME_DIMENSIONS = ("month", "quarter", "week")
ORDERED_DIMENSIONS = TIME_DIMENSIONS + ("weekday", "day_of_month")
CATEGORICAL_DIMENSIONS = ("category", "parent_group", "issuer", "merchant")

METRIC_DEFINITIONS: dict[str, str] = {
    "net_spend": "Net spend: purchases minus refunds and credits (sum of -amount); gold net_real_expense.",
    "gross_spend": "Gross spend: purchases only, before refunds (sum of -amount where amount < 0); gold gross_spending.",
    "refunds": "Refunds and credits: money returned to the card (sum of amount where amount > 0); gold total_refunds_and_credits.",
    "transaction_count": "Transaction count: number of purchases, returns, and adjustments; gold total_transaction_count.",
    "avg_transaction": "Average transaction: net spend divided by transaction count; gold average_transaction_amount.",
    "share_of_total": "Share of total: each group's net spend as a percent of total net spend for the same period and filters.",
}
METRIC_LABELS: dict[str, str] = {
    "net_spend": "Net spend",
    "gross_spend": "Gross spend",
    "refunds": "Refunds & credits",
    "transaction_count": "Transactions",
    "avg_transaction": "Average transaction",
    "share_of_total": "Share of total",
}
DIMENSION_LABELS: dict[str, str] = {
    "month": "month", "quarter": "quarter", "week": "week", "weekday": "weekday",
    "day_of_month": "day of month", "category": "category", "parent_group": "group",
    "issuer": "card", "merchant": "merchant",
}

# Chart readability limits shared by validate_and_repair and the executor's auto chart.
LINE_MIN_POINTS = 3
HEATMAP_MAX_ROWS = 15
HEATMAP_MAX_COLUMNS = 12
PIE_MAX_SLICES = 6
# Usable substitutes for a dimension the data profile marks limited or unusable, in order.
DIMENSION_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "merchant": ("category", "parent_group"),
    "category": ("parent_group",),
    "parent_group": ("category",),
    "issuer": (),
    "day_of_month": ("weekday",),
    "weekday": (),
}

_DATE_TEXT = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")


class SpecError(ValueError):
    """The spec cannot be executed; the message says why and lists valid alternatives."""


class Limitation(str):
    """A note about a data limitation or a profile-driven repair that the answer must mention.

    It is a plain ``str`` everywhere (lists, JSON); the executor copies these notes into the
    summary's ``limitations`` field, which compaction keeps as long as possible.
    """


def _as_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


class Filters(BaseModel):
    """Row filters. Amount bounds apply to the transaction size (absolute amount)."""

    model_config = ConfigDict(extra="forbid")

    category: list[str] = Field(default_factory=list, max_length=20, description="Keep only these categories.")
    exclude_category: list[str] = Field(default_factory=list, max_length=20, description="Drop these categories.")
    parent_group: list[str] = Field(default_factory=list, max_length=5, description="Keep only these groups (Needs, Wants, ...).")
    issuer: list[str] = Field(default_factory=list, max_length=5, description="Keep only these cards/issuers.")
    merchant: list[str] = Field(default_factory=list, max_length=20, description="Keep only these merchants (names from the catalog).")
    merchant_contains: str | None = Field(default=None, max_length=60, description="Keep merchants whose name contains this text.")
    min_amount: float | None = Field(default=None, ge=0, description="Minimum transaction size in dollars.")
    max_amount: float | None = Field(default=None, ge=0, description="Maximum transaction size in dollars.")

    @field_validator("category", "exclude_category", "parent_group", "issuer", "merchant", mode="before")
    @classmethod
    def _listify(cls, value: Any) -> Any:
        return _as_list(value)

    @field_validator("category", "exclude_category", "parent_group", "issuer", "merchant")
    @classmethod
    def _clean_items(cls, value: list[str]) -> list[str]:
        cleaned = [clean_label(item, 80) for item in value]
        return [item for item in cleaned if item]

    @field_validator("merchant_contains")
    @classmethod
    def _clean_contains(cls, value: str | None) -> str | None:
        return clean_label(value, 60) or None if value is not None else None

    @model_validator(mode="after")
    def _amount_order(self) -> "Filters":
        if self.min_amount is not None and self.max_amount is not None and self.min_amount > self.max_amount:
            raise ValueError("min_amount must not exceed max_amount")
        return self

    def active(self) -> dict[str, Any]:
        return {key: value for key, value in self.model_dump().items() if value not in (None, [], "")}


class AnalysisSpec(BaseModel):
    """One deterministic analysis over the user's card transactions."""

    model_config = ConfigDict(extra="forbid")

    metric: Metric = Field(default="net_spend", description="What to measure.")
    group_by: list[Dimension] = Field(default_factory=list, max_length=2, description="Up to two dimensions to break the metric down by.")
    period: Period = Field(default="last_12_months", description="Date range; resolved in Python relative to today.")
    date_from: str | None = Field(default=None, description="YYYY-MM or YYYY-MM-DD; used with period=custom (or focus_month).")
    date_to: str | None = Field(default=None, description="YYYY-MM or YYYY-MM-DD; used with period=custom.")
    filters: Filters = Field(default_factory=Filters)
    top_n: int | None = Field(default=None, ge=1, le=25, description="Keep the N largest groups; the rest become 'Other'.")
    sort: Sort = Field(default="auto", description="auto: chronological for time, largest first otherwise.")
    transform: Transform = Field(default="none", description="Applied along the time dimension.")
    compare_to: CompareTo = Field(default="none", description="Add a comparison period.")
    chart: Chart = Field(default="auto", description="Chart type; auto picks one from the dimensions.")
    title: str | None = Field(default=None, max_length=120, description="Optional chart title.")

    @field_validator("group_by", mode="before")
    @classmethod
    def _listify(cls, value: Any) -> Any:
        return _as_list(value)

    @field_validator("group_by")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("group_by dimensions must be distinct")
        return value

    @field_validator("date_from", "date_to")
    @classmethod
    def _date_text(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        value = value.strip()
        if not _DATE_TEXT.match(value):
            raise ValueError("dates must be YYYY-MM or YYYY-MM-DD")
        parse_date_text(value)  # raises ValueError for impossible dates
        return value

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str | None) -> str | None:
        return (clean_label(value, 120) or None) if value is not None else None


def model_json_schema() -> dict[str, Any]:
    """JSON schema of AnalysisSpec for model tool definitions."""
    return AnalysisSpec.model_json_schema()


# ----- dates -----


def parse_date_text(text: str, *, end: bool = False) -> date:
    """YYYY-MM -> first (or, with end=True, last) day of the month; YYYY-MM-DD -> that day."""
    parts = [int(part) for part in text.split("-")]
    if len(parts) == 2:
        year, month = parts
        return month_end(date(year, month, 1)) if end else date(year, month, 1)
    return date(*parts)


def month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def shift_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(value.day, calendar.monthrange(year, month + 1)[1]))


def custom_window(date_from: str | None, date_to: str | None, data_min: date, data_max: date) -> tuple[date, date, bool]:
    """Inclusive custom dates (YYYY-MM spans whole months); reversed inputs are swapped."""
    swapped = bool(date_from and date_to and parse_date_text(date_from) > parse_date_text(date_to, end=True))
    if swapped:
        date_from, date_to = date_to, date_from
    start = parse_date_text(date_from) if date_from else data_min
    end = parse_date_text(date_to, end=True) if date_to else data_max
    return start, end, swapped


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _quarter_start(value: date) -> date:
    return date(value.year, 3 * ((value.month - 1) // 3) + 1, 1)


def months_between(start: date, end: date) -> int:
    """Calendar months touched by [start, end] (inclusive)."""
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _fmt_day(value: date) -> str:
    return f"{value:%b} {value.day}, {value.year}"


# A month counts as complete when the data starts at most this many days after its first day
# and ends at most this many days before its last day (and the month is over).
COMPLETE_EDGE_DAYS = 2
RELATIVE_PERIODS = (
    "this_month", "last_month", "last_3_months", "last_6_months", "last_12_months",
    "year_to_date", "this_quarter", "last_quarter",
)


def month_coverage(data_min: date, data_max: date, today: date) -> tuple[list[date], list[date]]:
    """(complete, partial) first days of the months touched by [data_min, data_max]."""
    complete: list[date] = []
    partial: list[date] = []
    if data_min > data_max:
        return complete, partial
    month = _month_start(data_min)
    while month <= data_max:
        end = month_end(month)
        starts = data_min <= month + timedelta(days=COMPLETE_EDGE_DAYS)
        ends = data_max >= end - timedelta(days=COMPLETE_EDGE_DAYS) and today > end
        (complete if starts and ends else partial).append(month)
        month = shift_months(month, 1)
    return complete, partial


def latest_complete_month(data_min: date, data_max: date, today: date) -> date | None:
    complete, _ = month_coverage(data_min, data_max, today)
    return complete[-1] if complete else None


def period_label(start: date, end: date) -> str:
    """Readable label: 'September 2026', 'Jul–Sep 2026', 'Jan 2025–Dec 2025', or day ranges."""
    if start > end:
        return "no dates"
    whole_months = start.day == 1 and end == month_end(end)
    if whole_months:
        if (start.year, start.month) == (end.year, end.month):
            return f"{start:%B %Y}"
        if start.year == end.year:
            return f"{start:%b}–{end:%b %Y}"
        return f"{start:%b %Y}–{end:%b %Y}"
    if start == end:
        return _fmt_day(start)
    return f"{_fmt_day(start)} – {_fmt_day(end)}"


@dataclass(frozen=True)
class ResolvedPeriod:
    start: date
    end: date  # inclusive
    label: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def months(self) -> int:
        return months_between(self.start, self.end)

    @property
    def whole_months(self) -> bool:
        return self.start.day == 1 and self.end == month_end(self.end)


def _period(start: date, end: date, name: str | None = None, notes: Sequence[str] = ()) -> ResolvedPeriod:
    label = period_label(start, end)
    if name:
        label = f"{label} ({name})"
    return ResolvedPeriod(start, end, label, tuple(notes))


def resolve_period(
    spec: AnalysisSpec,
    today: date,
    data_min: date | None,
    data_max: date | None,
    focus_month: date | None = None,
) -> ResolvedPeriod:
    """Turn spec.period into concrete inclusive dates. "last N months" means complete months
    before the current one; to-date periods (this month/quarter, year to date) end today."""
    notes: list[str] = []
    data_min = data_min or today
    data_max = data_max or today
    this_month = _month_start(today)
    period = spec.period

    if period == "all":
        start, end, name = data_min, data_max, "all data"
    elif period == "this_month":
        start, end, name = this_month, today, "month to date"
    elif period == "last_month":
        start = shift_months(this_month, -1)
        start, end, name = start, month_end(start), None
    elif period in ("last_3_months", "last_6_months", "last_12_months"):
        count = int(period.split("_")[1])
        start, end, name = shift_months(this_month, -count), this_month - timedelta(days=1), f"last {count} months"
    elif period == "year_to_date":
        start, end, name = date(today.year, 1, 1), today, "year to date"
    elif period == "last_year":
        start, end, name = date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), None
    elif period == "this_quarter":
        start, end, name = _quarter_start(today), today, "quarter to date"
    elif period == "last_quarter":
        start = shift_months(_quarter_start(today), -3)
        start, end, name = start, shift_months(start, 3) - timedelta(days=1), "last quarter"
    elif period == "focus_month":
        anchor = parse_date_text(spec.date_from) if spec.date_from else (focus_month or data_max)
        start = _month_start(anchor)
        start, end, name = start, month_end(start), "focus month"
    else:  # custom
        start, end, swapped = custom_window(spec.date_from, spec.date_to, data_min, data_max)
        name = None
        if swapped:
            notes.append("date_from was after date_to, so the dates were swapped.")
        if start < data_min or end > data_max:
            clamped = (max(start, data_min), min(end, data_max))
            if clamped[0] <= clamped[1]:
                notes.append(
                    f"Custom dates were limited to the available data ({_fmt_day(data_min)} – {_fmt_day(data_max)})."
                )
                start, end = clamped

    if period in RELATIVE_PERIODS and data_max < this_month and end > data_max:
        anchored = _anchor_to_data_end(period, data_min, data_max, today)
        if anchored is not None:
            start, end, name, note = anchored
            notes.append(note)

    if end < data_min or start > data_max:
        notes.append(f"No data in this period; available data covers {_fmt_day(data_min)} – {_fmt_day(data_max)}.")
    else:
        if end > data_max and period not in ("all",):
            notes.append(f"Data currently ends {_fmt_day(data_max)}, so this period is incomplete.")
        if start < data_min and period not in ("all", "custom"):
            notes.append(f"Data starts {_fmt_day(data_min)}, so the start of this period is partial.")
    return _period(start, end, name, notes)


def _anchor_to_data_end(
    period: str, data_min: date, data_max: date, today: date
) -> tuple[date, date, str, Limitation] | None:
    """Re-anchor a relative period to the end of the data when the data stops before this month.

    "this month", "this quarter", and "year to date" end at the last data date; "last month" is
    the latest complete month and "last N months"/"last quarter" end with it.
    """
    latest = latest_complete_month(data_min, data_max, today)
    anchor = _month_start(data_max)
    ends = f"Data ends {_fmt_day(data_max)}"

    def whole(month: date) -> tuple[date, date]:
        return month, (month_end(month) if month == latest else min(month_end(month), data_max))

    if period == "this_month":
        start, end = whole(anchor)
        state = "complete" if anchor == latest else "partial"
        return start, end, "latest month with data", Limitation(f"{ends}; 'this month' = {anchor:%b %Y} ({state}).")
    if period == "last_month":
        month = latest or anchor
        start, end = whole(month)
        state = "the latest complete month" if month == latest else "partial; no complete month yet"
        return start, end, None, Limitation(f"{ends}; 'last month' = {month:%b %Y} ({state}).")
    if period in ("last_3_months", "last_6_months", "last_12_months"):
        count = int(period.split("_")[1])
        last = latest or anchor
        start, end = shift_months(last, -(count - 1)), whole(last)[1]
        label = period_label(start, end) if end == month_end(end) else f"{start:%b %Y}–{_fmt_day(end)}"
        return start, end, f"last {count} months", Limitation(
            f"{ends}; 'last {count} months' = {label} "
            f"({'ending with the latest complete month' if last == latest else 'no complete month yet'})."
        )
    if period == "year_to_date":
        start = date(data_max.year, 1, 1)
        return start, data_max, "year to date", Limitation(f"{ends}; 'year to date' = Jan 1 – {_fmt_day(data_max)}.")
    if period == "this_quarter":
        start = _quarter_start(data_max)
        return start, data_max, "latest quarter with data", Limitation(
            f"{ends}; 'this quarter' = {_fmt_day(start)} – {_fmt_day(data_max)} (partial)."
        )
    if period == "last_quarter" and latest is not None:
        last = latest
        while last.month % 3:
            last = shift_months(last, -1)
        start = shift_months(last, -2)
        complete, _ = month_coverage(data_min, data_max, today)
        if start not in complete or last not in complete:
            return None
        return start, month_end(last), "last quarter", Limitation(
            f"{ends}; 'last quarter' = {start:%b}–{last:%b %Y} (the latest complete quarter)."
        )
    return None


def comparison_period(current: ResolvedPeriod, compare_to: str) -> tuple[ResolvedPeriod, int, int] | None:
    """The comparison window and the shift (months, days) that maps it onto the current one."""
    if compare_to == "none":
        return None
    if compare_to == "same_period_last_year":
        months, days = 12, 0
    elif current.start.day == 1:
        months, days = months_between(current.start, current.end), 0
    else:
        months, days = 0, (current.end - current.start).days + 1
    if months:
        start = shift_months(current.start, -months)
        end = month_end(shift_months(current.end, -months)) if current.end == month_end(current.end) else shift_months(current.end, -months)
    else:
        start, end = current.start - timedelta(days=days), current.end - timedelta(days=days)
    name = "same period last year" if compare_to == "same_period_last_year" else "previous period"
    return _period(start, end, name), months, days


# ----- validation and repair -----


def _closest(value: str, options: Sequence[str], n: int = 5) -> list[str]:
    lowered = {option.lower(): option for option in options}
    matches = difflib.get_close_matches(value.lower(), list(lowered), n=n, cutoff=0.0)
    return [lowered[match] for match in matches]


def _match(value: str, options: Sequence[str], cutoff: float = 0.6) -> str | None:
    """Exact (case-insensitive), then unique prefix/substring, then difflib fuzzy match."""
    if not options:
        return None
    lowered = {option.lower(): option for option in options}
    key = value.lower().strip()
    if key in lowered:
        return lowered[key]
    contains = [option for low, option in lowered.items() if key and (low.startswith(key) or key in low)]
    if len(contains) == 1:
        return contains[0]
    fuzzy = difflib.get_close_matches(key, list(lowered), n=1, cutoff=cutoff)
    return lowered[fuzzy[0]] if fuzzy else None


def _unknown(kind: str, value: str, options: Sequence[str]) -> SpecError:
    closest = _closest(value, options)
    hint = f" Closest valid {kind} values: {', '.join(closest)}." if closest else f" No {kind} values are available."
    return SpecError(f"Unknown {kind} {value!r}.{hint}")


def _resolve_values(kind: str, values: list[str], options: Sequence[str], notes: list[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        match = _match(value, options)
        if match is None:
            raise _unknown(kind, value, options)
        if match != value:
            notes.append(f"Interpreted {kind} {value!r} as {match!r}.")
        if match not in resolved:
            resolved.append(match)
    return resolved


def _catalog_dates(catalog: Mapping[str, Any]) -> tuple[date | None, date | None]:
    dates = catalog.get("date_range") or {}
    to_date = lambda text: date.fromisoformat(text) if text else None  # noqa: E731
    return to_date(dates.get("min")), to_date(dates.get("max"))


def _profile_dimension(profile: Any, dimension: str) -> Any:
    dimensions = getattr(profile, "dimensions", None) or {}
    return dimensions.get(dimension)


def _status(profile: Any, dimension: str) -> tuple[str, str]:
    """(status, reason) from the data profile; everything is usable without a profile."""
    info = _profile_dimension(profile, dimension)
    return (info.status, info.reason) if info is not None else ("usable", "")


def heatmap_fits(sizes: Sequence[int]) -> bool:
    """A heatmap stays readable up to HEATMAP_MAX_ROWS x HEATMAP_MAX_COLUMNS cells (either way round)."""
    large, small = max(sizes), min(sizes)
    return large <= HEATMAP_MAX_ROWS and small <= HEATMAP_MAX_COLUMNS


def time_points(dimension: str, start: date, end: date) -> int:
    """Number of points a time dimension has over [start, end]."""
    if dimension == "quarter":
        return (end.year * 4 + (end.month - 1) // 3) - (start.year * 4 + (start.month - 1) // 3) + 1
    if dimension == "week":
        return ((end - start).days + start.weekday()) // 7 + 1
    return months_between(start, end)


def _repair_dimensions(group_by: list[str], filters: Mapping[str, Any], profile: Any, notes: list[str]) -> list[str]:
    """Swap dimensions the data profile marks limited/unusable for the best usable alternative.

    Merchant grouping is kept when the user filters to specific merchants. Time dimensions are
    kept (with a note): a month table is still correct, only thin.
    """
    if profile is None:
        return group_by
    repaired: list[str] = []
    for dim in group_by:
        status, reason = _status(profile, dim)
        if status == "usable":
            repaired.append(dim)
            continue
        label = DIMENSION_LABELS[dim]
        if dim == "merchant" and (filters.get("merchant") or filters.get("merchant_contains")):
            repaired.append(dim)
            continue
        if dim in TIME_DIMENSIONS:
            repaired.append(dim)
            notes.append(Limitation(f"Results by {label} are limited: {reason}."))
            continue
        alternative = next(
            (alt for alt in DIMENSION_ALTERNATIVES.get(dim, ())
             if alt not in group_by and alt not in repaired and _status(profile, alt)[0] == "usable"),
            None,
        )
        if alternative:
            repaired.append(alternative)
            notes.append(Limitation(
                f"Grouping by {label} isn't meaningful for this data ({reason}), so I grouped by "
                f"{DIMENSION_LABELS[alternative]} instead."
            ))
        elif status == "unusable":
            notes.append(Limitation(f"Grouping by {label} isn't possible with this data ({reason}), so it was left out."))
        else:
            repaired.append(dim)
            notes.append(Limitation(f"Results by {label} are limited: {reason}."))
    return repaired


def _estimated_slices(
    spec: AnalysisSpec, dimension: str, catalog: Mapping[str, Any], months: int, profile: Any = None
) -> int:
    filters = spec.filters
    folds = dimension in CATEGORICAL_DIMENSIONS or not any(dim in CATEGORICAL_DIMENSIONS for dim in spec.group_by)
    if spec.top_n and folds:  # top_n folds the first categorical dimension (else it keeps the top rows)
        return spec.top_n + 1
    if dimension == "category":
        return len(filters.category) or len(catalog.get("categories") or {}) - len(filters.exclude_category)
    if dimension == "parent_group":
        return len(filters.parent_group) or len(catalog.get("parent_groups") or [])
    if dimension == "issuer":
        return len(filters.issuer) or len(catalog.get("issuers") or [])
    if dimension == "merchant":
        info = _profile_dimension(profile, "merchant")
        return len(filters.merchant) or (info.distinct if info is not None else 99)
    return {"weekday": 7, "month": months, "quarter": max(1, months // 3), "week": months * 4, "day_of_month": 31}.get(dimension, 99)


def validate_and_repair(
    spec: AnalysisSpec | Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    known_merchants: Sequence[str] | None = None,
    today: date | None = None,
    focus_month: date | None = None,
    profile: Any = None,
) -> tuple[AnalysisSpec, list[str]]:
    """Return a runnable copy of the spec plus notes describing every repair.

    With a data ``profile`` (``profile.DataProfile``), group-by dimensions the profile marks
    limited or unusable are swapped for usable ones (merchant -> category) unless the user
    filtered to specific merchants; those notes are ``Limitation`` strings the answer must mention.

    Raises SpecError (a ValueError) when a filter value matches nothing in the catalog;
    the message lists the closest valid values so the model can retry.
    """
    if not isinstance(spec, AnalysisSpec):
        spec = AnalysisSpec.model_validate(spec)
    notes: list[str] = []
    data = spec.model_dump()
    filters = data["filters"]
    today = today or date.today()

    categories = list((catalog.get("categories") or {}).keys())
    filters["category"] = _resolve_values("category", filters["category"], categories, notes)
    filters["exclude_category"] = _resolve_values("category", filters["exclude_category"], categories, notes)
    filters["parent_group"] = _resolve_values("parent_group", filters["parent_group"], catalog.get("parent_groups") or [], notes)
    filters["issuer"] = _resolve_values("issuer", filters["issuer"], catalog.get("issuers") or [], notes)

    merchants = list(known_merchants if known_merchants is not None else catalog.get("merchants") or [])
    resolved_merchants: list[str] = []
    for value in filters["merchant"]:
        containing = [name for name in merchants if value.lower() in name.lower()]
        if len(containing) > 1 and not filters["merchant_contains"]:
            filters["merchant_contains"] = value
            notes.append(f"Several merchants match {value!r} ({', '.join(containing[:4])}); including all of them.")
            continue
        match = _match(value, merchants) or _match(normalize_merchant(value), merchants)
        if match is None:
            contains = [name for name in merchants if value.lower() in name.lower()]
            if contains and not filters["merchant_contains"]:
                filters["merchant_contains"] = value
                notes.append(f"No merchant is named {value!r}; matching merchants that contain it instead.")
                continue
            raise _unknown("merchant", value, merchants)
        if match != value:
            notes.append(f"Interpreted merchant {value!r} as {match!r}.")
        if match not in resolved_merchants:
            resolved_merchants.append(match)
    filters["merchant"] = resolved_merchants

    # Dates: anything given with a non-custom period becomes custom; clamp to data.
    data_min, data_max = _catalog_dates(catalog)
    if (data["date_from"] or data["date_to"]) and data["period"] not in ("custom", "focus_month"):
        notes.append(f"Explicit dates were given, so period {data['period']!r} was replaced by custom dates.")
        data["period"] = "custom"
    if data["period"] == "custom" and data_min and data_max:
        start, end, _ = custom_window(data["date_from"], data["date_to"], data_min, data_max)
        if end < data_min or start > data_max:
            raise SpecError(
                f"The requested dates are outside the available data ({data_min.isoformat()} to {data_max.isoformat()})."
            )
        if start < data_min or end > data_max:
            start, end = max(start, data_min), min(end, data_max)
            notes.append(f"Custom dates were limited to the available data ({data_min.isoformat()} to {data_max.isoformat()}).")
        data["date_from"], data["date_to"] = start.isoformat(), end.isoformat()

    group_by: list[str] = _repair_dimensions(list(data["group_by"]), filters, profile, notes)
    data["group_by"] = group_by
    if "merchant" in group_by and not data["top_n"]:
        data["top_n"] = 10
        notes.append("Showing the top 10 merchants; the rest are grouped as Other.")

    if data["compare_to"] != "none" and data["transform"] != "none":
        notes.append(f"Transform {data['transform']!r} cannot be combined with a comparison, so it was dropped.")
        data["transform"] = "none"
    if data["transform"] != "none" and not any(dim in TIME_DIMENSIONS for dim in group_by):
        notes.append(f"Transform {data['transform']!r} needs a month, quarter, or week dimension, so it was dropped.")
        data["transform"] = "none"

    candidate = AnalysisSpec.model_validate(data)
    period = resolve_period(candidate, today, data_min, data_max, focus_month)
    months = period.months

    chart = data["chart"]
    has_time = any(dim in TIME_DIMENSIONS for dim in group_by)
    if chart == "heatmap" and len(group_by) < 2:
        if not group_by:
            chart = "table"
            notes.append("A heatmap needs two dimensions; showing a table instead.")
        else:
            order = ["weekday", "month"] if data["metric"] == "transaction_count" else ["month", "weekday"]
            extra = next(
                (dim for dim in order if dim not in group_by and (dim != "month" or months >= 3)
                 and not (dim == "month" and has_time) and _status(profile, dim)[0] == "usable"),
                None,
            )
            if extra:
                group_by.append(extra)
                notes.append(f"A heatmap needs two dimensions, so {DIMENSION_LABELS[extra]} was added.")
            else:
                chart = "bar"
                notes.append("A heatmap needs two dimensions; showing a bar chart instead.")
    if chart == "heatmap" and len(group_by) == 2:
        sizes = [_estimated_slices(candidate, dim, catalog, months, profile) for dim in group_by]
        if not heatmap_fits(sizes):
            chart = "auto"
            notes.append(
                f"A heatmap of about {sizes[0]} x {sizes[1]} cells is hard to read (at most "
                f"{HEATMAP_MAX_ROWS} x {HEATMAP_MAX_COLUMNS}), so a simpler chart was chosen."
            )
    if chart == "line" and not has_time:
        chart = "bar"
        notes.append("A line chart needs a month, quarter, or week dimension; showing a bar chart instead.")
    elif chart == "line" and len(group_by) == 1:
        points = time_points(group_by[0], period.start, period.end)
        if points < LINE_MIN_POINTS:
            chart = "bar"
            notes.append(f"A line chart needs at least {LINE_MIN_POINTS} points and this period has {points}; showing bars instead.")
    if chart in ("stacked_bar", "grouped_bar") and len(group_by) < 2:
        chart = "bar"
        notes.append("Stacked and grouped bars need two dimensions; showing a bar chart instead.")
    if chart == "pie":
        if len(group_by) != 1:
            chart = "bar"
            notes.append("A pie chart needs exactly one dimension; showing a bar chart instead.")
        elif _estimated_slices(candidate, group_by[0], catalog, months, profile) > PIE_MAX_SLICES:
            chart = "bar"
            notes.append("A pie chart with more than 6 slices is hard to read; showing a bar chart instead.")
    data["chart"] = chart
    data["group_by"] = group_by
    return AnalysisSpec.model_validate(data), notes
