"""Deterministic execution of an AnalysisSpec over the transaction frame.

Metric math mirrors gold ``fact_monthly_spending`` exactly: only Sale/Return/Adjustment rows,
gross = sum of -amount over purchases, refunds = sum of positive amounts, net = sum(-amount),
count = rows, average = net / count.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis_spec import (
    CATEGORICAL_DIMENSIONS,
    DIMENSION_LABELS,
    LINE_MIN_POINTS,
    METRIC_DEFINITIONS,
    METRIC_LABELS,
    ORDERED_DIMENSIONS,
    PIE_MAX_SLICES,
    TIME_DIMENSIONS,
    AnalysisSpec,
    Limitation,
    ResolvedPeriod,
    comparison_period,
    heatmap_fits,
    resolve_period,
)
from .data import WEEKDAYS, analysis_rows

OTHER = "Other"
SUMMARY_MAX_BYTES = 1500
MAX_SERIES = 8
ADDITIVE_METRICS = ("net_spend", "gross_spend", "refunds", "transaction_count")
TRANSFORM_LABELS = {"cumulative": "cumulative", "rolling_3": "3-period rolling average", "pct_change": "% change from prior period"}
_SUMS = ("net", "gross", "refunds", "count")


@dataclass
class AnalysisContext:
    today: date = field(default_factory=date.today)
    focus_month: date | None = None
    data_min: date | None = None
    data_max: date | None = None
    notes: Sequence[str] = ()  # repair notes from validate_and_repair, carried into the result
    profile: Any = None  # profile.DataProfile: the auto chart avoids heatmaps on limited dimensions


@dataclass
class AnalysisResult:
    spec: AnalysisSpec
    table: pd.DataFrame
    summary: dict[str, Any]
    caption: str
    notes: list[str]
    chart: str
    period: ResolvedPeriod
    comparison: ResolvedPeriod | None
    dimensions: list[str]
    value_column: str
    compare_column: str | None
    unit: str  # "usd", "count", or "pct"


# ----- helpers -----


def _unit(metric: str, transform: str = "none") -> str:
    if transform == "pct_change" or metric == "share_of_total":
        return "pct"
    return "count" if metric == "transaction_count" else "usd"


def _round(series: pd.Series, unit: str) -> pd.Series:
    if unit == "count":
        return series.round(0).astype("Int64") if series.notna().all() else series.round(0)
    return series.round(1 if unit == "pct" else 2)


def _scalar(value: Any, unit: str) -> Any:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if pd.isna(value):
        return None
    if unit == "count":
        return int(round(float(value)))
    return round(float(value), 1 if unit == "pct" else 2)


def _apply_filters(rows: pd.DataFrame, spec: AnalysisSpec) -> pd.DataFrame:
    f = spec.filters
    mask = pd.Series(True, index=rows.index)
    if f.category:
        mask &= rows["category"].isin(f.category)
    if f.exclude_category:
        mask &= ~rows["category"].isin(f.exclude_category)
    if f.parent_group:
        mask &= rows["parent_group"].isin(f.parent_group)
    if f.issuer:
        mask &= rows["issuer"].isin(f.issuer)
    if f.merchant:
        wanted = {name.lower() for name in f.merchant}
        mask &= rows["merchant"].str.lower().isin(wanted)
    if f.merchant_contains:
        mask &= rows["merchant"].str.contains(f.merchant_contains, case=False, regex=False)
    size = rows["amount"].abs()
    if f.min_amount is not None:
        mask &= size >= f.min_amount
    if f.max_amount is not None:
        mask &= size <= f.max_amount
    return rows[mask]


def _in_period(rows: pd.DataFrame, period: ResolvedPeriod) -> pd.DataFrame:
    dates = rows["transaction_date"]
    return rows[(dates >= pd.Timestamp(period.start)) & (dates < pd.Timestamp(period.end + timedelta(days=1)))]


def _week_start(dates: pd.Series) -> pd.Series:
    return (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")


def _with_dimensions(rows: pd.DataFrame, dims: Sequence[str], keep: dict[str, set[str]]) -> pd.DataFrame:
    """Add one string/int label column per dimension (prefixed ``d_``) plus the sum inputs."""
    out = pd.DataFrame(index=rows.index)
    dates = rows["transaction_date"]
    for dim in dims:
        if dim == "month":
            out["d_month"] = dates.dt.strftime("%Y-%m")
        elif dim == "quarter":
            out["d_quarter"] = dates.dt.year.astype(str) + "-Q" + dates.dt.quarter.astype(str)
        elif dim == "week":
            out["d_week"] = _week_start(dates)
        elif dim == "weekday":
            out["d_weekday"] = rows["weekday"].astype(str)
        elif dim == "day_of_month":
            out["d_day_of_month"] = dates.dt.day.astype("int64")
        else:
            labels = rows[dim].astype(str)
            if dim in keep:
                labels = labels.where(labels.isin(keep[dim]), OTHER)
            out[f"d_{dim}"] = labels
    out["net"] = rows["spend"].astype(float)
    out["gross"] = rows["spend"].clip(lower=0).astype(float)
    out["refunds"] = rows["amount"].clip(lower=0).astype(float)
    out["count"] = 1
    return out


def _grid(dim: str, period: ResolvedPeriod) -> list[Any] | None:
    """Complete ordered labels for a dimension over the period (None for categorical dims)."""
    start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
    if dim == "month":
        return [str(p) for p in pd.period_range(start, end, freq="M")]
    if dim == "quarter":
        return [f"{p.year}-Q{p.quarter}" for p in pd.period_range(start, end, freq="Q")]
    if dim == "week":
        first = start - pd.Timedelta(days=start.dayofweek)
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(first, end, freq="7D")]
    if dim == "weekday":
        return list(WEEKDAYS)
    if dim == "day_of_month":
        last = end.day if (start.year, start.month) == (end.year, end.month) else 31
        return list(range(1, last + 1))
    return None


def _sums(frame: pd.DataFrame, dims: Sequence[str]) -> pd.DataFrame:
    cols = [f"d_{dim}" for dim in dims]
    if not cols:
        return frame[list(_SUMS)].sum().to_frame().T
    return frame.groupby(cols, sort=False)[list(_SUMS)].sum()


def _levels(dims: Sequence[str], period: ResolvedPeriod, *frames: pd.DataFrame) -> list[list[Any]]:
    levels = []
    for dim in dims:
        grid = _grid(dim, period)
        if grid is None:
            seen: list[Any] = []
            for frame in frames:
                for value in frame[f"d_{dim}"].unique().tolist():
                    if value not in seen:
                        seen.append(value)
            grid = seen
        levels.append(grid)
    return levels


def _reindex(sums: pd.DataFrame, dims: Sequence[str], levels: list[list[Any]], fill_grid: bool) -> pd.DataFrame:
    """Zero-fill: the full product of levels when a dimension is ordered, else observed rows."""
    if not dims:
        return sums
    names = [f"d_{dim}" for dim in dims]
    if fill_grid:
        index = pd.MultiIndex.from_product(levels, names=names) if len(dims) > 1 else pd.Index(levels[0], name=names[0])
        return sums.reindex(index, fill_value=0)
    return sums


def _values(sums: pd.DataFrame, metric: str, share_base: float) -> pd.Series:
    if metric == "net_spend":
        return sums["net"].astype(float)
    if metric == "gross_spend":
        return sums["gross"].astype(float)
    if metric == "refunds":
        return sums["refunds"].astype(float)
    if metric == "transaction_count":
        return sums["count"].astype(float)
    if metric == "avg_transaction":
        return sums["net"].astype(float) / sums["count"].replace(0, np.nan).astype(float)
    return sums["net"].astype(float) / share_base * 100 if share_base else sums["net"].astype(float) * np.nan


def _total(sums_row: pd.Series, metric: str) -> float | None:
    if metric == "share_of_total":
        return 100.0 if sums_row["net"] else None
    value = _values(sums_row.to_frame().T, metric, float(sums_row["net"]))
    return float(value.iloc[0]) if len(value) else None


def _transform(table: pd.DataFrame, column: str, transform: str, time_dim: str, other: list[str]) -> pd.Series:
    ordered = table.sort_values(other + [time_dim])
    groups = ordered.groupby(other, sort=False)[column] if other else None

    def apply(series: pd.Series) -> pd.Series:
        if transform == "cumulative":
            return series.cumsum()
        if transform == "rolling_3":
            return series.rolling(3, min_periods=3).mean()
        previous = series.shift(1)
        return ((series - previous) / previous.abs() * 100).replace([np.inf, -np.inf], np.nan)

    result = groups.transform(apply) if groups is not None else apply(ordered[column])
    return result.reindex(table.index)


def _label(row: pd.Series, dims: Sequence[str]) -> str:
    return " / ".join(str(row[dim])[:32] for dim in dims) if dims else "total"


def _sort(table: pd.DataFrame, spec: AnalysisSpec, dims: list[str], column: str) -> pd.DataFrame:
    if not dims:
        return table
    if spec.sort in ("desc", "asc"):
        return table.sort_values(column, ascending=spec.sort == "asc", na_position="last", kind="stable")
    if spec.sort == "label":
        return table.sort_values(dims, kind="stable")
    ordered = [dim for dim in dims if dim in ORDERED_DIMENSIONS]
    if ordered:
        keys = [table[dim].map({v: i for i, v in enumerate(WEEKDAYS)}) if dim == "weekday" else table[dim] for dim in ordered]
        frame = table.assign(**{f"_k{i}": key for i, key in enumerate(keys)})
        sort_cols = [f"_k{i}" for i in range(len(keys))]
        frame = frame.assign(_v=-frame[column].fillna(-np.inf)).sort_values(sort_cols + ["_v"], kind="stable")
        return frame.drop(columns=sort_cols + ["_v"])
    other_last = table[dims[0]].eq(OTHER)
    return table.assign(_o=other_last, _v=-table[column].fillna(-np.inf)).sort_values(["_o", "_v"], kind="stable").drop(columns=["_o", "_v"])


def _auto_chart(spec: AnalysisSpec, dims: list[str], table: pd.DataFrame, column: str, profile: Any = None) -> str:
    """Pick a readable chart: a line needs at least LINE_MIN_POINTS time points, a heatmap two
    usable dimensions within HEATMAP_MAX_ROWS x HEATMAP_MAX_COLUMNS, a pie at most PIE_MAX_SLICES
    non-negative slices; otherwise bars."""
    if not dims or table.empty:
        return "table"
    time_dim = next((dim for dim in dims if dim in TIME_DIMENSIONS), None)
    enough_points = time_dim is not None and table[time_dim].nunique() >= LINE_MIN_POINTS
    if spec.compare_to != "none":
        return "table" if len(dims) > 1 else ("line" if dims[0] in TIME_DIMENSIONS and enough_points else "bar")
    if len(dims) == 1:
        dim = dims[0]
        if dim in TIME_DIMENSIONS:
            return "bar" if spec.top_n or not enough_points else "line"
        if (dim in CATEGORICAL_DIMENSIONS and spec.metric == "share_of_total" and len(table) <= PIE_MAX_SLICES
                and (table[column] >= 0).all()):
            return "pie"
        return "bar"
    bars = "stacked_bar" if (table[column].fillna(0) >= 0).all() else "grouped_bar"
    ordered = [dim for dim in dims if dim in ORDERED_DIMENSIONS]
    if len(ordered) == 2 or "weekday" in dims or "day_of_month" in dims:
        usable = profile is None or all(profile.is_usable(dim) for dim in dims)
        if usable and heatmap_fits([table[dim].nunique() for dim in dims]):
            return "heatmap"
        return bars
    if time_dim is not None:
        if (spec.metric not in ADDITIVE_METRICS or spec.transform != "none") and enough_points:
            return "line"
        return bars
    return bars


# ----- main entry -----


def run_analysis(df: pd.DataFrame, spec: AnalysisSpec, ctx: AnalysisContext | None = None) -> AnalysisResult:
    """Execute a (validated) spec; never raises for empty data, it reports it in notes."""
    ctx = ctx or AnalysisContext()
    notes: list[str] = list(ctx.notes)
    dims = list(spec.group_by)
    metric, unit = spec.metric, _unit(spec.metric)

    rows = analysis_rows(df)
    data_min = ctx.data_min or (df["transaction_date"].min().date() if not df.empty else ctx.today)
    data_max = ctx.data_max or (df["transaction_date"].max().date() if not df.empty else ctx.today)
    period = resolve_period(spec, ctx.today, data_min, data_max, ctx.focus_month)
    notes.extend(note for note in period.notes if note not in notes)

    filtered = _apply_filters(rows, spec)
    current = _in_period(filtered, period)
    if current.empty:
        notes.append("No matching transactions in this period.")

    # top_n: fold the smaller groups of the (first) categorical dimension into "Other".
    keep: dict[str, set[str]] = {}
    categorical = next((dim for dim in dims if dim in CATEGORICAL_DIMENSIONS), None)
    top_n = spec.top_n
    if categorical and not top_n and len(dims) == 2 and current[categorical].nunique() > MAX_SERIES:
        top_n = MAX_SERIES - 1
        notes.append(f"Showing the top {top_n} {DIMENSION_LABELS[categorical]} values; the rest are grouped as Other.")
    if categorical and top_n:
        ranking = _values(_sums(_with_dimensions(current, [categorical], {}), [categorical]), metric,
                          float(current["spend"].sum()))
        ranking = ranking.sort_values(ascending=False, kind="stable")
        if len(ranking) > top_n:
            keep[categorical] = {str(label) for label in ranking.index[:top_n]}

    frame = _with_dimensions(current, dims, keep)
    share_base = float(frame["net"].sum())
    fill_grid = any(dim in ORDERED_DIMENSIONS for dim in dims)

    comp = comparison_period(period, spec.compare_to)
    comp_frame = None
    if comp is not None:
        comp_period, months, days = comp
        comp_rows = _in_period(filtered, comp_period).copy()
        if months:
            comp_rows["transaction_date"] = comp_rows["transaction_date"] + pd.DateOffset(months=months)
        else:
            comp_rows["transaction_date"] = comp_rows["transaction_date"] + pd.Timedelta(days=days)
        comp_rows["weekday"] = pd.Categorical.from_codes(
            comp_rows["transaction_date"].dt.dayofweek.to_numpy(dtype="int64"), categories=list(WEEKDAYS), ordered=True
        ) if len(comp_rows) else comp_rows.get("weekday")
        comp_frame = _with_dimensions(comp_rows, dims, keep)

    levels = _levels(dims, period, frame, *( [comp_frame] if comp_frame is not None else []))
    sums = _reindex(_sums(frame, dims), dims, levels, fill_grid)
    if comp_frame is not None and dims and not fill_grid:
        index = sums.index.union(_sums(comp_frame, dims).index)
        sums = sums.reindex(index, fill_value=0)
    table = sums.copy()
    table[metric] = _values(sums, metric, share_base)

    value_column, compare_column = metric, None
    if comp_frame is not None:
        comp_sums = _sums(comp_frame, dims)
        comp_sums = comp_sums.reindex(sums.index, fill_value=0) if dims else comp_sums
        comp_base = float(comp_frame["net"].sum())
        table["comparison"] = _values(comp_sums, metric, comp_base).to_numpy()
        table["change"] = table[metric] - table["comparison"]
        table["change_pct"] = (table["change"] / table["comparison"].abs() * 100).replace([np.inf, -np.inf], np.nan)
        compare_column = "comparison"

    table = table.reset_index(drop=not dims)
    table = table.rename(columns={f"d_{dim}": dim for dim in dims})
    keep_cols = dims + [metric] + (["comparison", "change", "change_pct"] if compare_column else [])
    table = table[keep_cols]

    transform = spec.transform
    time_dim = next((dim for dim in dims if dim in TIME_DIMENSIONS), None)
    if transform != "none":
        if time_dim is None:
            notes.append(f"Transform {transform!r} needs a month, quarter, or week dimension and was skipped.")
            transform = "none"
        elif transform == "cumulative" and metric not in ADDITIVE_METRICS:
            notes.append("A cumulative total only makes sense for sums and counts, so it was skipped.")
            transform = "none"
        else:
            value_column = f"{metric}_{transform}"
            table[value_column] = _transform(table, metric, transform, time_dim, [d for d in dims if d != time_dim])

    if not categorical and top_n and dims:
        table = table.sort_values(value_column, ascending=False, na_position="last", kind="stable").head(top_n)
        notes.append(f"Showing the top {top_n} rows by {METRIC_LABELS[metric].lower()}.")
    else:
        table = _sort(table, spec, dims, value_column)

    # Rounding by unit.
    for column in table.columns:
        if column in dims:
            continue
        col_unit = "pct" if column in ("change_pct",) or column.endswith("_pct_change") else unit
        if column == "change" and unit == "pct":
            col_unit = "pct"
        table[column] = _round(table[column].astype(float), col_unit)
    table = table.reset_index(drop=True)

    chart = spec.chart if spec.chart != "auto" else _auto_chart(spec, dims, table, value_column, ctx.profile)
    if chart == "pie":
        if len(table) > PIE_MAX_SLICES:
            chart = "bar"
            notes.append("A pie chart with more than 6 slices is hard to read; showing a bar chart instead.")
        elif (table[value_column].fillna(0) < 0).any():
            chart = "bar"
            notes.append("A pie chart cannot show negative values; showing a bar chart instead.")
    if chart == "treemap" and not (table[value_column].fillna(0) > 0).any():
        chart = "table"
        notes.append("A treemap needs positive values; showing a table instead.")
    if chart not in ("table", "none") and not dims:
        chart = "table"

    totals = _sums(frame, [])
    total_row = totals.iloc[0] if len(totals) else pd.Series({key: 0 for key in _SUMS})
    comp_total = None
    if comp_frame is not None:
        comp_totals = _sums(comp_frame, []).iloc[0]
        comp_total = _total(comp_totals, metric)

    result_unit = _unit(metric, transform)
    summary = _summary(
        spec, metric, transform, dims, table, value_column, period, comp[0] if comp else None,
        _total(total_row, metric), int(total_row["count"]), comp_total, notes, unit, result_unit,
    )
    caption = _caption(spec, metric, transform, dims, period, comp[0] if comp else None, top_n if categorical else None)
    return AnalysisResult(
        spec=spec, table=table, summary=summary, caption=caption, notes=notes, chart=chart,
        period=period, comparison=comp[0] if comp else None, dimensions=dims,
        value_column=value_column, compare_column=compare_column, unit=result_unit,
    )


# ----- summary and caption -----


def _summary(
    spec: AnalysisSpec, metric: str, transform: str, dims: list[str], table: pd.DataFrame, column: str,
    period: ResolvedPeriod, comp: ResolvedPeriod | None, total: float | None, count: int,
    comp_total: float | None, notes: list[str], unit: str, value_unit: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metric": metric,
        "unit": unit,
        "period": {"label": period.label, "start": period.start.isoformat(), "end": period.end.isoformat()},
        "group_by": dims,
        "total": _scalar(total, unit) if metric != "share_of_total" else 100.0,
        "transactions": count,
        "rows": int(len(table)),
    }
    if transform != "none":
        summary["transform"] = transform
    filters = spec.filters.active()
    if filters:
        summary["filters"] = {k: (v[:5] if isinstance(v, list) else v) for k, v in filters.items()}

    if dims and not table.empty:
        values = table[[*dims, column]].dropna(subset=[column])
        if not values.empty:
            hi = values.loc[values[column].idxmax()]
            lo = values.loc[values[column].idxmin()]
            summary["highest"] = {"label": _label(hi, dims), "value": _scalar(hi[column], value_unit)}
            summary["lowest"] = {"label": _label(lo, dims), "value": _scalar(lo[column], value_unit)}
            top = values.sort_values(column, ascending=False, kind="stable").head(5)
            summary["top"] = [[_label(row, dims), _scalar(row[column], value_unit)] for _, row in top.iterrows()]
        if transform != "none":
            last = table.dropna(subset=[column])
            if not last.empty:
                summary["latest"] = {"label": _label(last.iloc[-1], dims), "value": _scalar(last.iloc[-1][column], value_unit)}

    change = _largest_change(table, dims, metric, unit, comp is not None)
    if change:
        summary["largest_change"] = change
    if comp is not None:
        delta = None if total is None or comp_total is None else total - comp_total
        summary["comparison"] = {
            "label": comp.label,
            "total": _scalar(comp_total, unit),
            "change": _scalar(delta, unit if unit != "count" else "count"),
            "change_pct": _scalar(delta / abs(comp_total) * 100, "pct") if delta is not None and comp_total else None,
        }
    limitations = [note for note in notes if isinstance(note, Limitation)]
    others = [note for note in notes if not isinstance(note, Limitation)]
    if limitations:
        # Profile repairs and data-end anchoring: the answer must mention these.
        summary["limitations"] = [str(note)[:240] for note in limitations[:3]]
    if others:
        summary["notes"] = [note[:160] for note in others[:4]]
    return _fit(summary)


def _largest_change(table: pd.DataFrame, dims: list[str], metric: str, unit: str, compared: bool) -> dict[str, Any] | None:
    if table.empty:
        return None
    if compared and dims:
        frame = table.dropna(subset=["change"])
        if frame.empty:
            return None
        row = frame.loc[frame["change"].abs().idxmax()]
        return {
            "label": _label(row, dims), "current": _scalar(row[metric], unit), "comparison": _scalar(row["comparison"], unit),
            "change": _scalar(row["change"], unit), "change_pct": _scalar(row["change_pct"], "pct"),
        }
    time_dim = next((dim for dim in dims if dim in TIME_DIMENSIONS), None)
    if time_dim is None:
        return None
    if len(dims) == 1:
        series = table.set_index(time_dim)[metric]
    elif metric in ADDITIVE_METRICS:
        series = table.groupby(time_dim, sort=True)[metric].sum()
    else:
        return None
    series = series.sort_index()
    diffs = series.diff().dropna()
    if diffs.empty:
        return None
    label = diffs.abs().idxmax()
    position = list(series.index).index(label)
    before, after = series.iloc[position - 1], series.iloc[position]
    return {
        "from": str(series.index[position - 1]), "to": str(label),
        "before": _scalar(before, unit), "after": _scalar(after, unit),
        "change": _scalar(after - before, unit),
        "change_pct": _scalar((after - before) / abs(before) * 100, "pct") if before else None,
    }


def _size(data: Any) -> int:
    return len(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))


def _fit(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep the summary under SUMMARY_MAX_BYTES by trimming optional detail."""
    steps = [
        lambda s: s.get("notes") and s.__setitem__("notes", s["notes"][:2]),
        lambda s: s.get("top") and s.__setitem__("top", s["top"][:3]),
        lambda s: s.pop("filters", None),
        lambda s: s.pop("notes", None),
        lambda s: s.pop("lowest", None),
        lambda s: s.get("top") and s.__setitem__("top", [[label[:20], value] for label, value in s["top"]]),
        lambda s: s.get("limitations") and s.__setitem__("limitations", s["limitations"][:1]),
    ]
    for step in steps:
        if _size(summary) <= SUMMARY_MAX_BYTES:
            break
        step(summary)
    return summary


def _fmt_filters(filters: Mapping[str, Any]) -> str:
    parts = []
    names = {"category": "category is", "exclude_category": "excluding category", "parent_group": "group is",
             "issuer": "card is", "merchant": "merchant is"}
    for key, label in names.items():
        if filters.get(key):
            parts.append(f"{label} {', '.join(filters[key])}")
    if filters.get("merchant_contains"):
        parts.append(f"merchant contains '{filters['merchant_contains']}'")
    if filters.get("min_amount") is not None:
        parts.append(f"transactions of at least ${filters['min_amount']:,.2f}")
    if filters.get("max_amount") is not None:
        parts.append(f"transactions of at most ${filters['max_amount']:,.2f}")
    return "; ".join(parts)


def _caption(
    spec: AnalysisSpec, metric: str, transform: str, dims: list[str], period: ResolvedPeriod,
    comp: ResolvedPeriod | None, top_n: int | None,
) -> str:
    measure = METRIC_LABELS[metric]
    if transform != "none":
        measure = f"{measure} ({TRANSFORM_LABELS[transform]})"
    by = f" by {' and '.join(DIMENSION_LABELS[d] for d in dims)}" if dims else ""
    text = [f"{measure}{by}, {period.label}.", METRIC_DEFINITIONS[metric].split(";")[0].rstrip(".") + "."]
    text.append("Card payments are excluded (Sale, Return, and Adjustment transactions only).")
    filters = _fmt_filters(spec.filters.active())
    if filters:
        text.append(f"Filters: {filters}.")
    if top_n:
        text.append(f"Top {top_n} shown; the rest are grouped as Other.")
    if comp is not None:
        text.append(f"Compared with {comp.label}.")
    return " ".join(text)


# ----- user-facing table -----

DISPLAY_COLUMNS = {
    "month": "Month", "quarter": "Quarter", "week": "Week of", "weekday": "Weekday", "day_of_month": "Day of month",
    "category": "Category", "parent_group": "Group", "issuer": "Card", "merchant": "Merchant",
}
WEEKDAY_NAMES = dict(zip(WEEKDAYS, ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")))
UNIT_SUFFIX = {"usd": " ($)", "pct": " (%)", "count": ""}


def _display_label(dim: str, value: Any) -> Any:
    text = str(value)
    try:
        if dim == "month":
            return pd.Period(text, freq="M").strftime("%b %Y")
        if dim == "quarter":
            year, quarter = text.split("-")
            return f"{quarter} {year}"
        if dim == "week":
            day = pd.Timestamp(text)
            return f"{day:%b} {day.day}, {day.year}"
    except (ValueError, TypeError):
        return text
    if dim == "weekday":
        return WEEKDAY_NAMES.get(text, text)
    return value


def display_table(result: AnalysisResult) -> pd.DataFrame:
    """The result table with friendly column names and units ("Month", "Net spend ($)", "Transactions"),
    months as "Aug 2026" and full weekday names. ``result.table`` (what the summary is built from)
    is not changed."""
    table = result.table.copy()
    metric = result.spec.metric
    base_unit = _unit(metric)
    names: dict[str, str] = {}
    for dim in result.dimensions:
        table[dim] = table[dim].map(lambda value, dim=dim: _display_label(dim, value))
        names[dim] = DISPLAY_COLUMNS[dim]
    for column in table.columns:
        if column in names:
            continue
        if column == metric:
            names[column] = METRIC_LABELS[metric] + UNIT_SUFFIX[base_unit]
        elif column.startswith(metric + "_") and column[len(metric) + 1:] in TRANSFORM_LABELS:
            transform = column[len(metric) + 1:]
            unit = "pct" if transform == "pct_change" else base_unit
            names[column] = f"{METRIC_LABELS[metric]}, {TRANSFORM_LABELS[transform]}{UNIT_SUFFIX[unit]}"
        elif column == "comparison":
            label = result.comparison.label if result.comparison else "Comparison"
            names[column] = f"{label[:1].upper()}{label[1:]}{UNIT_SUFFIX[base_unit]}"
        elif column == "change":
            names[column] = "Change (pts)" if base_unit == "pct" else f"Change{UNIT_SUFFIX[base_unit]}"
        elif column == "change_pct":
            names[column] = "Change (%)"
    return table.rename(columns=names)


# ----- narrative guard -----

_DOLLAR = re.compile(r"(?<![\w.])(-?)\$\s?(-?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?([kKmM])?(?![\w])")
_PERCENT = re.compile(r"(?<![\w.$])(-?)(\d+(?:\.\d+)?)\s?%")


def _numbers(data: Any) -> list[float]:
    found: list[float] = []
    if isinstance(data, bool):
        return found
    if isinstance(data, (int, float)):
        if not (isinstance(data, float) and math.isnan(data)):
            found.append(abs(float(data)))
    elif isinstance(data, Mapping):
        for value in data.values():
            found.extend(_numbers(value))
    elif isinstance(data, (list, tuple)):
        for value in data:
            found.extend(_numbers(value))
    return found


def check_narrative_numbers(text: str, summary: Mapping[str, Any]) -> list[str]:
    """Flag dollar amounts and percentages in model text that match no summary value within rounding.

    The tolerance follows the precision written: "$1,235" may round 1234.56, "$1.2k" may
    round 1,234, "12%" may round 12.4. Signs are ignored ("down $50" vs change -50).
    """
    known = _numbers(summary)
    problems: list[str] = []

    def ok(value: float, tolerance: float) -> bool:
        return any(abs(value - number) <= tolerance + 1e-9 for number in known)

    for match in _DOLLAR.finditer(text or ""):
        whole, fraction, suffix = match.group(3).replace(",", ""), match.group(4) or "", (match.group(5) or "").lower()
        decimals = len(fraction) - 1 if fraction else 0
        scale = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
        value = float(whole + fraction) * scale
        tolerance = max(0.5 * 10 ** -decimals * scale, 0.01)
        if not ok(value, tolerance):
            problems.append(f"{match.group(0).strip()} does not match any value in the analysis summary.")
    for match in _PERCENT.finditer(text or ""):
        number = match.group(2)
        decimals = len(number.split(".")[1]) if "." in number else 0
        if not ok(float(number), 0.5 * 10 ** -decimals):
            problems.append(f"{match.group(0).strip()} does not match any value in the analysis summary.")
    return problems
