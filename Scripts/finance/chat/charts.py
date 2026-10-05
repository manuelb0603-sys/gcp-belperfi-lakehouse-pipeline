"""Plotly figures for analysis results.

Colors: a validated colorblind-safe categorical order (fixed slots, never cycled; "Other"
is neutral gray), a rose accent for single series, a single-hue rose ramp for non-negative
heatmaps, and a blue/red diverging ramp with a gray midpoint at 0 otherwise.
Every label passes through ``html.escape`` because plotly renders some HTML in text.
"""

from __future__ import annotations

import hashlib
import html
from typing import Any, Sequence

import pandas as pd

from .analysis_spec import DIMENSION_LABELS, METRIC_LABELS, ORDERED_DIMENSIONS, TIME_DIMENSIONS
from .data import WEEKDAYS
from .executor import OTHER, TRANSFORM_LABELS, AnalysisResult

# Categorical slots (validated adjacent-pair CVD-safe order), in fixed order.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
OTHER_COLOR = "#898781"
ACCENT = "#d6336c"  # rose, for single-series charts
COMPARISON_COLOR = "#c3c2b7"
SEQUENTIAL = [[0.0, "#fdf0f4"], [0.25, "#f8c6d7"], [0.5, "#ee8db0"], [0.75, "#d6336c"], [1.0, "#8f1d48"]]
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec7d7c"], [1.0, "#a3262a"]]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"

# Canonical slot per known label, so a category keeps its color across charts.
CANONICAL = {
    name: index for index, name in enumerate((
        "Groceries", "Food & Drink", "Shopping", "Automotive", "Bills & Utilities", "Health & Wellness",
        "Travel", "Entertainment",
    ))
}
CANONICAL.update({"Needs": 0, "Wants": 1, "Payments & Others": 2, "Amex": 0, "Chase": 1})


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _slot(label: str) -> int:
    if label in CANONICAL:
        return CANONICAL[label] % len(PALETTE)
    return int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % len(PALETTE)


def color_map(labels: Sequence[Any]) -> dict[str, str]:
    """Stable colors: each label prefers its canonical/hashed slot; collisions take the next free slot."""
    colors: dict[str, str] = {}
    used: set[int] = set()
    for label in dict.fromkeys(str(label) for label in labels):
        if label == OTHER:
            colors[label] = OTHER_COLOR
            continue
        slot = _slot(label)
        for step in range(len(PALETTE)):
            candidate = (slot + step) % len(PALETTE)
            if candidate not in used:
                slot = candidate
                break
        used.add(slot)
        colors[label] = PALETTE[slot]
    return {esc(label): color for label, color in colors.items()}


def _axis_format(unit: str) -> dict[str, Any]:
    if unit == "usd":
        return {"tickprefix": "$", "tickformat": ",.0f"}
    if unit == "pct":
        return {"ticksuffix": "%", "tickformat": ",.1f"}
    return {"tickformat": ",d"}


def _hover_value(unit: str, field: str = "y") -> str:
    if unit == "usd":
        return f"$%{{{field}:,.2f}}"
    if unit == "pct":
        return f"%{{{field}:,.1f}}%"
    return f"%{{{field}:,d}}"


def measure_label(result: AnalysisResult) -> str:
    label = METRIC_LABELS[result.spec.metric]
    transform = result.value_column.removeprefix(result.spec.metric + "_")
    if transform in TRANSFORM_LABELS:
        label = f"{label} ({TRANSFORM_LABELS[transform]})"
    return label


def chart_title(result: AnalysisResult) -> str:
    if result.spec.title:
        return esc(result.spec.title)
    dims = " and ".join(DIMENSION_LABELS[d] for d in result.dimensions)
    by = f" by {dims}" if dims else ""
    return esc(f"{measure_label(result)}{by} · {result.period.label}")


def _escaped_table(result: AnalysisResult) -> pd.DataFrame:
    table = result.table.copy()
    for dim in result.dimensions:
        table[dim] = table[dim].map(esc) if dim != "day_of_month" else table[dim].astype(int)
    return table


def _layout(fig, result: AnalysisResult, x_title: str | None, y_title: str | None, value_axis: str | None = "y") -> None:
    fig.update_layout(
        title={"text": chart_title(result), "x": 0, "xanchor": "left", "font": {"size": 16, "color": INK}},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": MUTED, "size": 12},
        margin={"l": 60, "r": 24, "t": 64, "b": 48},
        legend={"title": {"text": ""}, "orientation": "h", "y": -0.18},
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(title_text=esc(x_title) if x_title else None, gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=esc(y_title) if y_title else None, gridcolor=GRID, linecolor=GRID, zeroline=False)
    if value_axis:
        (fig.update_yaxes if value_axis == "y" else fig.update_xaxes)(**_axis_format(result.unit))


def _value_title(result: AnalysisResult) -> str:
    unit = {"usd": " ($)", "pct": " (%)", "count": ""}[result.unit]
    return f"{measure_label(result)}{unit}"


def _series_and_x(result: AnalysisResult) -> tuple[str, str | None]:
    """x dimension (time/ordered first) and optional series (color) dimension."""
    dims = result.dimensions
    if len(dims) == 1:
        return dims[0], None
    ordered = [d for d in dims if d in TIME_DIMENSIONS] or [d for d in dims if d in ORDERED_DIMENSIONS]
    x = ordered[0] if ordered else dims[0]
    return x, next(d for d in dims if d != x)


def build_figure(result: AnalysisResult):
    """A plotly Figure for the result's chart type, or None for table/none (or no data)."""
    chart = result.chart
    if chart in ("table", "none") or not result.dimensions or result.table.empty:
        return None
    import plotly.express as px
    import plotly.graph_objects as go

    table = _escaped_table(result)
    value = result.value_column
    unit = result.unit

    if chart == "heatmap":
        return _heatmap(result, table, px)
    if chart == "pie":
        dim = result.dimensions[0]
        colors = color_map(result.table[dim].tolist())
        fig = go.Figure(go.Pie(
            labels=table[dim], values=table[value], sort=False, hole=0.45,
            marker={"colors": [colors.get(str(label), OTHER_COLOR) for label in table[dim]], "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{label}<br>" + _hover_value(unit, "value") + "<extra></extra>",
            textinfo="percent",
        ))
        _layout(fig, result, None, None, value_axis=None)
        return fig
    if chart == "treemap":
        dims = result.dimensions
        data = table[table[value].fillna(0) > 0]
        colors = color_map(result.table[dims[0]].tolist())
        fig = px.treemap(data, path=dims, values=value, color=dims[0], color_discrete_map=colors)
        fig.update_traces(hovertemplate="%{label}<br>" + _hover_value(unit, "value") + "<extra></extra>")
        _layout(fig, result, None, None, value_axis=None)
        return fig

    x, series = _series_and_x(result)
    if chart == "line":
        return _line(result, table, x, series, go)
    return _bar(result, table, x, series, chart, go)


def _category_order(result: AnalysisResult, dim: str, table: pd.DataFrame) -> list[Any]:
    if dim == "weekday":
        return list(WEEKDAYS)
    return list(dict.fromkeys(table[dim].tolist()))


def _line(result: AnalysisResult, table: pd.DataFrame, x: str, series: str | None, go) -> Any:
    fig = go.Figure()
    value = result.value_column
    table = table.sort_values(x, kind="stable")
    hover = "%{x}<br>" + _hover_value(result.unit) + "<extra>%{fullData.name}</extra>"
    if series is None:
        fig.add_trace(go.Scatter(x=table[x], y=table[value], mode="lines+markers", name=esc(measure_label(result)),
                                 line={"color": ACCENT, "width": 2}, marker={"size": 8}, hovertemplate=hover))
        if result.compare_column:
            fig.add_trace(go.Scatter(x=table[x], y=table[result.compare_column], mode="lines+markers",
                                     name=esc(result.comparison.label if result.comparison else "Comparison"),
                                     line={"color": COMPARISON_COLOR, "width": 2, "dash": "dot"}, marker={"size": 8},
                                     hovertemplate=hover))
    else:
        colors = color_map(result.table[series].tolist())
        for label in dict.fromkeys(table[series].tolist()):
            part = table[table[series] == label]
            fig.add_trace(go.Scatter(x=part[x], y=part[value], mode="lines+markers", name=label,
                                     line={"color": colors.get(str(label), OTHER_COLOR), "width": 2}, marker={"size": 8},
                                     hovertemplate=hover))
    _layout(fig, result, DIMENSION_LABELS[x].capitalize(), _value_title(result))
    fig.update_xaxes(type="category")
    fig.update_layout(showlegend=series is not None or bool(result.compare_column))
    return fig


def _bar(result: AnalysisResult, table: pd.DataFrame, x: str, series: str | None, chart: str, go) -> Any:
    fig = go.Figure()
    value = result.value_column
    horizontal = series is None and x not in ORDERED_DIMENSIONS
    order = _category_order(result, x, table)
    if series is None:
        traces = [(esc(measure_label(result)), value, ACCENT)]
        if result.compare_column:
            traces.append((esc(result.comparison.label if result.comparison else "Comparison"), result.compare_column, COMPARISON_COLOR))
        for name, column, color in traces:
            if horizontal:
                fig.add_trace(go.Bar(y=table[x], x=table[column], orientation="h", name=name, marker_color=color,
                                     hovertemplate="%{y}<br>" + _hover_value(result.unit, "x") + "<extra>%{fullData.name}</extra>"))
            else:
                fig.add_trace(go.Bar(x=table[x], y=table[column], name=name, marker_color=color,
                                     hovertemplate="%{x}<br>" + _hover_value(result.unit) + "<extra>%{fullData.name}</extra>"))
        fig.update_layout(barmode="group", showlegend=bool(result.compare_column), bargap=0.25)
    else:
        colors = color_map(result.table[series].tolist())
        for label in dict.fromkeys(table[series].tolist()):
            part = table[table[series] == label]
            fig.add_trace(go.Bar(x=part[x], y=part[value], name=label, marker_color=colors.get(str(label), OTHER_COLOR),
                                 marker_line={"color": SURFACE, "width": 1},
                                 hovertemplate="%{x}<br>" + _hover_value(result.unit) + "<extra>%{fullData.name}</extra>"))
        fig.update_layout(barmode="stack" if chart == "stacked_bar" else "group", showlegend=True, bargap=0.25)

    if horizontal:
        _layout(fig, result, _value_title(result), DIMENSION_LABELS[x].capitalize(), value_axis="x")
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=list(reversed(order)))
    else:
        _layout(fig, result, DIMENSION_LABELS[x].capitalize(), _value_title(result))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    return fig


def _heatmap(result: AnalysisResult, table: pd.DataFrame, px) -> Any:
    x, y = _series_and_x(result)
    if x not in TIME_DIMENSIONS and y in TIME_DIMENSIONS:
        x, y = y, x
    value = result.value_column
    pivot = table.pivot_table(index=y, columns=x, values=value, aggfunc="sum", dropna=False, sort=False)
    if y == "weekday":
        pivot = pivot.reindex([day for day in WEEKDAYS if day in pivot.index])
    else:
        pivot = pivot.reindex(_category_order(result, y, table))
    pivot = pivot.reindex(columns=_category_order(result, x, table))
    values = pivot.to_numpy(dtype=float)
    has_negative = bool((pd.DataFrame(values).fillna(0) < 0).to_numpy().any())
    kwargs: dict[str, Any] = {"color_continuous_scale": DIVERGING if has_negative else SEQUENTIAL}
    if has_negative:
        kwargs["color_continuous_midpoint"] = 0
    fig = px.imshow(
        values, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], aspect="auto",
        labels={"x": esc(DIMENSION_LABELS[x]), "y": esc(DIMENSION_LABELS[y]), "color": esc(measure_label(result))},
        **kwargs,
    )
    fig.update_traces(
        hovertemplate="%{y} · %{x}<br>" + _hover_value(result.unit, "z") + "<extra></extra>",
        xgap=2, ygap=2,
    )
    _layout(fig, result, DIMENSION_LABELS[x].capitalize(), DIMENSION_LABELS[y].capitalize(), value_axis=None)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    colorbar = {"title": {"text": esc(_value_title(result))}}
    colorbar.update({"tickprefix": "$"} if result.unit == "usd" else {"ticksuffix": "%"} if result.unit == "pct" else {})
    fig.update_layout(coloraxis_colorbar=colorbar)
    return fig
