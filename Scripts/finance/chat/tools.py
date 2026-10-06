"""Model-callable tools for the finance chat and their dispatcher.

Three tools, defined provider-neutrally as ``{name, description, parameters}`` (JSON schema):

* ``analyze_spending``: the AnalysisSpec schema; validated by ``validate_and_repair`` and run by
  ``executor.run_analysis``.
* ``list_values``: valid names for category, parent_group, issuer, or merchant, with a search.
* ``get_month_snapshot``: ``report_metrics.calculate_report_snapshot`` for one month and its
  predecessor, built from the cached transaction frame (no report harness, no BigQuery call).

What goes back to the model is a compact JSON payload (at most ``MAX_RESULT_BYTES``), never rows.
Errors go back as tool results listing the closest valid values, so the model can retry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from typing import Any, Mapping

import pandas as pd
from pydantic import ValidationError

from ..report_metrics import calculate_report_snapshot
from ..report_models import ReportSnapshot
from ..settings import ReportSettings
from .analysis_spec import AnalysisSpec, SpecError, _closest, model_json_schema, validate_and_repair
from .data import analysis_rows
from .executor import AnalysisContext, AnalysisResult, run_analysis
from .merchants import clean_label
from .prompts import (
    LIST_VALUES_FIELD_DESCRIPTION,
    LIST_VALUES_SEARCH_DESCRIPTION,
    MONTH_SNAPSHOT_MONTH_DESCRIPTION,
    TOOL_DESCRIPTIONS,
)

MAX_RESULT_BYTES = 1536
LIST_LIMIT = 25
SNAPSHOT_CATEGORIES = 12
LIST_FIELDS = ("category", "parent_group", "issuer", "merchant")
TOOL_NAMES = ("analyze_spending", "list_values", "get_month_snapshot")
_MONTH = re.compile(r"^\d{4}-\d{2}$")


# ----- definitions -----


def _inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs and drop pydantic titles, so small local models get one flat schema."""
    defs = schema.get("$defs") or {}

    def walk(node: Any, names: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if names:  # a "properties" mapping: keys are field names, not schema keywords
            return {key: walk(value) for key, value in node.items()}
        if "$ref" in node:
            target = walk(defs[str(node["$ref"]).rsplit("/", 1)[-1]])
            extra = {key: walk(value) for key, value in node.items() if key not in ("$ref", "title")}
            return {**target, **extra}
        return {key: walk(value, names=key == "properties") for key, value in node.items() if key not in ("$defs", "title")}

    return walk(schema)


def analysis_parameters() -> dict[str, Any]:
    schema = _inline_schema(model_json_schema())
    schema.get("properties", {}).pop("dataset", None)  # the dataset is fixed by the app
    return schema


def tool_definitions() -> list[dict[str, Any]]:
    """Provider-neutral tool definitions (the model clients translate them per provider)."""
    return [
        {"name": "analyze_spending", "description": TOOL_DESCRIPTIONS["analyze_spending"], "parameters": analysis_parameters()},
        {
            "name": "list_values",
            "description": TOOL_DESCRIPTIONS["list_values"],
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": list(LIST_FIELDS), "description": LIST_VALUES_FIELD_DESCRIPTION},
                    "search": {"type": "string", "description": LIST_VALUES_SEARCH_DESCRIPTION},
                },
                "required": ["field"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_month_snapshot",
            "description": TOOL_DESCRIPTIONS["get_month_snapshot"],
            "parameters": {
                "type": "object",
                "properties": {"month": {"type": "string", "pattern": r"^\d{4}-\d{2}$", "description": MONTH_SNAPSHOT_MONTH_DESCRIPTION}},
                "additionalProperties": False,
            },
        },
    ]


# ----- context and outcome -----


@dataclass
class ToolContext:
    """Everything a tool needs; the frame stays on the server side and is never sent to the model."""

    frame: pd.DataFrame
    catalog: Mapping[str, Any]
    settings: ReportSettings
    today: date
    focus_month: date | None = None
    profile: Any = None  # profile.DataProfile; drives dimension repairs and the auto chart

    @cached_property
    def rows(self) -> pd.DataFrame:
        return analysis_rows(self.frame)

    @cached_property
    def merchants(self) -> list[str]:
        """Every merchant, largest net spend first (the catalog lists only the top ones)."""
        return self.rows.groupby("merchant")["spend"].sum().sort_values(ascending=False, kind="stable").index.tolist()

    @cached_property
    def months(self) -> list[str]:
        return sorted({str(period) for period in self.rows["month"].unique()})

    def data_bounds(self) -> tuple[date | None, date | None]:
        dates = self.catalog.get("date_range") or {}
        parse = lambda text: date.fromisoformat(text) if text else None  # noqa: E731
        return parse(dates.get("min")), parse(dates.get("max"))


@dataclass
class ToolOutcome:
    name: str
    arguments: dict[str, Any]
    ok: bool
    payload: dict[str, Any]
    result: AnalysisResult | None = None
    spec: AnalysisSpec | None = None
    notes: list[str] = field(default_factory=list)
    snapshot: ReportSnapshot | None = None

    def content(self) -> str:
        """The tool-result text sent to the model."""
        return _dumps(self.payload)


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))


def _size(data: Any) -> int:
    return len(_dumps(data).encode("utf-8"))


def compact(payload: dict[str, Any], limit: int = MAX_RESULT_BYTES) -> dict[str, Any]:
    """Trim optional detail until the payload fits ``limit`` bytes of JSON."""
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}

    def shorten(key: str, size: int) -> None:
        if isinstance(payload.get(key), str):
            payload[key] = payload[key][:size]

    steps = [
        lambda: payload.pop("notes", None),
        lambda: summary.pop("notes", None),
        lambda: summary.pop("filters", None),
        lambda: summary.pop("lowest", None),
        lambda: shorten("caption", 160),
        lambda: summary.get("top") and summary.__setitem__("top", summary["top"][:3]),
        lambda: payload.get("categories") and payload.__setitem__("categories", payload["categories"][:6]),
        lambda: payload.get("values") and payload.__setitem__("values", payload["values"][:10]),
        lambda: summary.pop("largest_change", None),
        lambda: summary.pop("top", None),
        lambda: shorten("caption", 80),
        lambda: shorten("error", 400),
        lambda: summary.get("limitations") and summary.__setitem__("limitations", summary["limitations"][:1]),
    ]
    for step in steps:
        if _size(payload) <= limit:
            break
        step()
    return payload


def _error(name: str, arguments: dict[str, Any], message: str) -> ToolOutcome:
    payload = {"ok": False, "error": message[:700], "hint": "Fix the arguments using the valid values listed and call the tool again."}
    return ToolOutcome(name, arguments, False, compact(payload))


# ----- analyze_spending -----


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:4]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "spec"
        message = str(error.get("msg", "invalid value"))
        if error.get("type") == "extra_forbidden":
            fields = sorted(AnalysisSpec.model_fields) if "." not in location else sorted(AnalysisSpec.model_fields["filters"].annotation.model_fields)
            message += f"; valid fields: {', '.join(fields)}"
        parts.append(f"{location}: {message}")
    return "Invalid analysis spec. " + " | ".join(parts)


def analyze_spending(arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
    args = {key: value for key, value in arguments.items() if key != "dataset"}
    if "__invalid_arguments__" in args:
        return _error("analyze_spending", arguments, "The arguments were not a valid JSON object matching the analysis spec.")
    try:
        spec, notes = validate_and_repair(
            args, ctx.catalog, known_merchants=ctx.merchants, today=ctx.today, focus_month=ctx.focus_month,
            profile=ctx.profile,
        )
    except ValidationError as exc:
        return _error("analyze_spending", arguments, _validation_message(exc))
    except SpecError as exc:
        return _error("analyze_spending", arguments, str(exc))
    data_min, data_max = ctx.data_bounds()
    result = run_analysis(
        ctx.frame, spec,
        AnalysisContext(today=ctx.today, focus_month=ctx.focus_month, data_min=data_min, data_max=data_max, notes=notes,
                        profile=ctx.profile),
    )
    payload = compact({"ok": True, "caption": result.caption, "chart": result.chart, "summary": dict(result.summary)})
    return ToolOutcome("analyze_spending", arguments, True, payload, result=result, spec=spec, notes=list(result.notes))


# ----- list_values -----


def _values_for(field_name: str, ctx: ToolContext) -> list[str]:
    if field_name == "merchant":
        return list(ctx.merchants)
    if field_name == "category":
        return list((ctx.catalog.get("categories") or {}).keys())
    if field_name == "parent_group":
        return list(ctx.catalog.get("parent_groups") or [])
    return list(ctx.catalog.get("issuers") or [])


def list_values(arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
    field_name = str(arguments.get("field") or "").strip().lower()
    if field_name not in LIST_FIELDS:
        return _error("list_values", arguments, f"Unknown field {field_name!r}. Valid fields: {', '.join(LIST_FIELDS)}.")
    values = _values_for(field_name, ctx)
    search = clean_label(arguments.get("search") or "", 60)
    matches = [value for value in values if search.lower() in value.lower()] if search else values
    payload: dict[str, Any] = {
        "ok": True, "field": field_name, "values": matches[:LIST_LIMIT], "total_matches": len(matches),
    }
    if search:
        payload["search"] = search
    if field_name == "category":
        categories = ctx.catalog.get("categories") or {}
        payload["groups"] = {name: categories.get(name, []) for name in payload["values"]}
    if not matches and search:
        payload["closest"] = _closest(search, values)
    return ToolOutcome("list_values", arguments, True, compact(payload))


# ----- get_month_snapshot -----


def _month_start(text: str) -> date:
    year, month = (int(part) for part in text.split("-"))
    return date(year, month, 1)


def monthly_rows(frame: pd.DataFrame, month: date) -> list[dict[str, Any]]:
    """gold.fact_monthly_spending-shaped rows (net_real_expense per month, category, group) for
    ``month`` and the month before it, built from the transaction frame."""
    rows = analysis_rows(frame)
    current = pd.Period(month, freq="M")
    selected = rows[rows["month"].isin([current, current - 1])]
    grouped = selected.groupby(["month", "category", "parent_group"], sort=True)["spend"].sum()
    return [
        {
            "transaction_month": period.to_timestamp().date(),
            "category": category,
            "parent_group": group,
            "net_real_expense": round(float(value), 2),
        }
        for (period, category, group), value in grouped.items()
    ]


def snapshot_for_month(ctx: ToolContext, month: date) -> ReportSnapshot:
    settings = ctx.settings
    goals = {name: float(value) for name, value in (settings.category_goals or {}).items() if value is not None}
    return calculate_report_snapshot(
        monthly_rows(ctx.frame, month),
        monthly_deposit=settings.monthly_budget or 0.0,
        fixed_costs=settings.fixed_costs or 0.0,
        savings_goal=settings.savings_goal_target,
        category_goals=goals,
        as_of=ctx.today,
    )


def _snapshot_payload(snapshot: ReportSnapshot, settings: ReportSettings) -> dict[str, Any]:
    categories = sorted(snapshot.categories, key=lambda item: (-item.current_spend, item.category))
    payload: dict[str, Any] = {
        "ok": True,
        "month": snapshot.latest_month[:7],
        "previous_month": snapshot.previous_month[:7] if snapshot.previous_month else None,
        "total_net_spend": snapshot.latest_total_net_expense,
        "previous_total_net_spend": snapshot.previous_total_net_expense,
        "mom_change_pct": snapshot.mom_change_percent,
        "top_discretionary": list(snapshot.top_discretionary_categories),
        "categories_columns": ["category", "group", "current", "previous", "mom_change_pct"],
        "categories": [
            [item.category, item.parent_group, item.current_spend, item.previous_spend, item.mom_change_percent]
            for item in categories[:SNAPSHOT_CATEGORIES]
        ],
    }
    if settings.monthly_budget is not None and settings.fixed_costs is not None:
        payload["budget"] = {
            "monthly_deposit": snapshot.monthly_deposit,
            "fixed_costs": snapshot.fixed_costs,
            "disposable_cash": snapshot.disposable_cash,
            "savings_goal": snapshot.savings_goal,
            "savings_shortfall": snapshot.savings_shortfall,
        }
    else:
        payload["budget"] = "Monthly_budget/Fixed_costs not configured; disposable cash is unavailable."
    return payload


def get_month_snapshot(arguments: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
    text = str(arguments.get("month") or "").strip()[:7]
    if not ctx.months:
        return _error("get_month_snapshot", arguments, "There is no spending data yet.")
    if not text:
        text = f"{ctx.focus_month:%Y-%m}" if ctx.focus_month else ctx.months[-1]
    if not _MONTH.match(text):
        return _error("get_month_snapshot", arguments, f"month must be YYYY-MM, got {text!r}.")
    if text not in ctx.months:
        target = _month_start(text) if 1 <= int(text[5:7]) <= 12 else None
        nearest = sorted(ctx.months, key=lambda m: abs((_month_start(m) - target).days) if target else 0)[:5]
        return _error("get_month_snapshot", arguments,
                      f"No spending data for {text}. Closest months with data: {', '.join(sorted(nearest))}.")
    snapshot = snapshot_for_month(ctx, _month_start(text))
    payload = compact(_snapshot_payload(snapshot, ctx.settings))
    return ToolOutcome("get_month_snapshot", arguments, True, payload, snapshot=snapshot)


# ----- dispatch -----

_TOOLS = {"analyze_spending": analyze_spending, "list_values": list_values, "get_month_snapshot": get_month_snapshot}


def dispatch(name: str, arguments: Mapping[str, Any] | None, ctx: ToolContext) -> ToolOutcome:
    """Run one tool call; unknown tools and bad arguments become error outcomes (never raise)."""
    args = dict(arguments or {})
    handler = _TOOLS.get(name)
    if handler is None:
        return _error(str(name), args, f"Unknown tool {name!r}. Available tools: {', '.join(TOOL_NAMES)}.")
    return handler(args, ctx)
