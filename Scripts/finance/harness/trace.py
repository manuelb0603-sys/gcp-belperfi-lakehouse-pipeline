"""Run trace, limits, and tool specs shared by the report harness."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from ..log import get_logger
from ..settings import DEFAULTS


ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]

logger = get_logger(__name__)


@dataclass
class ToolSpec:
    # A tool is a small, explicit, allowlisted action the model may request. The model
    # does not execute Python directly; it emits a structured tool call, and the harness
    # runs the bound function here.
    name: str
    description: str
    function: ToolFunction
    # Optional JSON schema (type "object") for the arguments. When set, the author's tool list
    # shows it and the harness validates arguments against it before running the tool.
    parameters: dict[str, Any] | None = None


_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def _schema_errors(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    """Errors for value against the JSON-schema subset the tools use (type, enum, bounds, pattern, properties)."""
    expected = schema.get("type")
    if expected:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_JSON_TYPES.get(name, lambda _: True)(value) for name in types):
            return [f"{path} must be {' or '.join(types)}, got {type(value).__name__} {value!r}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {', '.join(repr(item) for item in schema['enum'])}; got {value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be at least {schema['minimum']}; got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be at most {schema['maximum']}; got {value}")
    if isinstance(value, str):
        if "minLength" in schema and len(value.strip()) < schema["minLength"]:
            errors.append(f"{path} must be a non-empty string")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path} must match {schema['pattern']}; got {value!r}")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{key} is not a known argument" for key in value if key not in properties)
        for key, item in value.items():
            if key in properties:
                errors.extend(_schema_errors(properties[key], item, f"{path}.{key}"))
    return errors


def validate_tool_arguments(spec: ToolSpec, arguments: Any) -> list[str]:
    """Schema errors for one tool call's arguments ([] when valid or when the tool declares no schema)."""
    if not spec.parameters:
        return []
    return _schema_errors(spec.parameters, arguments, "arguments")


def tool_argument_summary(spec: ToolSpec) -> Any:
    """What the author's tool list shows for a tool's arguments: its schema, else the generic placeholder."""
    return spec.parameters if spec.parameters else "JSON object"


@dataclass
class HarnessLimits:
    # These limits keep the agent bounded: it can investigate enough to produce a good
    # report, but it cannot loop forever or consume unbounded tokens or data.
    max_agent_rounds: int = DEFAULTS["max_agent_rounds"]
    max_tool_calls: int = DEFAULTS["max_tool_calls"]
    max_calls_per_tool: int = DEFAULTS["max_calls_per_tool"]
    # Total validator reviews (validation cycles) allowed, so at most this value - 1 revisions; values < 1 mean 1.
    max_validation_cycles: int = DEFAULTS["max_validation_cycles"]
    max_report_bytes: int = DEFAULTS["max_report_bytes"]
    # Wall-clock budget for the whole run in seconds; None or <= 0 disables the deadline.
    max_run_seconds: float | None = DEFAULTS["max_run_seconds"]


# Guided guidance multiplies the tool budgets (rounds, cycles, and the run deadline are unchanged).
GUIDED_TOOL_LIMIT_FACTOR = 2


def effective_limits(limits: HarnessLimits, guidance: str | None = "strict") -> HarnessLimits:
    """The limits a run uses: the configured ones, with max_tool_calls and max_calls_per_tool doubled when guided."""
    if guidance != "guided":
        return limits
    return replace(
        limits,
        max_tool_calls=limits.max_tool_calls * GUIDED_TOOL_LIMIT_FACTOR,
        max_calls_per_tool=limits.max_calls_per_tool * GUIDED_TOOL_LIMIT_FACTOR,
    )


def limits_dict(limits: HarnessLimits) -> dict[str, Any]:
    """HarnessLimits as a plain dict (for trace events and run metadata)."""
    return asdict(limits)


@dataclass
class RunTrace:
    # Keep a full execution timeline so the run can be inspected later for debugging,
    # auditing, and artifact comparison across runs.
    events: list[dict[str, Any]] = field(default_factory=list)
    # Optional observers (for example a live progress recorder) notified of each event.
    # Listener failures are reported once and never interrupt the run.
    listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list, repr=False, compare=False)
    _listener_failed: bool = field(default=False, repr=False, compare=False)

    def add(self, event: str, **values: Any) -> None:
        """Record a single event in the agent timeline and notify any listeners."""
        item = {"event": event, "timestamp": time.time(), **values}
        self.events.append(item)
        for listener in self.listeners:
            try:
                listener(item)
            except Exception as exc:
                if not self._listener_failed:
                    self._listener_failed = True
                    logger.warning(f"Run trace listener failed; continuing without it: {exc}")
