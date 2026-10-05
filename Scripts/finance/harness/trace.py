"""Run trace, limits, and tool specs shared by the report harness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


ToolFunction = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ToolSpec:
    # A tool is a small, explicit, allowlisted action the model may request. The model
    # does not execute Python directly; it emits a structured tool call, and the harness
    # runs the bound function here.
    name: str
    description: str
    function: ToolFunction


@dataclass
class HarnessLimits:
    # These limits keep the agent bounded: it can investigate enough to produce a good
    # report, but it cannot loop forever or consume unbounded tokens or data.
    max_agent_rounds: int = 6
    max_tool_calls: int = 12
    max_calls_per_tool: int = 3
    # Total validator reviews (validation cycles) allowed, so at most this value - 1 revisions; values < 1 mean 1.
    max_validation_cycles: int = 3
    max_report_bytes: int = 250_000
    # Wall-clock budget for the whole run in seconds; None or <= 0 disables the deadline.
    max_run_seconds: float | None = 2700


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
                    print(f"Run trace listener failed; continuing without it: {exc}", flush=True)
