"""Bounded author/validator harness for the weekly financial report.

Modules: trace (RunTrace, limits, tool specs), html_safety (HTML gate), protocol (markers and
reply parsing), prompts (instruction text), feedback (revision feedback and cycle bookkeeping),
errors (stop conditions), transport (legacy query_ollama and the query-callable adapter; the model
clients live in finance.model_clients), fallback (deterministic report), and
harness (ReportHarness).
"""

from .harness import ReportHarness
from .trace import HarnessLimits, RunTrace, ToolSpec

__all__ = ["HarnessLimits", "ReportHarness", "RunTrace", "ToolSpec"]
