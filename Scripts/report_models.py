"""Typed, JSON-friendly models shared by the report harness."""

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class CategoryMetric:
    category: str
    parent_group: str
    current_spend: float
    previous_spend: float
    mom_change_percent: float | None
    goal_amount: float | None
    goal_percent: float | None


@dataclass(frozen=True)
class ReportSnapshot:
    as_of: str
    latest_month: str
    previous_month: str | None
    latest_total_net_expense: float
    previous_total_net_expense: float | None
    mom_change_percent: float | None
    monthly_deposit: float
    fixed_costs: float
    disposable_cash: float
    savings_goal: float | None
    categories: list[CategoryMetric] = field(default_factory=list)
    source_tables: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationFinding:
    severity: str
    category: str
    message: str
    evidence: str = ""
    revision_instruction: str = ""


@dataclass(frozen=True)
class ValidationDecision:
    decision: str
    findings: list[ValidationFinding] = field(default_factory=list)
    summary: str = ""

    @property
    def passed(self) -> bool:
        return self.decision.lower() == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def json_safe(value: Any) -> Any:
    """Convert common BigQuery/date values into JSON-safe values."""
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value