"""Shared project configuration: loading, validated ReportSettings, and money parsing.

Every config key the report code reads is turned into a typed field of ``ReportSettings``
once, by ``ReportSettings.from_config(config)``. Defaults for omitted keys live only in
``DEFAULTS`` below (README "Runtime config knobs" documents the same values; a test keeps
them in sync). Secrets stay plain fields but are masked in ``repr`` and never logged.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

# Code defaults for omitted config keys (the only place they are defined).
DEFAULTS: dict[str, Any] = {
    "report_mode": "legacy",
    # Report guidance: "strict" (exact headings/wording, minimal-edit revisions) or "guided"
    # (facts stay exact; wording, structure, and an optional alternative plan are freer).
    "guidance": "strict",
    "dry_run": False,
    "ollama_stream": False,
    "ollama_timeout": 600,
    "max_agent_rounds": 6,
    "max_tool_calls": 12,
    "max_calls_per_tool": 3,
    "max_validation_cycles": 3,
    "max_report_bytes": 250000,
    "max_run_seconds": 2700,
    "days_back": 60,
    "embedding_model_name": "nomic-embed-text",
    "memory_schema": "memory",
    "category_goals": {},
    "email_listings": [],
    "qa_report_context_chunks": 5,
    "qa_long_fact_limit": 5,
    "chat_data_cache_seconds": 900,
    "merchant_aliases": {},
    # Finance chat engine: recent turns sent verbatim, tool rounds per turn, seconds per turn.
    "chat_history_turns": 6,
    "chat_max_tool_rounds": 3,
    "chat_turn_seconds": 120,
    # Claude spend backstop (finance/budget.py): 0 or null disables a cap.
    "anthropic_daily_budget_usd": 15.0,
    "anthropic_run_budget_usd": 5.0,
    "anthropic_cost_ledger_path": "agent/runs/cost_ledger.jsonl",
    # snapshot.notable_changes thresholds: |MoM %| at least this, or |dollar change| at least this.
    "notable_change_pct": 50.0,
    "notable_change_usd": 250.0,
    # How the report author's BigQuery tools run: "mcp" (local stdio MCP server, finance/mcp_server.py)
    # or "direct" (in-process calls to finance/bigquery_tools.py).
    "tool_transport": "mcp",
}

REPORT_MODES = ("legacy", "agent")
GUIDANCE_MODES = ("strict", "guided")
TOOL_TRANSPORTS = ("mcp", "direct")
LIMIT_KEYS = ("max_agent_rounds", "max_tool_calls", "max_calls_per_tool", "max_validation_cycles", "max_report_bytes")
# Fallbacks used only by the finance app (Scripts/apps/finance_app.py) when the keys are missing.
APP_FALLBACK_MODEL_SERVER_URL = "http://localhost:11434/api/generate"
APP_FALLBACK_MODEL_NAME = "gemma4:e4b"
# Role fields that are safe to record in run metadata (no URLs or credentials).
PUBLIC_ROLE_KEYS = (
    "role", "provider", "model", "options", "think", "timeout", "max_tokens", "effort", "cache_system", "stream", "thinking",
    "guidance",
)

_SCHEMA_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class SettingsError(ValueError):
    """A config value failed validation; the message names the key."""


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load project settings from the requested path or the project config."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _validate_schema(memory_schema: Any) -> str:
    schema = str(memory_schema or DEFAULTS["memory_schema"])
    if not _SCHEMA_PATTERN.fullmatch(schema):
        raise SettingsError("memory_schema may contain only letters, numbers, and underscores")
    return schema


def memory_db_path(memory_schema: str | None = "memory") -> Path:
    """Resolve the SQLite memory database: MEMORY_DB_PATH, else agent/memory/<memory_schema>.sqlite3.

    The environment is read at call time; this is the only place memory paths are resolved.
    """
    if os.environ.get("MEMORY_DB_PATH"):
        return Path(os.environ["MEMORY_DB_PATH"])
    return PROJECT_ROOT / "agent" / "memory" / f"{_validate_schema(memory_schema)}.sqlite3"


def report_thread_key(config: "Mapping[str, Any] | ReportSettings") -> str:
    """Thread key for report history; matches the weekly email flow so all reports share one history."""
    settings = ReportSettings.coerce(config)
    return ", ".join(settings.email_listings) or settings.email_sender or "email:unknown"


def parse_money(value: Any) -> float:
    """Normalize numbers or amount strings (including k/m suffixes) to a float."""
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?\s*[kKmM]?", str(value or ""))
    if not matches:
        raise ValueError(f"Could not parse a financial amount from: {value!r}")
    total = 0.0
    for match in matches:
        normalized = match.replace(" ", "")
        multiplier = 1
        if normalized[-1:].lower() == "k":
            multiplier, normalized = 1_000, normalized[:-1]
        elif normalized[-1:].lower() == "m":
            multiplier, normalized = 1_000_000, normalized[:-1]
        total += float(normalized) * multiplier
    return total


# ----- validation helpers -----


def _get(config: Mapping[str, Any], key: str) -> Any:
    return copy.deepcopy(config[key]) if key in config else copy.deepcopy(DEFAULTS[key])


def _int(key: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise SettingsError(f"{key} must be an integer, got {value!r}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        value = int(value)
    if not isinstance(value, int):
        raise SettingsError(f"{key} must be an integer, got {value!r}")
    if value < minimum:
        raise SettingsError(f"{key} must be {'a positive integer' if minimum == 1 else f'at least {minimum}'}, got {value}")
    return value


def _positive_number(key: str, value: Any) -> float:
    """A number greater than 0 (ints, floats, or numeric strings)."""
    if isinstance(value, bool):
        raise SettingsError(f"{key} must be a positive number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{key} must be a positive number, got {value!r}") from None
    if number != number or number <= 0:
        raise SettingsError(f"{key} must be a positive number, got {value!r}")
    return number


def _tool_transport(value: Any) -> str:
    """The author tool transport ("mcp" or "direct", case-insensitive); anything else raises SettingsError."""
    if isinstance(value, str) and value.strip().lower() in TOOL_TRANSPORTS:
        return value.strip().lower()
    raise SettingsError(f"tool_transport must be one of {', '.join(TOOL_TRANSPORTS)}, got {value!r}")


def validate_guidance(key: str, value: Any) -> str:
    """A guidance mode ("strict" or "guided", case-insensitive); anything else raises SettingsError naming key."""
    if isinstance(value, str) and value.strip().lower() in GUIDANCE_MODES:
        return value.strip().lower()
    raise SettingsError(f"{key} must be one of {', '.join(GUIDANCE_MODES)}, got {value!r}")


def _optional_money(value: Any) -> float | None:
    """Parsed amount, or None when missing or unparseable (checked where it is required)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return parse_money(value)
    except ValueError:
        return None


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _budget_cap(key: str, value: Any) -> float | None:
    """A USD cap: a non-negative number; 0 or null disables it (returns None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise SettingsError(f"{key} must be a dollar amount, 0, or null, got {value!r}")
    try:
        amount = float(value)
    except (TypeError, ValueError):
        raise SettingsError(f"{key} must be a dollar amount, 0, or null, got {value!r}") from None
    if amount != amount or amount < 0:
        raise SettingsError(f"{key} must be 0 or more (0 or null disables it), got {value!r}")
    return amount if amount > 0 else None


def budget_settings(config: "Mapping[str, Any] | ReportSettings | None") -> tuple[float | None, float | None, str]:
    """(daily cap, run cap, ledger path) for a ReportSettings or a raw config mapping.

    Omitted keys use DEFAULTS (caps ON); only an explicit 0 or null disables a cap.
    """
    if isinstance(config, ReportSettings):
        return config.anthropic_daily_budget_usd, config.anthropic_run_budget_usd, config.anthropic_cost_ledger_path
    config = dict(config or {})
    return (
        _budget_cap("anthropic_daily_budget_usd", _get(config, "anthropic_daily_budget_usd")),
        _budget_cap("anthropic_run_budget_usd", _get(config, "anthropic_run_budget_usd")),
        str(_get(config, "anthropic_cost_ledger_path") or DEFAULTS["anthropic_cost_ledger_path"]),
    )


@dataclass(frozen=True)
class ReportSettings:
    """Validated, typed view of config.json for the report, email, app, Q&A, and compare tools."""

    # Workflow
    report_mode: str
    # Effective report guidance: model_roles.author.guidance, else top-level guidance (default "strict").
    guidance: str
    dry_run: bool
    project_name: str | None
    days_back: int
    # Models (resolved per role with model_clients.resolve_model_role)
    model_server_url: str | None
    model_name: str | None
    validator_model_name: str | None
    fact_model_name: str | None
    embedding_model_name: str
    ollama_stream: bool
    ollama_timeout: int
    ollama_options: dict[str, Any] | None
    model_roles: dict[str, Any] | None
    author_role: dict[str, Any]
    validator_role: dict[str, Any]
    eval_role: dict[str, Any]
    chat_role: dict[str, Any]
    # Harness limits
    max_agent_rounds: int
    max_tool_calls: int
    max_calls_per_tool: int
    max_validation_cycles: int
    max_report_bytes: int
    max_run_seconds: float | None
    # Memory
    memory_schema: str
    # Finance inputs (raw values are kept for the legacy prompt, which prints them as written)
    monthly_budget: float | None
    fixed_costs: float | None
    savings_goal_target: float | None
    monthly_budget_raw: Any
    fixed_costs_raw: Any
    savings_goal_target_raw: Any
    vehicle_info: Any
    category_goals: dict[str, Any]
    # Email
    email_sender: str | None
    email_listings: list[str]
    gmail_app_credentials: str | None
    # Finance app (Scripts/apps/finance_app.py)
    qa_report_context_chunks: int
    qa_long_fact_limit: int
    # Finance chat (Scripts/finance/chat/)
    chat_data_cache_seconds: int
    merchant_aliases: dict[str, str]
    chat_history_turns: int
    chat_max_tool_rounds: int
    chat_turn_seconds: int
    # Claude spend backstop (None = cap disabled); see finance/budget.py
    anthropic_daily_budget_usd: float | None
    anthropic_run_budget_usd: float | None
    anthropic_cost_ledger_path: str
    # snapshot.notable_changes thresholds (report_metrics.notable_changes)
    notable_change_pct: float = DEFAULTS["notable_change_pct"]
    notable_change_usd: float = DEFAULTS["notable_change_usd"]
    # Author BigQuery tool transport: "mcp" (stdio MCP server) or "direct" (in-process)
    tool_transport: str = DEFAULTS["tool_transport"]
    # The source mapping (for role re-resolution with caller overrides); never shown in repr.
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ----- construction -----

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ReportSettings":
        """Build and validate settings from a config mapping; raises SettingsError with the key name."""
        from .model_clients import resolve_model_role  # lazy: model_clients imports DEFAULTS from here

        config = dict(config or {})
        report_mode = str(_get(config, "report_mode") or DEFAULTS["report_mode"]).lower()
        if report_mode not in REPORT_MODES:
            raise SettingsError(f"report_mode must be one of {', '.join(REPORT_MODES)}, got {report_mode!r}")

        limits = {key: _int(key, _get(config, key), minimum=1) for key in LIMIT_KEYS}
        run_seconds = _get(config, "max_run_seconds")
        if run_seconds is not None:
            if isinstance(run_seconds, bool):
                raise SettingsError(f"max_run_seconds must be a number of seconds or null, got {run_seconds!r}")
            try:
                run_seconds = float(run_seconds)
            except (TypeError, ValueError):
                raise SettingsError(f"max_run_seconds must be a number of seconds or null, got {run_seconds!r}") from None
            if run_seconds < 0:
                raise SettingsError(f"max_run_seconds must be 0 or more (0 or null disables it), got {run_seconds:g}")
            run_seconds = run_seconds if run_seconds > 0 else None

        timeout = _get(config, "ollama_timeout")
        ollama_timeout = DEFAULTS["ollama_timeout"] if timeout is None else _int("ollama_timeout", timeout, minimum=1)

        savings_raw = config.get("Savings_goal_target")
        savings_goal = None
        if savings_raw is not None:
            if isinstance(savings_raw, bool):
                raise SettingsError(f"Savings_goal_target must be a number, got {savings_raw!r}")
            try:
                savings_goal = parse_money(savings_raw)
            except ValueError:
                raise SettingsError(f"Savings_goal_target must be a number, got {savings_raw!r}") from None

        category_goals = _get(config, "category_goals")
        if category_goals is None:
            category_goals = {}
        if not isinstance(category_goals, dict):
            raise SettingsError("category_goals must be an object mapping category names to amounts")

        merchant_aliases = _get(config, "merchant_aliases")
        if merchant_aliases is None:
            merchant_aliases = {}
        if not isinstance(merchant_aliases, dict):
            raise SettingsError("merchant_aliases must be an object mapping merchant names to display names")
        merchant_aliases = {str(key): str(value) for key, value in merchant_aliases.items()}

        listings = _get(config, "email_listings") or []
        if isinstance(listings, str) or not isinstance(listings, (list, tuple)):
            raise SettingsError("email_listings must be a list of email addresses")

        for key in ("model_roles", "ollama_options"):
            if config.get(key) is not None and not isinstance(config.get(key), dict):
                raise SettingsError(f"{key} must be an object")
        raw_guidance = config.get("guidance")
        guidance = validate_guidance("guidance", DEFAULTS["guidance"] if raw_guidance is None else raw_guidance)
        for role_name, entry in (config.get("model_roles") or {}).items():
            if role_name != "author" and isinstance(entry, dict) and "guidance" in entry:
                raise SettingsError(f"model_roles.{role_name}.guidance is not supported; set model_roles.author.guidance or top-level guidance")
        try:
            roles = {role: resolve_model_role(config, role) for role in ("author", "validator", "eval", "chat")}
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"model_roles: {exc}") from exc
        if "guidance" in roles["author"]:
            guidance = roles["author"]["guidance"]  # the author role's override wins

        return cls(
            report_mode=report_mode,
            guidance=guidance,
            dry_run=bool(_get(config, "dry_run")),
            project_name=_optional_text(config.get("project_name")),
            days_back=_int("days_back", _get(config, "days_back"), minimum=1),
            model_server_url=_optional_text(config.get("model_server_url")),
            model_name=_optional_text(config.get("model_name")),
            validator_model_name=_optional_text(config.get("validator_model_name")),
            fact_model_name=_optional_text(config.get("fact_model_name")),
            embedding_model_name=str(_get(config, "embedding_model_name") or DEFAULTS["embedding_model_name"]),
            ollama_stream=bool(_get(config, "ollama_stream")),
            ollama_timeout=ollama_timeout,
            ollama_options=copy.deepcopy(config.get("ollama_options")),
            model_roles=copy.deepcopy(config.get("model_roles")),
            author_role=roles["author"],
            validator_role=roles["validator"],
            eval_role=roles["eval"],
            chat_role=roles["chat"],
            max_run_seconds=run_seconds,
            memory_schema=_validate_schema(_get(config, "memory_schema")),
            monthly_budget=_optional_money(config.get("Monthly_budget")),
            fixed_costs=_optional_money(config.get("Fixed_costs")),
            savings_goal_target=savings_goal,
            monthly_budget_raw=config.get("Monthly_budget"),
            fixed_costs_raw=config.get("Fixed_costs"),
            savings_goal_target_raw=config.get("Savings_goal_target", "not configured"),
            vehicle_info=config.get("Vehicle_Info"),
            category_goals=category_goals,
            email_sender=_optional_text(config.get("email_sender")),
            email_listings=[str(item) for item in listings],
            gmail_app_credentials=_optional_text(config.get("Gmail_app_credentials")),
            qa_report_context_chunks=_int("qa_report_context_chunks", _get(config, "qa_report_context_chunks"), minimum=1),
            qa_long_fact_limit=_int("qa_long_fact_limit", _get(config, "qa_long_fact_limit"), minimum=1),
            chat_data_cache_seconds=_int("chat_data_cache_seconds", _get(config, "chat_data_cache_seconds"), minimum=0),
            merchant_aliases=merchant_aliases,
            chat_history_turns=_int("chat_history_turns", _get(config, "chat_history_turns"), minimum=1),
            chat_max_tool_rounds=_int("chat_max_tool_rounds", _get(config, "chat_max_tool_rounds"), minimum=1),
            chat_turn_seconds=_int("chat_turn_seconds", _get(config, "chat_turn_seconds"), minimum=1),
            **dict(zip(
                ("anthropic_daily_budget_usd", "anthropic_run_budget_usd", "anthropic_cost_ledger_path"),
                budget_settings(config),
            )),
            notable_change_pct=_positive_number("notable_change_pct", _get(config, "notable_change_pct")),
            notable_change_usd=_positive_number("notable_change_usd", _get(config, "notable_change_usd")),
            tool_transport=_tool_transport(_get(config, "tool_transport")),
            raw=config,
            **limits,
        )

    @classmethod
    def coerce(cls, value: "Mapping[str, Any] | ReportSettings | None") -> "ReportSettings":
        """Accept a ReportSettings or a config mapping (converted at the boundary)."""
        return value if isinstance(value, ReportSettings) else cls.from_config(value)

    def with_model_overrides(self, model_server_url: str | None = None, model_name: str | None = None) -> "ReportSettings":
        """Settings where a caller's Ollama URL and author model replace the legacy keys (roles re-resolved)."""
        overrides = {}
        if model_name:
            overrides["model_name"] = model_name
        if model_server_url:
            overrides["model_server_url"] = model_server_url
        if not overrides:
            return self
        return ReportSettings.from_config({**self.raw, **overrides})

    # ----- derived values -----

    @property
    def memory_db_path(self) -> Path:
        return memory_db_path(self.memory_schema)

    def role(self, role: str) -> dict[str, Any]:
        """Resolved settings for "author", "validator", "eval", or "chat" (a copy)."""
        resolved = {
            "author": self.author_role, "validator": self.validator_role, "eval": self.eval_role, "chat": self.chat_role,
        }.get(role)
        if resolved is None:
            raise ValueError(f"Unknown model role {role!r}")
        return copy.deepcopy(resolved)

    def public_role(self, role: str) -> dict[str, Any]:
        """Role settings safe for run metadata: provider, model, options, think, limits; no URLs or secrets."""
        resolved = self.role(role)
        return {key: resolved[key] for key in PUBLIC_ROLE_KEYS if key in resolved}

    def limits(self) -> dict[str, Any]:
        """Harness limits as a plain dict (HarnessLimits field names)."""
        return {
            "max_agent_rounds": self.max_agent_rounds,
            "max_tool_calls": self.max_tool_calls,
            "max_calls_per_tool": self.max_calls_per_tool,
            "max_validation_cycles": self.max_validation_cycles,
            "max_report_bytes": self.max_report_bytes,
            "max_run_seconds": self.max_run_seconds,
        }

    def __repr__(self) -> str:
        parts = []
        for item in fields(self):
            if item.name == "raw":
                continue
            value = getattr(self, item.name)
            if item.name == "gmail_app_credentials" and value:
                value = "***"
            parts.append(f"{item.name}={value!r}")
        return f"ReportSettings({', '.join(parts)})"

    __str__ = __repr__
