"""Hard dollar backstop for Claude (Anthropic) spend: an append-only cost ledger and a budget guard.

Every Anthropic call goes through ``model_clients.AnthropicClient``, which asks its ``BudgetGuard``
before each request (including retries and format retries) and records every billed call in
the ledger afterwards. The guard refuses a call, without contacting the API, when the call's
worst case (prompt at the list input price plus max_tokens output) would push today's spend
past ``anthropic_daily_budget_usd`` or this run's spend past ``anthropic_run_budget_usd``.

The ledger (``anthropic_cost_ledger_path``, default ``agent/runs/cost_ledger.jsonl``) has one
JSON line per Anthropic call. A missing ledger means nothing has been spent. The guard fails
closed: if the ledger cannot be read, or a spend record cannot be written, it refuses further
calls (``LedgerUnavailableError``, a ``BudgetExceededError``). Appends are single writes under
a lock file, so the Gradio app and the scheduled email run can share the ledger.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .pricing import cost_usd, usage_tokens, worst_case_call_usd

# Ledger run ids for calls that are not part of a report run. For these the run cap applies
# to the spend of one guard (one replay, one comparison, one chat session) instead.
GENERIC_RUN_IDS = frozenset({"replay", "compare", "chat", "other"})
LOCK_TIMEOUT_SECONDS = 10.0
STALE_LOCK_SECONDS = 30.0
# Trace events whose cost_usd counts toward a run's Claude cost.
COST_EVENTS = frozenset({"model_response", "model_call_failed"})


class BudgetExceededError(RuntimeError):
    """A Claude call was refused before it was made because it could exceed a spend cap.

    details holds the spend figures (spent_today_usd, daily_budget_usd, spent_run_usd,
    run_budget_usd, worst_case_usd, cap, run_id) for trace events and messages.
    """

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


class LedgerUnavailableError(BudgetExceededError):
    """The cost ledger could not be read or written, so Claude calls are refused (fail closed)."""


# Ledger paths whose last write failed in this process; every later guard check refuses.
_FAILED_LEDGERS: dict[str, str] = {}
_WARNED: set[tuple[Any, ...]] = set()


def _local_today() -> date:
    return datetime.now().date()


class CostLedger:
    """Append-only JSONL record of Anthropic calls and their cost."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _key(self) -> str:
        return str(self.path.resolve())

    def entries(self) -> list[dict[str, Any]]:
        """All ledger records; [] when the file does not exist yet.

        Raises LedgerUnavailableError when the file cannot be read, a complete line is not
        valid JSON, or an earlier write in this process failed. A final line without a newline
        is a write still in progress and is ignored if it does not parse.
        """
        failed = _FAILED_LEDGERS.get(self._key())
        if failed:
            raise LedgerUnavailableError(f"Claude cost ledger {self.path} is unavailable after a failed write ({failed}); refusing Claude calls")
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise LedgerUnavailableError(f"Claude cost ledger {self.path} could not be read ({exc}); refusing Claude calls") from exc
        lines = data.split(b"\n")
        partial = lines.pop() if lines else b""
        records: list[dict[str, Any]] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerUnavailableError(
                    f"Claude cost ledger {self.path} line {number} is not valid JSON ({exc}); fix or move the file. Refusing Claude calls"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
        if partial.strip():
            try:
                record = json.loads(partial.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                record = None
            if isinstance(record, dict):
                records.append(record)
        return records

    def spend_today(self, today: date | None = None) -> float:
        """USD recorded for today's local date."""
        day = (today or _local_today()).isoformat()
        return round(sum(_cost(item) for item in self.entries() if item.get("date") == day), 6)

    def spend_for_run(self, run_id: str) -> float:
        """USD recorded for one run id."""
        return round(sum(_cost(item) for item in self.entries() if item.get("run_id") == run_id), 6)

    def spend_by_run(self) -> dict[str, float]:
        """USD per run id from one read of the ledger (the run index uses it for old runs)."""
        totals: dict[str, float] = {}
        for item in self.entries():
            run_id = item.get("run_id")
            if run_id:
                totals[str(run_id)] = totals.get(str(run_id), 0.0) + _cost(item)
        return {run_id: round(total, 6) for run_id, total in totals.items()}

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one record as a single write under the lock file; failures make the ledger unavailable."""
        line = (json.dumps(dict(record), default=str) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _FileLock(self.lock_path):
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0))
                try:
                    size = os.fstat(fd).st_size
                    if size:
                        # Never glue a record onto a line left without its newline by a crashed writer.
                        with self.path.open("rb") as handle:
                            handle.seek(size - 1)
                            if handle.read(1) != b"\n":
                                line = b"\n" + line
                    written = 0
                    while written < len(line):
                        written += os.write(fd, line[written:])
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except (OSError, TimeoutError) as exc:
            _FAILED_LEDGERS[self._key()] = f"{type(exc).__name__}: {exc}"
            raise LedgerUnavailableError(
                f"Could not record Claude spend in {self.path} ({exc}); refusing further Claude calls"
            ) from exc


class _FileLock:
    """A lock file created exclusively; stale locks (older than STALE_LOCK_SECONDS) are removed."""

    def __init__(self, path: Path, timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        self.path = path
        self.timeout = timeout

    def __enter__(self) -> "_FileLock":
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))  # informational only
                except OSError:
                    pass
                finally:
                    os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > STALE_LOCK_SECONDS:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"cost ledger lock {self.path} is held by another process")
                time.sleep(0.05)
            except PermissionError:
                # Windows reports a lock file being deleted by another process this way.
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def __exit__(self, *exc_info: Any) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _cost(item: Mapping[str, Any]) -> float:
    try:
        return float(item.get("cost_usd") or 0)
    except (TypeError, ValueError):
        return 0.0


def _money(value: float | None) -> str:
    return "off" if value is None else f"${value:,.2f}"


class BudgetGuard:
    """Checks each Claude call against the daily and run caps and records its spend.

    daily_budget_usd / run_budget_usd of None disable that cap. run_id names the report run in
    the ledger ("replay", "compare", "chat", or "other" outside report runs; for those the run
    cap counts this guard's own spend).
    """

    def __init__(
        self,
        ledger: CostLedger,
        daily_budget_usd: float | None,
        run_budget_usd: float | None,
        run_id: str = "other",
        *,
        today: Callable[[], date] = _local_today,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.ledger = ledger
        self.daily_budget_usd = daily_budget_usd
        self.run_budget_usd = run_budget_usd
        self.run_id = str(run_id or "other")
        self.today = today
        self.now = now
        self.session_spend_usd = 0.0

    def spent_for_run(self) -> float:
        if self.run_id in GENERIC_RUN_IDS:
            return round(self.session_spend_usd, 6)
        return self.ledger.spend_for_run(self.run_id)

    def status(self) -> dict[str, Any]:
        """Spend figures for the run-start log line (reads the ledger; may raise LedgerUnavailableError)."""
        return {
            "spent_today_usd": self.ledger.spend_today(self.today()),
            "daily_budget_usd": self.daily_budget_usd,
            "spent_run_usd": self.spent_for_run(),
            "run_budget_usd": self.run_budget_usd,
            "run_id": self.run_id,
        }

    def check(self, model: str, prompt_chars: int, max_tokens: int, role: str | None = None) -> dict[str, Any]:
        """Raise BudgetExceededError when this call's worst case could pass a cap; else return the figures."""
        worst = worst_case_call_usd(model, prompt_chars, max_tokens)
        details = {**self.status(), "worst_case_usd": worst}
        what = f"{role + ' ' if role else ''}call to {model}"
        daily, spent_today = self.daily_budget_usd, details["spent_today_usd"]
        if daily is not None and spent_today + worst > daily:
            raise BudgetExceededError(
                f"Claude daily budget reached: ${spent_today:.2f} of ${daily:.2f} spent today; the next {what} "
                f"could cost up to ${worst:.2f}, so it was not made",
                {**details, "cap": "daily"},
            )
        run_cap, spent_run = self.run_budget_usd, details["spent_run_usd"]
        if run_cap is not None and spent_run + worst > run_cap:
            raise BudgetExceededError(
                f"Claude run budget reached: ${spent_run:.2f} of ${run_cap:.2f} spent in run {self.run_id}; the next {what} "
                f"could cost up to ${worst:.2f}, so it was not made",
                {**details, "cap": "run"},
            )
        return details

    def record(
        self,
        model: str,
        usage: Mapping[str, Any] | None,
        *,
        role: str | None = None,
        outcome: str = "ok",
        cost: float | None = None,
        estimated: bool = False,
    ) -> float:
        """Append one call to the ledger and return its cost (raises LedgerUnavailableError if it cannot)."""
        amount = cost_usd(model, usage) if cost is None else round(float(cost), 6)
        stamp = self.now()
        record: dict[str, Any] = {
            "timestamp": stamp.astimezone(timezone.utc).isoformat(),
            "date": self.today().isoformat(),
            "run_id": self.run_id,
            "role": role,
            "model": model,
            "outcome": outcome,
            "usage": usage_tokens(usage) if usage is not None else None,
            "cost_usd": amount,
        }
        if estimated:
            record["estimated"] = True
        self.session_spend_usd += amount
        self.ledger.append(record)
        return amount


def resolve_ledger_path(value: Any) -> Path:
    """A configured ledger path; relative paths are under the project root."""
    from .settings import DEFAULTS, PROJECT_ROOT

    path = Path(str(value or DEFAULTS["anthropic_cost_ledger_path"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_ledger_path(config: Any) -> Path | None:
    """The resolved ledger path for a config dict or ReportSettings; None if the budget settings are invalid."""
    from .settings import budget_settings

    try:
        return resolve_ledger_path(budget_settings(config)[2])
    except ValueError:
        return None


def guard_from_settings(config: Any, run_id: str | None = None) -> BudgetGuard:
    """The BudgetGuard for a config dict or ReportSettings (caps default ON: $15/day, $5/run)."""
    from .settings import budget_settings

    daily, per_run, ledger_path = budget_settings(config)
    warn_if_disabled(daily, per_run)
    return BudgetGuard(CostLedger(resolve_ledger_path(ledger_path)), daily, per_run, run_id or "other")


def warn_if_disabled(daily: float | None, per_run: float | None, echo: Callable[[str], None] = print) -> None:
    """Print a warning (once per process) when a Claude spend cap is explicitly disabled."""
    for name, value in (("anthropic_daily_budget_usd", daily), ("anthropic_run_budget_usd", per_run)):
        if value is None and (name,) not in _WARNED:
            _WARNED.add((name,))
            echo(f"WARNING: {name} is 0 or null, so that Claude spend cap is disabled.")


def status_line(status: Mapping[str, Any]) -> str:
    """"Claude budget: $X spent today of $Y daily; run cap $Z" for progress.log."""
    return (
        f"Claude budget: ${float(status.get('spent_today_usd') or 0):,.2f} spent today of "
        f"{_money(status.get('daily_budget_usd'))} daily; run cap {_money(status.get('run_budget_usd'))}"
    )


def anthropic_cost_from_events(events: Iterable[Mapping[str, Any]]) -> float | None:
    """Total Claude cost recorded in a run's trace events, or None when there were no Claude calls."""
    costs = [
        float(event["cost_usd"]) for event in events
        if event.get("event") in COST_EVENTS and isinstance(event.get("cost_usd"), (int, float))
    ]
    return round(sum(costs), 6) if costs else None
