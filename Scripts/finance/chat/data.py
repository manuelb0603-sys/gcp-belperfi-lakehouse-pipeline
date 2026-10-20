"""Transaction data for the finance chat: one parameterized gold query, a TTL cache, and a catalog.

The DataFrame is built from row dicts (db-dtypes/pyarrow are not installed): NUMERIC
Decimals become floats and dates become datetime64. Gold convention: purchases are
negative, so ``spend = -amount``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from ..settings import ReportSettings
from .analysis_spec import DIMENSIONS, METRIC_DEFINITIONS, PERIODS, shift_months
from .merchants import normalize_merchant

COLUMNS = ("issuer", "transaction_date", "description", "amount", "transaction_type", "category", "parent_group")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEKDAY_DTYPE = pd.CategoricalDtype(list(WEEKDAYS), ordered=True)
ANALYSIS_TYPES = ("Sale", "Return", "Adjustment")  # mirrors gold.fact_monthly_spending
DEFAULT_MONTHS_BACK = 24
CATALOG_TOP_MERCHANTS = 30

QUERY = """
    SELECT issuer, transaction_date, description, amount, transaction_type, category, parent_group
    FROM `{project}.gold.fact_transactions`
    WHERE transaction_date >= @start
    ORDER BY transaction_date
"""

# Monotonic clock; tests replace it to exercise the TTL.
clock: Callable[[], float] = time.monotonic
_CACHE: dict[tuple[Any, ...], tuple[float, pd.DataFrame]] = {}
# Catalog and data profile per cached frame (cache key, load time, today): built once per load.
_DERIVED: dict[tuple[Any, ...], tuple[dict[str, Any], Any]] = {}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class ChatData:
    """The frame plus what is derived from it once per load: the catalog and the data profile."""

    frame: pd.DataFrame
    catalog: dict[str, Any]
    profile: Any  # profile.DataProfile, or None if profiling failed


def default_start(today: date | None = None, months_back: int = DEFAULT_MONTHS_BACK) -> date:
    """First day of the month ``months_back`` months before today's month."""
    today = today or date.today()
    return shift_months(today.replace(day=1), -months_back)


def _bigquery_client(project: str):
    from google.cloud import bigquery

    return bigquery.Client(project=project)


def _job_config(start: date):
    try:
        from google.cloud import bigquery
    except ImportError:  # only fakes reach here without the library
        return {"query_parameters": [("start", "DATE", start)]}
    return bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("start", "DATE", start)])


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def frame_from_rows(rows: Iterable[Mapping[str, Any] | Any], aliases: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Build the analysis frame from BigQuery rows (or dicts) and add derived columns."""
    records = [{key: _plain(value) for key, value in dict(row).items()} for row in rows]
    frame = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    for column in ("issuer", "description", "transaction_type", "category", "parent_group"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["category"] = frame["category"].replace("", "Other")
    frame["parent_group"] = frame["parent_group"].replace("", "Payments & Others")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0).astype(float)
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"]).astype("datetime64[ns]")
    frame["merchant"] = [normalize_merchant(text, aliases) for text in frame["description"]]
    frame["month"] = frame["transaction_date"].dt.to_period("M")
    frame["weekday"] = pd.Categorical.from_codes(
        frame["transaction_date"].dt.dayofweek.to_numpy(dtype="int64"), dtype=WEEKDAY_DTYPE
    ) if len(frame) else pd.Categorical([], dtype=WEEKDAY_DTYPE)
    frame["spend"] = -frame["amount"]
    return frame.reset_index(drop=True)


def load_transactions(
    settings: ReportSettings | Mapping[str, Any],
    start: date | str | None = None,
    *,
    client: Any = None,
    today: date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Gold transactions since ``start`` (default 24 months back), cached per project and start.

    ``client`` is any object with ``query(sql, job_config=...)`` returning a job whose
    ``result()`` yields mapping-like rows (tests inject a fake).
    """
    settings = ReportSettings.coerce(settings)
    key = _cache_key(settings, start, today)
    ttl = settings.chat_data_cache_seconds
    project, aliases = key[0], settings.merchant_aliases
    start_date = key[1]

    if use_cache and ttl > 0:
        with _LOCK:
            cached = _CACHE.get(key)
            if cached and clock() - cached[0] < ttl:
                return cached[1].copy()

    client = client if client is not None else _bigquery_client(project)
    job = client.query(QUERY.format(project=project), job_config=_job_config(start_date))
    frame = frame_from_rows(job.result(), aliases)

    if use_cache and ttl > 0:
        with _LOCK:
            _CACHE[key] = (clock(), frame)
    return frame.copy()


def _cache_key(settings: ReportSettings, start: date | str | None, today: date | None) -> tuple[Any, ...]:
    project = settings.project_name
    if not project:
        raise ValueError("project_name is required to load transactions")
    start_date = date.fromisoformat(start) if isinstance(start, str) else (start or default_start(today))
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    return (project, start_date, tuple(sorted(settings.merchant_aliases.items())))


def load_chat_data(
    settings: ReportSettings | Mapping[str, Any],
    start: date | str | None = None,
    *,
    client: Any = None,
    today: date | None = None,
    use_cache: bool = True,
) -> ChatData:
    """``load_transactions`` plus the catalog and data profile, computed once per cached load."""
    from .profile import safe_profile

    settings = ReportSettings.coerce(settings)
    today = today or date.today()
    frame = load_transactions(settings, start, client=client, today=today, use_cache=use_cache)
    key = _cache_key(settings, start, today)
    with _LOCK:
        loaded = _CACHE.get(key)
        derived_key = (key, loaded[0], today) if loaded is not None and use_cache else None
        derived = _DERIVED.get(derived_key) if derived_key else None
    if derived is None:
        derived = (build_catalog(frame), safe_profile(frame, today=today))
        if derived_key:
            with _LOCK:
                _DERIVED.clear()  # only the latest load's derivations are worth keeping
                _DERIVED[derived_key] = derived
    return ChatData(frame, derived[0], derived[1])


def refresh(project: str | None = None) -> int:
    """Drop cached frames (all, or one project's); returns how many were dropped."""
    with _LOCK:
        keys = [key for key in _CACHE if project is None or key[0] == project]
        for key in keys:
            del _CACHE[key]
        _DERIVED.clear()
    return len(keys)


def analysis_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows counted by the spending metrics (gold fact_monthly_spending filter)."""
    return frame[frame["transaction_type"].isin(ANALYSIS_TYPES)]


def build_catalog(frame: pd.DataFrame, top_merchants: int = CATALOG_TOP_MERCHANTS) -> dict[str, Any]:
    """Compact description of what can be asked: dates, issuers, categories, top merchants, metrics."""
    rows = analysis_rows(frame)
    if frame.empty:
        date_range = {"min": None, "max": None, "months": 0}
    else:
        low, high = frame["transaction_date"].min().date(), frame["transaction_date"].max().date()
        date_range = {"min": low.isoformat(), "max": high.isoformat(),
                      "months": (high.year - low.year) * 12 + high.month - low.month + 1}
    categories: dict[str, list[str]] = {}
    for (category, group), _ in rows.groupby(["category", "parent_group"], sort=True):
        categories.setdefault(category, []).append(group)
    merchant_spend = rows.groupby("merchant")["spend"].sum().sort_values(ascending=False)
    return {
        "date_range": date_range,
        "transactions": int(len(rows)),
        "issuers": sorted(rows["issuer"].unique().tolist()),
        "categories": categories,
        "parent_groups": sorted(rows["parent_group"].unique().tolist()),
        "merchants": merchant_spend.head(top_merchants).index.tolist(),
        "merchant_count": int(merchant_spend.size),
        "metrics": dict(METRIC_DEFINITIONS),
        "dimensions": list(DIMENSIONS),
        "periods": list(PERIODS),
        "note": "Card payments are excluded; only Sale, Return, and Adjustment transactions are analyzed.",
    }
