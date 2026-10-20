"""Stdio stub of the finance MCP server for tests: the real tools over fake bigquery_tools.

Run as ``python -m tests.mcp_stub_server`` from ``Scripts/`` (test_mcp_tools.py does). Every
BigQuery entry point is replaced, so the stub can never reach BigQuery.
"""

from __future__ import annotations

from decimal import Decimal

from finance import bigquery_tools, mcp_server


def _blocked(*_args, **_kwargs):
    raise AssertionError("the stub MCP server must not reach BigQuery")


def fake_daily_spending(project_name, month, category=None, parent_group=None):
    if len(month) != 10:
        raise ValueError("month must be an ISO date")
    return {
        "source": "gold.fact_daily_spending",
        "project": project_name,
        "month": month,
        "category": category,
        "parent_group": parent_group,
        "rows": [{"transaction_date": "2026-09-01", "net_real_expense": Decimal("12.50")}],
    }


bigquery_tools._client = _blocked
for _name in mcp_server.TOOL_NAMES:
    setattr(bigquery_tools, _name, _blocked)
bigquery_tools.get_daily_spending = fake_daily_spending

if __name__ == "__main__":
    mcp_server.build_server("stub-project").run("stdio")
