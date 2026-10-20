"""Local MCP server (stdio) for the report author's read-only BigQuery tools.

Built on the MCP Python SDK's ``MCPServer`` (FastMCP in SDK 1.x). It exposes the
``finance.bigquery_tools`` functions the author's data tools use; each tool runs the same
parameterized SELECT as the in-process call. Snapshot-aware restrictions (report month and
predecessor, known categories) stay in ``finance/reporting.py``, which calls these tools through
``finance.mcp_client.McpToolClient`` when ``tool_transport`` is ``"mcp"``.

Results are JSON text serialized like the harness serializes tool results
(``json.dumps(..., default=str)``), so the author sees the same evidence either way. ValueError
messages (and other failures) come back as MCP tool errors with their text, so the author gets
actionable feedback.

The GCP project comes from the ``FINANCE_MCP_PROJECT`` environment variable (or ``--project``).
Run by hand from ``Scripts/``: ``python -m finance.mcp_server --project <gcp-project>``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import bigquery_tools

SERVER_NAME = "belperfi-finance"
PROJECT_ENV = "FINANCE_MCP_PROJECT"
TOOL_NAMES = (
    "get_category_detail",
    "get_transaction_detail",
    "get_daily_spending",
    "get_weekly_spending",
    "explain_category_change",
)


def _call(name: str, project_name: str, *args: Any) -> str:
    # Looked up at call time so tests can patch finance.bigquery_tools.
    fetch: Callable[..., dict[str, Any]] = getattr(bigquery_tools, name)
    try:
        return json.dumps(fetch(project_name, *args), default=str)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:  # BigQuery/auth failures: keep the message instead of a generic error
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


def build_server(project_name: str) -> MCPServer:
    """An MCPServer whose tools query ``project_name``'s gold tables (read-only)."""
    if not project_name:
        raise ValueError(f"A GCP project is required (set {PROJECT_ENV} or pass --project)")
    server = MCPServer(SERVER_NAME, log_level="WARNING")

    @server.tool()
    def get_category_detail(category: str, month: str, previous_month: str | None = None) -> str:
        """Monthly gold.fact_monthly_spending rows for one category in a month (YYYY-MM-01) and optionally its predecessor."""
        return _call("get_category_detail", project_name, category, month, previous_month)

    @server.tool()
    def get_transaction_detail(month: str, category: str | None = None, limit: int = 25) -> str:
        """Up to 100 gold.fact_transactions rows for a month (YYYY-MM-01), optionally one category."""
        return _call("get_transaction_detail", project_name, month, category, limit)

    @server.tool()
    def get_daily_spending(month: str, category: str | None = None, parent_group: str | None = None) -> str:
        """Per-day spending totals for a month (YYYY-MM-01) from gold.fact_daily_spending."""
        return _call("get_daily_spending", project_name, month, category, parent_group)

    @server.tool()
    def get_weekly_spending(month: str, category: str | None = None, parent_group: str | None = None) -> str:
        """Per-week (Monday-Sunday) spending totals for the weeks overlapping a month (YYYY-MM-01) from gold.fact_weekly_spending."""
        return _call("get_weekly_spending", project_name, month, category, parent_group)

    @server.tool()
    def explain_category_change(category: str, month: str, top_n: int = 5, parent_group: str | None = None) -> str:
        """Compact summary of what drove one category's change from the previous month to month (YYYY-MM-01)."""
        return _call("explain_category_change", project_name, category, month, top_n, parent_group)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Belperfi finance MCP server (stdio).")
    parser.add_argument("--project", default=os.environ.get(PROJECT_ENV), help=f"GCP project (default: ${PROJECT_ENV})")
    args = parser.parse_args(argv)
    build_server(args.project).run("stdio")


if __name__ == "__main__":
    main()
