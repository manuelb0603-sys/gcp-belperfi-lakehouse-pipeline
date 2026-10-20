import asyncio
import json
import sys
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from mcp import Client, StdioServerParameters

from finance import bigquery_tools, mcp_client, mcp_server, reporting, settings
from finance.mcp_client import McpToolClient
from finance.settings import PROJECT_ROOT, ReportSettings, SettingsError


def call_in_process(server, name, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


class McpServerTests(unittest.TestCase):
    def test_exposes_the_author_data_tools(self):
        async def names():
            async with Client(mcp_server.build_server("project")) as client:
                return sorted(tool.name for tool in (await client.list_tools()).tools)

        self.assertEqual(asyncio.run(names()), sorted(mcp_server.TOOL_NAMES))

    def test_tools_call_bigquery_tools_with_the_project_and_serialize_like_the_harness(self):
        server = mcp_server.build_server("project")
        result_rows = {"rows": [{"net_real_expense": Decimal("12.50"), "transaction_date": "2026-09-01"}]}
        with patch.object(bigquery_tools, "get_weekly_spending", return_value=result_rows) as weekly:
            result = call_in_process(server, "get_weekly_spending", {"month": "2026-09-01", "category": "Shopping"})
        weekly.assert_called_once_with("project", "2026-09-01", "Shopping", None)
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content[0].text), json.loads(json.dumps(result_rows, default=str)))

    def test_value_errors_come_back_as_tool_errors_with_their_message(self):
        server = mcp_server.build_server("project")
        with patch.object(bigquery_tools, "get_daily_spending", side_effect=ValueError("month must be an ISO date")):
            result = call_in_process(server, "get_daily_spending", {"month": "2026-09"})
        self.assertTrue(result.is_error)
        self.assertIn("month must be an ISO date", result.content[0].text)

    def test_a_project_is_required(self):
        with self.assertRaises(ValueError):
            mcp_server.build_server("")


class McpToolClientStdioTests(unittest.TestCase):
    """A real stdio session against tests/mcp_stub_server.py (fake bigquery_tools, no BigQuery)."""

    @classmethod
    def setUpClass(cls):
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "tests.mcp_stub_server"], cwd=str(PROJECT_ROOT / "Scripts")
        )
        cls.client = McpToolClient("ignored", timeout=60, server_params=params)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_round_trip_over_stdio(self):
        result = self.client.call_tool("get_daily_spending", {"month": "2026-09-01", "category": "Groceries"})
        self.assertEqual(result["project"], "stub-project")
        self.assertEqual((result["month"], result["category"], result["parent_group"]), ("2026-09-01", "Groceries", None))
        self.assertEqual(result["rows"][0]["net_real_expense"], "12.50")
        self.assertTrue(self.client.started)

    def test_tool_errors_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "month must be an ISO date"):
            self.client.call_tool("get_daily_spending", {"month": "2026-09"})

    def test_close_is_idempotent_and_the_client_restarts(self):
        client = McpToolClient("ignored", timeout=60, server_params=self.client._server_params)
        self.assertFalse(client.started)  # lazy: nothing launched before the first call
        client.call_tool("get_daily_spending", {"month": "2026-09-01"})
        client.close()
        client.close()
        self.assertFalse(client.started)
        self.assertEqual(client.call_tool("get_daily_spending", {"month": "2026-08-01"})["month"], "2026-08-01")
        client.close()


class McpServerParametersTests(unittest.TestCase):
    def test_launches_the_package_module_from_scripts_with_the_project_in_the_environment(self):
        params = mcp_client.server_parameters("my-project")
        self.assertEqual((params.command, params.args), (sys.executable, ["-m", "finance.mcp_server"]))
        self.assertEqual(params.cwd, str(PROJECT_ROOT / "Scripts"))
        self.assertEqual(params.env[mcp_server.PROJECT_ENV], "my-project")


class ReportingMcpRoutingTests(unittest.TestCase):
    def tools(self, call_tool):
        snapshot = SimpleNamespace(
            latest_month="2026-09-01", previous_month="2026-08-01",
            categories=[SimpleNamespace(category="Groceries", parent_group="Needs")],
        )
        return reporting._build_tools(
            config={}, project_name="project", snapshot=snapshot, rows=[], memory_module=None,
            thread_key="finance:test", ollama_url="url", model_name="model", call_tool=call_tool,
        )

    def test_data_tools_go_through_call_tool_with_keyword_arguments(self):
        calls = []
        tools = self.tools(lambda name, arguments: calls.append((name, arguments)) or {"ok": True})
        tools["get_category_detail"].function({"category": "Groceries"})
        tools["get_transaction_detail"].function({"month": "2026-08-01", "limit": 5})
        tools["get_daily_spending"].function({"parent_group": "Wants"})
        tools["get_weekly_spending"].function({})
        tools["explain_category_change"].function({"category": "Groceries"})
        self.assertEqual(calls, [
            ("get_category_detail", {"category": "Groceries", "month": "2026-09-01", "previous_month": "2026-08-01"}),
            ("get_transaction_detail", {"month": "2026-08-01", "category": None, "limit": 5}),
            ("get_daily_spending", {"month": "2026-09-01", "category": None, "parent_group": "Wants"}),
            ("get_weekly_spending", {"month": "2026-09-01", "category": None, "parent_group": None}),
            ("explain_category_change", {"category": "Groceries", "month": "2026-09-01", "top_n": 5, "parent_group": "Needs"}),
        ])

    def test_restrictions_apply_before_the_mcp_call(self):
        calls = []
        tools = self.tools(lambda name, arguments: calls.append(name))
        with self.assertRaises(ValueError):
            tools["get_daily_spending"].function({"month": "2025-01-01"})
        self.assertEqual(calls, [])

    def test_generate_finance_report_uses_mcp_by_default_and_closes_it(self):
        created = []

        class FakeMcpClient:
            def __init__(self, project_name):
                self.project_name, self.closed = project_name, False
                created.append(self)

            def call_tool(self, name, arguments):
                raise AssertionError("no data tool is called in this run")

            def close(self):
                self.closed = True

        class StopHarness(Exception):
            pass

        rows = [{"transaction_month": "2026-09-01", "category": "Groceries", "parent_group": "Needs", "net_real_expense": 10}]
        config = {"Monthly_budget": 1000, "Fixed_costs": 100, "Savings_goal_target": 50}
        with patch.object(reporting, "McpToolClient", FakeMcpClient), \
                patch.object(reporting.ReportHarness, "run", side_effect=StopHarness()):
            with self.assertRaises(StopHarness):
                reporting.generate_finance_report(config, "project", "url", "model", rows=rows)
            with self.assertRaises(StopHarness):
                reporting.generate_finance_report({**config, "tool_transport": "direct"}, "project", "url", "model", rows=rows)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].project_name, "project")
        self.assertTrue(created[0].closed)


class ToolTransportSettingTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(settings.DEFAULTS["tool_transport"], "mcp")
        self.assertEqual(ReportSettings.from_config({}).tool_transport, "mcp")
        self.assertEqual(ReportSettings.from_config({"tool_transport": " Direct "}).tool_transport, "direct")
        with self.assertRaisesRegex(SettingsError, "tool_transport"):
            ReportSettings.from_config({"tool_transport": "http"})


if __name__ == "__main__":
    unittest.main()
