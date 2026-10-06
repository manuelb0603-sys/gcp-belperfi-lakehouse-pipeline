"""Finance chat tools: definitions, dispatch, list_values, get_month_snapshot, compact summaries."""

import json
import unittest
from collections import defaultdict
from datetime import date
from decimal import Decimal

from tests.chat_test_fixtures import ROWS, TODAY, fixture_frame
from finance.chat.data import build_catalog
from finance.chat.tools import MAX_RESULT_BYTES, ToolContext, dispatch, monthly_rows, tool_definitions
from finance.report_metrics import calculate_report_snapshot
from finance.settings import ReportSettings

FRAME = fixture_frame()
CATALOG = build_catalog(FRAME)
SETTINGS = ReportSettings.from_config({
    "Monthly_budget": 5000, "Fixed_costs": 4000, "Savings_goal_target": 1000, "category_goals": {"Shopping": 120},
})


def context(focus_month=None, settings=SETTINGS):
    return ToolContext(FRAME, CATALOG, settings, TODAY, focus_month)


def gold_monthly_rows(months):
    """fact_monthly_spending computed independently from the raw fixture rows."""
    totals = defaultdict(Decimal)
    for day, _issuer, _desc, amount, kind, category, group in ROWS:
        if kind in ("Sale", "Return", "Adjustment") and day[:7] in months:
            totals[(date(int(day[:4]), int(day[5:7]), 1), category, group)] += -Decimal(amount)
    return [{"transaction_month": month, "category": category, "parent_group": group, "net_real_expense": float(value)}
            for (month, category, group), value in sorted(totals.items())]


class TestDefinitions(unittest.TestCase):
    def test_three_tools_with_flat_json_schemas(self):
        definitions = tool_definitions()
        self.assertEqual([d["name"] for d in definitions], ["analyze_spending", "list_values", "get_month_snapshot"])
        analyze = definitions[0]["parameters"]
        text = json.dumps(analyze)
        self.assertNotIn("$ref", text)
        self.assertNotIn("$defs", text)
        self.assertNotIn("dataset", analyze["properties"])
        self.assertIn("title", analyze["properties"])  # the spec's own title field survives title stripping
        self.assertEqual(analyze["properties"]["filters"]["properties"]["merchant"]["type"], "array")
        self.assertEqual(definitions[1]["parameters"]["properties"]["field"]["enum"], ["category", "parent_group", "issuer", "merchant"])
        for definition in definitions:
            self.assertEqual(definition["parameters"]["type"], "object")
            self.assertTrue(definition["description"])


class TestDispatch(unittest.TestCase):
    def test_analyze_spending_runs_the_validated_spec(self):
        outcome = dispatch("analyze_spending", {"metric": "net_spend", "group_by": "category", "period": "last_quarter",
                                                "filters": {"category": ["groceries", "Shoping"]}}, context())
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.spec.filters.category, ["Groceries", "Shopping"])
        self.assertIn("Interpreted category 'Shoping' as 'Shopping'.", outcome.notes)
        self.assertEqual(outcome.payload["summary"]["total"], 900.0)
        self.assertEqual(outcome.result.table["category"].tolist(), ["Groceries", "Shopping"])

    def test_errors_list_valid_values_and_never_raise(self):
        cases = {
            ("analyze_spending", json.dumps({"filters": {"category": ["Veterinary"]}})): "Closest valid category values",
            ("analyze_spending", json.dumps({"metric": "profit"})): "Input should be 'net_spend'",
            ("analyze_spending", json.dumps({"metric": "net_spend", "colour": "red"})): "valid fields:",
            ("analyze_spending", json.dumps({"__invalid_arguments__": "{oops"})): "not a valid JSON object",
            ("forecast", "{}"): "Unknown tool 'forecast'. Available tools: analyze_spending, list_values, get_month_snapshot.",
            ("list_values", json.dumps({"field": "store"})): "Valid fields: category, parent_group, issuer, merchant",
            ("get_month_snapshot", json.dumps({"month": "2025-01"})): "Closest months with data: 2026-06",
            ("get_month_snapshot", json.dumps({"month": "Sept"})): "month must be YYYY-MM",
        }
        for (name, arguments), expected in cases.items():
            outcome = dispatch(name, json.loads(arguments), context())
            self.assertFalse(outcome.ok, name)
            self.assertIn(expected, outcome.payload["error"], arguments)
            self.assertIsNone(outcome.result)

    def test_list_values_search(self):
        outcome = dispatch("list_values", {"field": "merchant", "search": "STAR"}, context())
        self.assertEqual(outcome.payload["values"], ["Starbucks"])
        categories = dispatch("list_values", {"field": "category"}, context()).payload
        self.assertEqual(categories["groups"]["Groceries"], ["Needs"])
        self.assertEqual(categories["total_matches"], len(CATALOG["categories"]))
        issuers = dispatch("list_values", {"field": "issuer", "search": "ame"}, context()).payload
        self.assertEqual(issuers["values"], ["Amex"])
        missing = dispatch("list_values", {"field": "merchant", "search": "Wholefoods"}, context()).payload
        self.assertEqual(missing["values"], [])
        self.assertIn("Wholefds Mkt", missing["closest"])
        # Every merchant is searchable, not only the catalog's top merchants.
        tiny = ToolContext(FRAME, {**CATALOG, "merchants": ["Amazon"]}, SETTINGS, TODAY)
        self.assertEqual(dispatch("list_values", {"field": "merchant", "search": "chev"}, tiny).payload["values"], ["Chevron"])

    def test_month_snapshot_equals_calculate_report_snapshot(self):
        outcome = dispatch("get_month_snapshot", {"month": "2026-09"}, context())
        self.assertTrue(outcome.ok)
        expected = calculate_report_snapshot(
            gold_monthly_rows({"2026-08", "2026-09"}), monthly_deposit=5000, fixed_costs=4000,
            savings_goal=1000, category_goals={"Shopping": 120.0}, as_of=TODAY,
        )
        self.assertEqual(outcome.snapshot.to_dict(), expected.to_dict())
        self.assertEqual(monthly_rows(FRAME, date(2026, 9, 1)), gold_monthly_rows({"2026-08", "2026-09"}))
        payload = outcome.payload
        self.assertEqual((payload["month"], payload["previous_month"]), ("2026-09", "2026-08"))
        self.assertEqual((payload["total_net_spend"], payload["previous_total_net_spend"]), (470.0, 490.5))
        self.assertEqual(payload["budget"]["disposable_cash"], expected.disposable_cash)
        self.assertEqual(payload["budget"]["savings_shortfall"], expected.savings_shortfall)
        self.assertEqual(payload["top_discretionary"], expected.top_discretionary_categories)

    def test_month_snapshot_defaults_to_focus_then_latest_month(self):
        self.assertEqual(dispatch("get_month_snapshot", {}, context()).payload["month"], "2026-10")
        self.assertEqual(dispatch("get_month_snapshot", {}, context(date(2026, 7, 1))).payload["month"], "2026-07")
        first = dispatch("get_month_snapshot", {"month": "2026-06"}, context()).payload
        self.assertIsNone(first["previous_month"])
        unconfigured = dispatch("get_month_snapshot", {"month": "2026-09"}, context(settings=ReportSettings.from_config({}))).payload
        self.assertIn("not configured", unconfigured["budget"])


class TestCompactSummaries(unittest.TestCase):
    SPECS = [
        {"metric": "net_spend", "group_by": ["merchant", "month"], "period": "all", "chart": "heatmap"},
        {"metric": "transaction_count", "group_by": ["weekday", "merchant"], "period": "all"},
        {"metric": "gross_spend", "group_by": ["week", "category"], "period": "all"},
        {"metric": "net_spend", "group_by": ["category"], "period": "last_quarter", "compare_to": "previous_period"},
        {"metric": "share_of_total", "group_by": ["merchant"], "period": "all",
         "filters": {"category": ["Groceries", "Food & Drink", "Shopping"], "min_amount": 5}},
        {"metric": "net_spend", "group_by": ["day_of_month", "issuer"], "period": "all", "transform": "none"},
    ]
    RAW_TEXT = ["WHOLEFDS MKT #10234", "800-555-0199", "AMZN Mktp", "2026-09-13", "2026-07-03", "AUTOPAY"]

    def test_results_are_small_and_contain_no_rows(self):
        ctx = context()
        outcomes = [dispatch("analyze_spending", spec, ctx) for spec in self.SPECS]
        outcomes += [dispatch("get_month_snapshot", {"month": month}, ctx) for month in ("2026-08", "2026-09")]
        outcomes += [dispatch("list_values", {"field": field}, ctx) for field in ("merchant", "category")]
        for outcome in outcomes:
            content = outcome.content()
            self.assertTrue(outcome.ok, content)
            self.assertLessEqual(len(content.encode("utf-8")), MAX_RESULT_BYTES, content)
            for raw in self.RAW_TEXT:
                self.assertNotIn(raw, content)
            self.assertNotIn("description", content)
            self.assertNotIn("transaction_date", content)

    def test_long_merchant_lists_are_trimmed_to_fit(self):
        many = [f"Merchant Number {index:03d} With A Long Name" for index in range(200)]
        ctx = context()
        ctx.__dict__["merchants"] = many  # override the cached property
        outcome = dispatch("list_values", {"field": "merchant"}, ctx)
        self.assertLessEqual(len(outcome.content().encode("utf-8")), MAX_RESULT_BYTES)
        self.assertEqual(outcome.payload["total_matches"], 200)
        self.assertGreaterEqual(len(outcome.payload["values"]), 10)


if __name__ == "__main__":
    unittest.main()
