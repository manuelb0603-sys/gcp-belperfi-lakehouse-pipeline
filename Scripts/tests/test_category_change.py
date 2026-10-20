import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from finance import bigquery_tools, reporting
from finance.category_change import (
    LIMITED_DESCRIPTIONS_NOTE,
    MAX_RESULT_BYTES,
    change_breakdown,
    explain_category_change,
    normalize_description,
)
from finance.harness.trace import validate_tool_arguments
from finance.report_metrics import calculate_report_snapshot
from finance.settings import PROJECT_ROOT

CURRENT, PREVIOUS = "2026-09-01", "2026-08-01"


def tx(month, day, description, spend, kind="Sale"):
    """A fact_transactions row; spend is positive for a purchase (amount = -spend)."""
    return {
        "transaction_month": f"{month[:7]}-{'30' if month.startswith('2026-09') else '31'}",
        "transaction_date": f"{month[:7]}-{day:02d}",
        "description": description,
        "amount": -spend,
        "transaction_type": kind,
    }


def size(result):
    return len(json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


class ChangeBreakdownTests(unittest.TestCase):
    def test_count_and_average_parts_sum_to_the_change(self):
        # Previous: 2 x 50 = 100. Current: 4 x 60 = 240. Count part (4-2)*50 = 100; average part (60-50)*4 = 40.
        result = change_breakdown([50, 50], [60, 60, 60, 60])
        self.assertEqual(result, {
            "change_usd": 140.0, "from_transaction_count": 100.0, "from_average_amount": 40.0, "transaction_count_change": 2,
        })

    def test_same_count_is_all_average(self):
        result = change_breakdown([10, 30], [100, 100])
        self.assertEqual((result["from_transaction_count"], result["from_average_amount"]), (0.0, 160.0))

    def test_empty_month_attributes_the_change_to_count(self):
        self.assertEqual(change_breakdown([], [25, 75])["from_transaction_count"], 100.0)
        self.assertEqual(change_breakdown([25, 75], [])["from_transaction_count"], -100.0)
        self.assertEqual(change_breakdown([25, 75], [])["from_average_amount"], 0.0)


class ExplainCategoryChangeTests(unittest.TestCase):
    def rows(self):
        return [
            tx(PREVIOUS, 3, "  whole foods  #12 ", 80), tx(PREVIOUS, 10, "WHOLE FOODS #12", 70),
            tx(PREVIOUS, 12, "Corner Market", 100), tx(PREVIOUS, 20, "Farm Stand", 30),
            tx(CURRENT, 2, "Whole Foods #12", 120), tx(CURRENT, 9, "whole foods #12", 110),
            tx(CURRENT, 15, "Whole Foods #12", 90), tx(CURRENT, 18, "Bulk Club", 250),
            tx(CURRENT, 21, "Corner Market", 40), tx(CURRENT, 25, "Bulk Club", -50, "Return"),
            tx(CURRENT, 26, "Card payment", 999, "Payment"),
            tx("2026-07-01", 5, "Whole Foods #12", 500),
        ]

    def test_month_stats_mirror_monthly_spending(self):
        result = explain_category_change(self.rows(), "Groceries", CURRENT, PREVIOUS)
        # Payment rows and other months are ignored; returns reduce spend (spend = -amount).
        self.assertEqual(result["current"], {
            "transaction_count": 6, "net_spend": 560.0, "average_transaction": 93.33, "largest_transaction": 250.0,
        })
        self.assertEqual(result["previous"], {
            "transaction_count": 4, "net_spend": 280.0, "average_transaction": 70.0, "largest_transaction": 100.0,
        })
        breakdown = result["change_breakdown"]
        self.assertEqual(breakdown["change_usd"], 280.0)
        self.assertEqual(breakdown["from_transaction_count"], 140.0)  # (6 - 4) * 70
        self.assertEqual(breakdown["from_average_amount"], 140.0)
        self.assertEqual(breakdown["from_transaction_count"] + breakdown["from_average_amount"], breakdown["change_usd"])

    def test_description_movers_and_new_descriptions(self):
        result = explain_category_change(self.rows(), "Groceries", CURRENT, PREVIOUS)
        self.assertEqual(normalize_description("  whole   foods #12 "), "WHOLE FOODS #12")
        self.assertEqual(result["top_increases"][0], {"description": "BULK CLUB", "current": 200.0, "previous": 0.0, "change": 200.0})
        self.assertEqual(result["top_increases"][1]["description"], "WHOLE FOODS #12")
        self.assertEqual(result["top_increases"][1]["change"], 170.0)
        self.assertEqual([item["description"] for item in result["top_decreases"]], ["CORNER MARKET", "FARM STAND"])
        self.assertEqual(result["top_decreases"][0]["change"], -60.0)
        self.assertEqual(result["new_descriptions"], ["BULK CLUB"])
        self.assertNotIn("limitations", result)

    def test_largest_transactions_this_month(self):
        result = explain_category_change(self.rows(), "Groceries", CURRENT, PREVIOUS)
        self.assertEqual(result["largest_transactions"][0], {"date": "2026-09-18", "description": "BULK CLUB", "amount": 250.0})
        self.assertEqual([item["amount"] for item in result["largest_transactions"]], [250.0, 120.0, 110.0])

    def test_top_n_limits_lists(self):
        result = explain_category_change(self.rows(), "Groceries", CURRENT, PREVIOUS, top_n=1)
        self.assertEqual(len(result["top_increases"]), 1)
        self.assertEqual(len(result["top_decreases"]), 1)
        self.assertEqual(len(result["largest_transactions"]), 1)

    def test_unique_descriptions_add_a_limitation(self):
        rows = [tx(CURRENT, day, f"POS PURCHASE REF {day:04d}", 10 + day) for day in range(1, 11)]
        rows += [tx(PREVIOUS, day, f"POS PURCHASE REF {day + 100:04d}", 5) for day in range(1, 4)]
        result = explain_category_change(rows, "Shopping", CURRENT, PREVIOUS)
        self.assertEqual(result["limitations"], LIMITED_DESCRIPTIONS_NOTE)
        self.assertIn("count, average transaction, and largest transactions are the reliable drivers", result["limitations"])

    def test_few_transactions_do_not_add_a_limitation(self):
        rows = [tx(CURRENT, 1, "A", 10), tx(PREVIOUS, 1, "B", 5)]
        self.assertNotIn("limitations", explain_category_change(rows, "Shopping", CURRENT, PREVIOUS))

    def test_result_is_capped_and_has_no_raw_rows(self):
        rows = [tx(CURRENT, 1 + day % 28, f"MERCHANT WITH A VERY LONG DESCRIPTOR NUMBER {day:05d} XYZ", 10 + day) for day in range(200)]
        rows += [tx(PREVIOUS, 1 + day % 28, f"OLD MERCHANT LONG DESCRIPTOR {day:05d} ABCDEFGHIJ", 5 + day) for day in range(200)]
        result = explain_category_change(rows, "Shopping", CURRENT, PREVIOUS, top_n=10)
        self.assertLessEqual(size(result), MAX_RESULT_BYTES)
        self.assertTrue(result["truncated"])
        self.assertIn("limitations", result)
        self.assertEqual(result["current"]["transaction_count"], 200)
        text = json.dumps(result)
        for raw_key in ("transaction_id", "issuer", "amount\": -", "transaction_type"):
            self.assertNotIn(raw_key, text)

    def test_no_rows_raises_a_correctable_error(self):
        with self.assertRaisesRegex(ValueError, "category name exactly as shown in the snapshot"):
            explain_category_change([tx("2026-07-01", 1, "A", 10)], "Nope", CURRENT, PREVIOUS)


class FakeBigQuery:
    class ScalarQueryParameter:
        def __init__(self, name, type_, value):
            self.name, self.type_, self.value = name, type_, value

    class ArrayQueryParameter:
        def __init__(self, name, type_, values):
            self.name, self.type_, self.values = name, type_, values

    class QueryJobConfig:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters


class FakeClient:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def query(self, query, job_config=None):
        self.queries.append((query, job_config))
        return SimpleNamespace(result=lambda: self.rows)


class ExplainCategoryChangeQueryTests(unittest.TestCase):
    def test_one_parameterized_read_only_select(self):
        rows = [
            {"transaction_month": date(2026, 9, 30), "transaction_date": date(2026, 9, 3), "description": "Shop",
             "amount": -40.0, "transaction_type": "Sale"},
            {"transaction_month": date(2026, 8, 31), "transaction_date": date(2026, 8, 3), "description": "Shop",
             "amount": -10.0, "transaction_type": "Sale"},
        ]
        client = FakeClient(rows)
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            result = bigquery_tools.explain_category_change("project", "Groceries'; DROP TABLE x --", "2026-09-01", 3)

        self.assertEqual(len(client.queries), 1)
        query, config = client.queries[0]
        self.assertNotIn("parent_group", query)
        statement = query.strip()
        self.assertTrue(statement.upper().startswith("SELECT"))
        for forbidden in ("INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "CREATE", "ALTER", "TRUNCATE"):
            self.assertNotIn(forbidden, statement.upper())
        self.assertIn("`project.gold.fact_transactions`", query)
        self.assertIn("@category", query)
        self.assertNotIn("Groceries", query)
        params = {param.name: param for param in config.query_parameters}
        self.assertEqual(params["category"].value, "Groceries'; DROP TABLE x --")
        self.assertEqual(params["months"].values, [date(2026, 9, 30), date(2026, 8, 31)])
        self.assertEqual(params["types"].values, ["Sale", "Return", "Adjustment"])
        self.assertEqual(result["source"], "gold.fact_transactions")
        self.assertEqual(result["change_breakdown"]["change_usd"], 30.0)
        self.assertEqual((result["current_month"], result["previous_month"]), ("2026-09-01", "2026-08-01"))
        self.assertLessEqual(size(result), MAX_RESULT_BYTES)

    def test_parent_group_filter_is_a_parameter(self):
        client = FakeClient([{"transaction_month": "2026-09-30", "transaction_date": "2026-09-02", "description": "A",
                              "amount": -5, "transaction_type": "Sale"}])
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            result = bigquery_tools.explain_category_change("project", "Health", "2026-09-01", 5, "Needs")
        query, config = client.queries[0]
        self.assertIn("AND parent_group = @parent_group", query)
        self.assertNotIn("Needs", query)
        params = {param.name: param for param in config.query_parameters}
        self.assertEqual(params["parent_group"].value, "Needs")
        self.assertEqual(result["parent_group"], "Needs")

    def test_january_compares_with_december(self):
        client = FakeClient([{"transaction_month": "2026-01-31", "transaction_date": "2026-01-02", "description": "A",
                              "amount": -5, "transaction_type": "Sale"}])
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            bigquery_tools.explain_category_change("project", "Groceries", "2026-01-01")
        params = {param.name: param for param in client.queries[0][1].query_parameters}
        self.assertEqual(params["months"].values, [date(2026, 1, 31), date(2025, 12, 31)])

    def test_argument_checks_happen_before_any_query(self):
        client = FakeClient([])
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            for args in (("", "2026-09-01"), ("Groceries", "2026-09"), ("Groceries", "2026-09-01", 11)):
                with self.assertRaises(ValueError):
                    bigquery_tools.explain_category_change("project", *args)
        self.assertEqual(client.queries, [])


class ExplainCategoryChangeToolTests(unittest.TestCase):
    def tools(self):
        rows = [
            {"transaction_month": "2026-09-01", "category": "Groceries", "parent_group": "Needs", "net_real_expense": 600},
            {"transaction_month": "2026-08-01", "category": "Groceries", "parent_group": "Needs", "net_real_expense": 300},
        ]
        snapshot = calculate_report_snapshot(rows, 5000, 1000, savings_goal=500)
        return reporting._build_tools(
            config={}, project_name="project", snapshot=snapshot, rows=rows, memory_module=None,
            thread_key="finance:test", ollama_url="http://ollama/api/generate", model_name="model",
        )

    def test_registered_with_schema(self):
        spec = self.tools()["explain_category_change"]
        schema = spec.parameters
        self.assertEqual(schema["required"], ["category"])
        self.assertEqual(schema["properties"]["month"]["enum"], ["2026-09-01"])
        top_n = schema["properties"]["top_n"]
        self.assertEqual((top_n["type"], top_n["minimum"], top_n["maximum"]), ("integer", 1, 10))
        self.assertIn("attribute causes only to what it returns", spec.description)
        self.assertEqual(reporting.tool_parameter_schemas("2026-09-01", "2026-08-01")["explain_category_change"], schema)

    def test_schema_rejects_bad_arguments(self):
        spec = self.tools()["explain_category_change"]
        self.assertEqual(validate_tool_arguments(spec, {"category": "Groceries"}), [])
        self.assertTrue(any("category is required" in error for error in validate_tool_arguments(spec, {})))
        self.assertTrue(validate_tool_arguments(spec, {"category": "Groceries", "month": "2026-08-01"}))
        self.assertTrue(validate_tool_arguments(spec, {"category": "Groceries", "top_n": 0}))
        self.assertTrue(validate_tool_arguments(spec, {"category": "Groceries", "top_n": 11}))

    def test_function_validates_before_querying(self):
        spec = self.tools()["explain_category_change"]
        with patch.object(reporting, "explain_category_change") as query:
            with self.assertRaisesRegex(ValueError, "requires a 'category'"):
                spec.function({})
            with self.assertRaisesRegex(ValueError, "restricted to the selected report month"):
                spec.function({"category": "Groceries", "month": "2026-08-01"})
            with self.assertRaisesRegex(ValueError, "use one of: Groceries"):
                spec.function({"category": "Grocery"})
            query.assert_not_called()
            spec.function({"category": "Groceries", "top_n": 3})
            # The snapshot has Groceries under one parent group, so the tool matches that row.
            query.assert_called_once_with("project", "Groceries", "2026-09-01", 3, "Needs")

    def test_category_under_several_parent_groups_needs_parent_group(self):
        rows = [
            {"transaction_month": "2026-09-01", "category": "Health", "parent_group": "Needs", "net_real_expense": 600},
            {"transaction_month": "2026-09-01", "category": "Health", "parent_group": "Wants", "net_real_expense": 100},
            {"transaction_month": "2026-08-01", "category": "Health", "parent_group": "Needs", "net_real_expense": 300},
        ]
        snapshot = calculate_report_snapshot(rows, 5000, 1000)
        spec = reporting._build_tools(
            config={}, project_name="project", snapshot=snapshot, rows=rows, memory_module=None,
            thread_key="t", ollama_url="http://ollama", model_name="m",
        )["explain_category_change"]
        with patch.object(reporting, "explain_category_change") as query:
            with self.assertRaisesRegex(ValueError, r"several parent groups \(Needs, Wants\)"):
                spec.function({"category": "Health"})
            with self.assertRaisesRegex(ValueError, "not 'Fun'"):
                spec.function({"category": "Health", "parent_group": "Fun"})
            query.assert_not_called()
            spec.function({"category": "Health", "parent_group": "Wants"})
            query.assert_called_once_with("project", "Health", "2026-09-01", 5, "Wants")

    def test_missing_category_becomes_tool_error_evidence(self):
        from finance.harness import HarnessLimits, ReportHarness

        responses = [
            '{"tool_calls":[{"name":"explain_category_change","arguments":{}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        prompts = []

        def query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        tools = self.tools()
        harness = ReportHarness("http://ollama", "test-model", limits=HarnessLimits(), query=query)
        snapshot = calculate_report_snapshot([
            {"transaction_month": "2026-09-01", "category": "Groceries", "parent_group": "Needs", "net_real_expense": 600},
        ], 5000, 1000)
        harness.run("Write it.", snapshot, "CTX", {"explain_category_change": tools["explain_category_change"]},
                    validator_context="VAL")
        self.assertIn("Invalid arguments for explain_category_change", prompts[1])
        self.assertIn("arguments.category is required", prompts[1])


class PromptInvitationTests(unittest.TestCase):
    def read(self, name):
        return (PROJECT_ROOT / "agent" / "prompts" / name).read_text(encoding="utf-8")

    def test_author_prompt_requires_investigating_spikes(self):
        ui = self.read("ui.system.md")
        self.assertIn("The snapshot lists notable_changes", ui)
        self.assertIn('a spike is a notable_changes entry with direction "increase"', ui)
        self.assertIn("call explain_category_change(category) with the entry's parent_group", ui)
        self.assertIn("call get_daily_spending or get_weekly_spending with that category", ui)
        self.assertIn("Attribute causes only to evidence the tools return.", ui)
        self.assertIn("say plainly that the spike is not yet explained instead of guessing", ui)
        self.assertIn("Never invent habits or merchants.", ui)

    def test_validator_requires_tool_evidence_for_change_drivers(self):
        validator = self.read("validator.system.md")
        self.assertIn("must be supported by explain_category_change or get_transaction_detail results in TOOL EVIDENCE", validator)

    def test_notable_changes_are_not_checklist_items(self):
        from finance.report_requirements import build_report_requirements

        rows = [
            {"transaction_month": "2026-09-01", "category": "Groceries", "parent_group": "Needs", "net_real_expense": 600},
            {"transaction_month": "2026-08-01", "category": "Groceries", "parent_group": "Needs", "net_real_expense": 300},
        ]
        snapshot = calculate_report_snapshot(rows, 5000, 1000, savings_goal=500)
        self.assertTrue(snapshot.notable_changes)
        for guidance in ("strict", "guided"):
            text = "\n".join(build_report_requirements(snapshot, guidance))
            self.assertNotIn("notable", text.lower())
            self.assertNotIn("explain_category_change", text)


if __name__ == "__main__":
    unittest.main()
