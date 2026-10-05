import os
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from finance import bigquery_tools
from finance import qa, reporting, settings
from finance.agent_harness import RunTrace
from report_models import ValidationDecision


class FakeBigQuery:
    class ArrayQueryParameter:
        def __init__(self, name, type_, values):
            self.name = name
            self.type_ = type_
            self.values = values

    class QueryJobConfig:
        def __init__(self, query_parameters):
            self.query_parameters = query_parameters


class FakeQueryJob:
    def __init__(self, rows):
        self.rows = rows

    def result(self):
        return self.rows


class FakeBigQueryClient:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def query(self, query, job_config=None):
        self.queries.append((query, job_config))
        return FakeQueryJob(self.results.pop(0))


class TestPeriodAwareBigQueryHelpers(unittest.TestCase):
    def test_explicit_period_fetches_month_and_prior_year_month(self):
        client = FakeBigQueryClient([[{"transaction_month": date(2026, 1, 1)}]])
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            result = bigquery_tools.get_report_period_spending("project", year=2026, month=1)

        self.assertEqual(result["report_month"], date(2026, 1, 1))
        self.assertEqual(result["previous_month"], date(2025, 12, 1))
        months = client.queries[0][1].query_parameters[0].values
        self.assertEqual(months, [date(2026, 1, 31), date(2025, 12, 31)])

    def test_latest_period_is_resolved_before_fetch(self):
        client = FakeBigQueryClient([
            [SimpleNamespace(latest_month=date(2025, 8, 31))],
            [{"transaction_month": date(2025, 8, 31)}, {"transaction_month": date(2025, 7, 31)}],
        ])
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            result = bigquery_tools.get_report_period_spending("project")

        self.assertEqual(result["report_month"], date(2025, 8, 1))
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["rows"][0]["transaction_month"], "2025-08-01")

    def test_leap_year_month_end_parameter_and_normalized_row(self):
        client = FakeBigQueryClient([[{"transaction_month": date(2024, 2, 29)}]])
        with patch.object(bigquery_tools, "_client", return_value=(client, FakeBigQuery)):
            result = bigquery_tools.get_report_period_spending("project", year=2024, month=2)

        month_values = client.queries[0][1].query_parameters[0].values
        self.assertEqual(month_values, [date(2024, 2, 29), date(2024, 1, 31)])
        self.assertEqual(result["rows"][0]["transaction_month"], "2024-02-01")

    def test_year_and_month_must_be_provided_together(self):
        with self.assertRaises(ValueError):
            bigquery_tools.get_report_period_spending("project", year=2026)


class TestSharedFinanceReportService(unittest.TestCase):
    def test_semantic_memory_empty_query_returns_actionable_error(self):
        snapshot = SimpleNamespace(latest_month="2026-02-01", previous_month="2026-01-01")
        tools = reporting._build_tools(
            config={},
            project_name="project",
            snapshot=snapshot,
            rows=[],
            memory_module=object(),
            thread_key="finance:test",
            ollama_url="http://ollama/api/generate",
            model_name="model",
        )

        with self.assertRaisesRegex(ValueError, "requires a non-empty 'query'"):
            tools["get_semantic_memory"].function({})

    def test_period_row_selector_does_not_use_predecessor_as_selected_month(self):
        previous_only = [{"transaction_month": "2026-01-01", "category": "Groceries", "net_real_expense": 10}]
        with patch.object(
            reporting,
            "get_report_period_spending",
            return_value={"report_month": date(2026, 2, 1), "rows": previous_only},
        ):
            selected_month, rows = reporting._report_rows({}, "project", 2026, 2)

        self.assertEqual(selected_month, date(2026, 2, 1))
        self.assertEqual(rows, [])

    def test_generator_returns_validated_html_for_requested_month(self):
        rows = [
            {"transaction_month": date(2026, 2, 1), "category": "Groceries", "parent_group": "Food", "net_real_expense": 450},
            {"transaction_month": date(2026, 1, 1), "category": "Groceries", "parent_group": "Food", "net_real_expense": 400},
        ]

        class FakeHarness:
            def __init__(self, **kwargs):
                self.trace = SimpleNamespace(events=[])
                self.tools = None

            def run(self, objective, snapshot, prompt_context, validator_context, tools, report_requirements=None):
                self.objective = objective
                self.report_requirements = report_requirements
                instances.append(self)
                self.tools = tools
                self.asserted_summary_rows = tools["get_spending_summary"].function({})["rows"]
                try:
                    tools["get_transaction_detail"].function({"month": "2025-01-01"})
                except ValueError:
                    pass
                else:
                    raise AssertionError("Agent transaction tool must stay within the selected report period")
                return (
                    {"report_html": "<html><body>February finance report</body></html>", "plain_text": "February report"},
                    ValidationDecision("pass", [], "Accepted"),
                    self.trace,
                )

        instances = []
        with patch.object(reporting, "ReportHarness", FakeHarness):
            result = reporting.generate_finance_report(
                config={
                    "Monthly_budget": 8500,
                    "Fixed_costs": 1000,
                    "Savings_goal_target": 2000,
                },
                project_name="project",
                ollama_url="http://ollama/api/generate",
                model_name="model",
                rows=rows,
                selected_report_month=date(2026, 2, 1),
            )

        self.assertTrue(result.passed)
        self.assertIn("February finance report", result.report_html)
        self.assertEqual(result.snapshot.latest_month, "2026-02-01")
        self.assertEqual(result.snapshot.previous_month, "2026-01-01")
        self.assertEqual(result.snapshot.savings_goal, 2000)
        requirements = instances[0].report_requirements
        self.assertTrue(requirements)
        self.assertIn('"Monthly Snapshot"', requirements[0])
        self.assertTrue(any("February 2026 total net expense $450.00" in line for line in requirements))


    def test_generator_reraises_harness_failure_with_trace_and_snapshot(self):
        rows = [
            {"transaction_month": date(2026, 2, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 450},
            {"transaction_month": date(2026, 1, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 400},
        ]

        class FailingHarness:
            def __init__(self, **kwargs):
                self.trace = RunTrace()

            def run(self, **kwargs):
                self.trace.add("run_aborted", error="boom")
                raise RuntimeError("boom")

        with patch.object(reporting, "ReportHarness", FailingHarness):
            with self.assertRaises(RuntimeError) as raised:
                reporting.generate_finance_report(
                    config={"Monthly_budget": 8500, "Fixed_costs": 1000},
                    project_name="project",
                    ollama_url="http://ollama/api/generate",
                    model_name="model",
                    rows=rows,
                    selected_report_month=date(2026, 2, 1),
                )

        events = [event["event"] for event in raised.exception.run_trace.events]
        self.assertIn("run_aborted", events)
        self.assertEqual(events[-1], "exception")
        self.assertEqual(raised.exception.report_snapshot.latest_month, "2026-02-01")

    def test_generator_passes_cycle_and_run_time_limits_to_the_harness(self):
        rows = [
            {"transaction_month": date(2026, 2, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 450},
            {"transaction_month": date(2026, 1, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 400},
        ]
        captured = []

        class CapturingHarness:
            def __init__(self, **kwargs):
                captured.append(kwargs["limits"])
                self.trace = RunTrace()

            def run(self, **kwargs):
                return {"report_html": "<p>ok</p>", "plain_text": "ok"}, ValidationDecision("pass", [], "ok"), self.trace

        base = {"Monthly_budget": 8500, "Fixed_costs": 1000}
        cases = [({}, 2700, 3), ({"max_run_seconds": 0}, None, 3), ({"max_run_seconds": None}, None, 3),
                 ({"max_run_seconds": 600, "max_validation_cycles": 4}, 600, 4)]
        with patch.object(reporting, "ReportHarness", CapturingHarness):
            for extra, _, _ in cases:
                reporting.generate_finance_report(
                    config={**base, **extra}, project_name="project", ollama_url="http://ollama/api/generate",
                    model_name="model", rows=rows, selected_report_month=date(2026, 2, 1),
                )

        self.assertEqual(
            [(limits.max_run_seconds, limits.max_validation_cycles) for limits in captured],
            [(seconds, cycles) for _, seconds, cycles in cases],
        )

    def test_tool_specs_use_their_registered_names(self):
        snapshot = SimpleNamespace(latest_month="2026-02-01", previous_month="2026-01-01")
        tools = reporting._build_tools({}, "project", snapshot, [], None, "finance:test", "url", "model")

        for key, spec in tools.items():
            self.assertEqual(spec.name, key)
        self.assertIn("historical", tools["get_episodic_memory"].description)
        self.assertIn("prior report", tools["get_report_history"].description)

    def test_report_history_excerpts_visible_text_not_css(self):
        class FakeMemory:
            def get_or_create_thread(self, channel, key):
                return "thread"

            def get_short_term(self, thread_id, limit):
                return [
                    {
                        "subject": "Old HTML-only report",
                        "response_html": None,
                        "response_text": "<html><head><style>body{color:#123}</style></head><body><h1>Hello from February</h1></body></html>",
                    },
                    {"subject": "Plain report", "response_html": "<p>ignored</p>", "response_text": "Plain   summary text"},
                    {"subject": "Empty", "response_html": "<style>x{}</style>", "response_text": ""},
                ]

        snapshot = SimpleNamespace(latest_month="2026-02-01", previous_month="2026-01-01")
        tools = reporting._build_tools({}, "project", snapshot, [], FakeMemory(), "finance:test", "url", "model")
        reports = tools["get_report_history"].function({})["reports"]

        self.assertEqual(len(reports), 2)
        self.assertIn("Hello from February", reports[0]["excerpt"])
        self.assertNotIn("color:#123", reports[0]["excerpt"])
        self.assertTrue(reports[0]["excerpt"].startswith("Previous report for reference:"))
        self.assertIn("Plain summary text", reports[1]["excerpt"])


class TestSharedLongFactSearch(unittest.TestCase):
    def test_search_backfills_embeddings_in_batches_and_combines_rankings(self):
        class FakeMemory:
            def __init__(self):
                self.backfilled = []

            def get_long_facts_needing_embedding(self, model, limit):
                return [
                    {"fact_key": "goal", "fact_text": "Save for travel"},
                    {"fact_key": "style", "fact_text": "Prefers concise reports"},
                ]

            def set_long_fact_embedding(self, key, vector, model):
                self.backfilled.append((key, vector, model))

            def search_long_facts(self, vector, model, limit):
                return [{"id": 1, "fact_key": "goal", "fact_text": "Save for travel", "similarity": 1.0}]

            def search_long_facts_bm25(self, query, limit):
                return [{"id": 1, "fact_key": "goal", "fact_text": "Save for travel", "bm25_score": 1.0}]

            def combine_fact_search_results(self, vectors, lexical, limit):
                self.sources = (vectors, lexical)
                return [{"fact_key": "goal", "retrieval_sources": ["vector", "bm25"]}]

        memory = FakeMemory()
        with (
            patch("finance.memory_search.embed_text", return_value=[1.0, 0.0]),
            patch("finance.memory_search.embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]) as batch_embed,
        ):
            facts, method = reporting.search_long_facts(
                "travel savings", memory, "http://ollama/api/generate", "nomic-embed-text", limit=3
            )

        self.assertEqual(method, "vector+bm25_rrf")
        self.assertEqual(facts[0]["retrieval_sources"], ["vector", "bm25"])
        batch_embed.assert_called_once()
        self.assertEqual([item[0] for item in memory.backfilled], ["goal", "style"])


class TestFinanceQuestionMemory(unittest.TestCase):
    def test_answer_uses_both_retrievers_and_saves_successful_exchange(self):
        class FakeMemory:
            def __init__(self):
                self.saved = []

            def get_or_create_thread(self, channel, key):
                self.thread = (channel, key)
                return "finance-thread"

            def save_short_term(self, **kwargs):
                self.saved.append(kwargs)

        memory = FakeMemory()
        report_passages = ["Selected-month dining spend is $80."]
        long_facts = [{"fact_key": "dining_goal", "fact_text": "User prefers reducing dining spending."}]
        captured_prompts = []
        with (
            patch.object(qa._REPORT_INDEX, "search", return_value=report_passages),
            patch.object(qa, "search_long_facts", return_value=(long_facts, "vector+bm25_rrf")),
            patch.object(qa, "generate_text", side_effect=lambda prompt, *args, **kwargs: captured_prompts.append(prompt) or "Try a weekly dining budget."),
        ):
            answer = qa.answer_finance_question(
                "How can I reduce dining costs?",
                {"selected_month": "February 2026", "snapshot": {"latest_total_net_expense": 500}},
                "<p>Selected month summary</p>",
                {"memory_schema": "test", "ollama_timeout": 5},
                "http://ollama/api/generate",
                "model",
                memory_module=memory,
            )

        self.assertEqual(answer, "Try a weekly dining budget.")
        self.assertIn("Selected-month dining spend", captured_prompts[0])
        self.assertIn("HISTORICAL LONG-TERM FACTS", captured_prompts[0])
        self.assertIn("User prefers reducing dining spending", captured_prompts[0])
        self.assertEqual(memory.thread[0], "finance_qa")
        self.assertEqual(len(memory.saved), 1)
        self.assertEqual(memory.saved[0]["prompt"], "How can I reduce dining costs?")
        self.assertEqual(memory.saved[0]["meta"]["selected_month"], "February 2026")
        self.assertIsNone(memory.saved[0]["response_html"])

    def test_empty_question_is_not_saved(self):
        self.assertEqual(
            qa.answer_finance_question(" ", {"selected_month": "2026-02"}, "", {}, "url", "model", memory_module=object()),
            "Enter a question about the selected financial summary.",
        )


class TestSharedMemoryResolution(unittest.TestCase):
    def test_schema_path_without_override(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_DB_PATH", None)
            path = settings.memory_db_path("finance_v2")
        self.assertEqual(path, settings.PROJECT_ROOT / "agent" / "memory" / "finance_v2.sqlite3")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_DB_PATH", None)
            self.assertEqual(settings.memory_db_path(None).name, "memory.sqlite3")
            with self.assertRaises(ValueError):
                settings.memory_db_path("../escape")

    def test_environment_override_wins(self):
        with patch.dict(os.environ, {"MEMORY_DB_PATH": "C:/tmp/custom.sqlite3"}):
            self.assertEqual(settings.memory_db_path("finance_v2"), Path("C:/tmp/custom.sqlite3"))

    def test_qa_memory_uses_shared_resolution(self):
        with (
            patch.dict(os.environ, {"MEMORY_DB_PATH": "C:/tmp/custom.sqlite3"}),
            patch("finance.memory.MemoryStore.init_db") as init_db,
        ):
            store = qa._memory_module({"memory_schema": "other"})
        self.assertTrue(hasattr(store, "get_or_create_thread"))
        init_db.assert_called_once()
        self.assertEqual(Path(store.db_path), Path("C:/tmp/custom.sqlite3"))

    def test_report_thread_key_matches_email_flow(self):
        self.assertEqual(settings.report_thread_key({"email_listings": ["a@x", "b@x"]}), "a@x, b@x")
        self.assertEqual(settings.report_thread_key({"email_sender": "s@x"}), "s@x")
        self.assertEqual(settings.report_thread_key({}), "email:unknown")


if __name__ == "__main__":
    unittest.main()
