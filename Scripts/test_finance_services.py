import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from finance import bigquery_tools
from finance import qa, reporting
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

            def run(self, objective, snapshot, prompt_context, validator_context, tools):
                self.objective = objective
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

        with patch.object(reporting, "ReportHarness", FakeHarness):
            result = reporting.generate_finance_report(
                config={"Monthly_budget": 8500, "Fixed_costs": 1000, "Savings_goal": 1000},
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


if __name__ == "__main__":
    unittest.main()
