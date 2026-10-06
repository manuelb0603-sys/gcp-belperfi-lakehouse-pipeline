"""Finance chat data profile: thresholds, profile-driven repairs, data-end anchoring, charts, tables, prompt, examples.

Two fixtures: (a) sparse merchants shaped like the mock gold data (531 nearly unique merchants, data ending
2026-08-19, today 2026-10-06) and (b) realistic repeating merchants. No BigQuery or model calls.
"""

import json
import unittest
from datetime import date
from unittest import mock

import pandas as pd

from chat_test_fixtures import (
    REPEATING_TODAY,
    SPARSE_TODAY,
    TODAY,
    fixture_frame,
    frame_of,
    raw_rows,
    repeating_merchant_rows,
    sparse_merchant_rows,
)
from finance.chat import data, prompts
from finance.chat.analysis_spec import (
    AnalysisSpec,
    Limitation,
    month_coverage,
    resolve_period,
    validate_and_repair,
)
from finance.chat.data import build_catalog
from finance.chat.engine import run_turn
from finance.chat.executor import AnalysisContext, display_table, run_analysis
from finance.chat.profile import (
    LIMITED,
    UNUSABLE,
    USABLE,
    _categorical_status,
    _month_status,
    build_data_profile,
    suggest_examples,
)
from finance.chat.session import ChatSession
from finance.chat.tools import ToolContext, dispatch
from finance.settings import ReportSettings
from test_chat_engine import FakeChatClient, call, reply

SETTINGS = ReportSettings.from_config({})

SPARSE = frame_of(sparse_merchant_rows())
SPARSE_CATALOG = build_catalog(SPARSE)
SPARSE_PROFILE = build_data_profile(SPARSE, today=SPARSE_TODAY)

REPEATING = frame_of(repeating_merchant_rows())
REPEATING_CATALOG = build_catalog(REPEATING)
REPEATING_PROFILE = build_data_profile(REPEATING, today=REPEATING_TODAY)


def sparse_ctx():
    return ToolContext(SPARSE, SPARSE_CATALOG, SETTINGS, SPARSE_TODAY, None, SPARSE_PROFILE)


def repeating_ctx():
    return ToolContext(REPEATING, REPEATING_CATALOG, SETTINGS, REPEATING_TODAY, None, REPEATING_PROFILE)


def sparse_repair(spec):
    ctx = sparse_ctx()
    return validate_and_repair(spec, SPARSE_CATALOG, known_merchants=ctx.merchants, today=SPARSE_TODAY,
                               profile=SPARSE_PROFILE)


class TestThresholds(unittest.TestCase):
    def test_categorical_status_rules(self):
        self.assertEqual(_categorical_status("issuer", 1, 50, 1.0)[0], UNUSABLE)
        self.assertEqual(_categorical_status("issuer", 2, 300, 1.0)[0], USABLE)
        self.assertEqual(_categorical_status("merchant", 40, 2, 0.9)[0], LIMITED)  # median under 3
        self.assertEqual(_categorical_status("merchant", 40, 3, 0.9)[0], USABLE)
        self.assertEqual(_categorical_status("merchant", 40, 5, 0.39)[0], LIMITED)  # top 10 under 40%
        self.assertEqual(_categorical_status("merchant", 40, 5, 0.40)[0], USABLE)
        self.assertEqual(_categorical_status("category", 8, 5, 0.1)[0], USABLE)  # 10 or fewer values: top-10 rule n/a
        status, reason = _categorical_status("merchant", 531, 1, 0.04)
        self.assertEqual(status, LIMITED)
        self.assertIn("531 merchants with a median of 1 transaction each", reason)
        self.assertNotIn("%", reason)

    def test_month_status_rules(self):
        self.assertEqual(_month_status(2, 3)[0], USABLE)
        self.assertEqual(_month_status(1, 3)[0], LIMITED)
        self.assertEqual(_month_status(0, 2)[0], LIMITED)
        self.assertEqual(_month_status(0, 1)[0], UNUSABLE)

    def test_month_coverage_edges(self):
        complete, partial = month_coverage(date(2025, 8, 20), date(2026, 8, 19), SPARSE_TODAY)
        self.assertEqual((complete[0], complete[-1], len(complete)), (date(2025, 9, 1), date(2026, 7, 1), 11))
        self.assertEqual(partial, [date(2025, 8, 1), date(2026, 8, 1)])
        # Within two days of either edge counts as complete; the current month never does.
        complete, partial = month_coverage(date(2026, 7, 3), date(2026, 9, 28), date(2026, 10, 1))
        self.assertEqual(complete, [date(2026, 7, 1), date(2026, 8, 1), date(2026, 9, 1)])
        complete, partial = month_coverage(date(2026, 9, 1), date(2026, 9, 30), date(2026, 9, 30))
        self.assertEqual((complete, partial), ([], [date(2026, 9, 1)]))

    def test_single_issuer_is_unusable_and_small_data_limits_dimensions(self):
        frame = frame_of([row for row in raw_rows() if row["issuer"] == "Chase"])
        profile = build_data_profile(frame, today=TODAY)
        self.assertEqual(profile.status("issuer"), UNUSABLE)
        self.assertEqual(profile.dimensions["issuer"].reason, "only 1 card in the data")
        self.assertEqual(profile.status("day_of_month"), LIMITED)

    def test_empty_frame(self):
        profile = build_data_profile(data.frame_from_rows([]), today=TODAY)
        self.assertEqual(profile.transactions, 0)
        self.assertIsNone(profile.data_max)
        self.assertEqual(suggest_examples(profile), list(prompts.STATIC_EXAMPLES))
        self.assertIn("no spending data", prompts.profile_block(profile))


class TestSparseMerchants(unittest.TestCase):
    """Fixture (a): shaped like the mock gold data."""

    def test_profile(self):
        p = SPARSE_PROFILE
        self.assertEqual((p.data_min, p.data_max), (date(2025, 8, 20), date(2026, 8, 19)))
        self.assertEqual(p.partial_months, ("2025-08", "2026-08"))
        self.assertEqual(len(p.complete_months), 11)
        self.assertEqual(p.latest_complete_month, "2026-07")
        self.assertEqual(p.data_end_lag_days, 48)
        self.assertTrue(p.anchored_to_data_end)
        self.assertEqual(p.transaction_types, {"Sale": 580, "Return": 20, "Payment": 12})
        self.assertEqual(p.analysis_types, ("Sale", "Return", "Adjustment"))
        self.assertEqual(p.status("merchant"), LIMITED)
        merchant = p.dimensions["merchant"]
        self.assertEqual((merchant.distinct, merchant.median_per_value, merchant.min_per_value), (531, 1.0, 1))
        self.assertLess(merchant.top10_transaction_share, 0.40)
        for name in ("category", "parent_group", "issuer", "weekday", "month", "day_of_month"):
            self.assertEqual(p.status(name), USABLE, name)
        self.assertEqual([c.name for c in p.schema],
                         ["issuer", "transaction_date", "description", "amount", "transaction_type", "category", "parent_group"])
        self.assertEqual(p.schema[3].dtype, "float64")
        json.dumps(p.to_dict())  # serializable

    def test_heatmap_by_store_is_repaired_to_category_by_weekday(self):
        spec, notes = sparse_repair({"metric": "transaction_count", "group_by": ["merchant", "weekday"], "chart": "heatmap"})
        self.assertEqual((spec.group_by, spec.chart, spec.top_n), (["category", "weekday"], "heatmap", None))
        limitation = next(note for note in notes if isinstance(note, Limitation))
        self.assertTrue(limitation.startswith("Grouping by merchant isn't meaningful for this data (531 merchants"))
        self.assertTrue(limitation.endswith("so I grouped by category instead."))
        # A model that only asks for "activity by store" as a heatmap gets category x weekday too.
        spec, _ = sparse_repair({"metric": "transaction_count", "group_by": ["merchant"], "chart": "heatmap"})
        self.assertEqual(spec.group_by, ["category", "weekday"])
        # Top merchants become top categories.
        spec, notes = sparse_repair({"group_by": ["merchant"], "top_n": 5})
        self.assertEqual((spec.group_by, spec.top_n), (["category"], 5))
        self.assertFalse(any(note.startswith("Showing the top 10 merchants") for note in notes))

    def test_explicit_merchant_filter_keeps_merchant_grouping(self):
        merchant = sparse_ctx().merchants[0]
        spec, notes = sparse_repair({"group_by": ["merchant"], "filters": {"merchant": [merchant]}})
        self.assertEqual(spec.group_by, ["merchant"])
        self.assertFalse(any(isinstance(note, Limitation) for note in notes))
        spec, _ = sparse_repair({"group_by": ["merchant", "month"], "filters": {"merchant_contains": "LLC"}})
        self.assertEqual(spec.group_by, ["merchant", "month"])

    def test_this_month_resolves_to_the_partial_last_data_month(self):
        outcome = dispatch("analyze_spending", {"metric": "net_spend", "period": "this_month"}, sparse_ctx())
        self.assertTrue(outcome.ok)
        period = outcome.result.period
        self.assertEqual((period.start, period.end), (date(2026, 8, 1), date(2026, 8, 19)))
        limitations = outcome.payload["summary"]["limitations"]
        self.assertIn("Data ends Aug 19, 2026; 'this month' = Aug 2026 (partial).", limitations)
        self.assertGreater(outcome.payload["summary"]["transactions"], 0)  # not empty
        self.assertNotIn("No matching transactions in this period.", outcome.notes)

    def test_relative_periods_anchor_to_data_end(self):
        def resolved(name):
            return resolve_period(AnalysisSpec(period=name), SPARSE_TODAY, SPARSE_PROFILE.data_min, SPARSE_PROFILE.data_max)

        expected = {
            "last_month": (date(2026, 7, 1), date(2026, 7, 31)),
            "last_3_months": (date(2026, 5, 1), date(2026, 7, 31)),
            "last_6_months": (date(2026, 2, 1), date(2026, 7, 31)),
            "year_to_date": (date(2026, 1, 1), date(2026, 8, 19)),
            "this_quarter": (date(2026, 7, 1), date(2026, 8, 19)),
            "last_quarter": (date(2026, 4, 1), date(2026, 6, 30)),
        }
        for name, window in expected.items():
            period = resolved(name)
            self.assertEqual((period.start, period.end), window, name)
            self.assertTrue(any(note.startswith("Data ends Aug 19, 2026;") for note in period.notes), name)
        # Explicit custom dates are kept as given (clamped), never re-anchored.
        custom = resolve_period(AnalysisSpec(period="custom", date_from="2026-03", date_to="2026-04"),
                                SPARSE_TODAY, SPARSE_PROFILE.data_min, SPARSE_PROFILE.data_max)
        self.assertEqual((custom.start, custom.end, custom.notes), (date(2026, 3, 1), date(2026, 4, 30), ()))

    def test_examples_fit_the_data(self):
        examples = suggest_examples(SPARSE_PROFILE)
        self.assertTrue(4 <= len(examples) <= 6)
        self.assertEqual(examples[:4], [
            "Monthly net spend by category, last 6 months of data",
            "Compare Amex vs Chase by month",
            "Which weekday do I spend the most?",
            "Top categories in Wants vs Needs",
        ])
        for text in examples:
            self.assertNotRegex(text.lower(), r"merchant|store|shop\b")
        self.assertIn("How did Jul 2026 compare with the month before?", examples)

    def test_engine_prompt_and_tool_result_carry_the_limitation(self):
        client = FakeChatClient(
            reply(calls=[call("analyze_spending", metric="transaction_count", group_by=["merchant", "weekday"], chart="heatmap")]),
            reply("Stores are mostly one-off here, so I grouped by category instead."),
        )
        events = list(run_turn(
            ChatSession("s"), "Heatmap of activity by store", client=client, df=SPARSE, catalog=SPARSE_CATALOG,
            settings=SETTINGS, today=SPARSE_TODAY, profile=SPARSE_PROFILE,
        ))
        system = client.calls[0]["system"]
        self.assertIn(prompts.PROFILE_HEADER, system)
        self.assertIn("- Limited (avoid unless asked; the app swaps them for a usable dimension): merchant: 531 merchants", system)
        self.assertIn("Merchants: 531 distinct names, too sparse to group by", system)
        self.assertNotIn("Top merchants by net spend", system)
        self.assertIn('If a tool result has "limitations", say each one in plain words', system)
        payload = json.loads([m for m in client.calls[1]["messages"] if m["role"] == "tool"][0]["content"])
        self.assertIn("so I grouped by category instead", payload["summary"]["limitations"][0])
        self.assertEqual(payload["summary"]["group_by"], ["category", "weekday"])
        done = next(event for event in events if event.status == "done")
        self.assertIn("grouped by category instead", done.content)
        table = events[-1].content
        self.assertEqual(list(table.columns), ["Category", "Weekday", "Transactions"])
        self.assertEqual(table["Weekday"].iloc[0], "Monday")


class TestRepeatingMerchants(unittest.TestCase):
    """Fixture (b): realistic recurring merchants."""

    def test_merchant_is_usable_and_heatmap_allowed(self):
        self.assertEqual(REPEATING_PROFILE.status("merchant"), USABLE)
        self.assertFalse(REPEATING_PROFILE.anchored_to_data_end)
        ctx = repeating_ctx()
        spec, notes = validate_and_repair(
            {"metric": "transaction_count", "group_by": ["merchant", "weekday"], "chart": "heatmap"}, REPEATING_CATALOG,
            known_merchants=ctx.merchants, today=REPEATING_TODAY, profile=REPEATING_PROFILE,
        )
        self.assertEqual((spec.group_by, spec.chart, spec.top_n), (["merchant", "weekday"], "heatmap", 10))
        self.assertFalse(any(isinstance(note, Limitation) for note in notes))
        outcome = dispatch("analyze_spending", {"metric": "transaction_count", "group_by": ["merchant", "weekday"],
                                                "period": "all"}, ctx)
        self.assertEqual(outcome.result.chart, "heatmap")  # auto: both usable, 11 x 7 fits
        self.assertNotIn("limitations", outcome.payload["summary"])

    def test_this_month_is_not_anchored_when_data_is_current(self):
        period = resolve_period(AnalysisSpec(period="this_month"), REPEATING_TODAY, REPEATING_PROFILE.data_min,
                                REPEATING_PROFILE.data_max)
        self.assertEqual((period.start, period.end), (date(2026, 10, 1), date(2026, 10, 6)))
        self.assertFalse(any(isinstance(note, Limitation) for note in period.notes))

    def test_examples_include_merchants(self):
        examples = suggest_examples(REPEATING_PROFILE)
        self.assertIn("Top 5 merchants in Sep 2026", examples)
        self.assertTrue(4 <= len(examples) <= 6)

    def test_prompt_lists_merchants_when_usable(self):
        system = prompts.build_system_prompt(REPEATING_CATALOG, today=REPEATING_TODAY, profile=REPEATING_PROFILE)
        self.assertIn("DATA PROFILE", system)
        self.assertIn("Top merchants by net spend", system)
        self.assertIn("merchant (14)", system)
        self.assertNotIn("- Limited", system)
        block = prompts.profile_block(SPARSE_PROFILE) + prompts.profile_block(REPEATING_PROFILE)
        self.assertNotIn("%", block)
        self.assertNotIn("$", block)
        # Without a profile the prompt is unchanged from before.
        plain = prompts.build_system_prompt(REPEATING_CATALOG, today=REPEATING_TODAY)
        self.assertNotIn(prompts.PROFILE_HEADER, plain)


class TestChartRules(unittest.TestCase):
    FRAME = fixture_frame()

    def run_spec(self, profile=None, **spec):
        return run_analysis(self.FRAME, AnalysisSpec(**spec), AnalysisContext(today=TODAY, profile=profile))

    def test_line_needs_three_points(self):
        self.assertEqual(self.run_spec(group_by=["month"], period="custom", date_from="2026-08", date_to="2026-09").chart, "bar")
        self.assertEqual(self.run_spec(group_by=["month"], period="last_quarter").chart, "line")
        spec, notes = validate_and_repair({"group_by": ["month"], "chart": "line", "period": "custom",
                                           "date_from": "2026-08", "date_to": "2026-09"}, build_catalog(self.FRAME), today=TODAY)
        self.assertEqual(spec.chart, "bar")
        self.assertIn("A line chart needs at least 3 points and this period has 2; showing bars instead.", notes)

    def test_heatmap_needs_usable_dimensions_and_a_small_pivot(self):
        profile = build_data_profile(self.FRAME, today=TODAY)
        self.assertEqual(self.run_spec(metric="transaction_count", group_by=["month", "weekday"], period="last_quarter").chart,
                         "heatmap")
        # day_of_month is limited in this small fixture, so auto avoids a heatmap on it.
        limited = self.run_spec(profile, metric="net_spend", group_by=["day_of_month", "weekday"], period="last_quarter")
        self.assertNotEqual(limited.chart, "heatmap")
        # 18 weeks x 7 weekdays is too big for a heatmap (at most 15 x 12).
        big = self.run_spec(metric="transaction_count", group_by=["week", "weekday"], period="all")
        self.assertEqual(big.chart, "stacked_bar")
        spec, notes = validate_and_repair({"group_by": ["week", "weekday"], "chart": "heatmap", "period": "all"},
                                          build_catalog(self.FRAME), today=TODAY)
        self.assertEqual(spec.chart, "auto")
        self.assertTrue(any("hard to read" in note for note in notes))

    def test_pie_at_most_six_slices(self):
        self.assertEqual(self.run_spec(metric="share_of_total", group_by=["issuer"], period="last_quarter").chart, "pie")
        result = self.run_spec(metric="share_of_total", group_by=["merchant"], period="all", chart="pie")
        self.assertEqual(result.chart, "bar")


class TestFriendlyTable(unittest.TestCase):
    FRAME = fixture_frame()

    def table(self, **spec):
        result = run_analysis(self.FRAME, AnalysisSpec(**spec), AnalysisContext(today=TODAY))
        return result, display_table(result)

    def test_columns_units_and_labels(self):
        result, table = self.table(metric="net_spend", group_by=["month", "category"], period="last_quarter")
        self.assertEqual(list(table.columns), ["Month", "Category", "Net spend ($)"])
        self.assertEqual(table["Month"].iloc[0], "Jul 2026")
        self.assertEqual(list(result.table.columns), ["month", "category", "net_spend"])  # raw table unchanged
        self.assertEqual(result.table["month"].iloc[0], "2026-07")
        _, table = self.table(metric="transaction_count", group_by=["weekday"], period="last_month")
        self.assertEqual(list(table.columns), ["Weekday", "Transactions"])
        self.assertEqual(table["Weekday"].tolist()[:2], ["Monday", "Tuesday"])
        _, table = self.table(metric="share_of_total", group_by=["issuer"], period="last_quarter")
        self.assertEqual(list(table.columns), ["Card", "Share of total (%)"])

    def test_comparison_transform_and_other_dimensions(self):
        _, table = self.table(group_by=["category"], period="last_month", compare_to="previous_period")
        self.assertEqual(list(table.columns),
                         ["Category", "Net spend ($)", "August 2026 (previous period) ($)", "Change ($)", "Change (%)"])
        _, table = self.table(group_by=["month"], period="last_quarter", transform="cumulative")
        self.assertEqual(list(table.columns), ["Month", "Net spend ($)", "Net spend, cumulative ($)"])
        _, table = self.table(group_by=["quarter"], period="custom", date_from="2026-06", date_to="2026-09")
        self.assertEqual(table["Quarter"].tolist(), ["Q2 2026", "Q3 2026"])
        _, table = self.table(group_by=["week"], period="last_month")
        self.assertEqual((table.columns[0], table["Week of"].iloc[0]), ("Week of", "Aug 31, 2026"))
        _, table = self.table(group_by=["parent_group", "merchant"], period="last_quarter", chart="table")
        self.assertEqual(list(table.columns)[:2], ["Group", "Merchant"])


class TestLoadChatData(unittest.TestCase):
    def setUp(self):
        data.refresh()
        self.addCleanup(data.refresh)

    def test_catalog_and_profile_are_built_once_per_load(self):
        from test_chat_data import FakeClient

        client = FakeClient(raw_rows())
        settings = ReportSettings.from_config({"project_name": "demo-project"})
        with mock.patch("finance.chat.profile.build_data_profile", wraps=build_data_profile) as built:
            first = data.load_chat_data(settings, "2026-06-01", client=client, today=TODAY)
            second = data.load_chat_data(settings, "2026-06-01", client=client, today=TODAY)
        self.assertEqual((len(client.calls), built.call_count), (1, 1))
        self.assertIs(first.profile, second.profile)
        self.assertEqual(first.catalog["transactions"], 32)
        self.assertIsInstance(first.frame, pd.DataFrame)
        data.refresh()
        with mock.patch("finance.chat.profile.build_data_profile", side_effect=RuntimeError("boom")):
            with self.assertLogs("finance.chat.profile", level="ERROR"):
                failed = data.load_chat_data(settings, "2026-06-01", client=client, today=TODAY)
        self.assertIsNone(failed.profile)
        self.assertEqual(suggest_examples(failed.profile), list(prompts.STATIC_EXAMPLES))


if __name__ == "__main__":
    unittest.main()
