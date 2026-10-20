import unittest
from datetime import date

from pydantic import ValidationError

from finance.chat.analysis_spec import (
    AnalysisSpec,
    SpecError,
    comparison_period,
    model_json_schema,
    resolve_period,
    validate_and_repair,
)

TODAY = date(2026, 10, 5)
DATA_MIN, DATA_MAX = date(2024, 10, 1), date(2026, 10, 3)

CATALOG = {
    "date_range": {"min": DATA_MIN.isoformat(), "max": DATA_MAX.isoformat(), "months": 25},
    "issuers": ["Amex", "Chase"],
    "categories": {
        "Groceries": ["Needs"], "Food & Drink": ["Wants"], "Shopping": ["Wants"], "Automotive": ["Needs"],
        "Bills & Utilities": ["Needs"], "Health & Wellness": ["Needs", "Wants"], "Travel": ["Wants"],
        "Entertainment": ["Wants"], "Payments": ["Payments & Others"],
    },
    "parent_groups": ["Needs", "Payments & Others", "Wants"],
    "merchants": ["Amazon", "Trader Joe S", "Starbucks", "Starbucks Seattle", "Sushi Zen", "Shell Oil", "Target"],
}


def period(**spec):
    return resolve_period(AnalysisSpec(**spec), TODAY, DATA_MIN, DATA_MAX)


def repair(spec, **kwargs):
    return validate_and_repair(spec, CATALOG, today=TODAY, **kwargs)


class TestResolvePeriod(unittest.TestCase):
    def test_named_periods_for_fixed_today(self):
        expected = {
            "this_month": (date(2026, 10, 1), date(2026, 10, 5)),
            "last_month": (date(2026, 9, 1), date(2026, 9, 30)),
            "last_3_months": (date(2026, 7, 1), date(2026, 9, 30)),
            "last_6_months": (date(2026, 4, 1), date(2026, 9, 30)),
            "last_12_months": (date(2025, 10, 1), date(2026, 9, 30)),
            "year_to_date": (date(2026, 1, 1), date(2026, 10, 5)),
            "last_year": (date(2025, 1, 1), date(2025, 12, 31)),
            "this_quarter": (date(2026, 10, 1), date(2026, 10, 5)),
            "last_quarter": (date(2026, 7, 1), date(2026, 9, 30)),
            "all": (DATA_MIN, DATA_MAX),
        }
        for name, (start, end) in expected.items():
            resolved = period(period=name)
            self.assertEqual((resolved.start, resolved.end), (start, end), name)

    def test_labels(self):
        self.assertEqual(period(period="last_quarter").label, "Jul–Sep 2026 (last quarter)")
        self.assertEqual(period(period="last_month").label, "September 2026")
        self.assertEqual(period(period="last_12_months").label, "Oct 2025–Sep 2026 (last 12 months)")

    def test_last_quarter_in_first_quarter_wraps_year(self):
        resolved = resolve_period(AnalysisSpec(period="last_quarter"), date(2026, 2, 10), DATA_MIN, DATA_MAX)
        self.assertEqual((resolved.start, resolved.end), (date(2025, 10, 1), date(2025, 12, 31)))

    def test_focus_month(self):
        resolved = resolve_period(AnalysisSpec(period="focus_month"), TODAY, DATA_MIN, DATA_MAX, focus_month=date(2026, 8, 17))
        self.assertEqual((resolved.start, resolved.end), (date(2026, 8, 1), date(2026, 8, 31)))
        self.assertEqual(period(period="focus_month", date_from="2026-03").start, date(2026, 3, 1))
        self.assertEqual(period(period="focus_month").start, date(2026, 10, 1))  # latest data month

    def test_custom_dates_month_and_day_forms(self):
        resolved = period(period="custom", date_from="2026-02", date_to="2026-04")
        self.assertEqual((resolved.start, resolved.end), (date(2026, 2, 1), date(2026, 4, 30)))
        resolved = period(period="custom", date_from="2026-02-10", date_to="2026-02-20")
        self.assertEqual((resolved.start, resolved.end), (date(2026, 2, 10), date(2026, 2, 20)))

    def test_custom_dates_clamped_and_swapped_with_notes(self):
        resolved = period(period="custom", date_from="2020-01", date_to="2026-12")
        self.assertEqual((resolved.start, resolved.end), (DATA_MIN, DATA_MAX))
        self.assertTrue(any("limited to the available data" in note for note in resolved.notes))
        swapped = period(period="custom", date_from="2026-05", date_to="2026-03")
        self.assertEqual((swapped.start, swapped.end), (date(2026, 3, 1), date(2026, 5, 31)))
        self.assertTrue(any("swapped" in note for note in swapped.notes))

    def test_period_without_data_is_noted(self):
        resolved = resolve_period(AnalysisSpec(period="last_year"), TODAY, date(2026, 6, 15), date(2026, 8, 20))
        self.assertTrue(any("No data" in note for note in resolved.notes))

    def test_stale_data_anchors_last_month_to_latest_complete_month(self):
        # Data ending before this month: "last month" is the latest complete month, not an empty September.
        resolved = resolve_period(AnalysisSpec(period="last_month"), TODAY, DATA_MIN, date(2026, 8, 20))
        self.assertEqual((resolved.start, resolved.end), (date(2026, 7, 1), date(2026, 7, 31)))
        self.assertIn("Data ends Aug 20, 2026; 'last month' = Jul 2026 (the latest complete month).", resolved.notes)
        self.assertFalse(any("No data" in note for note in resolved.notes))

    def test_comparison_periods(self):
        current = period(period="last_quarter")
        previous, months, days = comparison_period(current, "previous_period")
        self.assertEqual((previous.start, previous.end, months, days), (date(2026, 4, 1), date(2026, 6, 30), 3, 0))
        last_year, months, _ = comparison_period(current, "same_period_last_year")
        self.assertEqual((last_year.start, last_year.end, months), (date(2025, 7, 1), date(2025, 9, 30), 12))
        mtd, months, _ = comparison_period(period(period="this_month"), "previous_period")
        self.assertEqual((mtd.start, mtd.end, months), (date(2026, 9, 1), date(2026, 9, 5), 1))
        days_window = period(period="custom", date_from="2026-02-10", date_to="2026-02-20")
        prior, months, days = comparison_period(days_window, "previous_period")
        self.assertEqual((prior.start, prior.end, months, days), (date(2026, 1, 30), date(2026, 2, 9), 0, 11))
        self.assertIsNone(comparison_period(current, "none"))


class TestSpecModel(unittest.TestCase):
    def test_defaults_and_coercion(self):
        spec = AnalysisSpec(group_by="month", filters={"category": "Groceries"})
        self.assertEqual(spec.group_by, ["month"])
        self.assertEqual(spec.filters.category, ["Groceries"])
        self.assertEqual((spec.metric, spec.period, spec.chart), ("net_spend", "last_12_months", "auto"))

    def test_rejects_bad_values(self):
        bad = [
            {"metric": "profit"}, {"group_by": ["month", "category", "issuer"]}, {"group_by": ["month", "month"]},
            {"top_n": 0}, {"top_n": 26}, {"date_from": "2026/01"}, {"date_from": "2026-13"},
            {"filters": {"min_amount": 50, "max_amount": 10}}, {"unknown": 1}, {"filters": {"sql": "DROP"}},
        ]
        for spec in bad:
            with self.assertRaises(ValidationError, msg=spec):
                AnalysisSpec(**spec)

    def test_title_and_filters_are_cleaned(self):
        spec = AnalysisSpec(title="<b>Spend</b> [x](http://e.vil)", filters={"merchant_contains": "<i>bucks</i>"})
        self.assertNotIn("<", spec.title)
        self.assertNotIn("http", spec.title)
        self.assertEqual(spec.filters.merchant_contains, "i bucks /i")

    def test_json_schema_export(self):
        schema = model_json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("metric", schema["properties"])
        self.assertIn("Filters", schema.get("$defs", {}))
        self.assertIn("share_of_total", str(schema["properties"]["metric"]))


class TestValidateAndRepair(unittest.TestCase):
    def test_fuzzy_categories_issuers_and_groups(self):
        spec, notes = repair({"filters": {"category": ["grocery", "food and drink"], "issuer": "amex", "parent_group": "want"}})
        self.assertEqual(spec.filters.category, ["Groceries", "Food & Drink"])
        self.assertEqual(spec.filters.issuer, ["Amex"])
        self.assertEqual(spec.filters.parent_group, ["Wants"])
        self.assertTrue(any("'grocery'" in note and "'Groceries'" in note for note in notes))

    def test_unknown_category_lists_closest_values(self):
        with self.assertRaises(SpecError) as caught:
            repair({"filters": {"category": ["Crypto Mining"]}})
        message = str(caught.exception)
        self.assertIn("Unknown category 'Crypto Mining'", message)
        self.assertIn("Closest valid category values:", message)
        self.assertIsInstance(caught.exception, ValueError)

    def test_merchant_matching(self):
        spec, _ = repair({"filters": {"merchant": ["trader joes"]}})
        self.assertEqual(spec.filters.merchant, ["Trader Joe S"])
        spec, _ = repair({"filters": {"merchant": ["SQ *SUSHI ZEN #12"]}})
        self.assertEqual(spec.filters.merchant, ["Sushi Zen"])
        # A name shared by several merchants (store locations) matches all of them.
        spec, notes = repair({"filters": {"merchant": ["Starbucks"]}})
        self.assertEqual((spec.filters.merchant, spec.filters.merchant_contains), ([], "Starbucks"))
        self.assertTrue(notes)
        spec, _ = repair({"filters": {"merchant": ["shell"]}})
        self.assertEqual(spec.filters.merchant, ["Shell Oil"])
        spec, _ = repair({"filters": {"merchant": ["Costco"]}}, known_merchants=["Costco Whse", "Amazon"])
        self.assertEqual(spec.filters.merchant, ["Costco Whse"])
        with self.assertRaises(SpecError) as caught:
            repair({"filters": {"merchant": ["Zzyzx Holdings"]}})
        self.assertIn("Closest valid merchant values", str(caught.exception))

    def test_heatmap_gains_a_dimension(self):
        spec, notes = repair({"metric": "transaction_count", "group_by": ["month"], "chart": "heatmap"})
        self.assertEqual(spec.group_by, ["month", "weekday"])
        spec, _ = repair({"metric": "transaction_count", "group_by": ["category"], "chart": "heatmap"})
        self.assertEqual(spec.group_by, ["category", "weekday"])
        spec, notes = repair({"metric": "net_spend", "group_by": ["category"], "chart": "heatmap", "period": "last_6_months"})
        self.assertEqual(spec.group_by, ["category", "month"])
        self.assertTrue(any("month was added" in note for note in notes))
        # One month of data is too short for a month axis, so weekday is used.
        spec, _ = repair({"metric": "net_spend", "group_by": ["category"], "chart": "heatmap", "period": "last_month"})
        self.assertEqual(spec.group_by, ["category", "weekday"])
        spec, _ = repair({"group_by": [], "chart": "heatmap"})
        self.assertEqual(spec.chart, "table")

    def test_chart_repairs(self):
        spec, notes = repair({"group_by": ["category"], "chart": "line"})
        self.assertEqual(spec.chart, "bar")
        self.assertTrue(notes)
        spec, _ = repair({"group_by": ["month"], "chart": "line"})
        self.assertEqual(spec.chart, "line")
        spec, _ = repair({"group_by": ["category"], "chart": "pie"})
        self.assertEqual(spec.chart, "bar")  # 9 categories
        spec, _ = repair({"group_by": ["parent_group"], "chart": "pie"})
        self.assertEqual(spec.chart, "pie")
        spec, _ = repair({"group_by": ["category"], "chart": "pie", "top_n": 5})
        self.assertEqual(spec.chart, "pie")
        spec, _ = repair({"group_by": ["category"], "chart": "stacked_bar"})
        self.assertEqual(spec.chart, "bar")

    def test_merchant_grouping_gets_top_10(self):
        spec, notes = repair({"group_by": ["merchant"]})
        self.assertEqual(spec.top_n, 10)
        self.assertTrue(any("top 10" in note for note in notes))
        spec, _ = repair({"group_by": ["merchant"], "top_n": 5})
        self.assertEqual(spec.top_n, 5)

    def test_custom_dates_clamped_and_explicit_dates_become_custom(self):
        spec, notes = repair({"period": "custom", "date_from": "2019-01", "date_to": "2030-12-31"})
        self.assertEqual((spec.date_from, spec.date_to), (DATA_MIN.isoformat(), DATA_MAX.isoformat()))
        self.assertTrue(any("limited" in note for note in notes))
        spec, notes = repair({"period": "last_month", "date_from": "2026-01", "date_to": "2026-03"})
        self.assertEqual((spec.period, spec.date_from, spec.date_to), ("custom", "2026-01-01", "2026-03-31"))
        with self.assertRaises(SpecError):
            repair({"period": "custom", "date_from": "2019-01", "date_to": "2019-06"})

    def test_transform_rules(self):
        spec, notes = repair({"group_by": ["category"], "transform": "cumulative"})
        self.assertEqual(spec.transform, "none")
        spec, _ = repair({"group_by": ["month"], "transform": "rolling_3", "compare_to": "previous_period"})
        self.assertEqual((spec.transform, spec.compare_to), ("none", "previous_period"))
        spec, _ = repair({"group_by": ["month"], "transform": "rolling_3"})
        self.assertEqual(spec.transform, "rolling_3")


if __name__ == "__main__":
    unittest.main()
