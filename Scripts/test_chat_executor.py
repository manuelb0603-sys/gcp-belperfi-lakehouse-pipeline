import json
import math
import unittest
from collections import defaultdict
from datetime import date
from decimal import Decimal

from chat_test_fixtures import ROWS, TODAY, fixture_frame
from finance.chat.analysis_spec import AnalysisSpec
from finance.chat.executor import AnalysisContext, check_narrative_numbers, run_analysis


def run(**spec):
    return run_analysis(FRAME, AnalysisSpec(**spec), AnalysisContext(today=TODAY))


def records(result, *columns):
    table = result.table
    columns = columns or tuple(table.columns)
    out = []
    for row in table[list(columns)].itertuples(index=False):
        out.append(tuple(None if isinstance(v, float) and math.isnan(v) else (float(v) if not isinstance(v, str) else v) for v in row))
    return out


FRAME = fixture_frame()


class TestMetrics(unittest.TestCase):
    def test_net_spend_by_month(self):
        result = run(metric="net_spend", group_by=["month"], period="last_quarter")
        self.assertEqual(records(result), [("2026-07", 301.5), ("2026-08", 490.5), ("2026-09", 470.0)])
        self.assertEqual(result.chart, "line")
        self.assertEqual(result.summary["total"], 1262.0)
        self.assertEqual(result.summary["transactions"], 29)
        self.assertEqual(result.summary["largest_change"]["change"], 189.0)

    def test_each_metric_by_category_last_quarter(self):
        expected = {
            "net_spend": {"Groceries": 490.0, "Shopping": 410.0, "Food & Drink": 202.0, "Automotive": 170.0, "Other": 5.0, "Payments": -15.0},
            "gross_spend": {"Groceries": 490.0, "Shopping": 490.0, "Food & Drink": 202.0, "Automotive": 170.0, "Other": 5.0, "Payments": 0.0},
            "refunds": {"Shopping": 80.0, "Payments": 15.0, "Groceries": 0.0, "Food & Drink": 0.0, "Automotive": 0.0, "Other": 0.0},
            "transaction_count": {"Groceries": 8, "Food & Drink": 9, "Shopping": 6, "Automotive": 4, "Other": 1, "Payments": 1},
            "avg_transaction": {"Groceries": 61.25, "Shopping": 68.33, "Food & Drink": 22.44, "Automotive": 42.5, "Other": 5.0, "Payments": -15.0},
        }
        for metric, values in expected.items():
            result = run(metric=metric, group_by=["category"], period="last_quarter")
            self.assertEqual(dict(records(result)), {k: float(v) for k, v in values.items()}, metric)
        totals = {"net_spend": 1262.0, "gross_spend": 1357.0, "refunds": 95.0, "transaction_count": 29, "avg_transaction": 43.52}
        for metric, total in totals.items():
            self.assertEqual(run(metric=metric, period="last_quarter").summary["total"], total, metric)

    def test_default_sort_is_largest_first(self):
        result = run(metric="net_spend", group_by=["category"], period="last_quarter")
        self.assertEqual([row[0] for row in records(result)][:4], ["Groceries", "Shopping", "Food & Drink", "Automotive"])
        ascending = run(metric="net_spend", group_by=["category"], period="last_quarter", sort="asc")
        self.assertEqual(records(ascending)[0], ("Payments", -15.0))

    def test_share_of_total(self):
        result = run(metric="share_of_total", group_by=["parent_group"], period="last_quarter")
        self.assertEqual(records(result), [("Needs", 52.3), ("Wants", 48.5), ("Payments & Others", -0.8)])
        self.assertAlmostEqual(sum(v for _, v in records(result)), 100.0, delta=0.2)
        self.assertEqual(result.unit, "pct")

    def test_no_dimensions_gives_single_value_table(self):
        result = run(metric="avg_transaction", period="last_quarter", filters={"category": ["Groceries"]})
        self.assertEqual(records(result), [(61.25,)])
        self.assertEqual(result.chart, "table")


class TestGrids(unittest.TestCase):
    def test_zero_fill_months(self):
        result = run(metric="net_spend", group_by=["month"], period="last_6_months", filters={"category": ["Automotive"]})
        self.assertEqual(records(result), [
            ("2026-04", 0.0), ("2026-05", 0.0), ("2026-06", 0.0), ("2026-07", 40.0), ("2026-08", 45.0), ("2026-09", 85.0),
        ])

    def test_weekday_grid_is_ordered_and_complete(self):
        result = run(metric="transaction_count", group_by=["weekday"], period="last_month")
        self.assertEqual(records(result), [("Mon", 2.0), ("Tue", 3.0), ("Wed", 1.0), ("Thu", 0.0), ("Fri", 2.0), ("Sat", 2.0), ("Sun", 2.0)])

    def test_two_dimensions_zero_fill_product(self):
        result = run(metric="gross_spend", group_by=["month", "category"], period="last_quarter")
        self.assertEqual(len(result.table), 3 * 6)
        rows = {(m, c): v for m, c, v in records(result)}
        self.assertEqual(rows[("2026-07", "Other")], 0.0)
        self.assertEqual(rows[("2026-09", "Shopping")], 210.0)
        self.assertEqual(result.chart, "stacked_bar")

    def test_heatmap_auto_for_weekday_by_month(self):
        result = run(metric="transaction_count", group_by=["month", "weekday"], period="last_quarter")
        self.assertEqual(result.chart, "heatmap")
        self.assertEqual(len(result.table), 21)
        self.assertEqual(result.table["transaction_count"].sum(), 29)

    def test_week_and_quarter_labels(self):
        result = run(metric="net_spend", group_by=["quarter"], period="custom", date_from="2026-06", date_to="2026-09")
        self.assertEqual(records(result), [("2026-Q2", 70.0), ("2026-Q3", 1262.0)])
        weeks = run(metric="net_spend", group_by=["week"], period="last_month")
        self.assertEqual(weeks.table["week"].iloc[0], "2026-08-31")
        self.assertEqual(weeks.table["net_spend"].sum(), 470.0)


class TestTopNAndFilters(unittest.TestCase):
    def test_top_n_folds_rest_into_other(self):
        result = run(metric="net_spend", group_by=["merchant"], period="last_quarter", top_n=3)
        self.assertEqual(records(result), [("Amazon", 410.0), ("Wholefds Mkt", 310.0), ("Sushi Zen", 155.0), ("Other", 387.0)])
        self.assertIn("Top 3 shown", result.caption)

    def test_top_n_with_time_dimension_keeps_other_series(self):
        result = run(metric="net_spend", group_by=["month", "category"], period="last_quarter", top_n=2)
        self.assertEqual(sorted(result.table["category"].unique()), ["Groceries", "Other", "Shopping"])
        self.assertEqual(result.table["net_spend"].sum(), 1262.0)

    def test_payments_are_excluded(self):
        self.assertNotIn("Card Payment", run(group_by=["merchant"], period="all", top_n=25).table["merchant"].tolist())
        result = run(metric="transaction_count", period="all")
        self.assertEqual(result.summary["total"], len([r for r in ROWS if r[4] != "Payment"]))
        empty = run(period="all", filters={"merchant": ["Card Payment"]})
        self.assertEqual(empty.summary["total"], 0.0)
        self.assertIn("No matching transactions in this period.", empty.notes)

    def test_filters(self):
        self.assertEqual(run(period="last_quarter", filters={"merchant_contains": "star"}).summary["total"], 21.5)
        self.assertEqual(run(period="last_quarter", filters={"issuer": ["Amex"]}).summary["total"], 430.0)
        self.assertEqual(run(period="last_quarter", filters={"exclude_category": ["Groceries", "Shopping"]}).summary["total"], 362.0)
        self.assertEqual(run(metric="transaction_count", period="last_quarter", filters={"min_amount": 100}).summary["total"], 4)
        self.assertEqual(run(metric="transaction_count", period="last_quarter", filters={"max_amount": 9}).summary["total"], 6)
        self.assertEqual(run(period="last_quarter", filters={"merchant": ["sushi zen"]}).summary["total"], 155.0)
        self.assertIn("Filters: card is Amex.", run(period="last_quarter", filters={"issuer": ["Amex"]}).caption)


class TestTransformsAndComparisons(unittest.TestCase):
    def test_cumulative(self):
        result = run(group_by=["month"], period="last_quarter", transform="cumulative")
        self.assertEqual(result.value_column, "net_spend_cumulative")
        self.assertEqual(records(result, "month", "net_spend_cumulative"), [("2026-07", 301.5), ("2026-08", 792.0), ("2026-09", 1262.0)])

    def test_rolling_and_pct_change(self):
        rolling = run(group_by=["month"], period="last_6_months", transform="rolling_3")
        self.assertEqual([v for _, v in records(rolling, "month", "net_spend_rolling_3")], [None, None, 23.33, 123.83, 287.33, 420.67])
        pct = run(group_by=["month"], period="last_6_months", transform="pct_change")
        self.assertEqual([v for _, v in records(pct, "month", "net_spend_pct_change")], [None, None, None, 330.7, 62.7, -4.2])
        self.assertEqual(pct.unit, "pct")

    def test_previous_period_by_category(self):
        result = run(group_by=["category"], period="last_month", compare_to="previous_period")
        rows = {r[0]: r[1:] for r in records(result)}
        self.assertEqual(rows["Shopping"], (150.0, 200.0, -50.0, -25.0))
        self.assertEqual(rows["Payments"], (0.0, -15.0, 15.0, 100.0))
        self.assertEqual(result.summary["comparison"], {"label": "August 2026 (previous period)", "total": 490.5, "change": -20.5, "change_pct": -4.2})
        self.assertEqual(result.summary["largest_change"]["label"], "Shopping")
        self.assertIn("Compared with August 2026", result.caption)

    def test_previous_period_by_month_aligns_shifted_months(self):
        result = run(group_by=["month"], period="custom", date_from="2026-08", date_to="2026-09", compare_to="previous_period")
        self.assertEqual(records(result, "month", "net_spend", "comparison"), [("2026-08", 490.5, 70.0), ("2026-09", 470.0, 301.5)])


class TestReconciliation(unittest.TestCase):
    def test_matches_fact_monthly_spending_formulas(self):
        """Recompute gold fact_monthly_spending in plain Python and compare every row."""
        gold = defaultdict(lambda: {"gross": Decimal(0), "refunds": Decimal(0), "net": Decimal(0), "count": 0})
        for day, _issuer, _desc, amount, kind, category, _group in ROWS:
            if kind not in ("Return", "Sale", "Adjustment"):
                continue
            amount = Decimal(amount)
            row = gold[(day[:7], category)]
            row["gross"] += -amount if amount < 0 else 0
            row["refunds"] += amount if amount > 0 else 0
            row["net"] += -amount
            row["count"] += 1
        for metric, key in (("gross_spend", "gross"), ("refunds", "refunds"), ("net_spend", "net"),
                            ("transaction_count", "count"), ("avg_transaction", "avg")):
            result = run(metric=metric, group_by=["month", "category"], period="all")
            ours = {(m, c): v for m, c, v in records(result)}
            for group, values in gold.items():
                expected = round(float(values["net"]) / values["count"], 2) if key == "avg" else float(values[key])
                self.assertAlmostEqual(ours[group], expected, places=2, msg=(metric, group))
            extra = [v for g, v in ours.items() if g not in gold]
            self.assertTrue(all(v in (0.0, None) for v in extra), metric)


class TestSummaryAndNarrative(unittest.TestCase):
    def test_summary_is_compact(self):
        result = run(group_by=["merchant", "month"], period="all", top_n=25, filters={"exclude_category": ["Other"]})
        self.assertLessEqual(len(json.dumps(result.summary).encode("utf-8")), 1500)
        result = run(group_by=["week", "category"], period="all", compare_to="previous_period")
        self.assertLessEqual(len(json.dumps(result.summary).encode("utf-8")), 1500)

    def test_caption_names_metric_period_and_exclusions(self):
        result = run(metric="gross_spend", group_by=["category"], period="last_quarter")
        self.assertTrue(result.caption.startswith("Gross spend by category, Jul–Sep 2026 (last quarter)."))
        self.assertIn("Card payments are excluded", result.caption)

    def test_narrative_numbers(self):
        summary = run(group_by=["month"], period="last_quarter").summary
        text = ("You spent $1,262 over the quarter ($1.3k), peaking at $490.50 in August, up 63% (62.7%) from July's "
                "$301.5. Groceries alone were $999 and dining rose 12%.")
        self.assertEqual(check_narrative_numbers(text, summary), [
            "$999 does not match any value in the analysis summary.",
            "12% does not match any value in the analysis summary.",
        ])
        self.assertEqual(check_narrative_numbers("Spending fell by $189 (-$189.00).", summary), [])
        self.assertEqual(check_narrative_numbers("$1,262.40", summary), ["$1,262.40 does not match any value in the analysis summary."])
        self.assertEqual(check_narrative_numbers("No numbers here, just 3 months.", summary), [])


if __name__ == "__main__":
    unittest.main()
