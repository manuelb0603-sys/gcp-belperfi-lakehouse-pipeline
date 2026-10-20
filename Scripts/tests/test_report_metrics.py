import json
import unittest

from finance.report_metrics import calculate_report_snapshot


LATEST = "2026-08-01"
PREVIOUS = "2026-07-01"


def row(month, category, parent_group, amount):
    return {
        "transaction_month": month,
        "category": category,
        "parent_group": parent_group,
        "net_real_expense": amount,
    }


def by_key(snapshot):
    return {(item.category, item.parent_group): item for item in snapshot.categories}


class ReportMetricsGoalTests(unittest.TestCase):
    def rows(self):
        return [
            row(LATEST, "Groceries", "Needs", 600),
            row(LATEST, "Food & Drink", "Wants", 300),
            row(LATEST, "Shopping", "Wants", 200),
            row(LATEST, "Entertainment", "Wants", 100),
            row(LATEST, "Travel", "Wants", 50),
            row(PREVIOUS, "Groceries", "Needs", 500),
            row(PREVIOUS, "Food & Drink", "Wants", 250),
            row(PREVIOUS, "Shopping", "Wants", 260),
            row(PREVIOUS, "Entertainment", "Wants", 100),
        ]

    def test_shortfall_is_allocated_across_all_discretionary_categories(self):
        # Total 1250; disposable = 3000 - (1250 + 1500) = 250; shortfall = 500 - 250 = 250.
        snapshot = calculate_report_snapshot(self.rows(), 3000, 1500, savings_goal=500)
        self.assertEqual(snapshot.projected_savings, 250)
        self.assertEqual(snapshot.savings_shortfall, 250)
        self.assertEqual(snapshot.eligible_discretionary_total, 650)
        self.assertEqual(snapshot.goal_method, "shortfall_allocation")
        self.assertFalse(snapshot.discretionary_cuts_insufficient)
        self.assertEqual(snapshot.remaining_gap_after_cuts, 0)
        items = by_key(snapshot)
        food = items[("Food & Drink", "Wants")]
        self.assertEqual(food.suggested_cap, round(300 - 250 * 300 / 650, 2))
        self.assertEqual(food.cap_source, "suggested")
        self.assertEqual(food.percent_of_cap, round(300 / food.suggested_cap * 100, 2))
        travel = items[("Travel", "Wants")]
        self.assertEqual(travel.suggested_cap, round(50 - 250 * 50 / 650, 2))
        caps = sum(item.suggested_cap for item in snapshot.categories if item.is_discretionary)
        self.assertAlmostEqual(caps, 650 - 250, places=1)
        groceries = items[("Groceries", "Needs")]
        self.assertFalse(groceries.is_discretionary)
        self.assertIsNone(groceries.suggested_cap)
        self.assertEqual(snapshot.top_discretionary_categories, ["Food & Drink", "Shopping", "Entertainment"])

    def test_shortfall_larger_than_discretionary_spend_zeroes_caps(self):
        snapshot = calculate_report_snapshot(self.rows(), 2000, 1500, savings_goal=1000)
        # disposable = 2000 - 2750 = -750; shortfall = 1750 > 650 discretionary.
        self.assertEqual(snapshot.savings_shortfall, 1750)
        self.assertTrue(snapshot.discretionary_cuts_insufficient)
        self.assertEqual(snapshot.remaining_gap_after_cuts, 1100)
        for item in snapshot.categories:
            if item.is_discretionary:
                self.assertEqual(item.suggested_cap, 0)
                self.assertIsNone(item.percent_of_cap)

    def test_target_met_uses_trend_maintenance_caps(self):
        snapshot = calculate_report_snapshot(self.rows(), 6000, 1500, savings_goal=500)
        self.assertEqual(snapshot.savings_shortfall, 0)
        self.assertEqual(snapshot.goal_method, "trend_maintenance")
        self.assertFalse(snapshot.discretionary_cuts_insufficient)
        items = by_key(snapshot)
        self.assertEqual(items[("Food & Drink", "Wants")].suggested_cap, 250)  # increased
        self.assertEqual(items[("Food & Drink", "Wants")].mom_direction, "increase")
        self.assertEqual(items[("Shopping", "Wants")].suggested_cap, 200)  # decreased
        self.assertEqual(items[("Shopping", "Wants")].mom_direction, "decrease")
        self.assertEqual(items[("Entertainment", "Wants")].suggested_cap, 100)
        self.assertEqual(items[("Entertainment", "Wants")].mom_direction, "flat")
        self.assertEqual(items[("Travel", "Wants")].suggested_cap, 50)  # no prior spend

    def test_configured_goal_takes_precedence(self):
        snapshot = calculate_report_snapshot(
            self.rows(), 3000, 1500, savings_goal=500, category_goals={"Shopping": 160}
        )
        shopping = by_key(snapshot)[("Shopping", "Wants")]
        self.assertEqual(shopping.cap_source, "configured")
        self.assertEqual(shopping.suggested_cap, 160)
        self.assertEqual(shopping.percent_of_cap, 125)
        self.assertEqual(shopping.goal_amount, 160)
        self.assertEqual(shopping.goal_percent, 125)
        self.assertEqual(snapshot.goal_method, "shortfall_allocation")

    def test_all_discretionary_configured_reports_configured_method(self):
        goals = {"Food & Drink": 280, "Shopping": 180, "Entertainment": 90, "Travel": 40}
        snapshot = calculate_report_snapshot(self.rows(), 3000, 1500, savings_goal=500, category_goals=goals)
        self.assertEqual(snapshot.goal_method, "configured")

    def test_no_savings_target(self):
        snapshot = calculate_report_snapshot(self.rows(), 3000, 1500)
        self.assertEqual(snapshot.goal_method, "no_savings_target")
        self.assertIsNone(snapshot.savings_shortfall)
        self.assertIsNone(snapshot.remaining_gap_after_cuts)
        for item in snapshot.categories:
            self.assertIsNone(item.suggested_cap)
            self.assertIsNone(item.cap_source)

    def test_refunds_do_not_distort_bar_scale(self):
        rows = self.rows() + [row(LATEST, "Payments", "Payments & Others", -900)]
        snapshot = calculate_report_snapshot(rows, 3000, 1500, savings_goal=500)
        self.assertEqual(snapshot.max_positive_spend, 600)
        items = by_key(snapshot)
        refund = items[("Payments", "Payments & Others")]
        self.assertTrue(refund.is_refund)
        self.assertEqual(refund.bar_width_percent, 0)
        self.assertEqual(items[("Groceries", "Needs")].bar_width_percent, 100)
        self.assertEqual(items[("Food & Drink", "Wants")].bar_width_percent, 50)
        self.assertEqual(items[("Travel", "Wants")].bar_width_percent, round(50 / 600 * 100, 2))

    def test_negative_discretionary_spend_is_not_capped_or_eligible(self):
        rows = self.rows() + [row(LATEST, "Gifts & Donations", "Wants", -40)]
        snapshot = calculate_report_snapshot(rows, 3000, 1500, savings_goal=500)
        gifts = by_key(snapshot)[("Gifts & Donations", "Wants")]
        self.assertIsNone(gifts.suggested_cap)
        self.assertEqual(snapshot.eligible_discretionary_total, 650)
        self.assertNotIn("Gifts & Donations", snapshot.top_discretionary_categories)

    def test_same_category_in_two_groups(self):
        rows = self.rows() + [
            row(LATEST, "Health & Wellness", "Needs", 80),
            row(LATEST, "Health & Wellness", "Wants", 400),
        ]
        snapshot = calculate_report_snapshot(rows, 3000, 1500, savings_goal=500)
        items = by_key(snapshot)
        self.assertFalse(items[("Health & Wellness", "Needs")].is_discretionary)
        self.assertIsNone(items[("Health & Wellness", "Needs")].suggested_cap)
        self.assertTrue(items[("Health & Wellness", "Wants")].is_discretionary)
        self.assertIsNotNone(items[("Health & Wellness", "Wants")].suggested_cap)
        self.assertEqual(snapshot.top_discretionary_categories[0], "Health & Wellness")

    def test_single_month_has_no_prior_direction(self):
        rows = [row(LATEST, "Shopping", "Wants", 100)]
        snapshot = calculate_report_snapshot(rows, 6000, 1500, savings_goal=500)
        shopping = snapshot.categories[0]
        self.assertEqual(shopping.mom_direction, "no_prior")
        self.assertEqual(shopping.suggested_cap, 100)

    def test_snapshot_is_json_serializable(self):
        snapshot = calculate_report_snapshot(self.rows(), 3000, 1500, savings_goal=500)
        payload = json.loads(json.dumps(snapshot.to_dict()))
        self.assertIn("bar_width_percent", payload["categories"][0])
        self.assertIn("top_discretionary_categories", payload)


class NotableChangeTests(unittest.TestCase):
    def notable(self, rows, **kwargs):
        snapshot = calculate_report_snapshot(rows, 5000, 1000, savings_goal=500, **kwargs)
        return {item.category: item for item in snapshot.notable_changes}, snapshot

    def test_percent_only_change_is_notable(self):
        # +60% but only +$60: meets the percent threshold, not the dollar threshold.
        found, _ = self.notable([row(LATEST, "Books", "Wants", 160), row(PREVIOUS, "Books", "Wants", 100)])
        item = found["Books"]
        self.assertEqual((item.change_usd, item.change_pct, item.direction), (60.0, 60.0, "increase"))
        self.assertEqual((item.current_spend, item.previous_spend, item.parent_group), (160.0, 100.0, "Wants"))

    def test_dollar_only_change_is_notable(self):
        # -$300 is only -10%: meets the dollar threshold, not the percent threshold.
        found, _ = self.notable([row(LATEST, "Rent", "Needs", 2700), row(PREVIOUS, "Rent", "Needs", 3000)])
        self.assertEqual((found["Rent"].change_usd, found["Rent"].change_pct, found["Rent"].direction), (-300.0, -10.0, "decrease"))

    def test_change_meeting_both_thresholds_and_one_below_both(self):
        found, _ = self.notable([
            row(LATEST, "Groceries", "Needs", 1000), row(PREVIOUS, "Groceries", "Needs", 500),
            row(LATEST, "Gas", "Needs", 120), row(PREVIOUS, "Gas", "Needs", 100),
        ])
        self.assertEqual(found["Groceries"].change_pct, 100.0)
        self.assertNotIn("Gas", found)

    def test_previous_zero_has_no_percent_and_needs_the_dollar_threshold(self):
        found, _ = self.notable([
            row(LATEST, "Travel", "Wants", 400), row(PREVIOUS, "Travel", "Wants", 0),
            row(LATEST, "Gifts", "Wants", 40), row(PREVIOUS, "Gifts", "Wants", 0),
            row(LATEST, "Rent", "Needs", 1), row(PREVIOUS, "Rent", "Needs", 1),
        ])
        self.assertIsNone(found["Travel"].change_pct)
        self.assertEqual(found["Travel"].change_usd, 400.0)
        self.assertNotIn("Gifts", found)

    def test_no_previous_month_or_trivial_base_gives_no_notable_changes(self):
        _, snapshot = self.notable([row(LATEST, "Travel", "Wants", 900)])
        self.assertEqual(snapshot.notable_changes, [])
        found, _ = self.notable([row(LATEST, "Refunds", "Needs", -400), row(PREVIOUS, "Refunds", "Needs", -10), row(PREVIOUS, "Rent", "Needs", 1)])
        self.assertNotIn("Refunds", found)

    def test_sorted_by_absolute_dollar_change_and_capped_at_five(self):
        rows = []
        for index, delta in enumerate([300, -900, 500, 260, -700, 1000, 400]):
            rows += [row(LATEST, f"Cat{index}", "Wants", 1000 + delta), row(PREVIOUS, f"Cat{index}", "Wants", 1000)]
        _, snapshot = self.notable(rows)
        self.assertEqual([item.change_usd for item in snapshot.notable_changes], [1000.0, -900.0, -700.0, 500.0, 400.0])

    def test_custom_thresholds(self):
        rows = [row(LATEST, "Books", "Wants", 160), row(PREVIOUS, "Books", "Wants", 100)]
        found, _ = self.notable(rows, notable_change_pct=75, notable_change_usd=100)
        self.assertEqual(found, {})
        found, _ = self.notable(rows, notable_change_pct=75, notable_change_usd=50)
        self.assertIn("Books", found)

    def test_notable_changes_are_in_the_snapshot_json(self):
        _, snapshot = self.notable([row(LATEST, "Books", "Wants", 160), row(PREVIOUS, "Books", "Wants", 100)])
        payload = json.loads(json.dumps(snapshot.to_dict()))
        self.assertEqual(payload["notable_changes"][0]["category"], "Books")
        self.assertEqual(set(payload["notable_changes"][0]), {
            "category", "parent_group", "current_spend", "previous_spend", "change_usd", "change_pct", "direction",
        })


if __name__ == "__main__":
    unittest.main()
