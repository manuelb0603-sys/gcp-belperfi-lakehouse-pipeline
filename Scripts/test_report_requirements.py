import json
import unittest

from finance.agent_harness import REQUIREMENTS_HEADER, HarnessLimits, ReportHarness
from finance.report_metrics import calculate_report_snapshot
from finance.report_requirements import (
    REQUIRED_SECTION_HEADINGS,
    build_report_requirements,
    format_money,
)


LATEST = "2026-08-01"
PREVIOUS = "2026-07-01"


def row(month, category, parent_group, amount):
    return {
        "transaction_month": month,
        "category": category,
        "parent_group": parent_group,
        "net_real_expense": amount,
    }


def real_run_rows():
    # Reproduces the real-run goal numbers: total 3269.84, eligible discretionary 1603.17.
    return [
        row(LATEST, "Groceries", "Needs", 1100.00),
        row(LATEST, "Transport", "Needs", 600.00),
        row(LATEST, "Health", "Needs", -33.33),
        row(LATEST, "Food & Drink", "Wants", 812.40),
        row(LATEST, "Shopping", "Wants", 540.27),
        row(LATEST, "Entertainment", "Wants", 250.50),
        row(PREVIOUS, "Groceries", "Needs", 1000.00),
        row(PREVIOUS, "Transport", "Needs", 600.00),
        row(PREVIOUS, "Health", "Needs", 20.00),
        row(PREVIOUS, "Food & Drink", "Wants", 700.00),
        row(PREVIOUS, "Shopping", "Wants", 600.00),
        row(PREVIOUS, "Entertainment", "Wants", 250.50),
    ]


def small_rows():
    return [
        row(LATEST, "Groceries", "Needs", 600),
        row(LATEST, "Food & Drink", "Wants", 300),
        row(LATEST, "Shopping", "Wants", 200),
        row(LATEST, "Entertainment", "Wants", 100),
        row(PREVIOUS, "Groceries", "Needs", 500),
        row(PREVIOUS, "Food & Drink", "Wants", 250),
        row(PREVIOUS, "Shopping", "Wants", 260),
        row(PREVIOUS, "Entertainment", "Wants", 100),
    ]


def text(lines):
    return "\n".join(lines)


class FormatMoneyTests(unittest.TestCase):
    def test_money_style(self):
        self.assertEqual(format_money(1234.5), "$1,234.50")
        self.assertEqual(format_money(-1769.84), "-$1,769.84")
        self.assertEqual(format_money(0), "$0.00")


class ReportRequirementsTests(unittest.TestCase):
    def test_real_run_shortfall_with_cuts_insufficient(self):
        snapshot = calculate_report_snapshot(real_run_rows(), 5000, 3500, savings_goal=2000)
        self.assertEqual(snapshot.disposable_cash, -1769.84)
        self.assertEqual(snapshot.savings_shortfall, 3769.84)
        self.assertEqual(snapshot.eligible_discretionary_total, 1603.17)
        self.assertEqual(snapshot.remaining_gap_after_cuts, 2166.67)
        self.assertTrue(snapshot.discretionary_cuts_insufficient)
        lines = build_report_requirements(snapshot)
        body = text(lines)
        self.assertLessEqual(len(lines), 20)
        self.assertIn(
            "discretionary cuts alone cannot reach the $2,000.00 savings target", body
        )
        self.assertIn("remaining gap after cutting all discretionary spend is $2,166.67", body)
        self.assertIn("shortfall $3,769.84", body)
        self.assertIn("eligible discretionary spend $1,603.17", body)
        # Sushi Stack values with month labels, direction word and the sign kept.
        self.assertIn("August 2026 total net expense $3,269.84", body)
        self.assertIn("July 2026 total net expense $3,170.50", body)
        self.assertIn("month-over-month change 3.13% increase", body)
        self.assertIn("disposable cash -$1,769.84", body)
        self.assertIn("keep the minus sign", body)
        # Top three rows: $0.00 caps with no percent.
        top_lines = [line for line in lines if line[:3] in {"1. ", "2. ", "3. "}]
        self.assertEqual([line.split(":")[0] for line in top_lines], ["1. Food & Drink", "2. Shopping", "3. Entertainment"])
        for line in top_lines:
            self.assertIn("advisor-suggested cap $0.00", line)
            self.assertIn("no percent of cap", line)
            self.assertNotIn("percent of cap 0", line)
        self.assertIn("1. Food & Drink: current $812.40; previous (July 2026) $700.00; MoM 16.06% increase", body)
        # Snapshot bars: every category with exact amount; refund labeled; Needs before Wants.
        self.assertIn("Needs: Groceries $1,100.00, Health -$33.33 (label as refund/credit), Transport $600.00.", body)
        self.assertIn("Wants: Entertainment $250.50, Food & Drink $812.40, Shopping $540.27.", body)
        self.assertLess(body.index("Needs:"), body.index("Wants:"))
        self.assertNotIn("proportionally", body)

    def test_headings_are_listed_in_order_with_exact_wording(self):
        snapshot = calculate_report_snapshot(small_rows(), 3000, 1500, savings_goal=500)
        first = build_report_requirements(snapshot)[0]
        self.assertIn("greeting", first)
        positions = [first.index(f'"{heading}"') for heading in REQUIRED_SECTION_HEADINGS]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(
            REQUIRED_SECTION_HEADINGS,
            (
                "Monthly Snapshot",
                "Spending Trends & Analysis",
                "Mogumogu's Financial Health Check",
                "The Sushi Stack",
                "Top 3 Discretionary Categories",
                "Mogumogu's Tips for the Week",
            ),
        )

    def test_shortfall_with_cuts_sufficient(self):
        # disposable 300; shortfall 200 < 600 eligible.
        snapshot = calculate_report_snapshot(small_rows(), 3000, 1500, savings_goal=500)
        body = text(build_report_requirements(snapshot))
        self.assertEqual(snapshot.goal_method, "shortfall_allocation")
        self.assertNotIn("cannot reach", body)
        self.assertIn("shortfall $200.00", body)
        self.assertIn("share the $200.00 shortfall proportionally", body)
        food = next(m for m in snapshot.categories if m.category == "Food & Drink")
        self.assertIn(f"advisor-suggested cap {format_money(food.suggested_cap)}", body)
        self.assertIn(f"percent of cap {food.percent_of_cap:.2f}%", body)

    def test_target_met_uses_trend_maintenance(self):
        snapshot = calculate_report_snapshot(small_rows(), 6000, 1500, savings_goal=500)
        body = text(build_report_requirements(snapshot))
        self.assertEqual(snapshot.goal_method, "trend_maintenance")
        self.assertIn("meet the $500.00 savings target (shortfall $0.00)", body)
        self.assertIn("trend-based maintenance caps", body)
        self.assertIn("1. Food & Drink: current $300.00; previous (July 2026) $250.00; MoM 20.00% increase; "
                      "advisor-suggested cap $250.00; percent of cap 120.00%.", body)
        self.assertNotIn("cannot reach", body)

    def test_no_savings_target(self):
        snapshot = calculate_report_snapshot(small_rows(), 6000, 1500, savings_goal=None)
        body = text(build_report_requirements(snapshot))
        self.assertIn("no numeric savings target is configured", body)
        self.assertIn("no cap and no percent; state why: No numeric savings target is configured.", body)
        self.assertNotIn("advisor-suggested cap $", body)

    def test_configured_goals(self):
        goals = {"Food & Drink": 280, "Shopping": 250, "Entertainment": 0}
        snapshot = calculate_report_snapshot(small_rows(), 6000, 1500, savings_goal=500, category_goals=goals)
        body = text(build_report_requirements(snapshot))
        self.assertEqual(snapshot.goal_method, "configured")
        self.assertIn("Label every cap as a configured goal", body)
        self.assertIn("configured goal $280.00; percent of cap 107.14%", body)
        self.assertIn("configured goal $0.00; no percent of cap", body)
        self.assertNotIn("advisor-suggested cap $", body)

    def test_no_eligible_discretionary_categories(self):
        rows = [row(LATEST, "Rent", "Needs", 1000)]
        snapshot = calculate_report_snapshot(rows, 3000, 500, savings_goal=100)
        body = text(build_report_requirements(snapshot))
        self.assertIn("no eligible Wants/Discretionary categories", body)
        self.assertIn("previous month total net expense not available", body)


class RequirementsPromptWiringTests(unittest.TestCase):
    def _run(self, requirements, responses):
        prompts = []

        def fake_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        harness = ReportHarness(
            "http://ollama", "test-model", limits=HarnessLimits(max_validation_cycles=2), query=fake_query
        )
        snapshot = calculate_report_snapshot(real_run_rows(), 5000, 3500, savings_goal=2000)
        _, decision, trace = harness.run(
            "objective", snapshot, "author context", {}, report_requirements=requirements
        )
        calls = [
            (event["role"], event["prompt"]) for event in trace.events
            if event.get("event") == "model_call" and event.get("role") in {"author", "validator"}
        ]
        self.assertEqual(len(calls), len(prompts))
        return decision, calls, snapshot

    def test_checklist_appears_once_in_author_revision_and_validator_prompts(self):
        fail = json.dumps({
            "decision": "fail",
            "summary": "Heading renamed.",
            "findings": [{
                "severity": "error",
                "category": "completeness",
                "message": "Heading renamed",
                "evidence": 'report "August Spending Overview" vs checklist "Monthly Snapshot"',
                "revision_instruction": "Use the exact heading.",
            }],
        })
        passed = json.dumps({"decision": "pass", "summary": "ok", "findings": []})
        report = "<<<REPORT_HTML>>>\n<html><body><h2>Monthly Snapshot</h2></body></html>\n<<<END_REPORT_HTML>>>"
        snapshot = calculate_report_snapshot(real_run_rows(), 5000, 3500, savings_goal=2000)
        requirements = build_report_requirements(snapshot)
        decision, calls, _ = self._run(requirements, [report, fail, report, passed])
        self.assertTrue(decision.passed)
        self.assertEqual([role for role, _ in calls], ["author", "validator", "author", "validator"])
        for role, prompt in calls:
            self.assertEqual(prompt.count(REQUIREMENTS_HEADER), 1, role)
            self.assertIn("remaining gap after cutting all discretionary spend is $2,166.67", prompt)
            # The checklist follows the snapshot.
            self.assertLess(prompt.index("AUTHORITATIVE SNAPSHOT"), prompt.index(REQUIREMENTS_HEADER))
        validator_prompts = [prompt for role, prompt in calls if role == "validator"]
        for prompt in validator_prompts:
            self.assertIn("Check every checklist item above explicitly", prompt)
        author_prompts = [prompt for role, prompt in calls if role == "author"]
        self.assertIn("PREVIOUS REPORT HTML", author_prompts[1])
        self.assertNotIn("Check every checklist item above explicitly", author_prompts[0])

    def test_harness_works_without_checklist(self):
        passed = json.dumps({"decision": "pass", "summary": "ok", "findings": []})
        report = "<<<REPORT_HTML>>>\n<html><body>ok</body></html>\n<<<END_REPORT_HTML>>>"
        decision, calls, _ = self._run(None, [report, passed])
        self.assertTrue(decision.passed)
        for _, prompt in calls:
            self.assertNotIn(REQUIREMENTS_HEADER, prompt)


if __name__ == "__main__":
    unittest.main()
