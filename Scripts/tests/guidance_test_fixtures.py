"""Scripted harness runs and snapshots for test_guidance.py and its strict golden prompt capture.

Uses only APIs that existed before the guidance setting, so ``capture_strict_prompts()`` can be
run against older code to regenerate ``testdata/strict_prompts_golden.json``:
    python -c "import sys; sys.path.insert(0, 'Scripts'); from tests import guidance_test_fixtures as f; f.write_golden()"
"""

import json
from datetime import date
from functools import partial
from pathlib import Path

from finance.harness import HarnessLimits, ReportHarness, ToolSpec
from finance.model_clients import ModelReply
from finance.report_metrics import calculate_report_snapshot
from finance.report_requirements import build_report_requirements

GOLDEN_PATH = Path(__file__).resolve().parent / "testdata" / "strict_prompts_golden.json"

LATEST, PREVIOUS = "2026-08-01", "2026-07-01"
AS_OF = date(2026, 9, 1)


def _row(month, category, group, amount):
    return {"transaction_month": month, "category": category, "parent_group": group, "net_real_expense": amount}


def cuts_insufficient_rows():
    # Total 3269.84, eligible discretionary 1603.17; with deposit 5000, fixed 3500, target 2000 the
    # shortfall 3769.84 exceeds the eligible total (remaining gap 2166.67).
    return [
        _row(LATEST, "Groceries", "Needs", 1100.00), _row(LATEST, "Transport", "Needs", 600.00),
        _row(LATEST, "Health", "Needs", -33.33), _row(LATEST, "Food & Drink", "Wants", 812.40),
        _row(LATEST, "Shopping", "Wants", 540.27), _row(LATEST, "Entertainment", "Wants", 250.50),
        _row(PREVIOUS, "Groceries", "Needs", 1000.00), _row(PREVIOUS, "Transport", "Needs", 600.00),
        _row(PREVIOUS, "Health", "Needs", 20.00), _row(PREVIOUS, "Food & Drink", "Wants", 700.00),
        _row(PREVIOUS, "Shopping", "Wants", 600.00), _row(PREVIOUS, "Entertainment", "Wants", 250.50),
    ]


def small_rows():
    return [
        _row(LATEST, "Groceries", "Needs", 600), _row(LATEST, "Food & Drink", "Wants", 300),
        _row(LATEST, "Shopping", "Wants", 200), _row(LATEST, "Entertainment", "Wants", 100),
        _row(PREVIOUS, "Groceries", "Needs", 500), _row(PREVIOUS, "Food & Drink", "Wants", 250),
        _row(PREVIOUS, "Shopping", "Wants", 260), _row(PREVIOUS, "Entertainment", "Wants", 100),
    ]


def scenarios():
    """{name: snapshot} covering every goal method (as_of pinned so the golden does not depend on today)."""
    snap = partial(calculate_report_snapshot, as_of=AS_OF)
    return {
        "cuts_insufficient": snap(cuts_insufficient_rows(), 5000, 3500, savings_goal=2000),
        "shortfall_allocation": snap(small_rows(), 3000, 1500, savings_goal=500),
        "trend_maintenance": snap(small_rows(), 6000, 1500, savings_goal=500),
        "no_target": snap(small_rows(), 6000, 1500, savings_goal=None),
        "configured": snap(
            small_rows(), 6000, 1500, savings_goal=500, category_goals={"Food & Drink": 280, "Shopping": 250, "Entertainment": 0}
        ),
        "no_eligible": snap([_row(LATEST, "Rent", "Needs", 1000)], 3000, 500, savings_goal=100),
    }


REPORT = "<<<REPORT_HTML>>>\n<html><body><h2>Monthly Snapshot</h2><p>Food & Drink $812.40</p></body></html>\n<<<END_REPORT_HTML>>>"
REBUTTED_REPORT = REPORT + '\n<<<REBUTTALS>>>\n[{"finding": "Missing gap", "quote": "Food & Drink $812.40"}]\n<<<END_REBUTTALS>>>'
TOOL_REQUEST = json.dumps({"tool_calls": [{"name": "get_spending_summary", "arguments": {}}]})
PASS = json.dumps({"decision": "pass", "summary": "ok", "findings": []})
# Error categories in non-priority order, so strict keeps arithmetic, completeness, evidence.
FIVE_ERRORS = [
    ("visual_design", "Bars not pink"),
    ("arithmetic", "Shortfall wrong for Shopping"),
    ("completeness", "Missing remaining gap"),
    ("usability", "Low contrast"),
    ("evidence", "Unsupported coffee claim"),
]


def finding(category, message, severity="error"):
    return {"severity": severity, "category": category, "message": message,
            "evidence": f'report "x" vs snapshot.disposable_cash ({message})', "revision_instruction": "Fix it."}


FAIL_FIVE = json.dumps({"decision": "fail", "summary": "Five errors.", "findings": [
    *(finding(category, message) for category, message in FIVE_ERRORS),
    finding("visual_design", "Minor spacing", severity="warning"),
]})
FAIL_ONE = json.dumps({"decision": "fail", "summary": "One error.", "findings": [finding("completeness", "Missing remaining gap")]})


class ScriptedClient:
    """A ModelClient fake: records {"role", "system", "user"} and replays scripted replies."""

    provider = "fake"

    def __init__(self, model, replies, calls):
        self.model = model
        self.replies = replies
        self.calls = calls

    def generate(self, prompt, *, system=None, json_schema=None, timeout=None):
        self.calls.append({"role": self.model, "system": system, "user": prompt})
        return ModelReply(self.replies.pop(0), done_reason="stop")


def run_scripted(snapshot, requirements, limits=None, **harness_kwargs):
    """Initial tool round, report, five-error review, rebutted revision, one-error review, revision, pass.

    Returns (calls, decision, trace, harness).
    """
    calls = []
    harness = ReportHarness(
        limits=limits or HarnessLimits(max_validation_cycles=3),
        author_client=ScriptedClient("author", [TOOL_REQUEST, REPORT, REBUTTED_REPORT, REPORT], calls),
        validator_client=ScriptedClient("validator", [FAIL_FIVE, FAIL_ONE, PASS], calls),
        **harness_kwargs,
    )
    tools = {"get_spending_summary": ToolSpec("get_spending_summary", "Spending rows.", lambda _: {"rows": [1, 2]})}
    _, decision, trace = harness.run(
        "Write the August 2026 report.", snapshot, "AUTHOR CONTEXT", tools,
        validator_context="VALIDATOR CONTEXT", report_requirements=requirements,
    )
    return calls, decision, trace, harness


def capture_strict_prompts(**harness_kwargs):
    """Every strict checklist, and every prompt (system and user) of the scripted cuts-insufficient run."""
    snapshots = scenarios()
    captured = {"requirements": {name: build_report_requirements(snapshot) for name, snapshot in snapshots.items()}}
    for name in ("cuts_insufficient",):
        snapshot = snapshots[name]
        calls, decision, _, _ = run_scripted(snapshot, build_report_requirements(snapshot), **harness_kwargs)
        captured[name] = {"decision": decision.decision, "calls": calls}
    return captured


def write_golden(path=GOLDEN_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(capture_strict_prompts(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
