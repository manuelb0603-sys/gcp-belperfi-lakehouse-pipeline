"""Claude spend backstop tests: pricing, cost ledger, budget guard, and every entry point's refusal.

Fake SDK clients and temp ledgers only: no Anthropic, Ollama, or BigQuery calls, and the real
agent/runs/cost_ledger.jsonl is never touched.
"""

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from finance import budget, pricing, replay, reporting, run_compare
from finance.budget import BudgetExceededError, BudgetGuard, CostLedger, LedgerUnavailableError
from finance.harness import HarnessLimits, ReportHarness
from finance.model_clients import AnthropicClient, EmptyModelResponse, OllamaClient, build_model_client
from finance.run_archive import ProgressRecorder, _describe_event, update_index
from finance.settings import DEFAULTS, PROJECT_ROOT, ReportSettings, SettingsError
from tests.test_model_clients import MARKED_REPORT, FakeAnthropic, FakeRequests, _message, _snapshot, _with_requests
from tests.test_replay import CONFIG as REPLAY_CONFIG
from tests.test_replay import _write_run as write_replay_run
from tests.test_run_compare import make_new_run

TODAY = date(2026, 10, 5)
SCRIPTS = Path(__file__).resolve().parents[1]
REAL_LEDGER = PROJECT_ROOT / "agent" / "runs" / "cost_ledger.jsonl"


def _usage(input_tokens=1000, output_tokens=500, cache_read=0, cache_write=0):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)


class LedgerTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.ledger_path = self.dir / "cost_ledger.jsonl"
        self.ledger = CostLedger(self.ledger_path)
        budget._FAILED_LEDGERS.clear()

    def tearDown(self):
        budget._FAILED_LEDGERS.clear()
        self.temp.cleanup()

    def guard(self, daily=15.0, per_run=5.0, run_id="20261005T120000_run"):
        return BudgetGuard(self.ledger, daily, per_run, run_id, today=lambda: TODAY)

    def seed(self, cost, run_id="earlier_run", day=TODAY):
        self.ledger.append({"timestamp": "2026-10-05T10:00:00+00:00", "date": day.isoformat(), "run_id": run_id,
                            "role": "validator", "model": "claude-opus-5-5", "usage": None, "cost_usd": cost})

    def lines(self):
        if not self.ledger_path.exists():
            return []
        return [json.loads(line) for line in self.ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestPricing(unittest.TestCase):
    def test_cost_includes_cache_reads_and_writes(self):
        usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 10000, "cache_creation_input_tokens": 8000}
        # haiku $1/$5: 1000*1 + 10000*0.1 + 8000*1.25 + 500*5 = 14500 per million
        self.assertAlmostEqual(pricing.cost_usd("claude-haiku-4-5", usage), 0.0145)
        self.assertAlmostEqual(pricing.cost_usd("claude-opus-5-5", {"input_tokens": 10**6, "output_tokens": 10**6}), 24.0)
        self.assertEqual(pricing.cost_usd("claude-sonnet-5-5", None), 0.0)
        for model, price in (("claude-opus-5-5", (4, 20)), ("claude-sonnet-5-5", (2, 10)), ("claude-haiku-4-5", (1, 5)),
                             ("claude-fable-5-1", (10, 50)), ("claude-opus-5", (5, 25)), ("claude-sonnet-5", (2, 10)),
                             ("claude-haiku-4-5-20251001", (1, 5)), ("claude-opus-5-20260101", (5, 25))):
            self.assertEqual(pricing.price_for(model), price, model)

    def test_worst_case_estimates_chars_over_3_5_plus_max_tokens(self):
        # 3500 chars -> 1000 input tokens at $2, plus 1000 output tokens at $10
        self.assertAlmostEqual(pricing.worst_case_call_usd("claude-sonnet-5-5", 3500, 1000), 0.012)
        self.assertEqual(pricing.estimate_input_tokens(10), 3)  # rounded up
        self.assertAlmostEqual(pricing.worst_case_call_usd("claude-sonnet-5-5", 1000, 1000, tokens=True), 0.012)
        self.assertAlmostEqual(pricing.worst_case_call_usd("claude-opus-5-5", 0, 64000), 1.28)

    def test_unknown_claude_model_is_priced_at_fable_never_free(self):
        self.assertIsNone(pricing.known_price("claude-mystery-9"))
        self.assertEqual(pricing.price_for("claude-mystery-9"), (10.0, 50.0))
        self.assertAlmostEqual(pricing.cost_usd("claude-mystery-9", {"input_tokens": 10**6}), 10.0)
        self.assertGreater(pricing.worst_case_call_usd(None, 100, 100), 0)

    def test_replay_and_compare_use_the_shared_table(self):
        self.assertFalse(hasattr(run_compare, "ANTHROPIC_PRICES"))
        self.assertAlmostEqual(run_compare.estimate_cost("anthropic", "claude-fable-5-1", 10**6, 0), 10.0)
        self.assertEqual(replay.worst_case_cost("anthropic", "claude-opus-5", 3500, 1000),
                         round(pricing.worst_case_call_usd("claude-opus-5", 3500, 1000), 4))


class TestLedger(LedgerTestCase):
    def test_missing_ledger_is_zero_and_appends_are_read_back(self):
        self.assertEqual(self.ledger.entries(), [])
        self.assertEqual(self.ledger.spend_today(TODAY), 0)
        guard = self.guard(run_id="run_a")
        cost = guard.record("claude-haiku-4-5", {"input_tokens": 1000, "output_tokens": 500}, role="validator")
        self.assertAlmostEqual(cost, 0.0035)
        self.seed(1.0, run_id="run_b")
        self.seed(2.0, run_id="run_a", day=date(2026, 10, 4))  # yesterday
        self.assertAlmostEqual(self.ledger.spend_today(TODAY), 1.0035)
        self.assertAlmostEqual(self.ledger.spend_for_run("run_a"), 2.0035)
        line = self.lines()[0]
        self.assertEqual((line["date"], line["run_id"], line["role"], line["model"], line["outcome"]),
                         ("2026-10-05", "run_a", "validator", "claude-haiku-4-5", "ok"))
        self.assertEqual(line["usage"]["output_tokens"], 500)
        self.assertTrue(datetime.fromisoformat(line["timestamp"]).tzinfo is not None)

    def test_unreadable_or_corrupt_ledger_fails_closed(self):
        self.ledger_path.write_text('{"cost_usd": 1, "date": "2026-10-05"}\nnot json\n{"cost_usd": 1}\n', encoding="utf-8")
        with self.assertRaises(LedgerUnavailableError):
            self.ledger.spend_today(TODAY)
        with self.assertRaises(LedgerUnavailableError):
            self.guard().check("claude-haiku-4-5", 100, 100)
        directory = CostLedger(self.dir)  # a directory cannot be read as a file
        with self.assertRaises(LedgerUnavailableError):
            directory.entries()

    def test_a_partial_last_line_is_ignored_and_never_glued_to_the_next_record(self):
        self.seed(1.0)
        with self.ledger_path.open("ab") as handle:
            handle.write(b'{"cost_usd": 5, "da')  # a write still in progress (or a crashed writer)
        self.assertAlmostEqual(self.ledger.spend_today(TODAY), 1.0)
        self.seed(2.0)
        raw = self.ledger_path.read_bytes().split(b"\n")
        self.assertEqual(raw[1], b'{"cost_usd": 5, "da')  # terminated, not merged into the new record
        self.assertEqual(json.loads(raw[2])["cost_usd"], 2.0)
        # A broken complete line means unknown spend: refuse until someone fixes the ledger.
        with self.assertRaises(LedgerUnavailableError):
            self.ledger.entries()

    def test_failed_write_refuses_further_calls(self):
        blocker = self.dir / "file"
        blocker.write_text("x", encoding="utf-8")
        ledger = CostLedger(blocker / "ledger.jsonl")  # the parent is a file, so the write fails
        guard = BudgetGuard(ledger, 15.0, 5.0, "run", today=lambda: TODAY)
        with self.assertRaises(LedgerUnavailableError):
            guard.record("claude-haiku-4-5", {"input_tokens": 1})
        with self.assertRaises(LedgerUnavailableError):
            guard.check("claude-haiku-4-5", 10, 10)

    def test_concurrent_appends_keep_every_line(self):
        def writer(index):
            guard = BudgetGuard(CostLedger(self.ledger_path), None, None, f"run_{index}", today=lambda: TODAY)
            for _ in range(25):
                guard.record("claude-haiku-4-5", {"input_tokens": 1000})

        threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(self.lines()), 100)
        self.assertAlmostEqual(self.ledger.spend_today(TODAY), 0.1)
        self.assertFalse(self.ledger.lock_path.exists())


class TestSettings(unittest.TestCase):
    def test_caps_default_on_and_only_explicit_zero_or_null_disables(self):
        s = ReportSettings.from_config({})
        self.assertEqual((s.anthropic_daily_budget_usd, s.anthropic_run_budget_usd), (15.0, 5.0))
        self.assertEqual(s.anthropic_cost_ledger_path, "agent/runs/cost_ledger.jsonl")
        self.assertEqual(budget.resolve_ledger_path(s.anthropic_cost_ledger_path), REAL_LEDGER)
        off = ReportSettings.from_config({"anthropic_daily_budget_usd": 0, "anthropic_run_budget_usd": None})
        self.assertEqual((off.anthropic_daily_budget_usd, off.anthropic_run_budget_usd), (None, None))
        self.assertEqual(ReportSettings.from_config({"anthropic_run_budget_usd": "2.5"}).anthropic_run_budget_usd, 2.5)
        for bad in (-1, True, "lots"):
            with self.assertRaises(SettingsError):
                ReportSettings.from_config({"anthropic_daily_budget_usd": bad})
        self.assertEqual(budget.guard_from_settings({}, "r").daily_budget_usd, 15.0)

    def test_disabled_cap_warns_once(self):
        out = io.StringIO()
        with patch.object(budget, "_WARNED", set()), contextlib.redirect_stdout(out):
            guard = budget.guard_from_settings({"anthropic_daily_budget_usd": 0}, "r")
            budget.guard_from_settings({"anthropic_daily_budget_usd": 0}, "r")
        self.assertIsNone(guard.daily_budget_usd)
        self.assertEqual(out.getvalue().count("WARNING: anthropic_daily_budget_usd is 0 or null"), 1)
        self.assertNotIn("anthropic_run_budget_usd", out.getvalue())


class TestGuardedClient(LedgerTestCase):
    def test_daily_cap_refuses_without_calling_the_sdk(self):
        self.seed(14.99)
        sdk = FakeAnthropic(_message())
        client = AnthropicClient("claude-sonnet-5-5", client=sdk, budget_guard=self.guard(), role="validator")
        with self.assertRaises(BudgetExceededError) as raised:
            client.generate("p" * 350, system="s")
        self.assertEqual((sdk.calls, sdk.stream_calls, sdk.timeouts), ([], [], []))
        self.assertIn("Claude daily budget reached: $14.99 of $15.00", str(raised.exception))
        details = raised.exception.details
        self.assertEqual((details["cap"], details["spent_today_usd"], details["daily_budget_usd"]), ("daily", 14.99, 15.0))
        self.assertGreater(details["worst_case_usd"], 0.01)
        self.assertEqual(len(self.lines()), 1)  # nothing new recorded

    def test_run_cap_refuses_without_calling_the_sdk(self):
        self.seed(4.95, run_id="run_x")
        self.seed(4.0, run_id="other_run")  # other runs only count toward the daily cap
        sdk = FakeAnthropic(_message())
        client = AnthropicClient("claude-opus-5-5", max_tokens=8000, client=sdk, budget_guard=self.guard(run_id="run_x"))
        with self.assertRaises(BudgetExceededError) as raised:
            client.generate("p")
        self.assertEqual(sdk.calls, [])
        self.assertIn("Claude run budget reached: $4.95 of $5.00 spent in run run_x", str(raised.exception))
        allowed = AnthropicClient("claude-opus-5-5", max_tokens=8000, client=sdk, budget_guard=self.guard(run_id="run_y"))
        allowed.generate("p")
        self.assertEqual(len(sdk.calls), 1)

    def test_generic_run_ids_cap_one_invocation(self):
        self.seed(4.99, run_id="replay")  # an earlier replay does not use up this replay's cap
        guard = self.guard(run_id="replay")
        self.assertEqual(guard.spent_for_run(), 0)
        guard.record("claude-opus-5-5", {"output_tokens": 249_000})  # $4.98 in this invocation
        with self.assertRaises(BudgetExceededError):
            guard.check("claude-opus-5-5", 100, 2000)

    def test_successful_and_refused_replies_are_recorded_with_cost(self):
        sdk = FakeAnthropic(_message("ok", usage=_usage(1000, 500, cache_read=10000, cache_write=8000)),
                            _message("", stop_reason="refusal", stop_details={"category": "x"}))
        client = AnthropicClient("claude-haiku-4-5", client=sdk, budget_guard=self.guard(), role="eval")
        reply = client.generate("p", system="s")
        self.assertAlmostEqual(reply.metadata["cost_usd"], 0.0145)
        refused = client.generate("p")
        self.assertEqual(refused.done_reason, "refusal")
        lines = self.lines()
        self.assertEqual([line["outcome"] for line in lines], ["stop", "refusal"])
        self.assertEqual(lines[0]["role"], "eval")
        self.assertAlmostEqual(lines[0]["cost_usd"], 0.0145)
        self.assertEqual(lines[0]["usage"]["cache_read_input_tokens"], 10000)

    def test_streamed_calls_are_checked_and_recorded(self):
        sdk = FakeAnthropic(_message(MARKED_REPORT, usage=_usage(2000, 20000)))
        client = AnthropicClient("claude-opus-5-5", max_tokens=32000, client=sdk, budget_guard=self.guard())
        reply = client.generate("p", system="s")
        self.assertTrue(reply.metadata["streamed"])
        self.assertEqual((len(sdk.stream_calls), sdk.calls), (1, []))
        line = self.lines()[0]
        self.assertAlmostEqual(line["cost_usd"], (2000 * 4 + 20000 * 20) / 1e6)
        self.assertEqual(line["usage"]["output_tokens"], 20000)
        # A streamed call over the cap is refused before the stream opens.
        self.seed(14.9)
        with self.assertRaises(BudgetExceededError):
            client.generate("p")
        self.assertEqual(len(sdk.stream_calls), 1)

    def test_error_calls_with_usage_are_recorded(self):
        sdk = FakeAnthropic(_message("", stop_reason="end_turn", usage=_usage(3000, 50)))
        client = AnthropicClient("claude-sonnet-5-5", client=sdk, budget_guard=self.guard(), role="author")
        with self.assertRaises(EmptyModelResponse) as raised:
            client.generate("p")
        line = self.lines()[0]
        self.assertEqual(line["outcome"], "empty")
        self.assertAlmostEqual(line["cost_usd"], (3000 * 2 + 50 * 10) / 1e6)
        self.assertAlmostEqual(raised.exception.cost_usd, line["cost_usd"])

    def test_timeouts_record_an_estimate_and_rejected_requests_record_nothing(self):
        client = AnthropicClient("claude-sonnet-5-5", client=FakeAnthropic(TimeoutError("slow")), budget_guard=self.guard())
        with self.assertRaises(TimeoutError):
            client.generate("p" * 3500)
        line = self.lines()[0]
        self.assertTrue(line["estimated"])
        self.assertAlmostEqual(line["cost_usd"], 0.002)  # 1000 estimated input tokens at $2
        rejected = AnthropicClient("claude-sonnet-5-5", client=FakeAnthropic(ValueError("400 bad request")), budget_guard=self.guard())
        with self.assertRaises(ValueError):
            rejected.generate("p")
        self.assertEqual(len(self.lines()), 1)

    def test_a_failed_ledger_write_fails_the_call_and_blocks_the_next(self):
        sdk = FakeAnthropic(_message())
        guard = self.guard()
        client = AnthropicClient("claude-haiku-4-5", client=sdk, budget_guard=guard)
        with patch("finance.budget.os.write", side_effect=OSError("disk full")):
            with self.assertRaises(LedgerUnavailableError):
                client.generate("p")
        with self.assertRaises(LedgerUnavailableError):
            client.generate("p")
        self.assertEqual(len(sdk.calls), 1)

    def test_real_sdk_client_is_never_built_without_a_guard(self):
        with self.assertRaises(RuntimeError) as raised:
            AnthropicClient("claude-sonnet-5-5").generate("p")
        self.assertIn("BudgetGuard", str(raised.exception))

    def test_build_model_client_attaches_the_configured_guard(self):
        config = {"model_roles": {"eval": {"provider": "anthropic", "model": "claude-sonnet-5-5"}},
                  "anthropic_cost_ledger_path": str(self.ledger_path), "anthropic_run_budget_usd": 2}
        client = build_model_client(config, "eval", anthropic_client=FakeAnthropic(_message()), run_id="compare")
        self.assertEqual((client.budget_guard.run_id, client.budget_guard.daily_budget_usd, client.budget_guard.run_budget_usd),
                         ("compare", 15.0, 2.0))
        self.assertEqual(client.budget_guard.ledger.path, self.ledger_path)
        client.generate("p")
        self.assertEqual(self.lines()[0]["run_id"], "compare")
        settings_client = build_model_client(ReportSettings.from_config(config), "eval", anthropic_client=FakeAnthropic(_message()))
        self.assertEqual(settings_client.budget_guard.run_id, "other")

    def test_ollama_is_never_budgeted(self):
        self.ledger_path.write_text("corrupt\n", encoding="utf-8")
        config = {"model_server_url": "http://h/api/generate", "model_name": "gemma4:e4b",
                  "anthropic_cost_ledger_path": str(self.ledger_path), "anthropic_daily_budget_usd": 0.000001}
        client = build_model_client(config, "author")
        self.assertIsInstance(client, OllamaClient)
        self.assertFalse(hasattr(client, "budget_guard"))
        with _with_requests(FakeRequests({"response": "hello", "done_reason": "stop"})):
            self.assertEqual(client.generate("p"), "hello")
        self.assertEqual(self.ledger_path.read_text(encoding="utf-8"), "corrupt\n")


class FakeAuthor:
    provider = "ollama"
    model = "gemma4:e4b"

    def generate(self, prompt, *, system=None, json_schema=None, timeout=None):
        return MARKED_REPORT


class TestHarnessStop(LedgerTestCase):
    def test_validator_over_budget_stops_with_fail_and_keeps_the_sanitized_report(self):
        self.seed(14.99)
        sdk = FakeAnthropic(_message('{"decision":"pass","summary":"ok","findings":[]}'))
        validator = AnthropicClient("claude-opus-5-5", client=sdk, budget_guard=self.guard(), role="validator")
        harness = ReportHarness(limits=HarnessLimits(max_validation_cycles=3), author_client=FakeAuthor(), validator_client=validator)
        report, decision, trace = harness.run("objective", _snapshot(), "author ctx", {}, validator_context="validator ctx")
        self.assertEqual(sdk.calls, [])
        self.assertFalse(decision.passed)
        self.assertIn("Claude daily budget reached", decision.summary)
        self.assertIn("<h1>Hi</h1>", report["report_html"])
        events = [event["event"] for event in trace.events]
        self.assertEqual(events.count("budget_exceeded"), 1)  # never retried
        self.assertNotIn("model_call_retry", events)
        self.assertNotIn("validator_format_retry", events)
        stop = next(event for event in trace.events if event["event"] == "budget_exceeded")
        self.assertEqual((stop["role"], stop["cap"], stop["spent_today_usd"], stop["daily_budget_usd"]),
                         ("validator", "daily", 14.99, 15.0))
        result = next(event for event in trace.events if event["event"] == "validation_cycle_result")
        self.assertEqual(result["stage"], "budget_exceeded")
        self.assertEqual(trace.events[-1]["event"], "run_completed")
        line = _describe_event(stop)
        self.assertTrue(line.startswith("Claude budget guard refused the validator call: Claude daily budget reached"))
        self.assertTrue(line.endswith("stopping with a fail decision"))

    def test_author_over_budget_returns_the_fallback(self):
        self.seed(14.99)
        author = AnthropicClient("claude-opus-5-5", client=FakeAnthropic(_message(MARKED_REPORT)), budget_guard=self.guard())
        harness = ReportHarness(author_client=author, validator_client=author)
        report, decision, trace = harness.run("objective", _snapshot(), "author ctx", {})
        self.assertFalse(decision.passed)
        self.assertIn("Claude spend cap", report["report_html"] + report.get("plain_text", ""))
        self.assertIn("budget_exceeded", [event["event"] for event in trace.events])


ROWS = [
    {"transaction_month": date(2026, 2, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 450},
    {"transaction_month": date(2026, 1, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 400},
]


class TestReportAndEmailRun(LedgerTestCase):
    def config(self, **extra):
        return {
            "Monthly_budget": 8500, "Fixed_costs": 1000, "report_mode": "agent", "dry_run": False,
            "model_roles": {
                "author": {"provider": "anthropic", "model": "claude-haiku-4-5", "max_tokens": 1000},
                "validator": {"provider": "anthropic", "model": "claude-opus-5-5", "max_tokens": 64000},
            },
            # The author's worst case fits under $0.50; the validator's (64000 Opus tokens = $1.28) does not.
            "anthropic_daily_budget_usd": 0.5,
            "anthropic_cost_ledger_path": str(self.ledger_path),
            **extra,
        }

    def generate(self, sdk, **kwargs):
        return reporting.generate_finance_report(
            rows=ROWS, selected_report_month=date(2026, 2, 1), anthropic_client=sdk, **kwargs
        )

    def test_report_records_cost_and_budget_stop_in_metrics(self):
        sdk = FakeAnthropic(_message(MARKED_REPORT, usage=_usage(1000, 500)))
        result = self.generate(sdk, config=self.config(), project_name="p", ollama_url="http://h", model_name="m",
                               run_id="20261005T120000_test")
        self.assertEqual(len(sdk.calls), 1)  # the author only; the validator was refused
        self.assertFalse(result.passed)
        self.assertEqual(result.metrics["status"], "budget_exceeded")
        self.assertIn("Claude daily budget reached", result.metrics["budget_exceeded"])
        self.assertAlmostEqual(result.metrics["anthropic_cost_usd"], 0.0035)
        start = next(event for event in result.trace.events if event["event"] == "anthropic_budget")
        self.assertEqual((start["daily_budget_usd"], start["run_budget_usd"], start["run_id"]), (0.5, 5.0, "20261005T120000_test"))
        response = next(event for event in result.trace.events if event["event"] == "model_response")
        self.assertAlmostEqual(response["cost_usd"], 0.0035)
        self.assertEqual(self.lines()[0]["run_id"], "20261005T120000_test")

    def test_scheduled_email_run_over_budget_sends_nothing_and_shows_it(self):
        from reports import weekly_report as weekly

        sdk = FakeAnthropic(_message(MARKED_REPORT, usage=_usage(1000, 500)))
        real = reporting.generate_finance_report
        runs = self.dir / "runs"
        with (
            patch.object(weekly, "RUN_ARCHIVE_ROOT", runs),
            patch("finance.reporting.generate_finance_report",
                  side_effect=lambda **kw: real(rows=ROWS, selected_report_month=date(2026, 2, 1), anthropic_client=sdk, **kw)),
            patch.object(weekly, "send_email") as send_email,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            weekly.run_agent_weekly_report(self.config(), [], None, "s", ["r"], "r", "pw", "project", "http://h", "m", "memory")
        send_email.assert_not_called()
        self.assertIn("Report was not sent: Claude daily budget reached", out.getvalue())
        run_dir = next(path for path in runs.iterdir() if path.is_dir())
        log = (run_dir / "progress.log").read_text(encoding="utf-8")
        self.assertRegex(log, r"Claude budget: \$0\.00 spent today of \$0\.50 daily; run cap \$5\.00")
        self.assertIn("author response: parse ok", log)
        self.assertIn("cost $0.0035", log)
        self.assertIn("Claude budget guard refused the validator call: Claude daily budget reached", log)
        self.assertIn("status=budget_exceeded", log)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["status"], "budget_exceeded")
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(progress["anthropic_cost_usd"], 0.0035)
        self.assertIn("Claude daily budget reached", progress["budget_exceeded"])
        index = (runs / "index.html").read_text(encoding="utf-8")
        self.assertIn("<td>budget_exceeded</td>", index)
        self.assertIn("<td>$0.0035</td>", index)
        self.assertEqual(self.lines()[0]["run_id"], run_dir.name)


class TestGradioApp(LedgerTestCase):
    def load_app(self):
        """Import the app with gradio, config, memory, and BigQuery stubbed out."""
        path = SCRIPTS / "apps" / "finance_app.py"
        spec = importlib.util.spec_from_file_location("finance_app_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict(sys.modules, {"gradio": MagicMock()}),
            patch("finance.settings.load_config", return_value={"project_name": "p", "Monthly_budget": 8500, "Fixed_costs": 1000}),
            patch("finance.memory.open_memory", return_value=None),
            patch("finance.bigquery_tools.get_spending_years", return_value=[2026]),
        ):
            spec.loader.exec_module(module)
        return module

    def test_budget_refusal_shows_a_clear_message(self):
        app = self.load_app()
        period = {"rows": ROWS, "report_month": date(2026, 2, 1)}
        stopped = SimpleNamespace(
            metrics={"budget_exceeded": "Claude daily budget reached: $14.99 of $15.00 spent today; ..."},
            report_html="<p>fallback</p>", plain_text="", passed=False, decision=None,
            snapshot=SimpleNamespace(latest_month="2026-02-01", to_dict=lambda: {}),
        )
        with patch.object(app, "get_report_period_spending", return_value=period), \
                patch.object(app, "generate_finance_report", return_value=stopped):
            html, _, status = app.load_finance_summary(2026, 2)
        self.assertIn("Claude budget reached", html)
        self.assertTrue(status.startswith("Claude daily budget reached: $14.99 of $15.00"))
        with patch.object(app, "get_report_period_spending", return_value=period), \
                patch.object(app, "generate_finance_report", side_effect=BudgetExceededError("Claude daily budget reached: $15.00 of $15.00")):
            html, context, status = app.load_finance_summary(2026, 2)
        self.assertEqual(status, "Claude daily budget reached: $15.00 of $15.00")
        self.assertNotIn("Traceback", status)


class TestReplayAndCompareRefuse(LedgerTestCase):
    def over_budget_client(self, model="claude-haiku-4-5", run_id="replay"):
        self.seed(15.0)
        self.sdk = FakeAnthropic(_message())
        return AnthropicClient(model, client=self.sdk, budget_guard=self.guard(run_id=run_id))

    def test_replay_prints_the_refusal_and_exits_non_zero(self):
        root = self.dir / "runs"
        write_replay_run(root)
        client = self.over_budget_client()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = replay.main(["20261005T133631_gemma4_e4b", "--role-model", "anthropic:claude-haiku-4-5"],
                               config=REPLAY_CONFIG, runs_root=root, prompt_dir=self.dir, client_factory=lambda cfg, spec: client)
        self.assertEqual(code, 2)
        self.assertIn("Error: Claude daily budget reached: $15.00 of $15.00", err.getvalue())
        self.assertEqual(self.sdk.calls, [])
        self.assertFalse((root / "replays").exists())

    def test_compare_prints_the_refusal_and_exits_non_zero(self):
        root = self.dir / "runs"
        make_new_run(root, "20261005T180000_a")
        make_new_run(root, "20261005T190000_b")
        config = self.dir / "config.json"
        config.write_text(json.dumps({"model_roles": {"eval": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}}), encoding="utf-8")
        client = self.over_budget_client("claude-sonnet-5-5", run_id="compare")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_compare.main(["20261005T180000_a", "20261005T190000_b", "--config", str(config)], client=client, runs_root=root)
        self.assertEqual(code, 2)
        self.assertIn("Claude daily budget reached", err.getvalue())
        self.assertIn("Nothing was saved", err.getvalue())
        self.assertEqual(self.sdk.calls, [])
        self.assertFalse((root / "comparisons").exists())

    def test_default_clients_are_guarded_and_recorded_as_replay_and_compare(self):
        captured = []
        real = build_model_client

        def spy(config, role, **kwargs):
            captured.append(kwargs.get("run_id"))
            client = real(config, role, anthropic_client=FakeAnthropic(_message("## Summary\nok")), **kwargs)
            self.assertIsNotNone(client.budget_guard)
            return client

        root = self.dir / "runs"
        write_replay_run(root)
        make_new_run(root, "20261005T180000_a")
        make_new_run(root, "20261005T190000_b")
        ledger = {"anthropic_cost_ledger_path": str(self.ledger_path)}
        with patch.object(replay, "build_model_client", side_effect=spy), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            replay.main(["20261005T133631_gemma4_e4b", "--role-model", "anthropic:claude-haiku-4-5"],
                        config={**REPLAY_CONFIG, **ledger}, runs_root=root, prompt_dir=self.dir)
        config = {"model_roles": {"eval": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}, **ledger}
        with patch.object(run_compare, "build_model_client", side_effect=spy):
            run_compare.compare_runs(["20261005T180000_a", "20261005T190000_b"], config=config, runs_root=root, echo=lambda _: None)
        self.assertEqual(captured, ["replay", "compare"])
        self.assertEqual([line["run_id"] for line in self.lines()], ["replay", "compare"])


class TestIndexCost(unittest.TestCase):
    def test_cost_column_is_blank_without_claude_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_id, cost in (("20261005T100000_claude", 0.1234), ("20261005T090000_local", None)):
                (root / run_id).mkdir()
                (root / run_id / "metrics.json").write_text(json.dumps({"status": "validated", "anthropic_cost_usd": cost}), encoding="utf-8")
            running = root / "20261005T110000_live"
            recorder = ProgressRecorder(running, running.name, "m", "v", echo=False)
            recorder({"event": "model_response", "timestamp": 1.0, "role": "validator", "cost_usd": 0.5})
            recorder({"event": "model_call_failed", "timestamp": 2.0, "role": "validator", "cost_usd": 0.25})
            html = update_index(root).read_text(encoding="utf-8")
        headers = re.findall(r"<th>([^<]*)</th>", html.split("<tbody>")[0])
        self.assertIn("cost", headers)
        position = headers.index("cost")
        rows = {}
        for row in html.split("<tbody>")[1].split("</tbody>")[0].split("</tr>")[:-1]:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row)
            run_id = re.search(r'value="([^"]+)"', row).group(1)
            rows[run_id] = cells[position]
        self.assertEqual(rows["20261005T100000_claude"], "$0.1234")
        self.assertEqual(rows["20261005T090000_local"], "")
        self.assertEqual(rows["20261005T110000_live"], "$0.7500")

    def test_progress_log_lines(self):
        self.assertEqual(
            _describe_event({"event": "anthropic_budget", "spent_today_usd": 3.2, "daily_budget_usd": 15.0, "run_budget_usd": None}),
            "Claude budget: $3.20 spent today of $15.00 daily; run cap off",
        )
        line = _describe_event({"event": "model_response", "role": "validator", "parse_status": "ok", "response_chars": 10,
                                "elapsed_seconds": 2, "eval_count": 5, "prompt_eval_count": 7, "cost_usd": 0.01234})
        self.assertIn("tokens in/out 7/5, cost $0.0123", line)
        self.assertNotIn("cost", _describe_event({"event": "model_response", "role": "author", "parse_status": "ok"}))


class TestNoUnguardedAnthropicCalls(unittest.TestCase):
    PATTERN = re.compile(r"\banthropic\.(?:Async)?Anthropic\s*\(|\.messages\.(?:create|stream)\s*\(|\bfrom\s+anthropic\s+import\b")
    ALLOWED = {SCRIPTS / "finance" / "model_clients.py"}

    def test_only_the_guarded_client_module_touches_the_sdk(self):
        offenders = []
        for path in SCRIPTS.rglob("*.py"):
            if path.name.startswith("test_") or path.name == "chat_test_fixtures.py" or "__pycache__" in path.parts:
                continue
            if path in self.ALLOWED:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if self.PATTERN.search(line):
                    offenders.append(f"{path.relative_to(SCRIPTS)}:{number}: {line.strip()}")
        self.assertEqual(offenders, [], "Anthropic SDK used outside the guarded AnthropicClient")

    def test_the_guarded_module_builds_the_sdk_client_only_behind_the_guard(self):
        source = (SCRIPTS / "finance" / "model_clients.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"=\s*anthropic\.Anthropic\(", source)), 1)
        guard_check = source.index("if self.budget_guard is None:\n                # Real API calls")
        self.assertLess(guard_check, source.index("anthropic.Anthropic(max_retries=1)"))
        self.assertLess(source.index("self.budget_guard.check("), source.index("client.messages.create(**kwargs)"))

    def test_default_ledger_lives_under_agent_runs(self):
        self.assertEqual(DEFAULTS["anthropic_cost_ledger_path"], "agent/runs/cost_ledger.jsonl")
        self.assertEqual((DEFAULTS["anthropic_daily_budget_usd"], DEFAULTS["anthropic_run_budget_usd"]), (15.0, 5.0))


if __name__ == "__main__":
    unittest.main()
