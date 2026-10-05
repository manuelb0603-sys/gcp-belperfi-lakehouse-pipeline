import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import email_weekly_report as weekly
from agent_harness import ReportHarness, RunTrace
from report_models import ReportSnapshot
from run_archive import ProgressRecorder, save_run, update_index


def _snapshot():
    return ReportSnapshot(
        as_of="2026-09-15",
        latest_month="2026-08",
        previous_month="2026-07",
        latest_total_net_expense=1000,
        previous_total_net_expense=900,
        mom_change_percent=11.1,
        monthly_deposit=5000,
        fixed_costs=2200,
        disposable_cash=1600,
        savings_goal=500,
        categories=[],
        source_tables=[],
    )


def _passing_harness():
    responses = [
        '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
        '{"decision":"pass","summary":"Accepted","findings":[]}',
    ]
    return ReportHarness(
        "http://localhost:11434/api/generate",
        "test-model",
        query=lambda *args, **kwargs: responses.pop(0),
    )


def _read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


class TestTraceListeners(unittest.TestCase):
    def test_listener_receives_each_event_with_model_timing(self):
        harness = _passing_harness()
        received = []
        harness.trace.listeners.append(received.append)

        _, decision, trace = harness.run("objective", _snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        self.assertEqual(received, trace.events)
        responses = [event for event in received if event["event"] == "model_response"]
        self.assertEqual({event["role"] for event in responses}, {"author", "validator"})
        self.assertTrue(all(isinstance(event["elapsed_seconds"], float) for event in responses))

    def test_listener_failure_does_not_break_run(self):
        harness = _passing_harness()

        def broken_listener(event):
            raise OSError("disk full")

        harness.trace.listeners.append(broken_listener)
        with redirect_stdout(io.StringIO()) as output:
            _, decision, trace = harness.run("objective", _snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        self.assertEqual(trace.events[-1]["event"], "run_completed")
        self.assertEqual(output.getvalue().count("listener failed"), 1)


class TestProgressRecorder(unittest.TestCase):
    def test_recorder_streams_trace_progress_and_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run1"
            recorder = ProgressRecorder(run_dir, "run1", "author-model", "validator-model", echo=False)
            self.assertEqual(json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))["status"], "running")

            trace = RunTrace(listeners=[recorder])
            trace.add("agent_round", round=1)
            trace.add("model_call", role="author", model="author-model", prompt="p", prompt_chars=1200)
            self.assertEqual(len(_read_lines(run_dir / "trace.jsonl")), 2)
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["phase"], "author_model_call")
            self.assertEqual(progress["agent_round"], 1)

            trace.add("model_response", role="author", parse_status="ok", response_chars=50, elapsed_seconds=2.5)
            trace.add("tool_call", name="get_spending_summary", arguments={}, result={}, phase="initial")
            trace.add("validation_cycle_start", cycle=1, report_html_bytes=300)
            trace.add("validation_cycle_result", cycle=1, decision="fail", stage="validator", error_count=2, warning_count=0)
            recorder.finish("validation_failed", delivery="not_sent")

            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "validation_failed")
            self.assertEqual(progress["tool_calls"], 1)
            self.assertEqual(progress["validation_cycle"], 1)
            self.assertEqual(progress["last_validation_error_count"], 2)
            self.assertEqual(progress["delivery"], "not_sent")
            log = (run_dir / "progress.log").read_text(encoding="utf-8")
            self.assertIn("author model call started (author-model, 1,200 prompt chars)", log)
            self.assertIn("author response: parse ok, 50 chars in 2.5s", log)
            self.assertIn("Validation cycle 1 result (validator): fail, 2 errors", log)
            self.assertIn("Run finished: status=validation_failed", log)
            self.assertEqual(len(_read_lines(run_dir / "trace.jsonl")), 6)

            # The final archive rewrites the complete trace once instead of appending duplicates.
            save_run(Path(temp_dir), "run1", {}, trace.events, "<p>r</p>", "r", {"status": "validation_failed"})
            self.assertEqual(len(_read_lines(run_dir / "trace.jsonl")), 6)

    def test_rebuttal_and_stall_events_are_logged_and_counted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run1"
            recorder = ProgressRecorder(run_dir, "run1", "author-model", "validator-model", echo=False)
            trace = RunTrace(listeners=[recorder])
            trace.add("author_rebuttals", responding_to_cycle=1, rebuttal_count=1,
                      rebuttals=[{"finding": "Cuts note missing", "quote": "would not meet your goal"}])
            trace.add("author_rebuttals_ignored", responding_to_cycle=2, reason="Malformed", error="not JSON")
            trace.add("validation_stalled", cycle=2, repeated_error_count=2, rebuttal_count=1, findings=[])
            trace.add("validation_cycle_result", cycle=2, decision="fail", stage="validator",
                      error_count=2, warning_count=0, repeated_error_count=2)

            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["author_rebuttals"], 1)
            self.assertEqual(progress["stalled_validation_cycles"], 1)
            self.assertEqual(progress["last_repeated_error_count"], 2)
            log = (run_dir / "progress.log").read_text(encoding="utf-8")
            self.assertIn("Author disputed 1 finding(s) from validation cycle 1: Cuts note missing", log)
            self.assertIn("Author rebuttals ignored: not JSON", log)
            self.assertIn("Validation stalled: cycle 2 repeated 2 error finding(s)", log)
            self.assertIn("fail, 2 errors, 0 warnings, 2 repeated from the previous cycle", log)

    def test_deadline_and_resolved_findings_are_logged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run1"
            recorder = ProgressRecorder(run_dir, "run1", "author-model", "validator-model", echo=False)
            trace = RunTrace(listeners=[recorder])
            trace.add("run_started", tool_names=["get_spending_summary"], deadline_seconds=2700.0)
            trace.add("validation_cycle_result", cycle=2, decision="fail", stage="validator",
                      error_count=1, warning_count=0, repeated_error_count=0, resolved_finding_count=2)
            trace.add("run_deadline_exceeded", role="author", elapsed_seconds=2712.5, limit_seconds=2700.0)

            log = (run_dir / "progress.log").read_text(encoding="utf-8")
            self.assertIn("Harness started; tools: get_spending_summary; time limit 2700s", log)
            self.assertIn("fail, 1 errors, 0 warnings, 2 earlier finding(s) resolved", log)
            self.assertIn("Run time limit reached (2712.5s elapsed of 2700.0s) at a author model call", log)

    def test_index_lists_running_run_from_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ProgressRecorder(Path(temp_dir) / "20260101T000000_live", "20260101T000000_live", "m", "v", echo=False)
            index = update_index(Path(temp_dir)).read_text(encoding="utf-8")
            self.assertIn("20260101T000000_live", index)
            self.assertIn("running", index)

    def test_move_to_renames_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ProgressRecorder(Path(temp_dir) / "run", "run", "m", "v", echo=False)
            self.assertTrue(recorder.move_to(Path(temp_dir) / "run_failed"))
            recorder.log("after move")
            self.assertFalse((Path(temp_dir) / "run").exists())
            self.assertIn("after move", (Path(temp_dir) / "run_failed" / "progress.log").read_text(encoding="utf-8"))


class TestEmailRunDirectories(unittest.TestCase):
    def _run(self, temp_dir, generate):
        with (
            patch.object(weekly, "RUN_ARCHIVE_ROOT", Path(temp_dir)),
            patch("finance.reporting.generate_finance_report", side_effect=generate),
            patch.object(weekly, "send_email") as send_email,
            redirect_stdout(io.StringIO()),
        ):
            weekly.run_agent_weekly_report(
                {"dry_run": True}, [], None, "s", ["r"], "r", "pw", "project", "url", "model:tag", "memory"
            )
        send_email.assert_not_called()
        return sorted(path for path in Path(temp_dir).iterdir() if path.is_dir())

    def test_successful_dry_run_uses_one_live_directory(self):
        def generate(**kwargs):
            trace = RunTrace(listeners=kwargs["trace_listeners"])
            trace.add("agent_round", round=1)
            trace.add("run_completed", decision="pass", tool_calls=0)
            return SimpleNamespace(
                passed=True,
                metrics={"status": "validated"},
                snapshot=SimpleNamespace(to_dict=lambda: {}),
                trace=trace,
                report_html="<p>Report</p>",
                plain_text="Report",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dirs = self._run(temp_dir, generate)
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.endswith("_model_tag"))
            progress = json.loads((run_dirs[0] / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "validated")
            self.assertEqual(progress["delivery"], "dry_run")
            self.assertEqual(len(_read_lines(run_dirs[0] / "trace.jsonl")), 2)
            self.assertTrue((run_dirs[0] / "metrics.json").exists())

    def test_failed_run_is_renamed_not_duplicated(self):
        def generate(**kwargs):
            trace = RunTrace(listeners=kwargs["trace_listeners"])
            trace.add("agent_round", round=1)
            error = RuntimeError("Unexpected harness failure")
            error.run_trace = trace
            raise error

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dirs = self._run(temp_dir, generate)
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.endswith("_model_tag_failed"))
            progress = json.loads((run_dirs[0] / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "exception")
            self.assertIn("Unexpected harness failure", progress["error"])
            events = [json.loads(line)["event"] for line in _read_lines(run_dirs[0] / "trace.jsonl")]
            self.assertEqual(events, ["agent_round", "exception", "run_aborted"])
            self.assertIn("status=exception", (run_dirs[0] / "progress.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
