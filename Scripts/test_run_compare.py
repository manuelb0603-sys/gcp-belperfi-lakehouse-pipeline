import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from finance import run_compare
from finance.model_clients import ModelReply
from finance.run_archive import COMPARISONS_DIR, ProgressRecorder, run_times, trace_first_last_timestamps, update_index


def _local(dt):
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _trace_lines(events):
    return "".join(json.dumps(event) + "\n" for event in events)


START = datetime(2026, 10, 5, 18, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 10, 5, 18, 3, 20, tzinfo=timezone.utc)


def make_new_run(root, run_id="20261005T180000_author_a", findings=None, report_text="Hello report", extra_events=()):
    """A run in the current format: progress.json, metrics.json, trace.jsonl with float timestamps."""
    run_dir = root / run_id
    finding_list = findings if findings is not None else [
        {"severity": "error", "category": "completeness", "message": "Missing goal disclosure", "evidence": "Snapshot remaining_gap_after_cuts=2166.67"},
        {"severity": "warning", "category": "usability", "message": "Small font", "evidence": "font-size:10px"},
    ]
    base = START.timestamp()
    events = [
        {"event": "run_started", "timestamp": base, "author_model": "author-a", "validator_model": "val-b",
         "limits": {"max_validation_cycles": 3}, "deadline_seconds": 2700.0},
        {"event": "model_call", "timestamp": base + 1, "role": "author", "model": "author-a", "prompt": "x" * 50, "prompt_chars": 50},
        {"event": "model_response", "timestamp": base + 61, "role": "author", "parse_status": "ok", "raw_response": "r",
         "response_chars": 900, "elapsed_seconds": 60.0, "done_reason": "length", "eval_count": 4000, "prompt_eval_count": 5000},
        {"event": "model_call", "timestamp": base + 62, "role": "validator", "model": "val-b", "prompt_chars": 70, "validation_cycle": 1},
        {"event": "model_response", "timestamp": base + 90, "role": "validator", "parse_status": "ok", "validation_cycle": 1,
         "response_chars": 300, "elapsed_seconds": 28.0, "done_reason": "stop", "eval_count": 500, "prompt_eval_count": 7000},
        {"event": "validation_cycle_result", "timestamp": base + 91, "cycle": 1, "decision": "fail", "stage": "validator",
         "summary": "Fails on disclosure", "findings": finding_list, "error_count": 1, "warning_count": 1},
        {"event": "validation_stalled", "timestamp": base + 92, "cycle": 2, "repeated_error_count": 1, "rebuttal_count": 0},
        *extra_events,
        {"event": "run_completed", "timestamp": base + 200, "decision": "fail", "passed": False},
    ]
    run_dir.mkdir(parents=True)
    (run_dir / "trace.jsonl").write_text(_trace_lines(events), encoding="utf-8")
    _write_json(run_dir / "progress.json", {
        "run_id": run_id, "status": "validation_failed", "model": "author-a", "validator_model": "val-b",
        "started_at": START.isoformat(), "updated_at": END.isoformat(), "elapsed_seconds": 200.0, "tool_calls": 0,
    })
    _write_json(run_dir / "metrics.json", {
        "model": "author-a", "validator_model": "val-b", "author_provider": "ollama", "validator_provider": "ollama",
        "mode": "agent", "status": "validation_failed", "validation": "fail", "runtime_seconds": 199.5,
        "agent_rounds": 1, "tool_calls": 0, "validation_cycles": 1, "final_validation_errors": 1, "report_bytes": 1234,
    })
    _write_json(run_dir / "run.json", {"run_id": run_id, "mode": "agent", "status": "validation_failed", "snapshot": {"big": "x" * 5000}})
    (run_dir / "final_report.txt").write_text(report_text, encoding="utf-8")
    return run_dir


def make_old_run(root, run_id="20260915T133925_old"):
    """An early-format run: no progress.json, trace timestamps only (one stringly long last line)."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    first = datetime(2026, 9, 15, 20, 39, 25, tzinfo=timezone.utc).timestamp()
    events = [
        {"event": "agent_round", "timestamp": first, "round": 1},
        {"event": "model_call", "timestamp": first + 1, "role": "author", "model": "m", "prompt_chars": 10},
        {"event": "model_response", "timestamp": first + 123.5, "role": "author", "raw_response": "y" * 200000, "response_chars": 200000},
    ]
    (run_dir / "trace.jsonl").write_text(_trace_lines(events), encoding="utf-8")
    _write_json(run_dir / "metrics.json", {"model": "m", "mode": "agent", "status": "failed"})
    return run_dir, first


class FakeClient:
    provider = "anthropic"
    model = "claude-sonnet-5-5"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate(self, prompt, *, system=None, json_schema=None, timeout=None):
        self.calls.append({"prompt": prompt, "system": system, "timeout": timeout})
        if self.error:
            raise self.error
        return self.reply


def _good_reply():
    return ModelReply(
        "## Summary\n\nRun **20261005T180000_author_a** was best.\n\n| run | outcome |\n|---|---|\n| a | fail |\n\n- one\n- two\n",
        done_reason="stop", eval_count=800, prompt_eval_count=9000, provider="anthropic", model="claude-sonnet-5-5",
        usage={"input_tokens": 9000, "output_tokens": 800, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    )


class RunMetadataTests(unittest.TestCase):
    def test_git_code_version_reports_commit_and_dirty_flag(self):
        from finance.run_archive import git_code_version

        outputs = {"rev-parse": "abc1234\n", "status": " M Scripts/x.py\n"}
        version = git_code_version(Path("."), runner=lambda args, cwd: outputs[args[0]])
        self.assertEqual(version, {"commit": "abc1234", "dirty": True})
        clean = git_code_version(Path("."), runner=lambda args, cwd: "abc1234\n" if args[0] == "rev-parse" else "")
        self.assertEqual(clean, {"commit": "abc1234", "dirty": False})

    def test_git_code_version_tolerates_missing_git(self):
        from finance.run_archive import git_code_version

        def missing(args, cwd):
            raise FileNotFoundError("git")

        version = git_code_version(Path("."), runner=missing)
        self.assertIsNone(version["commit"])
        self.assertIsNone(version["dirty"])
        self.assertIn("FileNotFoundError", version["error"])

    def test_collect_run_metadata_has_roles_and_limits_without_secrets(self):
        from finance.run_archive import collect_run_metadata
        from finance.settings import ReportSettings

        settings = ReportSettings.from_config({
            "model_server_url": "http://private-host:11434", "Gmail_app_credentials": "app-password",
            "max_validation_cycles": 5,
            "model_roles": {
                "author": {"provider": "ollama", "model": "gemma4:e4b", "options": {"num_ctx": 32768}},
                "validator": {"provider": "ollama", "model": "gemma4:12b", "think": False, "options": {"num_predict": 4000}},
            },
        })
        meta = collect_run_metadata(settings, code_version={"commit": "abc1234", "dirty": False})
        self.assertEqual(meta["code_version"], {"commit": "abc1234", "dirty": False})
        self.assertEqual(meta["author_role"]["model"], "gemma4:e4b")
        self.assertEqual(meta["validator_role"]["options"], {"num_predict": 4000})
        self.assertIs(meta["validator_role"]["think"], False)
        self.assertEqual(meta["harness_limits"]["max_validation_cycles"], 5)
        text = json.dumps(meta)
        self.assertNotIn("private-host", text)
        self.assertNotIn("app-password", text)

    def test_email_run_records_metadata_at_start_and_in_manifest(self):
        from contextlib import redirect_stdout as quiet
        from types import SimpleNamespace
        from unittest.mock import patch

        import email_weekly_report as weekly
        from finance.agent_harness import RunTrace

        seen_at_start = {}

        def generate(**kwargs):
            run_dir = next(Path(temp_dir).iterdir())
            seen_at_start.update(json.loads((run_dir / "progress.json").read_text(encoding="utf-8")))
            trace = RunTrace(listeners=kwargs["trace_listeners"])
            trace.add("run_completed", decision="pass", tool_calls=0)
            return SimpleNamespace(passed=True, metrics={"status": "validated"}, snapshot=SimpleNamespace(to_dict=lambda: {}),
                                   trace=trace, report_html="<p>r</p>", plain_text="r")

        config = {"dry_run": True, "model_roles": {"validator": {"provider": "ollama", "model": "gemma4:12b", "think": False}}}
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(weekly, "RUN_ARCHIVE_ROOT", Path(temp_dir)),
                patch("finance.reporting.generate_finance_report", side_effect=generate),
                patch("finance.run_archive.git_code_version", return_value={"commit": "abc1234", "dirty": True}),
                patch.object(weekly, "send_email") as send_email,
                quiet(io.StringIO()),
            ):
                weekly.run_agent_weekly_report(config, [], None, "s", ["r"], "r", "pw", "project", "http://host:11434", "gemma4:e4b", "memory")
            send_email.assert_not_called()
            run_dir = next(path for path in Path(temp_dir).iterdir() if path.is_dir())
            manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            summary = run_compare.gather_run(run_dir)
        for source in (seen_at_start, manifest):
            self.assertEqual(source["code_version"], {"commit": "abc1234", "dirty": True})
            self.assertEqual(source["author_role"]["model"], "gemma4:e4b")
            self.assertEqual(source["validator_role"]["model"], "gemma4:12b")
            self.assertIs(source["validator_role"]["think"], False)
            self.assertEqual(source["harness_limits"]["max_validation_cycles"], 3)
            self.assertNotIn("http://host", json.dumps(source["author_role"]))
        self.assertNotIn("pw", json.dumps({key: manifest[key] for key in ("author_role", "validator_role", "harness_limits")}))
        self.assertEqual(summary["code_version"], {"commit": "abc1234", "dirty": True})
        self.assertEqual(summary["options"]["validator_role"]["model"], "gemma4:12b")
        self.assertEqual(summary["limits"]["max_validation_cycles"], 3)

    def test_compare_prompt_includes_code_version_and_instruction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            new = make_new_run(root, "20261005T180000_a")
            manifest = json.loads((new / "run.json").read_text(encoding="utf-8"))
            manifest["code_version"] = {"commit": "abc1234", "dirty": False}
            _write_json(new / "run.json", manifest)
            make_new_run(root, "20261005T190000_b")
            system, user = run_compare.build_prompt(run_compare.gather_runs(run_compare.resolve_runs(["20261005T180000_a", "20261005T190000_b"], root)))
        self.assertIn("code_version", system)
        self.assertIn("code or prompt changes rather than the models", system)
        data = json.loads(user.split("RUN DATA (JSON):\n", 1)[1])
        self.assertEqual(data["runs"][0]["code_version"], {"commit": "abc1234", "dirty": False})
        self.assertEqual(data["runs"][1]["code_version"], "not recorded")


class RunIndexTests(unittest.TestCase):
    def test_index_has_number_start_end_for_new_and_old_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_new_run(root)
            _, first = make_old_run(root)
            ProgressRecorder(root / "20261006T000000_live", "20261006T000000_live", "m", "v", echo=False)
            index = update_index(root).read_text(encoding="utf-8")
            self.assertIn("<th>No.</th>", index)
            self.assertIn("<th>start</th>", index)
            self.assertIn("<th>end</th>", index)
            rows = re.findall(r"<tr><td>(\d+)</td><td><input[^>]*value=\"([^\"]+)\"", index)
            self.assertEqual(rows, [("1", "20261006T000000_live"), ("2", "20261005T180000_author_a"), ("3", "20260915T133925_old")])
            self.assertIn(f'<td class="nowrap">{_local(START)}</td><td class="nowrap">{_local(END)}</td>', index)
            old_start = datetime.fromtimestamp(first, tz=timezone.utc)
            old_end = datetime.fromtimestamp(first + 123.5, tz=timezone.utc)
            self.assertIn(f'<td class="nowrap">{_local(old_start)}</td><td class="nowrap">{_local(old_end)}</td>', index)
            live_row = index[index.index('value="20261006T000000_live"'):].split("</tr>")[0]
            self.assertIn('<td class="nowrap">running</td>', live_row)
            # The compare panel offers a copyable compare_runs.py command.
            self.assertIn("compare_runs.py", index)
            self.assertIn("navigator.clipboard", index)

    def test_trace_first_last_reads_long_last_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, first = make_old_run(Path(temp_dir))
            self.assertEqual(trace_first_last_timestamps(run_dir / "trace.jsonl"), (first, first + 123.5))
            self.assertEqual(run_times(run_dir)["duration_seconds"], 123.5)

    def test_comparisons_listed_and_excluded_from_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_new_run(root)
            folder = root / COMPARISONS_DIR
            _write_json(folder / "20261001T100000_2runs.json", {
                "created_at": "2026-10-01T17:00:00+00:00", "run_ids": ["r1", "r2"], "eval_provider": "ollama",
                "eval_model": "gemma4:12b", "html_file": "20261001T100000_2runs.html"})
            _write_json(folder / "20261003T100000_3runs.json", {
                "created_at": "2026-10-03T17:00:00+00:00", "run_ids": ["r3", "r4", "<b>r5</b>"], "eval_provider": "anthropic",
                "eval_model": "claude-sonnet-5-5", "html_file": "20261003T100000_3runs.html"})
            # A progress.json inside comparisons must not turn it into a run row.
            _write_json(folder / "progress.json", {"status": "running"})
            index = update_index(root).read_text(encoding="utf-8")
            self.assertNotIn('value="comparisons"', index)
            self.assertIn("<h2>Comparisons</h2>", index)
            newer = index.index("comparisons/20261003T100000_3runs.html")
            older = index.index("comparisons/20261001T100000_2runs.html")
            self.assertLess(newer, older)
            self.assertIn("anthropic:claude-sonnet-5-5", index)
            self.assertIn("r3, r4, &lt;b&gt;r5&lt;/b&gt;", index)


class GatherTests(unittest.TestCase):
    def test_gather_collects_cycles_events_tokens_and_excerpt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = make_new_run(Path(temp_dir))
            summary = run_compare.gather_run(run_dir)
            self.assertEqual(summary["models"]["author"], {"provider": "ollama", "model": "author-a"})
            self.assertEqual(summary["models"]["validator"]["model"], "val-b")
            self.assertEqual(summary["start"], _local(START))
            self.assertEqual(summary["duration_seconds"], 200.0)
            self.assertEqual(summary["outcome"]["status"], "validation_failed")
            cycle = summary["validation_cycles"][0]
            self.assertEqual(cycle["decision"], "fail")
            self.assertEqual(cycle["errors"], [{"category": "completeness", "message": "Missing goal disclosure", "evidence": "Snapshot remaining_gap_after_cuts=2166.67"}])
            totals = summary["model_call_totals"]
            self.assertEqual(totals["author"]["length_stops"], 1)
            self.assertEqual(totals["author"]["out_tokens"], 4000)
            self.assertEqual(totals["validator"]["in_tokens"], 7000)
            self.assertEqual([event["event"] for event in summary["events"]], ["validation_stalled"])
            self.assertEqual(summary["report_excerpt"], "Hello report")
            self.assertNotIn("snapshot", json.dumps(summary))

    def test_gather_is_bounded_and_truncates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            findings = [
                {"severity": "error", "category": "arithmetic", "message": f"M{i} " + "m" * 2000, "evidence": "E " + "e" * 5000}
                for i in range(30)
            ]
            failures = [
                {"event": "model_call_failed", "timestamp": 1.0 * i, "role": "author", "error": "boom " * 500, "error_type": "ReadTimeout"}
                for i in range(40)
            ]
            run_dir = make_new_run(Path(temp_dir), findings=findings, report_text="Report line\n" * 2000, extra_events=failures)
            summary = run_compare.gather_run(run_dir, budget=12000)
            size = len(json.dumps(summary, ensure_ascii=False, indent=1))
            self.assertLessEqual(size, 12000)
            cycle = summary["validation_cycles"][0]
            self.assertLessEqual(len(cycle["errors"]), 6)
            self.assertGreater(cycle["errors_omitted"], 0)
            self.assertTrue(all(len(item["message"]) <= 300 for item in cycle["errors"]))
            self.assertLessEqual(len(summary["report_excerpt"]), 1500)
            self.assertLessEqual(len(summary["events"]), 15)

    def test_five_large_runs_stay_under_prompt_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            findings = [{"severity": "error", "category": "html", "message": "x" * 900, "evidence": "y" * 900} for _ in range(12)]
            ids = []
            for number in range(5):
                run_id = f"2026100{number}T000000_run"
                make_new_run(root, run_id=run_id, findings=findings, report_text="z" * 50000)
                ids.append(run_id)
            system, user = run_compare.build_prompt(run_compare.gather_runs(run_compare.resolve_runs(ids, root)))
            self.assertLess(len(system) + len(user), 60000)

    def test_prompt_contains_run_data_and_instructions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_new_run(root, "20261005T180000_a")
            make_new_run(root, "20261005T190000_b", report_text="Top 3 table text")
            folders = run_compare.resolve_runs(["20261005T180000_a", "20261005T190000_b"], root)
            system, user = run_compare.build_prompt(run_compare.gather_runs(folders))
            self.assertIn("never invent metrics", system)
            self.assertIn("false positives", system)
            self.assertIn("Recommendations", system)
            data = json.loads(user.split("RUN DATA (JSON):\n", 1)[1])
            self.assertEqual([run["run_id"] for run in data["runs"]], ["20261005T180000_a", "20261005T190000_b"])
            self.assertIn("Missing goal disclosure", user)
            self.assertIn("Top 3 table text", user)

    def test_repeated_findings_are_not_repeated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            again = {"event": "validation_cycle_result", "timestamp": START.timestamp() + 150, "cycle": 2, "decision": "fail",
                     "summary": "Fails on disclosure",
                     "findings": [{"severity": "error", "category": "completeness", "message": "Missing goal disclosure", "evidence": "other"}]}
            run_dir = make_new_run(Path(temp_dir), extra_events=[again])
            cycles = run_compare.gather_run(run_dir)["validation_cycles"]
            self.assertEqual(cycles[1]["errors_same_as_cycle"], 1)
            self.assertEqual(cycles[1]["summary_same_as_cycle"], 1)
            self.assertNotIn("errors", cycles[1])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "runs"
        make_new_run(self.root, "20261005T180000_a")
        make_new_run(self.root, "20261005T190000_b")
        self.config = Path(self.temp.name) / "config.json"
        _write_json(self.config, {"model_roles": {"eval": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}})

    def tearDown(self):
        self.temp.cleanup()

    def _main(self, args, client=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_compare.main(args + ["--config", str(self.config)], client=client, runs_root=self.root)
        return code, out.getvalue(), err.getvalue()

    def test_cli_writes_html_json_and_refreshes_index(self):
        client = FakeClient(_good_reply())
        code, out, err = self._main(["20261005T180000_a", "20261005T190000_b"], client)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("RUN DATA (JSON)", client.calls[0]["prompt"])
        self.assertTrue(client.calls[0]["system"])
        pages = sorted((self.root / COMPARISONS_DIR).glob("*_2runs.html"))
        metas = sorted((self.root / COMPARISONS_DIR).glob("*_2runs.json"))
        self.assertEqual(len(pages), 1)
        self.assertEqual(len(metas), 1)
        self.assertIn(str(pages[0]), out)
        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        self.assertEqual(meta["run_ids"], ["20261005T180000_a", "20261005T190000_b"])
        self.assertEqual(meta["eval_provider"], "anthropic")
        self.assertEqual(meta["eval_model"], "claude-sonnet-5-5")
        self.assertEqual(meta["done_reason"], "stop")
        self.assertEqual(meta["usage"]["output_tokens"], 800)
        self.assertGreater(meta["prompt_chars"], 1000)
        self.assertIn("## Summary", meta["markdown"])
        # 9000 input * $2/MTok + 800 output * $10/MTok
        self.assertAlmostEqual(meta["estimated_cost_usd"], 0.026)
        self.assertIn("estimated cost $0.0260", out)
        page = pages[0].read_text(encoding="utf-8")
        self.assertIn("<h2>Summary</h2>", page)
        self.assertIn("<strong>20261005T180000_author_a</strong>", page)
        self.assertIn("<table><thead><tr><th>run</th>", page)
        index = (self.root / "index.html").read_text(encoding="utf-8")
        self.assertIn(f"comparisons/{pages[0].name}", index)
        self.assertNotIn('value="comparisons"', index)

    def test_dry_run_prompt_makes_no_call(self):
        client = FakeClient(_good_reply())
        code, out, _ = self._main(["20261005T180000_a", "20261005T190000_b", "--dry-run-prompt"], client)
        self.assertEqual(code, 0)
        self.assertEqual(client.calls, [])
        self.assertIn("=== SYSTEM ===", out)
        self.assertIn("20261005T190000_b", out)
        self.assertIn("No model call was made.", out)
        self.assertIn("anthropic:claude-sonnet-5-5", out)
        self.assertFalse((self.root / COMPARISONS_DIR).exists())

    def test_unknown_run_id_is_reported(self):
        client = FakeClient(_good_reply())
        code, _, err = self._main(["20261005T180000_a", "nope_1", "nope_2"], client)
        self.assertEqual(code, 2)
        self.assertIn("nope_1, nope_2", err)
        self.assertEqual(client.calls, [])
        code, _, err = self._main(["20261005T180000_a"], client)
        self.assertEqual(code, 2)
        self.assertIn("at least two", err)

    def test_model_failure_saves_nothing(self):
        for client in (
            FakeClient(error=ConnectionError("server down")),
            FakeClient(ModelReply("", done_reason="refusal", provider="anthropic", model="claude-sonnet-5-5")),
            FakeClient(ModelReply("   ", done_reason="stop")),
        ):
            code, _, err = self._main(["20261005T180000_a", "20261005T190000_b"], client)
            self.assertEqual(code, 2)
            self.assertTrue(err.startswith("Error:"))
            self.assertFalse((self.root / COMPARISONS_DIR).exists())
        self.assertFalse((self.root / "index.html").exists())


class MarkdownTests(unittest.TestCase):
    def test_raw_html_and_scripts_are_escaped(self):
        markdown = (
            "# Title <script>alert(1)</script>\n\n"
            "Text with <img src=x onerror=alert(1)> and **bold** and *it* and `<b>code</b>`.\n\n"
            "[click](javascript:alert(1))\n\n"
            "| a | b |\n|---|---|\n| <i>x</i> | **y** |\n\n"
            "- item <svg onload=alert(1)>\n  - nested\n1. first\n\n"
            "```\n<script>bad()</script>\n```\n\n> quote <iframe>\n"
        )
        output = run_compare.markdown_to_html(markdown)
        self.assertNotIn("<script", output)
        self.assertNotIn("<img", output)
        self.assertNotIn("<svg", output)
        self.assertNotIn("<iframe", output)
        self.assertNotIn("<a ", output)
        self.assertNotIn("javascript:", output)
        self.assertNotIn("href", output)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", output)
        self.assertIn("<strong>bold</strong>", output)
        self.assertIn("<em>it</em>", output)
        self.assertIn("<code>&lt;b&gt;code&lt;/b&gt;</code>", output)
        self.assertIn("<td>&lt;i&gt;x&lt;/i&gt;</td><td><strong>y</strong></td>", output)
        self.assertIn("<pre><code>&lt;script&gt;bad()&lt;/script&gt;</code></pre>", output)
        self.assertIn("<ul><li>item &lt;svg onload=alert(1)&gt;<ul><li>nested</li></ul></li></ul>", output.replace("\n", ""))
        self.assertIn("<ol><li>first</li></ol>", output.replace("\n", ""))
        # Every tag in the output is one the converter writes, without attributes.
        tags = set(re.findall(r"</?([a-z0-9]+)[^>]*>", output))
        self.assertLessEqual(tags, {"h1", "p", "strong", "em", "code", "table", "thead", "tbody", "tr", "th", "td", "ul", "ol", "li", "pre", "blockquote", "br"})
        self.assertEqual(re.findall(r"<[a-z0-9]+\s", output), [])

    def test_run_ids_with_underscores_are_not_italicized(self):
        output = run_compare.markdown_to_html("Run 20261005_gemma4_e4b_failed and a_b_c")
        self.assertEqual(output, "<p>Run 20261005_gemma4_e4b_failed and a_b_c</p>")


if __name__ == "__main__":
    unittest.main()
