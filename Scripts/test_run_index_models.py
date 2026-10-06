import json
import re
import tempfile
import unittest
from pathlib import Path

from finance.run_archive import run_role_models, trace_run_started, update_index


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_trace(run_dir, events):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _row(index_html, run_id):
    match = re.search(r"<tr><td>\d+</td><td><input[^>]*value=\"" + re.escape(run_id) + r"\".*?</tr>", index_html)
    assert match, f"row for {run_id} not found"
    return match.group(0)


def _cells(row):
    return re.findall(r"<td[^>]*>(.*?)</td>", row)


def _headers(index_html):
    first_table = index_html.split("<tbody>", 1)[0]
    return re.findall(r"<th>(.*?)</th>", first_table)


def _column(index_html, run_id, header):
    return _cells(_row(index_html, run_id))[_headers(index_html).index(header)]


class RunIndexModelColumnsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "runs"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_role_metadata_from_run_json_wins(self):
        run_dir = self.root / "20261005T155123_new"
        _write_json(run_dir / "run.json", {
            "author_role": {"role": "author", "provider": "ollama", "model": "gemma4:e4b"},
            "validator_role": {"role": "validator", "provider": "anthropic", "model": "claude-haiku-4-5"},
        })
        # Flat metrics fields disagree on purpose: resolved role metadata comes first.
        _write_json(run_dir / "metrics.json", {"model": "other", "validator_model": "other-v", "mode": "agent", "status": "passed"})
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertEqual(_column(index, run_dir.name, "author model"), "ollama:gemma4:e4b")
        self.assertEqual(_column(index, run_dir.name, "validator model"), "anthropic:claude-haiku-4-5")
        self.assertIn('title="validator: anthropic:claude-haiku-4-5"', _row(index, run_dir.name))
        self.assertIn('class="nowrap" title="author: ollama:gemma4:e4b"', _row(index, run_dir.name))
        self.assertIn('class="table-scroll"', index)
        self.assertNotIn("<th>model</th>", index)

    def test_role_metadata_from_progress_for_running_run(self):
        run_dir = self.root / "20261005T160000_live"
        _write_json(run_dir / "progress.json", {
            "status": "running", "model": "gemma4:e4b",
            "author_role": {"provider": "ollama", "model": "gemma4:e4b"},
            "validator_role": {"provider": "anthropic", "model": "claude-sonnet-5-5"},
        })
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertEqual(_column(index, run_dir.name, "validator model"), "anthropic:claude-sonnet-5-5")
        self.assertEqual(_column(index, run_dir.name, "end"), "running")

    def test_metrics_only_run(self):
        run_dir = self.root / "20261005T150000_metrics"
        _write_json(run_dir / "metrics.json", {
            "model": "gemma4:e4b", "validator_model": "claude-sonnet-5-5",
            "author_provider": "ollama", "validator_provider": "anthropic", "mode": "agent", "status": "passed",
            "anthropic_cost_usd": 0.0123,
        })
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertEqual(_column(index, run_dir.name, "author model"), "ollama:gemma4:e4b")
        self.assertEqual(_column(index, run_dir.name, "validator model"), "anthropic:claude-sonnet-5-5")
        self.assertEqual(_column(index, run_dir.name, "cost"), "$0.0123")
        self.assertEqual(_column(index, run_dir.name, "No."), "1")

    def test_trace_only_old_run_reads_run_started(self):
        run_dir = self.root / "20261005T121741_old"
        _write_trace(run_dir, [
            {"event": "run_started", "timestamp": 1791227872.3, "author_model": "gemma4:e4b", "validator_model": "qwen3.5:9b"},
            {"event": "model_call", "timestamp": 1791227873.0, "role": "author", "prompt": "x" * 200000},
            {"event": "run_completed", "timestamp": 1791227999.0},
        ])
        _write_json(run_dir / "metrics.json", {"mode": "agent", "status": "passed"})
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertEqual(_column(index, run_dir.name, "author model"), "gemma4:e4b")
        self.assertEqual(_column(index, run_dir.name, "validator model"), "qwen3.5:9b")
        self.assertIn('title="validator: qwen3.5:9b (provider not recorded)"', _row(index, run_dir.name))
        self.assertTrue(_column(index, run_dir.name, "start"))

    def test_trace_supplies_missing_provider_for_same_model(self):
        run_dir = self.root / "20261005T130000_mixed"
        _write_trace(run_dir, [{"event": "run_started", "timestamp": 1.0, "author_model": "a", "author_provider": "ollama",
                                "validator_model": "v", "validator_provider": "anthropic"}])
        roles = run_role_models(run_dir, metrics={"model": "a", "validator_model": "v"})
        self.assertEqual(roles["author"], {"provider": "ollama", "model": "a"})
        self.assertEqual(roles["validator"], {"provider": "anthropic", "model": "v"})

    def test_trace_head_read_is_bounded(self):
        run_dir = self.root / "late"
        events = [{"event": "agent_round", "timestamp": i} for i in range(6)]
        events.append({"event": "run_started", "timestamp": 9, "author_model": "late"})
        _write_trace(run_dir, events)
        self.assertIsNone(trace_run_started(run_dir / "trace.jsonl"))
        self.assertIsNone(trace_run_started(run_dir / "missing.jsonl"))

    def test_missing_data_is_blank(self):
        run_dir = self.root / "20261005T110000_bare"
        _write_json(run_dir / "metrics.json", {"mode": "agent", "status": "failed"})
        (run_dir / "trace.jsonl").write_text("not json\n", encoding="utf-8")
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertEqual(_column(index, run_dir.name, "author model"), "")
        self.assertEqual(_column(index, run_dir.name, "validator model"), "")
        self.assertEqual(_column(index, run_dir.name, "guidance"), "")
        self.assertNotIn("title=", _row(index, run_dir.name))

    def test_model_names_are_html_escaped(self):
        run_dir = self.root / "20261005T100000_xss"
        _write_json(run_dir / "run.json", {
            "author_role": {"provider": "ollama", "model": '<script>alert("x")</script>'},
            "validator_role": {"provider": "anth\"ropic", "model": "v&1"},
        })
        _write_json(run_dir / "metrics.json", {"mode": "agent", "guidance": "<b>strict</b>"})
        index = update_index(self.root).read_text(encoding="utf-8")
        row = _row(index, run_dir.name)
        self.assertNotIn("<script>alert", row)
        self.assertIn("ollama:&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", row)
        self.assertIn("anth&quot;ropic:v&amp;1", row)
        self.assertIn("&lt;b&gt;strict&lt;/b&gt;", row)
        # The embedded metrics JSON cannot close the script element either.
        script = index.split("const metrics = ", 1)[1]
        self.assertNotIn("</script>alert", script)
        self.assertIn("<\\/script>", script)

    def test_guidance_column_present_and_absent(self):
        with_guidance = self.root / "20261005T090000_guided"
        _write_json(with_guidance / "run.json", {"guidance": "coach"})
        _write_json(with_guidance / "metrics.json", {"model": "m", "mode": "agent", "guidance": "ignored"})
        from_progress = self.root / "20261005T080000_progress"
        _write_json(from_progress / "progress.json", {"status": "passed", "guidance": {"mode": "light", "notes": 2}})
        flag = self.root / "20261005T070000_flag"
        _write_json(flag / "metrics.json", {"model": "m", "guidance": False})
        without = self.root / "20261005T060000_plain"
        _write_json(without / "metrics.json", {"model": "m", "mode": "agent"})
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertIn("guidance", _headers(index))
        self.assertEqual(_column(index, with_guidance.name, "guidance"), "coach")
        self.assertEqual(_column(index, from_progress.name, "guidance"), "light")
        self.assertEqual(_column(index, flag.name, "guidance"), "off")
        self.assertEqual(_column(index, without.name, "guidance"), "")

    def test_compare_table_includes_model_and_guidance_fields(self):
        run_dir = self.root / "20261005T155328_cmp"
        _write_json(run_dir / "metrics.json", {"model": "gemma4:e4b", "author_provider": "ollama",
                                               "validator_model": "claude-sonnet-5-5", "validator_provider": "anthropic",
                                               "guidance": "coach"})
        index = update_index(self.root).read_text(encoding="utf-8")
        self.assertIn("const keys = ['start','end','author model','validator model','guidance',", index)
        metrics = json.loads(index.split("const metrics = ", 1)[1].split(";\n", 1)[0])
        entry = metrics[run_dir.name]
        self.assertEqual(entry["author model"], "ollama:gemma4:e4b")
        self.assertEqual(entry["validator model"], "anthropic:claude-sonnet-5-5")
        self.assertEqual(entry["guidance"], "coach")
        self.assertIn("#comparison td,#comparison th{white-space:nowrap}", index)


if __name__ == "__main__":
    unittest.main()
