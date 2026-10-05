import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import email_weekly_report as weekly
import run_archive


def _plain_part(message):
    alternative = message.get_payload()[0]
    plain, html = alternative.get_payload()
    return plain.get_payload(decode=True).decode("utf-8"), html.get_payload(decode=True).decode("utf-8")


class TestEmailMessage(unittest.TestCase):
    def test_agent_plain_text_is_used_for_text_part(self):
        message = weekly.build_email_message(
            "sender@example.com", "you@example.com", "<html><body><p>Hi</p></body></html>", plain_text="Hi in text"
        )
        plain, html = _plain_part(message)
        self.assertEqual(plain, "Hi in text")
        self.assertIn("<p>Hi</p>", html)

    def test_legacy_text_part_is_unchanged_without_plain_text(self):
        report = "First paragraph\n\nSecond paragraph"
        legacy = _plain_part(weekly.build_email_message("s", "r", report))[0]
        blank = _plain_part(weekly.build_email_message("s", "r", report, plain_text="  "))[0]
        self.assertIn("First paragraph", legacy)
        self.assertNotIn("Second paragraph", legacy)
        self.assertEqual(legacy, blank)


class FakeMemory:
    def __init__(self):
        self.saved = []
        self.facts = []

    def get_or_create_thread(self, channel, key):
        return "thread"

    def save_short_term(self, **kwargs):
        self.saved.append(kwargs)
        return 7

    def upsert_long_fact(self, key, text, **kwargs):
        self.facts.append((key, text))


class TestRememberReport(unittest.TestCase):
    def _remember(self, **kwargs):
        memory = FakeMemory()
        prompts = []
        with (
            patch.object(weekly, "query_ollama", side_effect=lambda prompt, *a, **k: prompts.append(prompt) or "[]"),
            patch.object(weekly, "log_status"),
        ):
            weekly._remember_report(
                memory, "s", ["r"], "r", "prompt", "<html><body>HTML report</body></html>", "model", "url", {}, **kwargs
            )
        return memory, prompts[0]

    def test_agent_mode_stores_html_and_plain_text_separately(self):
        memory, fact_prompt = self._remember(report_plain_text="Plain report")
        self.assertEqual(memory.saved[0]["response_html"], "<html><body>HTML report</body></html>")
        self.assertEqual(memory.saved[0]["response_text"], "Plain report")
        self.assertIn("plain-text report", fact_prompt)
        self.assertIn("Plain report", fact_prompt)
        self.assertNotIn("<html>", fact_prompt)

    def test_legacy_mode_keeps_html_for_both_fields(self):
        memory, fact_prompt = self._remember()
        self.assertEqual(memory.saved[0]["response_text"], memory.saved[0]["response_html"])
        self.assertIn("following HTML report", fact_prompt)
        self.assertIn("<html>", fact_prompt)


class TestAgentRunFailureArchive(unittest.TestCase):
    def test_harness_failure_is_archived_as_exception_with_harness_trace(self):
        error = RuntimeError("Unexpected harness failure")
        error.run_trace = SimpleNamespace(events=[{"event": "agent_round", "round": 1}])
        error.report_snapshot = SimpleNamespace(to_dict=lambda: {"latest_month": "2026-02-01"})
        saved = []
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(weekly, "RUN_ARCHIVE_ROOT", Path(temp_dir)),
            patch("finance.reporting.generate_finance_report", side_effect=error),
            patch.object(run_archive, "save_run", side_effect=lambda *args: saved.append(args)),
            patch.object(weekly, "send_email") as send_email,
        ):
            weekly.run_agent_weekly_report(
                {}, [], None, "s", ["r"], "r", "pw", "project", "url", "model:tag", "memory"
            )

        send_email.assert_not_called()
        root, run_id, manifest, events, html, text, metrics = saved[0]
        self.assertTrue(run_id.endswith("_model_tag_failed"))
        self.assertEqual(manifest["status"], "exception")
        self.assertEqual(metrics["status"], "exception")
        self.assertEqual(metrics["agent_rounds"], 1)
        self.assertEqual(events[0]["event"], "agent_round")
        self.assertIn("traceback", events[-2])
        self.assertIn("RuntimeError", events[-2]["traceback"])


class TestMemoryPath(unittest.TestCase):
    def test_memory_db_path_environment_override(self):
        with (
            patch.dict(os.environ, {"MEMORY_DB_PATH": "C:/tmp/custom.sqlite3"}),
            patch("finance.memory.MemoryStore.init_db") as init_db,
            patch.object(weekly, "log_status"),
        ):
            store = weekly._init_memory_module("memory")

        self.assertIsNotNone(store)
        init_db.assert_called_once()
        self.assertEqual(Path(store.db_path), Path("C:/tmp/custom.sqlite3"))

    def test_memory_init_failure_returns_none(self):
        with patch("finance.memory.MemoryStore.init_db", side_effect=OSError("locked")):
            self.assertIsNone(weekly._init_memory_module("memory"))


if __name__ == "__main__":
    unittest.main()
