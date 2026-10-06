"""Import smoke test for the Gradio app with real gradio and faked BigQuery, memory, and model clients."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.chat_test_fixtures import fixture_frame
from tests.test_chat_engine import FakeChatClient, call, reply

SCRIPTS = Path(__file__).resolve().parents[1]

try:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr
except ImportError:  # pragma: no cover - environments without the app dependencies
    gr = None


@unittest.skipIf(gr is None, "gradio is not installed")
class TestChatApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("finance_chat_app_under_test", SCRIPTS / "apps" / "finance_app.py")
        module = importlib.util.module_from_spec(spec)
        with (
            patch("finance.settings.load_config", return_value={"project_name": "p", "model_name": "gemma4:12b",
                                                                 "model_server_url": "http://fake:11434"}),
            patch("finance.memory.open_memory", return_value=None),
            patch("finance.bigquery_tools.get_spending_years", return_value=[2026]),
        ):
            spec.loader.exec_module(module)
        cls.app = module

    def test_blocks_build_with_tabs_and_chat(self):
        self.assertIsInstance(self.app.app, gr.Blocks)
        config = self.app.app.get_config_file()
        labels = {str(component.get("props", {}).get("label")) for component in config["components"]}
        self.assertIn("Monthly report", labels)
        self.assertIn("Chat", labels)
        self.assertIn("Focus month", labels)
        self.assertEqual(self.app.focus_month_choices()[0], "Latest month")
        self.assertEqual(len(self.app.focus_month_choices()), 26)

    def test_chat_respond_yields_messages_with_plot_and_dataframe(self):
        client = FakeChatClient(
            reply(calls=[call("analyze_spending", metric="net_spend", group_by=["month"], period="last_quarter")]),
            reply("Net spend last quarter was $1,262.00."),
        )
        request = SimpleNamespace(session_hash="smoke")
        with (
            patch.object(self.app.chat_data, "load_transactions", return_value=fixture_frame()),
            patch.object(self.app, "build_model_client", return_value=client) as build,
            patch("finance.chat.engine.date") as fake_date,
        ):
            from tests.chat_test_fixtures import TODAY
            fake_date.today.return_value = TODAY
            outputs = list(self.app.chat_respond("Monthly spend last quarter", [], request))
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs, {"run_id": "chat"})
        self.assertEqual(build.call_args.args[1], "chat")
        first, last = outputs[0], outputs[-1]
        self.assertEqual(first[0].metadata["status"], "pending")
        self.assertEqual(len(last), 4)
        explanation, text, plot, table = last
        self.assertEqual(explanation.metadata, {"title": "How I computed this", "status": "done"})
        self.assertEqual(text.content, "Net spend last quarter was $1,262.00.")
        self.assertIsInstance(plot.content, gr.Plot)
        self.assertIsInstance(table.content, gr.Dataframe)
        # The chatbot accepts the component messages (postprocess as the UI would).
        chatbot = gr.Chatbot()
        chatbot.postprocess([{"role": "user", "content": "Monthly spend last quarter"}, *last])
        session = self.app.chat_session(request)
        self.assertEqual(len(session.turns), 1)
        self.assertIsNotNone(session.client)

    def test_data_and_model_setup_errors_are_messages(self):
        request = SimpleNamespace(session_hash="errors")
        with patch.object(self.app.chat_data, "load_transactions", side_effect=RuntimeError("bigquery down")):
            [[message]] = list(self.app.chat_respond("hi", [], request))
        self.assertEqual(message.content, "Could not load your transactions: bigquery down")
        with (
            patch.object(self.app.chat_data, "load_transactions", return_value=fixture_frame()),
            patch.object(self.app, "build_model_client", side_effect=ValueError("No model configured for role 'chat'")),
        ):
            [[message]] = list(self.app.chat_respond("hi", [], request))
        self.assertIn("No model configured", message.content)

    def test_example_chips_come_from_the_data_profile(self):
        from tests.chat_test_fixtures import SPARSE_TODAY, frame_of, sparse_merchant_rows
        from finance.chat.data import ChatData, build_catalog
        from finance.chat.profile import build_data_profile

        frame = frame_of(sparse_merchant_rows())
        loaded = ChatData(frame, build_catalog(frame), build_data_profile(frame, today=SPARSE_TODAY))
        self.assertEqual(self.app.CHAT_EXAMPLES, list(self.app.chat_prompts.STATIC_EXAMPLES))
        with patch.object(self.app.chat_data, "load_chat_data", return_value=loaded):
            update = self.app.load_chat_examples()
            focus, status, chips = self.app.refresh_chat_data()
        texts = [example["text"] for example in update["examples"]]
        self.assertIn("Compare Amex vs Chase by month", texts)
        self.assertFalse(any("merchant" in text.lower() or "store" in text.lower() for text in texts))
        self.assertEqual(chips, update)
        self.assertIn("Loaded 600 transactions from 2025-08-20 to 2026-08-19.", status)
        with patch.object(self.app.chat_data, "load_chat_data", side_effect=RuntimeError("bigquery down")):
            fallback = self.app.load_chat_examples()
        self.assertEqual([e["text"] for e in fallback["examples"]], list(self.app.chat_prompts.STATIC_EXAMPLES))

    def test_focus_month_is_per_session(self):
        request = SimpleNamespace(session_hash="focus")
        self.app.set_focus_month("2026-08", request)
        self.assertEqual(str(self.app.chat_session(request).focus_month), "2026-08-01")
        self.app.set_focus_month("Latest month", request)
        self.assertIsNone(self.app.chat_session(request).focus_month)


if __name__ == "__main__":
    unittest.main()
