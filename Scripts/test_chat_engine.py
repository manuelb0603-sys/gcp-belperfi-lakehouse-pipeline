"""Finance chat engine tests with a scripted fake chat client (no Ollama, Claude, or BigQuery)."""

import copy
import json
import unittest

from chat_test_fixtures import TODAY, fixture_frame
from finance.budget import BudgetExceededError
from finance.chat import prompts
from finance.chat.data import build_catalog
from finance.chat.engine import run_turn
from finance.chat.session import ChatSession, Turn
from finance.model_clients import ChatReply
from finance.settings import ReportSettings

FRAME = fixture_frame()
CATALOG = build_catalog(FRAME)
QUARTER_SPEC = {"metric": "net_spend", "group_by": ["month"], "period": "last_quarter"}


def reply(text="", calls=None, done=None):
    calls = calls or []
    return ChatReply(text, calls, done_reason=done or ("tool_use" if calls else "stop"), provider="fake", model="fake-model")


def call(name, call_id="call_1", **arguments):
    return {"id": call_id, "name": name, "arguments": arguments}


class FakeChatClient:
    """Scripted chat client: returns (or raises) the queued replies in order and records each call."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, *replies, think=None, default_timeout=600, on_call=None):
        self.replies = list(replies)
        self.calls = []
        self.think = think
        self.default_timeout = default_timeout
        self.on_call = on_call

    def chat(self, messages, *, system=None, tools=None, json_schema=None, timeout=None, think=None):
        self.calls.append({"messages": copy.deepcopy(messages), "system": system, "tools": tools,
                           "timeout": timeout, "think": think})
        if self.on_call:
            self.on_call(len(self.calls))
        if not self.replies:
            raise AssertionError("FakeChatClient ran out of scripted replies")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def settings(**overrides):
    return ReportSettings.from_config({"Monthly_budget": 5000, "Fixed_costs": 2000, **overrides})


def run(client, question, session=None, clock=None, **settings_overrides):
    session = session or ChatSession("test")
    kwargs = {"clock": clock} if clock else {}
    events = list(run_turn(session, question, client=client, df=FRAME, catalog=CATALOG,
                           settings=settings(**settings_overrides), today=TODAY, **kwargs))
    return session, events


def kinds(events):
    return [event.kind + (f":{event.status}" if event.status else "") for event in events]


def text_of(events):
    return next(event.content for event in events if event.kind == "text")


class TestToolLoop(unittest.TestCase):
    def test_tool_call_then_narrative_yields_explanation_text_figure_table(self):
        client = FakeChatClient(
            reply(calls=[call("analyze_spending", **QUARTER_SPEC)]),
            reply("Net spend last quarter was $1,262.00, highest in 2026-08 at $490.50. 🍣"),
        )
        session, events = run(client, "How much did I spend each month last quarter?")
        self.assertEqual(kinds(events), ["explanation:pending", "explanation:pending", "explanation:done", "text", "figure", "table"])
        self.assertEqual(text_of(events), "Net spend last quarter was $1,262.00, highest in 2026-08 at $490.50. 🍣")
        explanation = events[2].content
        self.assertIn("Net spend by month, Jul–Sep 2026 (last quarter).", explanation)
        self.assertIn('Spec: `{"group_by": ["month"], "period": "last_quarter"}`', explanation)
        table = events[-1].content
        self.assertEqual(table["month"].tolist(), ["2026-07", "2026-08", "2026-09"])
        # The second call carries the tool result: compact JSON summary, no transaction rows.
        tool_messages = [m for m in client.calls[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        payload = json.loads(tool_messages[0]["content"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["total"], 1262.0)
        self.assertNotIn("WHOLEFDS", tool_messages[0]["content"])
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        # Tools and the catalog-bearing system prompt are sent on every call.
        self.assertEqual([tool["name"] for tool in client.calls[0]["tools"]], ["analyze_spending", "list_values", "get_month_snapshot"])
        self.assertIn("Today: 2026-10-05", client.calls[0]["system"])
        self.assertEqual(session.last_spec, {"group_by": ["month"], "period": "last_quarter"})

    def test_bad_spec_gets_an_error_result_then_a_corrected_call(self):
        client = FakeChatClient(
            reply(calls=[call("analyze_spending", filters={"merchant": ["Qwerty Plaza"]}, period="last_quarter")]),
            reply(calls=[call("analyze_spending", "call_2", filters={"merchant": ["Starbucks"]}, period="last_quarter")]),
            reply("You spent $21.50 at Starbucks last quarter."),
        )
        _, events = run(client, "Starbucks last quarter?")
        first_result = [m for m in client.calls[1]["messages"] if m["role"] == "tool"][0]
        self.assertTrue(first_result["is_error"])
        error = json.loads(first_result["content"])
        self.assertFalse(error["ok"])
        self.assertIn("Closest valid merchant values", error["error"])
        self.assertEqual(text_of(events), "You spent $21.50 at Starbucks last quarter.")
        done = next(event for event in events if event.status == "done")
        self.assertIn("rejected and corrected", done.content)
        self.assertEqual(events[-1].content["net_spend"].tolist(), [21.5])

    def test_direct_answer_without_tools(self):
        client = FakeChatClient(reply("I can answer questions about your card spending, like monthly totals by category."))
        _, events = run(client, "Hi there")
        self.assertEqual(kinds(events), ["explanation:pending", "explanation:done", "text"])
        self.assertEqual(events[1].content, prompts.NO_TOOLS_TEXT)

    def test_empty_length_reply_with_think_retries_once_without_thinking(self):
        client = FakeChatClient(reply("", done="length"), reply("Ask me about your spending."), think=True)
        _, events = run(client, "Hello")
        self.assertEqual([c["think"] for c in client.calls], [None, False])
        self.assertEqual(text_of(events), "Ask me about your spending.")
        # Without thinking there is no retry; the friendly length message is shown.
        client = FakeChatClient(reply("", done="length"), think=False)
        _, events = run(client, "Hello")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(text_of(events), prompts.LENGTH_MESSAGE)

    def test_tool_round_cap(self):
        tool_reply = lambda n: reply(calls=[call("analyze_spending", f"call_{n}", **QUARTER_SPEC)])  # noqa: E731
        client = FakeChatClient(tool_reply(1), tool_reply(2), tool_reply(3))
        _, events = run(client, "Loop forever", chat_max_tool_rounds=2)
        self.assertEqual(len(client.calls), 3)
        last_messages = client.calls[2]["messages"]
        self.assertEqual(last_messages[-1], {"role": "user", "content": prompts.FINAL_ROUND_NOTE})
        self.assertTrue(text_of(events).startswith(prompts.FALLBACK_INTRO))
        self.assertEqual(sum(1 for m in last_messages if m["role"] == "tool"), 2)  # the third call never ran
        self.assertEqual(events[-1].kind, "table")

    def test_deadline_stops_the_turn_with_a_friendly_message(self):
        now = [0.0]
        client = FakeChatClient(reply(calls=[call("analyze_spending", **QUARTER_SPEC)]),
                                on_call=lambda n: now.__setitem__(0, now[0] + 31))
        _, events = run(client, "Slow question", clock=lambda: now[0], chat_turn_seconds=30)
        self.assertEqual(len(client.calls), 1)
        self.assertLessEqual(client.calls[0]["timeout"], 30)
        text = text_of(events)
        self.assertTrue(text.startswith(prompts.DEADLINE_MESSAGE.format(seconds=30)))
        self.assertIn(prompts.FALLBACK_INTRO, text)
        self.assertEqual(events[-1].kind, "table")

    def test_budget_refusal_and_unavailable_model_are_friendly(self):
        client = FakeChatClient(BudgetExceededError("Claude daily budget reached: $15.00 of $15.00 spent today"))
        session, events = run(client, "Spend by month?")
        text = text_of(events)
        self.assertTrue(text.startswith("The Claude budget guard stopped this answer"))
        self.assertIn("$15.00 of $15.00", text)
        self.assertNotIn("Traceback", text)
        self.assertEqual(session.turns[-1].assistant_text, text)
        _, events = run(FakeChatClient(ConnectionError("connection refused")), "Spend by month?")
        self.assertTrue(text_of(events).startswith("The chat model is not reachable right now (connection refused)"))
        with self.assertLogs("finance.chat.engine", level="ERROR"):
            _, events = run(FakeChatClient(KeyError("boom")), "Spend by month?")
        self.assertTrue(text_of(events).startswith("Something went wrong while answering (KeyError"))

    def test_narrative_guard_regenerates_once(self):
        client = FakeChatClient(
            reply(calls=[call("analyze_spending", **QUARTER_SPEC)]),
            reply("You spent $999.99 last quarter."),
            reply("You spent $1,262.00 last quarter."),
        )
        _, events = run(client, "Total last quarter?")
        self.assertEqual(text_of(events), "You spent $1,262.00 last quarter.")
        fix = client.calls[2]["messages"][-1]
        self.assertEqual(fix["role"], "user")
        self.assertIn("$999.99 does not match", fix["content"])
        self.assertEqual(client.calls[2]["messages"][-2]["content"], "You spent $999.99 last quarter.")

    def test_narrative_guard_flags_a_second_mismatch(self):
        client = FakeChatClient(
            reply(calls=[call("analyze_spending", **QUARTER_SPEC)]),
            reply("You spent $999.99 last quarter."),
            reply("You spent about $5,000 last quarter."),
        )
        _, events = run(client, "Total last quarter?")
        self.assertEqual(text_of(events), "You spent about $5,000 last quarter.\n\n" + prompts.AUTHORITATIVE_NOTE)
        self.assertEqual(len(client.calls), 3)

    def test_month_snapshot_tool_feeds_the_guard(self):
        client = FakeChatClient(
            reply(calls=[call("get_month_snapshot", month="2026-09")]),
            reply("September net spend was $470.00, down 4.18% from August."),
        )
        _, events = run(client, "How was September?")
        self.assertEqual(text_of(events), "September net spend was $470.00, down 4.18% from August.")
        self.assertIn("Monthly snapshot for 2026-09 compared with 2026-08", events[-2].content)
        self.assertEqual(kinds(events)[-2:], ["explanation:done", "text"])


class TestSessionHistory(unittest.TestCase):
    def test_follow_up_sees_the_previous_spec_and_turns(self):
        session = ChatSession("s")
        run(FakeChatClient(reply(calls=[call("analyze_spending", **QUARTER_SPEC)]), reply("Done: $1,262.00.")),
            "Monthly spend last quarter", session=session)
        client = FakeChatClient(reply("Sure."))
        run(client, "Now only Amex", session=session)
        messages = client.calls[0]["messages"]
        self.assertEqual(messages[0], {"role": "user", "content": "Monthly spend last quarter"})
        self.assertEqual(messages[1], {"role": "assistant", "content": "Done: $1,262.00."})
        self.assertIn(prompts.CONTEXT_LAST_SPEC_HEADER, messages[-1]["content"])
        self.assertIn('{"group_by": ["month"], "period": "last_quarter"}', messages[-1]["content"])
        self.assertTrue(messages[-1]["content"].endswith("USER QUESTION:\nNow only Amex"))

    def test_history_truncation_on_undo_or_retry(self):
        session = ChatSession("s")
        session.add_turn(Turn("q1", specs=[{"metric": "refunds"}], assistant_text="a1"))
        session.add_turn(Turn("q2", specs=[{"metric": "transaction_count"}], assistant_text="a2"))
        session.add_turn(Turn("q3", assistant_text="a3"))
        ui_history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        self.assertEqual(session.sync_history(ui_history), 2)
        self.assertEqual([turn.user_text for turn in session.turns], ["q1"])
        self.assertEqual(session.last_spec, {"metric": "refunds"})
        self.assertEqual(session.sync_history(ui_history), 0)
        self.assertEqual(session.sync_history([]), 1)
        self.assertIsNone(session.last_spec)

    def test_retry_through_the_engine_replaces_the_last_turn(self):
        session = ChatSession("s")
        run(FakeChatClient(reply("first answer")), "q1", session=session)
        run(FakeChatClient(reply("second answer")), "q2", session=session)
        session.sync_history([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "first answer"}])
        client = FakeChatClient(reply("retried answer"))
        run(client, "q2 again", session=session)
        self.assertEqual([t.user_text for t in session.turns], ["q1", "q2 again"])
        self.assertEqual([m["content"] for m in client.calls[0]["messages"][:2]], ["q1", "first answer"])

    def test_rolling_summary_of_older_turns(self):
        session = ChatSession("s", history_turns=2)
        for number in range(1, 5):
            session.add_turn(Turn(f"question {number}", captions=[f"Caption {number}. More."], assistant_text=f"answer {number}"))
        summary = session.rolling_summary()
        self.assertEqual(summary.splitlines(), [
            "- Q: question 1 | computed: Caption 1. More. | answer: answer 1",
            "- Q: question 2 | computed: Caption 2. More. | answer: answer 2",
        ])
        self.assertEqual(summary, session.rolling_summary())  # deterministic
        self.assertEqual([m["content"] for m in session.history_messages()], ["question 3", "answer 3", "question 4", "answer 4"])
        client = FakeChatClient(reply("ok"))
        run(client, "question 5", session=session)
        current = client.calls[0]["messages"][-1]["content"]
        self.assertTrue(current.startswith(prompts.CONTEXT_SUMMARY_HEADER + "\n- Q: question 1"))
        # The summary is bounded: the oldest lines are dropped first.
        long_session = ChatSession("s", history_turns=1)
        for number in range(40):
            long_session.add_turn(Turn("x" * 200, assistant_text="y" * 200))
        bounded = long_session.rolling_summary()
        self.assertLessEqual(len(bounded), 1300)
        self.assertTrue(bounded.startswith("- ("))


if __name__ == "__main__":
    unittest.main()
