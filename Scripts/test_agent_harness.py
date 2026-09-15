import unittest

from agent_harness import HarnessLimits, ReportHarness, ToolSpec, parse_json_response, sanitize_html
from report_models import ReportSnapshot


VALIDATOR_STYLE_RESPONSE = '''```json
{
  "decision": "pass",
  "summary": "Looks good.",
  "findings": [
    {
      "severity": "info",
      "category": "html",
      "message": "The report uses a custom 'daifuku' inspired palette and layout as requested.",
      "evidence": "HTML/CSS used to create a themed, responsive layout.",
      "revision_instruction": null
    }
  ]
}
```'''

AUTHOR_RESPONSE_WITH_LITERAL_NEWLINE = '''```json
{
  "report_html": "<html>
<body><h1>Weekly report</h1></body>
</html>",
  "plain_text": "Weekly report"
}
```'''

AUTHOR_RESPONSE_WITH_OUTER_QUOTES = "'```json\\n{\\n  \\\"report_html\\\": \\\"<html>\\n<body>Weekly report</body>\\n</html>\\\",\\n  \\\"plain_text\\\": \\\"Weekly report\\\"\\n}\\n```'"


class TestAgentHarness(unittest.TestCase):
    def test_sanitize_html_removes_void_blocked_tags_and_preserves_report(self):
        cleaned = sanitize_html(
            '<html><head><meta charset="UTF-8"></head><body><p>Report</p></body></html>'
        )

        self.assertNotIn("<meta", cleaned.lower())
        self.assertIn("<p>Report</p>", cleaned)

    def test_author_call_accepts_tool_call_response(self):
        tool_response = '{"tool_calls": [{"name": "get_report_history", "arguments": {}}]}'
        harness = ReportHarness(
            "http://localhost:11434/api/generate",
            "test-model",
            query=lambda *args, **kwargs: tool_response,
        )

        result = harness._author_call("prompt")

        self.assertEqual(result["tool_calls"][0]["name"], "get_report_history")

    def test_parse_json_response_rejects_literal_newlines_inside_report_fields(self):
        with self.assertRaises(ValueError):
            parse_json_response(
                AUTHOR_RESPONSE_WITH_LITERAL_NEWLINE,
                required_keys={"report_html", "plain_text"},
            )

    def test_parse_json_response_rejects_repr_like_fenced_response(self):
        with self.assertRaises(ValueError):
            parse_json_response(
                AUTHOR_RESPONSE_WITH_OUTER_QUOTES,
                required_keys={"report_html", "plain_text"},
            )

    def test_parse_json_response_rejects_validator_payload_without_report_keys(self):
        with self.assertRaises(ValueError):
            parse_json_response(VALIDATOR_STYLE_RESPONSE, required_keys={"report_html", "plain_text"})

    def test_author_call_logs_raw_response_for_debugging(self):
        harness = ReportHarness(
            "http://localhost:11434/api/generate",
            "test-model",
            query=lambda *args, **kwargs: VALIDATOR_STYLE_RESPONSE,
        )

        result = harness._author_call("prompt")

        self.assertEqual(result.get("report_html"), "")
        call_events = [event for event in harness.trace.events if event.get("event") == "model_call"]
        self.assertEqual(call_events[-1]["prompt"], "prompt")
        self.assertEqual(call_events[-1]["prompt_chars"], len("prompt"))
        response_events = [event for event in harness.trace.events if event.get("event") == "model_response"]
        self.assertEqual(response_events[-1]["parse_status"], "error")
        self.assertEqual(response_events[-1]["raw_response"], VALIDATOR_STYLE_RESPONSE)
        self.assertEqual(response_events[-1]["response_chars"], len(VALIDATOR_STYLE_RESPONSE))

    def test_query_ollama_requests_json_format(self):
        import agent_harness

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": '{"report_html":"<p>ok</p>","plain_text":"ok"}'}

        class FakeRequests:
            def __init__(self):
                self.payload = None

            def post(self, url, json, timeout):
                self.payload = json
                return FakeResponse()

        fake_requests = FakeRequests()
        original_requests = agent_harness.__dict__.get("requests")
        import sys
        sys.modules["requests"] = fake_requests
        try:
            agent_harness.query_ollama("prompt", "http://ollama", "model", timeout=10)
        finally:
            if original_requests is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original_requests

        self.assertEqual(fake_requests.payload["format"], "json")

    def test_model_call_exception_is_added_to_trace(self):
        failure = TimeoutError("model timed out")

        def failing_query(*args, **kwargs):
            raise failure

        harness = ReportHarness("http://ollama", "test-model", query=failing_query)

        with self.assertRaises(TimeoutError):
            harness._author_call("prompt")

        exception_events = [event for event in harness.trace.events if event.get("event") == "exception"]
        self.assertEqual(exception_events[-1]["phase"], "author_model_call")
        self.assertEqual(exception_events[-1]["exception_type"], "TimeoutError")
        self.assertEqual(exception_events[-1]["error"], "model timed out")

    def test_tool_exception_is_added_to_trace(self):
        responses = ['{"tool_calls":[{"name":"broken_tool","arguments":{}}]}']

        def failing_query(*args, **kwargs):
            return responses.pop(0)

        def broken_tool(arguments):
            raise RuntimeError("tool failed")

        snapshot = ReportSnapshot(
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
        harness = ReportHarness("http://ollama", "test-model", query=failing_query)

        with self.assertRaises(RuntimeError):
            harness.run("objective", snapshot, "author context", {"broken_tool": ToolSpec("broken_tool", "broken", broken_tool)})

        exception_events = [event for event in harness.trace.events if event.get("event") == "exception"]
        self.assertEqual(exception_events[-1]["phase"], "initial_tool_call")
        self.assertEqual(exception_events[-1]["tool"], "broken_tool")

    def test_malformed_validator_findings_are_added_to_trace(self):
        harness = ReportHarness(
            "http://ollama",
            "test-model",
            query=lambda *args, **kwargs: '{"decision":"fail","findings":[{"unexpected":"field"}]}',
        )

        decision = harness._validator_call("validator prompt")

        self.assertFalse(decision.passed)
        exception_events = [event for event in harness.trace.events if event.get("event") == "exception"]
        self.assertEqual(exception_events[-1]["phase"], "validator_payload")

    def test_revision_can_request_additional_tool_evidence(self):
        responses = [
            '{"report_html":"<html><body>Initial</body></html>","plain_text":"Initial"}',
            '{"decision":"fail","summary":"Need more evidence.","findings":[]}',
            '{"tool_calls":[{"name":"get_spending_summary","arguments":{}}]}',
            '{"report_html":"<html><body>Revised</body></html>","plain_text":"Revised"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        tool_results = []

        def fake_query(*args, **kwargs):
            return responses.pop(0)

        def get_summary(arguments):
            tool_results.append(arguments)
            return {"summary": "additional evidence"}

        snapshot = ReportSnapshot(
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
        harness = ReportHarness(
            "http://ollama",
            "test-model",
            limits=HarnessLimits(max_agent_rounds=2, max_tool_calls=2, max_calls_per_tool=2, max_validation_cycles=1),
            query=fake_query,
        )

        report, decision, trace = harness.run(
            "objective",
            snapshot,
            "author context",
            {"get_spending_summary": ToolSpec("get_spending_summary", "summary", get_summary)},
            validator_context="validator context",
        )

        self.assertTrue(decision.passed)
        self.assertEqual(report["plain_text"], "Revised")
        self.assertEqual(tool_results, [{}])
        author_prompts = [event["prompt"] for event in trace.events if event.get("event") == "model_call" and event.get("role") == "author"]
        self.assertIn("additional evidence", author_prompts[-1])
        self.assertIn("FAILED REPORT FOR REVISION", author_prompts[1])
        self.assertIn("<html><body>Initial</body></html>", author_prompts[1])
        validator_prompts = [event["prompt"] for event in trace.events if event.get("event") == "model_call" and event.get("role") == "validator"]
        self.assertIn("<html><body>Initial</body></html>", validator_prompts[0])


if __name__ == "__main__":
    unittest.main()
