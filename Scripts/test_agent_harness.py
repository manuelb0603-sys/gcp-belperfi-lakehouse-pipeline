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

    def test_author_call_derives_plain_text_when_only_html_is_returned(self):
        harness = ReportHarness(
            "http://localhost:11434/api/generate",
            "test-model",
            query=lambda *args, **kwargs: '{"report_html":"<html><head><style>body{color:red}</style></head><body><h1>Hello</h1><p>Report</p></body></html>"}',
        )

        result = harness._author_call("prompt")

        self.assertNotIn("error", result)
        self.assertIn("Hello", result["plain_text"])
        self.assertIn("Report", result["plain_text"])
        self.assertNotIn("color:red", result["plain_text"])

    def test_malformed_author_json_is_retried_before_validation(self):
        responses = [
            '{"report_html":"<html><body>broken',
            '{"report_html":"<html><body>Recovered report</body></html>","plain_text":"Recovered report"}',
            '{"decision":"pass","summary":"Accepted","findings":[]}',
        ]
        prompts = []

        def retrying_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

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
            limits=HarnessLimits(max_agent_rounds=2),
            query=retrying_query,
        )

        report, decision, trace = harness.run("objective", snapshot, "author context", {})

        self.assertTrue(decision.passed)
        self.assertIn("Recovered report", report["report_html"])
        self.assertIn("AUTHOR OUTPUT FORMAT ERROR", prompts[1])
        retry_events = [event for event in trace.events if event.get("event") == "author_format_retry"]
        self.assertEqual(retry_events[0]["phase"], "initial")
        validator_prompts = [event["prompt"] for event in trace.events if event.get("event") == "model_call" and event.get("role") == "validator"]
        self.assertIn("Recovered report", validator_prompts[0])
        self.assertNotIn("Missing report_html in author output", validator_prompts[0])
        cycle_results = [event for event in trace.events if event.get("event") == "validation_cycle_result"]
        self.assertEqual(len(cycle_results), 1)
        self.assertEqual(cycle_results[0]["cycle"], 1)
        self.assertEqual(cycle_results[0]["decision"], "pass")

    def test_malformed_validator_json_gets_one_format_retry(self):
        responses = [
            '{"decision":"fail",',
            '{"decision":"pass","summary":"Accepted","findings":[]}',
        ]
        prompts = []

        def retrying_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        harness = ReportHarness("http://ollama", "test-model", query=retrying_query)

        decision = harness._validator_call("validator prompt")

        self.assertTrue(decision.passed)
        self.assertIn("FORMAT RETRY", prompts[1])
        retry_events = [event for event in harness.trace.events if event.get("event") == "validator_format_retry"]
        self.assertEqual(len(retry_events), 1)

    def test_failed_validation_cycle_is_logged_when_validator_json_stays_malformed(self):
        responses = [
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"fail",',
            '{"decision":"fail",',
        ]

        def fake_query(*args, **kwargs):
            return responses.pop(0)

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
            limits=HarnessLimits(max_validation_cycles=0),
            query=fake_query,
        )

        report, decision, trace = harness.run("objective", snapshot, "author context", {})

        self.assertFalse(decision.passed)
        self.assertIn("Report", report["report_html"])
        cycle_results = [event for event in trace.events if event.get("event") == "validation_cycle_result"]
        self.assertEqual(len(cycle_results), 1)
        self.assertEqual(cycle_results[0]["decision"], "fail")
        self.assertIn("malformed JSON twice", cycle_results[0]["summary"])
        validator_calls = [event for event in trace.events if event.get("event") == "model_call" and event.get("role") == "validator"]
        self.assertEqual([event.get("attempt") for event in validator_calls], [1, 2])

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

    def test_query_ollama_passes_json_schema_format(self):
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
        response_schema = {
            "type": "object",
            "properties": {"accepted": {"type": "boolean"}},
            "required": ["accepted"],
        }
        original_requests = agent_harness.__dict__.get("requests")
        import sys
        sys.modules["requests"] = fake_requests
        try:
            agent_harness.query_ollama(
                "prompt",
                "http://ollama",
                "model",
                timeout=10,
                response_format=response_schema,
            )
        finally:
            if original_requests is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original_requests

        self.assertEqual(fake_requests.payload["format"], response_schema)

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

    def test_unhandled_run_exception_records_traceback_and_aborted_event(self):
        def failing_query(*args, **kwargs):
            raise ConnectionError("ollama endpoint unavailable")

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

        with self.assertRaisesRegex(ConnectionError, "endpoint unavailable"):
            harness.run("objective", snapshot, "author context", {})

        events = harness.trace.events
        self.assertEqual(events[0]["event"], "run_started")
        failure = next(event for event in events if event.get("event") == "exception" and event.get("phase") == "harness_unhandled")
        self.assertEqual(failure["exception_type"], "ConnectionError")
        self.assertIn("Traceback", failure["traceback"])
        aborted = events[-1]
        self.assertEqual(aborted["event"], "run_aborted")
        self.assertEqual(aborted["exception_type"], "ConnectionError")

    def test_tool_exception_is_added_to_trace(self):
        responses = [
            '{"tool_calls":[{"name":"broken_tool","arguments":{}}]}',
            '{"tool_calls":[{"name":"broken_tool","arguments":{"query":"saving preferences"}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"pass","summary":"Accepted after retry.","findings":[]}',
        ]
        prompts = []

        def retrying_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        def broken_tool(arguments):
            query = arguments.get("query")
            if not query:
                raise ValueError("query is required; retry with a focused search query")
            return {"query": query, "facts": []}

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
        harness = ReportHarness("http://ollama", "test-model", query=retrying_query)

        report, decision, _ = harness.run(
            "objective",
            snapshot,
            "author context",
            {"broken_tool": ToolSpec("broken_tool", "broken", broken_tool)},
        )

        exception_events = [event for event in harness.trace.events if event.get("event") == "exception"]
        self.assertEqual(exception_events[-1]["phase"], "initial_tool_call")
        self.assertEqual(exception_events[-1]["tool"], "broken_tool")
        self.assertTrue(decision.passed)
        self.assertEqual(report["plain_text"], "Report")
        self.assertIn("query is required", prompts[1])
        self.assertIn('"arguments": {}', prompts[1])
        tool_errors = [event for event in harness.trace.events if event.get("event") == "tool_error"]
        self.assertEqual(tool_errors[0]["error_type"], "ValueError")
        recoveries = [event for event in harness.trace.events if event.get("event") == "tool_error_recovery"]
        self.assertEqual(recoveries[0]["disposition"], "retry_same_tool")
        self.assertEqual(recoveries[0]["failed_tools"][0]["tool"], "broken_tool")
        actions = [event for event in harness.trace.events if event.get("event") == "agent_action"]
        self.assertEqual(actions[0]["action"], "request_tools")
        self.assertEqual(actions[-1]["action"], "report_ready")
        self.assertEqual(harness.trace.events[-1]["event"], "run_completed")

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
            '{"decision":"fail","summary":"Need more evidence.","findings":[{"severity":"error","category":"evidence","message":"Need category evidence.","evidence":"No comparison evidence.","revision_instruction":"Fetch category details."}]}',
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
        self.assertIn("resolved_error_count_by_category", author_prompts[1])
        self.assertIn("do not reintroduce a fixed issue", author_prompts[1])
        self.assertIn("<html><body>Initial</body></html>", author_prompts[1])
        validator_prompts = [event["prompt"] for event in trace.events if event.get("event") == "model_call" and event.get("role") == "validator"]
        self.assertIn("<html><body>Initial</body></html>", validator_prompts[0])
        self.assertIn("PREVIOUS VALIDATION CYCLE", validator_prompts[1])
        self.assertIn("Need category evidence.", validator_prompts[1])
        self.assertIn("independently re-check", validator_prompts[1])
        progress = [event for event in trace.events if event.get("event") == "validation_progress"]
        self.assertEqual([event["error_count"] for event in progress], [1, 0])
        self.assertEqual(progress[1]["error_count_delta"], 1)
        self.assertEqual(progress[1]["resolved_error_count_by_category"], {"evidence": 1})
        self.assertEqual(progress[1]["new_error_count_by_category"], {})

    def _snapshot(self):
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

    def _scripted_harness(self, responses, limits=None):
        prompts = []

        def fake_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        return ReportHarness("http://ollama", "test-model", limits=limits or HarnessLimits(), query=fake_query), prompts

    def test_missing_history_label_is_fed_back_for_revision(self):
        responses = [
            '{"tool_calls":[{"name":"get_report_history","arguments":{}}]}',
            '{"report_html":"<html><body>Unlabeled</body></html>","plain_text":"Unlabeled"}',
            '{"report_html":"<html><body><p>Previous report for reference: last month was calm.</p></body></html>","plain_text":"Labeled"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        harness, prompts = self._scripted_harness(responses)
        tools = {"get_report_history": ToolSpec("get_report_history", "history", lambda _: {"reports": []})}

        report, decision, trace = harness.run("objective", self._snapshot(), "author context", tools)

        self.assertTrue(decision.passed)
        self.assertIn("Previous report for reference:", report["report_html"])
        self.assertIn("HISTORY LABEL REQUIREMENT", prompts[1])
        self.assertIn("FAILED REPORT FOR REVISION", prompts[2])
        self.assertIn('"category": "history"', prompts[2])
        cycle_results = [event for event in trace.events if event.get("event") == "validation_cycle_result"]
        self.assertEqual(
            [(event["decision"], event["stage"]) for event in cycle_results],
            [("fail", "sanitization"), ("pass", "validator")],
        )
        validator_calls = [event for event in trace.events if event.get("event") == "model_call" and event.get("role") == "validator"]
        self.assertEqual(len(validator_calls), 1)
        self.assertEqual(trace.events[-1]["event"], "run_completed")

    def test_oversize_report_is_never_returned_when_cycles_exhaust(self):
        oversize = "<html><body>" + "x" * 500 + "</body></html>"
        responses = [
            '{"report_html":"%s","plain_text":"Big"}' % oversize,
            '{"report_html":"%s","plain_text":"Still big"}' % oversize,
        ]
        harness, prompts = self._scripted_harness(responses, HarnessLimits(max_report_bytes=200, max_validation_cycles=1))

        report, decision, trace = harness.run("objective", self._snapshot(), "author context", {})

        self.assertFalse(decision.passed)
        self.assertNotIn("x" * 500, report["report_html"])
        self.assertIn("Fallback report", report["report_html"])
        self.assertIn("[omitted:", prompts[1])
        self.assertEqual(decision.findings[0].category, "html")
        self.assertEqual(trace.events[-1]["event"], "run_completed")

    def test_per_tool_budget_returns_error_evidence_without_executing(self):
        executed = []
        responses = [
            '{"tool_calls":[{"name":"get_spending_summary","arguments":{}}]}',
            '{"tool_calls":[{"name":"get_spending_summary","arguments":{"again":true}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        harness, prompts = self._scripted_harness(responses, HarnessLimits(max_calls_per_tool=1))
        tools = {"get_spending_summary": ToolSpec("get_spending_summary", "summary", lambda args: executed.append(args) or {"ok": True})}

        _, decision, trace = harness.run("objective", self._snapshot(), "author context", tools)

        self.assertTrue(decision.passed)
        self.assertEqual(executed, [{}])
        self.assertIn("ToolBudgetExceeded", prompts[2])
        tool_errors = [event for event in trace.events if event.get("event") == "tool_error"]
        self.assertEqual(tool_errors[0]["error_type"], "ToolBudgetExceeded")
        self.assertEqual(sum(event.get("event") == "tool_call" for event in trace.events), 1)

    def test_total_tool_budget_marks_budget_exhausted_in_prompt(self):
        executed = []
        responses = [
            '{"tool_calls":[{"name":"get_spending_summary","arguments":{}},{"name":"get_spending_summary","arguments":{"second":true}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        harness, prompts = self._scripted_harness(responses, HarnessLimits(max_tool_calls=1))
        tools = {"get_spending_summary": ToolSpec("get_spending_summary", "summary", lambda args: executed.append(args) or {"ok": True})}

        _, decision, _ = harness.run("objective", self._snapshot(), "author context", tools)

        self.assertTrue(decision.passed)
        self.assertEqual(executed, [{}])
        self.assertIn("TOOL BUDGET EXHAUSTED", prompts[1])
        self.assertIn("ToolBudgetExceeded", prompts[1])

    def test_unknown_tool_returns_error_evidence_and_run_completes(self):
        responses = [
            '{"tool_calls":[{"name":"delete_everything","arguments":{}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        harness, prompts = self._scripted_harness(responses)

        _, decision, trace = harness.run("objective", self._snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        self.assertIn("ToolNotAllowlisted", prompts[1])
        tool_errors = [event for event in trace.events if event.get("event") == "tool_error"]
        self.assertEqual(tool_errors[0]["error_type"], "ToolNotAllowlisted")
        self.assertEqual(trace.events[-1]["event"], "run_completed")

    def test_author_prompt_includes_each_tool_result_once(self):
        responses = [
            '{"tool_calls":[{"name":"first_tool","arguments":{}}]}',
            '{"tool_calls":[{"name":"second_tool","arguments":{}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"fail","summary":"Fix.","findings":[{"severity":"error","category":"evidence","message":"First issue.","evidence":"e","revision_instruction":"Fix first."}]}',
            '{"report_html":"<html><body>Report 2</body></html>","plain_text":"Report 2"}',
            '{"decision":"fail","summary":"Fix again.","findings":[{"severity":"error","category":"evidence","message":"Second issue.","evidence":"e","revision_instruction":"Fix second."}]}',
            '{"report_html":"<html><body>Report 3</body></html>","plain_text":"Report 3"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        harness, _ = self._scripted_harness(responses, HarnessLimits(max_validation_cycles=2))
        tools = {
            "first_tool": ToolSpec("first_tool", "first", lambda _: {"marker": "FIRST_RESULT_MARKER"}),
            "second_tool": ToolSpec("second_tool", "second", lambda _: {"marker": "SECOND_RESULT_MARKER"}),
        }

        _, decision, trace = harness.run("objective", self._snapshot(), "author context", tools)

        self.assertTrue(decision.passed)
        author_prompts = [event["prompt"] for event in trace.events if event.get("event") == "model_call" and event.get("role") == "author"]
        self.assertEqual(author_prompts[-1].count("FIRST_RESULT_MARKER"), 1)
        self.assertEqual(author_prompts[-1].count("SECOND_RESULT_MARKER"), 1)
        self.assertEqual(author_prompts[-1].count("FAILED REPORT FOR REVISION"), 1)
        self.assertIn("Second issue.", author_prompts[-1])
        self.assertNotIn("First issue.", author_prompts[-1])


if __name__ == "__main__":
    unittest.main()
