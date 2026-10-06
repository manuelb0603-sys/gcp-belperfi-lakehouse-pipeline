import unittest

from finance.agent_harness import (
    HarnessLimits,
    ModelReply,
    ModelUnavailableError,
    ReportHarness,
    ToolSpec,
    focused_feedback,
    parse_author_response,
    parse_json_response,
    sanitize_html,
)
from finance.report_models import ReportSnapshot


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
            limits=HarnessLimits(max_validation_cycles=0),  # values < 1 mean a single review
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
        from finance import agent_harness

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

        with self.assertRaises(ModelUnavailableError):
            harness._author_call("prompt")

        exception_events = [event for event in harness.trace.events if event.get("event") == "exception"]
        self.assertEqual(len(exception_events), 2)
        self.assertEqual(exception_events[-1]["phase"], "author_model_call")
        self.assertEqual(exception_events[-1]["exception_type"], "TimeoutError")
        self.assertEqual(exception_events[-1]["error"], "model timed out")
        names = [event["event"] for event in harness.trace.events]
        self.assertEqual(names.count("model_call_failed"), 2)
        self.assertEqual(names.count("model_call_retry"), 1)

    def test_unhandled_run_exception_records_traceback_and_aborted_event(self):
        # A non-transport error (a client bug, not a timeout/connection failure) still aborts the run.
        def failing_query(*args, **kwargs):
            raise RuntimeError("ollama client misconfigured")

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

        with self.assertRaisesRegex(RuntimeError, "client misconfigured"):
            harness.run("objective", snapshot, "author context", {})

        events = harness.trace.events
        self.assertEqual(events[0]["event"], "run_started")
        failure = next(event for event in events if event.get("event") == "exception" and event.get("phase") == "harness_unhandled")
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertIn("Traceback", failure["traceback"])
        aborted = events[-1]
        self.assertEqual(aborted["event"], "run_aborted")
        self.assertEqual(aborted["exception_type"], "RuntimeError")

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
            limits=HarnessLimits(max_agent_rounds=2, max_tool_calls=2, max_calls_per_tool=2, max_validation_cycles=2),
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
        self.assertIn("PREVIOUS REPORT HTML", author_prompts[1])
        self.assertIn("PRIORITY FIXES FROM THE REVIEW", author_prompts[1])
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
        self.assertIn("PREVIOUS REPORT HTML", prompts[2])
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
        harness, prompts = self._scripted_harness(responses, HarnessLimits(max_report_bytes=200, max_validation_cycles=2))

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

    def test_tool_arguments_are_validated_against_the_schema_before_running(self):
        executed = []
        schema = {
            "type": "object",
            "properties": {
                "month": {"type": "string", "enum": ["2026-02-01", "2026-01-01"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["month"],
        }
        responses = [
            '{"tool_calls":[{"name":"get_transaction_detail","arguments":{"month":"2025-12-01","limit":500}},'
            '{"name":"get_transaction_detail","arguments":{"month":"2026-02-01","limit":10,"extra":1}}]}',
            '{"report_html":"<html><body>Report</body></html>","plain_text":"Report"}',
            '{"decision":"pass","summary":"Accepted.","findings":[]}',
        ]
        harness, prompts = self._scripted_harness(responses)
        tools = {"get_transaction_detail": ToolSpec(
            "get_transaction_detail", "rows", lambda args: executed.append(args) or {"ok": True}, schema)}

        _, decision, trace = harness.run("objective", self._snapshot(), "author context", tools)

        self.assertTrue(decision.passed)
        # The invalid call never ran; the valid one did (unknown extra keys are ignored).
        self.assertEqual(executed, [{"month": "2026-02-01", "limit": 10, "extra": 1}])
        errors = [event for event in trace.events if event.get("event") == "tool_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "ValueError")
        self.assertIn("arguments.month must be one of '2026-02-01', '2026-01-01'", errors[0]["error"])
        self.assertIn("arguments.limit must be at most 100", errors[0]["error"])
        self.assertIn("Invalid arguments for get_transaction_detail", prompts[1])
        # The tool list shows the declared schema instead of the generic placeholder.
        self.assertIn('"enum": [', prompts[0])
        self.assertNotIn('"arguments": "JSON object"', prompts[0])

    def test_tools_without_a_schema_keep_the_generic_argument_placeholder(self):
        prompt = ReportHarness._tool_prompt({"t": ToolSpec("t", "d", lambda _: {})})
        self.assertIn('"arguments": "JSON object"', prompt)

    def test_schema_type_errors_and_missing_required_arguments(self):
        from finance.harness.trace import validate_tool_arguments

        spec = ToolSpec("t", "d", lambda _: {}, {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        })
        self.assertEqual(validate_tool_arguments(spec, {"query": "goals", "limit": 3}), [])
        self.assertEqual(validate_tool_arguments(spec, {"limit": 3}), ["arguments.query is required"])
        self.assertEqual(validate_tool_arguments(spec, {"query": "x", "limit": True}),
                         ["arguments.limit must be integer, got bool True"])
        self.assertEqual(validate_tool_arguments(spec, "not an object"),
                         ["arguments must be object, got str 'not an object'"])

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
        harness, _ = self._scripted_harness(responses, HarnessLimits(max_validation_cycles=3))
        tools = {
            "first_tool": ToolSpec("first_tool", "first", lambda _: {"marker": "FIRST_RESULT_MARKER"}),
            "second_tool": ToolSpec("second_tool", "second", lambda _: {"marker": "SECOND_RESULT_MARKER"}),
        }

        _, decision, trace = harness.run("objective", self._snapshot(), "author context", tools)

        self.assertTrue(decision.passed)
        author_prompts = [event["prompt"] for event in trace.events if event.get("event") == "model_call" and event.get("role") == "author"]
        self.assertEqual(author_prompts[-1].count("FIRST_RESULT_MARKER"), 1)
        self.assertEqual(author_prompts[-1].count("SECOND_RESULT_MARKER"), 1)
        self.assertEqual(author_prompts[-1].count("PREVIOUS REPORT HTML"), 1)
        self.assertIn("Second issue.", author_prompts[-1])
        # The old revision request was replaced: the first finding is no longer a priority fix,
        # only a one-line entry in the resolved list.
        self.assertNotIn('"message": "First issue."', author_prompts[-1])
        self.assertIn("ALREADY FIXED IN EARLIER CYCLES (do not reintroduce):\n- evidence: First issue.", author_prompts[-1])


def _protocol_snapshot() -> ReportSnapshot:
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


MARKED_REPORT = '<<<REPORT_HTML>>>\n<html><body><h1>Hello "quoted"</h1><p>Report</p></body></html>\n<<<END_REPORT_HTML>>>'
PASS = '{"decision":"pass","summary":"Accepted","findings":[]}'


def _error(category: str, message: str) -> dict:
    return {"severity": "error", "category": category, "message": message, "evidence": "seen", "revision_instruction": "fix it"}


class TestAuthorProtocol(unittest.TestCase):
    def test_marker_report_is_parsed_and_plain_text_derived(self):
        result = parse_author_response("Here you go:\n" + MARKED_REPORT + "\nThanks!")

        self.assertEqual(result["report_html"], '<html><body><h1>Hello "quoted"</h1><p>Report</p></body></html>')
        self.assertIn("Hello", result["plain_text"])
        self.assertNotIn("<h1>", result["plain_text"])

    def test_missing_end_marker_is_a_truncation_error(self):
        with self.assertRaisesRegex(ValueError, "missing; the output was likely truncated"):
            parse_author_response("<<<REPORT_HTML>>>\n<html><body>cut off")

    def test_fenced_tool_call_json_is_accepted(self):
        result = parse_author_response('```json\n{"tool_calls":[{"name":"get_spending_summary","arguments":{}}]}\n```')

        self.assertEqual(result["tool_calls"][0]["name"], "get_spending_summary")

    def test_prose_around_tool_call_json_is_accepted(self):
        result = parse_author_response(
            'I need more data first. {"tool_calls":[{"name":"get_category_detail","arguments":{"category":"Shopping"}}]} Done.'
        )

        self.assertEqual(result["tool_calls"][0]["arguments"], {"category": "Shopping"})

    def test_whole_html_document_without_markers_is_accepted_but_unclosed_is_not(self):
        result = parse_author_response("<!DOCTYPE html><html><body><p>Report</p></body></html>")
        self.assertIn("<p>Report</p>", result["report_html"])

        with self.assertRaisesRegex(ValueError, "neither a report"):
            parse_author_response("<html><body><p>Report")

    def test_response_without_markers_or_tool_calls_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "neither a report"):
            parse_author_response("Sure! I will write the report now.")

    def test_author_uses_no_format_schema_and_role_options_are_passed(self):
        calls = []

        def fake_query(prompt, url, model, **kwargs):
            calls.append((model, kwargs))
            return MARKED_REPORT if len(calls) == 1 else PASS

        harness = ReportHarness(
            "http://ollama",
            "author-model",
            validator_model="validator-model",
            query=fake_query,
            model_options={
                "author": {"temperature": 0.3, "num_predict": 12000, "think": False},
                "validator": {"temperature": 0.1, "num_ctx": 32768, "repeat_penalty": None},
            },
        )

        report, decision, _ = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        author_model, author_kwargs = calls[0]
        self.assertEqual(author_model, "author-model")
        self.assertIsNone(author_kwargs["response_format"])
        self.assertEqual(author_kwargs["options"], {"temperature": 0.3, "num_predict": 12000})
        self.assertIs(author_kwargs["think"], False)
        validator_model, validator_kwargs = calls[1]
        self.assertEqual(validator_model, "validator-model")
        self.assertIsInstance(validator_kwargs["response_format"], dict)
        self.assertEqual(validator_kwargs["options"], {"temperature": 0.1, "num_ctx": 32768})
        self.assertNotIn("think", validator_kwargs)
        author_prompt = next(event["prompt"] for event in harness.trace.events if event.get("event") == "model_call")
        self.assertIn("<<<END_REPORT_HTML>>>", author_prompt)
        self.assertNotIn('"report_html"', author_prompt)

    def test_no_options_configured_sends_no_extras(self):
        captured = {}

        def fake_query(prompt, url, model, **kwargs):
            captured.update(kwargs)
            return '{"tool_calls":[]}'

        ReportHarness("http://ollama", "m", query=fake_query)._author_call("prompt")

        self.assertNotIn("options", captured)
        self.assertNotIn("think", captured)

    def test_query_ollama_returns_empty_reply_when_token_limit_spent_on_thinking(self):
        import sys

        from finance import agent_harness

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": "", "thinking": "x" * 500, "done_reason": "length", "eval_count": 4000}

        class FakeRequests:
            def post(self, url, json, timeout):
                return FakeResponse()

        original = sys.modules.get("requests")
        sys.modules["requests"] = FakeRequests()
        try:
            reply = agent_harness.query_ollama("prompt", "http://ollama", "model")
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(str(reply), "")
        self.assertEqual(reply.done_reason, "length")

    def test_query_ollama_sends_options_think_and_omits_format_when_none(self):
        import sys

        from finance import agent_harness

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": MARKED_REPORT, "done_reason": "stop", "eval_count": 42, "prompt_eval_count": 1000}

        class FakeRequests:
            payload = None

            def post(self, url, json, timeout):
                FakeRequests.payload = json
                return FakeResponse()

        original = sys.modules.get("requests")
        sys.modules["requests"] = FakeRequests()
        try:
            reply = agent_harness.query_ollama(
                "prompt", "http://ollama", "model", response_format=None,
                options={"temperature": 0.3}, think=False,
            )
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertNotIn("format", FakeRequests.payload)
        self.assertEqual(FakeRequests.payload["options"], {"temperature": 0.3})
        self.assertIs(FakeRequests.payload["think"], False)
        self.assertEqual(reply, MARKED_REPORT)
        self.assertEqual((reply.done_reason, reply.eval_count, reply.prompt_eval_count), ("stop", 42, 1000))

    def test_author_output_cut_by_token_limit_is_a_format_error(self):
        truncated = ModelReply("<<<REPORT_HTML>>>\n<html><body>partial", done_reason="length", eval_count=12000, prompt_eval_count=9000)
        harness = ReportHarness("http://ollama", "m", query=lambda *a, **k: truncated)

        result = harness._author_call("prompt")

        self.assertIn("token limit", result["error"])
        response = [event for event in harness.trace.events if event.get("event") == "model_response"][-1]
        self.assertEqual(response["done_reason"], "length")
        self.assertEqual(response["eval_count"], 12000)
        self.assertEqual(response["prompt_eval_count"], 9000)

    def test_validator_output_cut_by_token_limit_is_malformed(self):
        truncated = ModelReply('{"decision":"fail","summary":"long', done_reason="length")
        harness = ReportHarness("http://ollama", "m", query=lambda *a, **k: truncated)

        decision = harness._validator_call("validator prompt", cycle=1)

        self.assertFalse(decision.passed)
        self.assertIn("malformed JSON twice", decision.summary)
        errors = [event["error"] for event in harness.trace.events if event.get("event") == "model_response"]
        self.assertTrue(all("token limit" in error for error in errors))


class TestFocusedFeedback(unittest.TestCase):
    def test_top_three_errors_are_prioritized_with_snapshot_values(self):
        from finance.report_models import ValidationDecision, ValidationFinding

        findings = [
            ValidationFinding(**_error("visual_design", "Bars are purple.")),
            ValidationFinding(**_error("usability", "Small text.")),
            ValidationFinding(**_error("evidence", "Unsupported coffee claim.")),
            ValidationFinding(**_error("arithmetic", "Disposable cash shows -1779.84.")),
            ValidationFinding(**_error("completeness", "Missing Sushi Stack.")),
            ValidationFinding(severity="warning", category="arithmetic", message="Minor rounding.", evidence="", revision_instruction=""),
        ]
        feedback = focused_feedback(ValidationDecision("fail", findings, "Several issues"), _protocol_snapshot())

        self.assertEqual([item["category"] for item in feedback["fix_these"]], ["arithmetic", "completeness", "evidence"])
        self.assertEqual(feedback["remaining_lower_priority_errors"], 2)
        self.assertEqual(feedback["fix_these"][0]["authoritative_values"], {"disposable_cash": 1600})
        self.assertNotIn("authoritative_values", feedback["fix_these"][1])

    def test_revision_prompt_lists_only_top_three(self):
        failing = {
            "decision": "fail", "summary": "Issues", "audit": {},
            "findings": [_error("visual_design", f"Design issue {index}.") for index in range(4)]
            + [_error("arithmetic", "MoM value is wrong.")],
        }
        responses = [MARKED_REPORT, __import__("json").dumps(failing), MARKED_REPORT, PASS]
        prompts = []

        def fake_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        harness = ReportHarness("http://ollama", "m", limits=HarnessLimits(max_validation_cycles=2), query=fake_query)
        _, decision, _ = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        revision_prompt = prompts[2]
        self.assertIn("MoM value is wrong.", revision_prompt)
        self.assertIn('"mom_change_percent": 11.1', revision_prompt)
        self.assertIn("Design issue 0.", revision_prompt)
        self.assertNotIn("Design issue 2.", revision_prompt)
        self.assertIn('"remaining_lower_priority_errors": 2', revision_prompt)
        self.assertIn("minimal edits", revision_prompt)


class TestModelCallFailures(unittest.TestCase):
    def _harness(self, responses, cycles=2):
        def fake_query(*args, **kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return ReportHarness("http://ollama", "m", limits=HarnessLimits(max_validation_cycles=cycles), query=fake_query)

    def test_author_timeout_is_retried_once(self):
        harness = self._harness([TimeoutError("slow"), MARKED_REPORT, PASS])

        report, decision, trace = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        names = [event["event"] for event in trace.events]
        self.assertEqual(names.count("model_call_failed"), 1)
        self.assertEqual(names.count("model_call_retry"), 1)

    def test_author_unavailable_initially_returns_fallback_with_fail(self):
        harness = self._harness([TimeoutError("slow"), ConnectionError("down")])

        report, decision, trace = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertFalse(decision.passed)
        self.assertIn("Author model unavailable", decision.summary)
        self.assertIn("Fallback report", report["report_html"])
        self.assertTrue(any(event["event"] == "model_unavailable" for event in trace.events))
        self.assertEqual(trace.events[-1]["event"], "run_completed")

    def test_author_unavailable_during_revision_keeps_last_sanitized_report(self):
        failing = '{"decision":"fail","summary":"Fix it","audit":{},"findings":[%s]}' % __import__("json").dumps(_error("arithmetic", "Wrong total."))
        harness = self._harness([MARKED_REPORT, failing, TimeoutError("slow"), TimeoutError("slow again")])

        report, decision, trace = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertFalse(decision.passed)
        self.assertIn("unavailable during revision", decision.summary)
        self.assertIn('Hello "quoted"', report["report_html"])
        self.assertEqual(len(decision.findings), 1)

    def test_validator_unavailable_returns_fail_without_crashing(self):
        harness = self._harness([MARKED_REPORT, TimeoutError("slow"), TimeoutError("slow again")])

        report, decision, trace = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertFalse(decision.passed)
        self.assertIn("Validator unavailable", decision.summary)
        self.assertIn("<p>Report</p>", report["report_html"])
        result = [event for event in trace.events if event.get("event") == "validation_cycle_result"][-1]
        self.assertEqual(result["stage"], "validator_unavailable")
        author_calls = [event for event in trace.events if event.get("event") == "model_call" and event.get("role") == "author"]
        self.assertEqual(len(author_calls), 1)


class TestAgentRoundsExhausted(unittest.TestCase):
    TOOL_REQUEST = '{"tool_calls":[{"name":"broken","arguments":{}}]}'

    def _run(self, responses):
        def fake_query(*args, **kwargs):
            return responses.pop(0)

        def broken(arguments):
            raise ValueError("tool failed")

        harness = ReportHarness(
            "http://ollama", "m", limits=HarnessLimits(max_agent_rounds=2, max_validation_cycles=2), query=fake_query
        )
        return harness.run("objective", _protocol_snapshot(), "author context", {"broken": ToolSpec("broken", "fails", broken)})

    def test_initial_round_exhaustion_returns_fallback_with_fail(self):
        report, decision, trace = self._run([self.TOOL_REQUEST, self.TOOL_REQUEST])

        self.assertFalse(decision.passed)
        self.assertIn("agent rounds", decision.summary)
        self.assertIn("Fallback report", report["report_html"])
        names = [event["event"] for event in trace.events]
        self.assertIn("agent_rounds_exhausted", names)
        self.assertNotIn("run_aborted", names)
        self.assertEqual(names[-1], "run_completed")
        exhausted = next(event for event in trace.events if event["event"] == "agent_rounds_exhausted")
        self.assertEqual(exhausted["phase"], "initial")
        unresolved = next(event for event in trace.events if event["event"] == "tool_errors_unresolved")
        self.assertEqual(unresolved["failed_tools"][0]["tool"], "broken")

    def test_revision_round_exhaustion_keeps_last_sanitized_report(self):
        failing = '{"decision":"fail","summary":"Fix it","audit":{},"findings":[%s]}' % __import__("json").dumps(_error("arithmetic", "Wrong total."))
        report, decision, trace = self._run([MARKED_REPORT, failing, self.TOOL_REQUEST, self.TOOL_REQUEST])

        self.assertFalse(decision.passed)
        self.assertIn("revision agent rounds", decision.summary)
        self.assertIn("Fix it", decision.summary)
        self.assertEqual([finding.message for finding in decision.findings], ["Wrong total."])
        self.assertIn('Hello "quoted"', report["report_html"])
        exhausted = next(event for event in trace.events if event["event"] == "agent_rounds_exhausted")
        self.assertEqual(exhausted["phase"], "revision")
        self.assertEqual(trace.events[-1]["event"], "run_completed")
        validator_calls = [event for event in trace.events if event.get("event") == "model_call" and event.get("role") == "validator"]
        self.assertEqual(len(validator_calls), 1)


class TestQueryOllamaUrl(unittest.TestCase):
    def test_base_and_full_generate_urls_post_to_generate_endpoint(self):
        from unittest.mock import MagicMock, patch

        from finance import agent_harness

        fake_requests = MagicMock()
        fake_requests.post.return_value.json.return_value = {"response": "ok"}
        with patch.dict("sys.modules", {"requests": fake_requests}):
            for url in ("http://host:11434", "http://host:11434/", "http://host:11434/api/generate"):
                agent_harness.query_ollama("prompt", url, "model", timeout=5)

        posted = [call.args[0] for call in fake_requests.post.call_args_list]
        self.assertEqual(posted, ["http://host:11434/api/generate"] * 3)


REBUTTAL_JSON = '[{"finding": "Cuts-insufficient note missing", "quote": "would not meet your goal"}]'
REPORT_WITH_QUOTE = (
    "<<<REPORT_HTML>>>\n<html><body><p>Even cutting everything would not meet your goal.</p></body></html>\n"
    "<<<END_REPORT_HTML>>>"
)


class TestAuthorRebuttals(unittest.TestCase):
    def test_valid_rebuttals_are_parsed_and_kept_out_of_report_html(self):
        for wrapper in ("{block}", "<!-- {block} -->"):
            block = f"<<<REBUTTALS>>>\n{REBUTTAL_JSON}\n<<<END_REBUTTALS>>>"
            result = parse_author_response(REPORT_WITH_QUOTE + "\n" + wrapper.format(block=block))

            self.assertEqual(result["rebuttals"], [{"finding": "Cuts-insufficient note missing", "quote": "would not meet your goal"}])
            self.assertNotIn("rebuttal_error", result)
            self.assertNotIn("REBUTTALS", result["report_html"])
            self.assertNotIn("<!--", result["report_html"])
            self.assertNotIn("Cuts-insufficient", result["plain_text"])

    def test_rebuttal_block_inside_report_markers_is_stripped(self):
        text = (
            "<<<REPORT_HTML>>>\n<html><body><p>Report</p></body></html>\n"
            f"<!-- <<<REBUTTALS>>>{REBUTTAL_JSON}<<<END_REBUTTALS>>> -->\n<<<END_REPORT_HTML>>>"
        )
        result = parse_author_response(text)

        self.assertEqual(result["report_html"], "<html><body><p>Report</p></body></html>")
        self.assertEqual(len(result["rebuttals"]), 1)

    def test_malformed_rebuttals_are_ignored_without_failing_the_parse(self):
        cases = [
            "<<<REBUTTALS>>>\n[{\"finding\": \"x\", \"quote\": \n<<<END_REBUTTALS>>>",  # invalid JSON
            "<<<REBUTTALS>>>\n[{\"finding\": \"x\"}, \"text\"]\n<<<END_REBUTTALS>>>",  # no usable entries
            "<<<REBUTTALS>>>\n" + REBUTTAL_JSON,  # unclosed block
        ]
        for suffix in cases:
            result = parse_author_response(REPORT_WITH_QUOTE + "\n" + suffix)

            self.assertIn("would not meet your goal", result["report_html"])
            self.assertNotIn("REBUTTALS", result["report_html"])
            self.assertNotIn("rebuttals", result)
            self.assertIn("rebuttal_error", result)

    def test_absent_rebuttals_leave_result_unchanged(self):
        result = parse_author_response(REPORT_WITH_QUOTE)

        self.assertEqual(set(result), {"report_html", "plain_text"})

    def _run(self, responses, cycles=2):
        prompts = []

        def fake_query(prompt, *args, **kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        harness = ReportHarness("http://ollama", "m", limits=HarnessLimits(max_validation_cycles=cycles), query=fake_query)
        report, decision, trace = harness.run("objective", _protocol_snapshot(), "author context", {})
        return report, decision, trace, prompts

    def test_rebuttals_reach_next_validator_prompt_and_trace(self):
        import json

        failing = json.dumps({"decision": "fail", "summary": "Issues", "audit": {},
                              "findings": [_error("completeness", "Cuts-insufficient note missing.")]})
        revision = REPORT_WITH_QUOTE + f"\n<<<REBUTTALS>>>\n{REBUTTAL_JSON}\n<<<END_REBUTTALS>>>"
        report, decision, trace, prompts = self._run([REPORT_WITH_QUOTE, failing, revision, PASS])

        self.assertTrue(decision.passed)
        self.assertIn("<<<REBUTTALS>>>", prompts[2])  # the revision request documents the format
        self.assertNotIn("AUTHOR REBUTTALS", prompts[1])
        second_validator_prompt = prompts[3]
        self.assertIn("AUTHOR REBUTTALS (verify each quote against the report)", second_validator_prompt)
        self.assertIn('"quote": "would not meet your goal"', second_validator_prompt)
        self.assertNotIn("REBUTTALS>>>", report["report_html"])
        self.assertNotIn("rebuttals", report)
        events = [event for event in trace.events if event.get("event") == "author_rebuttals"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["responding_to_cycle"], 1)
        self.assertEqual(events[0]["rebuttals"][0]["quote"], "would not meet your goal")

    def test_malformed_rebuttals_during_revision_are_traced_and_ignored(self):
        import json

        failing = json.dumps({"decision": "fail", "summary": "Issues", "audit": {},
                              "findings": [_error("completeness", "Missing note.")]})
        revision = REPORT_WITH_QUOTE + "\n<<<REBUTTALS>>>not json<<<END_REBUTTALS>>>"
        _, decision, trace, prompts = self._run([REPORT_WITH_QUOTE, failing, revision, PASS])

        self.assertTrue(decision.passed)
        self.assertNotIn("AUTHOR REBUTTALS", prompts[3])
        names = [event["event"] for event in trace.events]
        self.assertIn("author_rebuttals_ignored", names)
        self.assertNotIn("author_rebuttals", names)

    def test_repeated_findings_record_validation_stalled(self):
        import json

        def failing(message):
            return json.dumps({"decision": "fail", "summary": "Issues", "audit": {},
                               "findings": [_error("completeness", message)]})

        responses = [
            REPORT_WITH_QUOTE, failing("Report fails to disclose that cuts are insufficient."),
            REPORT_WITH_QUOTE, failing("Report fails to disclose that cuts are insufficient!"),
            REPORT_WITH_QUOTE, failing("Sushi Stack is missing."),
        ]
        _, decision, trace, _ = self._run(responses, cycles=3)

        self.assertFalse(decision.passed)
        stalled = [event for event in trace.events if event.get("event") == "validation_stalled"]
        self.assertEqual([event["cycle"] for event in stalled], [2])
        self.assertEqual(stalled[0]["repeated_error_count"], 1)
        progress = [event for event in trace.events if event.get("event") == "validation_progress"]
        self.assertEqual([event["repeated_error_count"] for event in progress], [None, 1, 0])


def _failing_review(*messages, category="completeness"):
    import json

    return json.dumps({"decision": "fail", "summary": "Issues", "audit": {},
                       "findings": [_error(category, message) for message in messages]})


def _model_calls(trace, role):
    return [event for event in trace.events if event.get("event") == "model_call" and event.get("role") == role]


class TestValidationCycleLimit(unittest.TestCase):
    def _run_always_failing(self, cycles):
        counter = {"validator": 0}

        def fake_query(prompt, url, model, response_format=None, **kwargs):
            if response_format is None:
                return MARKED_REPORT
            counter["validator"] += 1
            return _failing_review(f"Issue number {counter['validator']}.")

        harness = ReportHarness("http://ollama", "m", limits=HarnessLimits(max_validation_cycles=cycles), query=fake_query)
        return harness.run("objective", _protocol_snapshot(), "author context", {})

    def test_max_validation_cycles_is_the_exact_number_of_reviews(self):
        for cycles, expected_reviews in ((1, 1), (2, 2), (4, 4), (0, 1), (-3, 1)):
            with self.subTest(cycles=cycles):
                report, decision, trace = self._run_always_failing(cycles)

                self.assertFalse(decision.passed)
                self.assertIn("Validation cycles were exhausted", decision.summary)
                self.assertIn('Hello "quoted"', report["report_html"])
                self.assertEqual(len(_model_calls(trace, "validator")), expected_reviews)
                author_calls = _model_calls(trace, "author")
                self.assertEqual(len(author_calls), expected_reviews)  # initial report + (reviews - 1) revisions
                revisions = [event for event in author_calls if "PREVIOUS REPORT HTML" in event["prompt"]]
                self.assertEqual(len(revisions), expected_reviews - 1)
                results = [event for event in trace.events if event.get("event") == "validation_cycle_result"]
                self.assertEqual([event["cycle"] for event in results], list(range(1, expected_reviews + 1)))
                self.assertEqual(trace.events[-1]["validation_cycles"], expected_reviews)


class TestResolvedFindings(unittest.TestCase):
    def _run(self, responses, cycles=3):
        def fake_query(prompt, *args, **kwargs):
            return responses.pop(0)

        harness = ReportHarness("http://ollama", "m", limits=HarnessLimits(max_validation_cycles=cycles), query=fake_query)
        return harness.run("objective", _protocol_snapshot(), "author context", {})

    def test_resolved_findings_reach_author_and_validator_prompts(self):
        long_message = "Sushi Stack is missing the disposable cash card " + "x" * 200
        responses = [
            MARKED_REPORT, _failing_review("Greeting is missing.", long_message),
            MARKED_REPORT, _failing_review(long_message),
            MARKED_REPORT, PASS,
        ]
        _, decision, trace = self._run(responses)

        self.assertTrue(decision.passed)
        author_prompts = [event["prompt"] for event in _model_calls(trace, "author")]
        validator_prompts = [event["prompt"] for event in _model_calls(trace, "validator")]
        self.assertNotIn("ALREADY FIXED IN EARLIER CYCLES", author_prompts[1])
        self.assertNotIn("FINDINGS RESOLVED IN EARLIER CYCLES", validator_prompts[1])
        self.assertIn("ALREADY FIXED IN EARLIER CYCLES (do not reintroduce):\n- completeness: Greeting is missing.", author_prompts[2])
        self.assertEqual(author_prompts[2].count("ALREADY FIXED IN EARLIER CYCLES"), 1)
        self.assertNotIn("completeness: Sushi Stack", author_prompts[2].split("ALREADY FIXED IN EARLIER CYCLES")[1].split("Fix only")[0])
        self.assertIn(
            "FINDINGS RESOLVED IN EARLIER CYCLES (you accepted these as fixed; re-flag only with new verbatim "
            "evidence from the current report):\n- completeness: Greeting is missing.",
            validator_prompts[2],
        )
        progress = [event for event in trace.events if event.get("event") == "validation_progress"]
        self.assertEqual(progress[0]["resolved_findings"], [])
        self.assertEqual(progress[1]["resolved_findings"], ["completeness: Greeting is missing."])
        self.assertEqual(len(progress[2]["resolved_findings"]), 2)
        label = progress[2]["resolved_findings"][1]
        self.assertTrue(label.startswith("completeness: Sushi Stack"))
        self.assertLessEqual(len(label), len("completeness: ") + 101)
        results = [event for event in trace.events if event.get("event") == "validation_cycle_result"]
        self.assertEqual([event["resolved_finding_count"] for event in results], [0, 1, 2])

    def test_reappearing_finding_leaves_list_and_review_without_items_changes_nothing(self):
        responses = [
            MARKED_REPORT, _failing_review("Greeting is missing.", "Top 3 table is missing."),
            MARKED_REPORT, _failing_review("Top 3 table is missing."),
            MARKED_REPORT, '{"decision":"fail",', '{"decision":"fail",',  # malformed twice: no itemized findings
            MARKED_REPORT, _failing_review("Greeting is missing."),
        ]
        _, decision, trace = self._run(responses, cycles=4)

        self.assertFalse(decision.passed)
        progress = [event["resolved_findings"] for event in trace.events if event.get("event") == "validation_progress"]
        self.assertEqual(progress, [
            [],
            ["completeness: Greeting is missing."],
            ["completeness: Greeting is missing."],
            ["completeness: Top 3 table is missing."],
        ])

    def test_resolved_list_is_bounded(self):
        from finance.agent_harness import MAX_RESOLVED_FINDINGS, _update_resolved_findings

        earlier = {("completeness", f"issue {index}"): f"completeness: Issue {index}." for index in range(15)}
        resolved = {}
        _update_resolved_findings(resolved, earlier, set())

        self.assertEqual(len(resolved), MAX_RESOLVED_FINDINGS)
        self.assertEqual(list(resolved.values())[-1], "completeness: Issue 14.")
        _update_resolved_findings(resolved, earlier, {("completeness", "issue 14")})
        self.assertNotIn(("completeness", "issue 14"), resolved)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class TestRunDeadline(unittest.TestCase):
    def _harness(self, fake_query, max_run_seconds, clock, cycles=3, timeout=600):
        return ReportHarness(
            "http://ollama", "m",
            limits=HarnessLimits(max_validation_cycles=cycles, max_run_seconds=max_run_seconds),
            timeout=timeout, query=fake_query, clock=clock,
        )

    def test_deadline_passed_before_first_call_returns_fallback_with_fail(self):
        clock = FakeClock()
        calls = []
        harness = self._harness(lambda *a, **k: calls.append(1) or MARKED_REPORT, 100, clock)
        # Time jumps past the deadline right after the run starts, before any model call.
        harness.trace.listeners.append(lambda event: event["event"] == "run_started" and setattr(clock, "now", clock.now + 500))

        report, decision, trace = harness.run("objective", _protocol_snapshot(), "author context", {})

        self.assertEqual(calls, [])
        self.assertFalse(decision.passed)
        self.assertIn("Fallback report", report["report_html"])
        self.assertIn("time limit", decision.summary)
        self.assertEqual(trace.events[0]["deadline_seconds"], 100)
        exceeded = [event for event in trace.events if event.get("event") == "run_deadline_exceeded"]
        self.assertEqual(len(exceeded), 1)
        self.assertEqual((exceeded[0]["elapsed_seconds"], exceeded[0]["limit_seconds"], exceeded[0]["role"]), (500, 100, "author"))
        self.assertEqual(trace.events[-1]["event"], "run_completed")

    def test_deadline_during_revision_keeps_last_sanitized_report(self):
        clock = FakeClock()
        responses = [MARKED_REPORT, _failing_review("Wrong total.", category="arithmetic"), MARKED_REPORT, PASS]

        def fake_query(*args, **kwargs):
            clock.now += 60  # each call takes a minute; the revision call would start after the 100 s limit
            return responses.pop(0)

        report, decision, trace = self._harness(fake_query, 100, clock).run("objective", _protocol_snapshot(), "author context", {})

        self.assertFalse(decision.passed)
        self.assertIn("during revision", decision.summary)
        self.assertIn("Issues", decision.summary)
        self.assertEqual([finding.message for finding in decision.findings], ["Wrong total."])
        self.assertIn('Hello "quoted"', report["report_html"])
        self.assertEqual(len(responses), 2)  # the revision and second review never ran
        exceeded = next(event for event in trace.events if event.get("event") == "run_deadline_exceeded")
        self.assertEqual((exceeded["elapsed_seconds"], exceeded["limit_seconds"]), (120, 100))
        self.assertNotIn("model_call_retry", [event["event"] for event in trace.events])

    def test_per_call_timeout_is_capped_at_time_remaining_with_a_floor(self):
        clock = FakeClock()
        timeouts = []
        steps = iter([700, 280, 0, 0])  # time spent by each call
        responses = [MARKED_REPORT, _failing_review("Missing greeting."), MARKED_REPORT, PASS]

        def fake_query(*args, timeout=None, **kwargs):
            timeouts.append(timeout)
            clock.now += next(steps)
            return responses.pop(0)

        _, decision, _ = self._harness(fake_query, 1000, clock).run("objective", _protocol_snapshot(), "author context", {})

        self.assertTrue(decision.passed)
        # 1000 s left > ollama_timeout; then 300 s left; then 20 s left, raised to the 30 s floor.
        self.assertEqual(timeouts, [600, 300, 30, 30])

    def test_timeout_of_a_capped_call_is_a_deadline_stop_not_a_retry(self):
        clock = FakeClock()
        calls = []

        def fake_query(*args, timeout=None, **kwargs):
            calls.append(timeout)
            if len(calls) == 1:
                clock.now += 900
                return MARKED_REPORT
            clock.now += timeout
            raise TimeoutError("read timed out")

        report, decision, trace = self._harness(fake_query, 1000, clock).run("objective", _protocol_snapshot(), "author context", {})

        self.assertEqual(calls, [600, 100])
        self.assertFalse(decision.passed)
        self.assertIn("time limit", decision.summary)
        self.assertIn("not reviewed", decision.summary)
        self.assertIn('Hello "quoted"', report["report_html"])
        names = [event["event"] for event in trace.events]
        self.assertNotIn("model_call_retry", names)
        self.assertEqual(names.count("run_deadline_exceeded"), 1)
        result = [event for event in trace.events if event.get("event") == "validation_cycle_result"][-1]
        self.assertEqual(result["stage"], "run_deadline")

    def test_deadline_disabled_when_zero_or_none(self):
        for limit in (0, None):
            with self.subTest(limit=limit):
                clock = FakeClock()
                timeouts = []
                responses = [MARKED_REPORT, PASS]

                def fake_query(*args, timeout=None, **kwargs):
                    timeouts.append(timeout)
                    clock.now += 100_000
                    return responses.pop(0)

                _, decision, trace = self._harness(fake_query, limit, clock).run(
                    "objective", _protocol_snapshot(), "author context", {}
                )

                self.assertTrue(decision.passed)
                self.assertEqual(timeouts, [600, 600])
                self.assertIsNone(trace.events[0]["deadline_seconds"])
                self.assertNotIn("run_deadline_exceeded", [event["event"] for event in trace.events])


class TestValidatorPrompt(unittest.TestCase):
    def test_prompt_keeps_gates_and_adds_evidence_rule(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "agent" / "prompts" / "validator.system.md").read_text(encoding="utf-8")
        for gate in (
            "1. FINANCIAL MATH", "2. SAVINGS-DRIVEN CATEGORY GOALS", "3. REQUIRED CONTENT AND PERSONALITY",
            "4. MATERIAL 3-INSPIRED VISUAL INSPECTION", "5. READABILITY AND ACCESSIBILITY", "6. EVIDENCE AND HTML SAFETY",
            "Decision rules:", "never downgrade a failed hard requirement",
        ):
            self.assertIn(gate, text)
        self.assertIn("verbatim", text)
        self.assertIn("AUTHOR REBUTTALS", text)
        self.assertIn("Do not repeat a prior finding unchanged", text)
        self.assertIn("FINDINGS RESOLVED IN EARLIER CYCLES", text)


if __name__ == "__main__":
    unittest.main()
