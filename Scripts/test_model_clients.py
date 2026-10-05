"""Model-client layer, per-role config, and HTML-to-text tests. Every client is fake: no Ollama or paid API calls."""

import json
import re
import tempfile
import unittest
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from finance import qa, reporting
from finance.harness import HarnessLimits, ReportHarness
from finance.harness.errors import ModelUnavailableError, _is_model_call_failure, _is_timeout
from finance.harness.html_safety import _plain_text_from_html
from finance.harness.prompts import PromptParts, author_base_prompt
from finance.harness.protocol import VALIDATOR_RESPONSE_SCHEMA
from finance.harness.trace import RunTrace
from finance.html_text import html_to_text
from finance.model_clients import (
    AnthropicClient,
    EmptyModelResponse,
    ModelClient,
    ModelReply,
    OllamaClient,
    build_model_client,
    resolve_model_role,
)
from finance.ollama import embed_text, embed_texts, generate_text
from report_models import ReportSnapshot, ValidationDecision
from run_archive import ProgressRecorder

try:  # Only used to build real SDK exception objects; nothing calls the API.
    import anthropic
    import httpx
except ImportError:  # pragma: no cover - Ollama-only environments
    anthropic = None


MARKED_REPORT = "<<<REPORT_HTML>>>\n<html><body><h1>Hi</h1><p>Report</p></body></html>\n<<<END_REPORT_HTML>>>"
PASS = '{"decision":"pass","summary":"Accepted","findings":[]}'


def _snapshot() -> ReportSnapshot:
    return ReportSnapshot(
        as_of="2026-09-15", latest_month="2026-08", previous_month="2026-07",
        latest_total_net_expense=1000, previous_total_net_expense=900, mom_change_percent=11.1,
        monthly_deposit=5000, fixed_costs=2200, disposable_cash=1600, savings_goal=500,
        categories=[], source_tables=[],
    )


class FakeRequests:
    """Stands in for the requests module: records posts and returns queued JSON bodies."""

    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        response = MagicMock()
        response.json.return_value = body
        return response


def _with_requests(fake):
    return patch.dict("sys.modules", {"requests": fake})


# ----- fake Anthropic SDK -----


def _message(text="ok", stop_reason="end_turn", usage=None, stop_details=None):
    return SimpleNamespace(
        id="msg_test",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=usage or SimpleNamespace(
            input_tokens=120, output_tokens=30, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
    )


class FakeAnthropic:
    """Mimics anthropic.Anthropic: with_options(timeout=) and messages.create(**kwargs)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.timeouts = []
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, timeout=None, **_):
        self.timeouts.append(timeout)
        return self

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def _sdk_error_classes():
    """Classes named and moduled like the SDK's, so classification never needs the real package."""
    module = {"__module__": "anthropic._exceptions"}
    base = type("AnthropicError", (Exception,), module)
    api_error = type("APIError", (base,), module)
    status_error = type("APIStatusError", (api_error,), module)
    connection = type("APIConnectionError", (api_error,), module)
    return {
        "APITimeoutError": type("APITimeoutError", (connection,), module),
        "APIConnectionError": connection,
        "RateLimitError": type("RateLimitError", (status_error,), module),
        "InternalServerError": type("InternalServerError", (status_error,), module),
        "BadRequestError": type("BadRequestError", (status_error,), module),
        "APIStatusError": status_error,
    }


SDK = _sdk_error_classes()


def _status_error(name, status):
    exc = SDK[name](f"{name} {status}")
    exc.status_code = status
    return exc


class TestOllamaClient(unittest.TestCase):
    def test_generate_request_shape_with_schema_options_think_and_timeout(self):
        fake = FakeRequests({"response": "{}", "done_reason": "stop", "eval_count": 5, "prompt_eval_count": 9, "eval_duration": 7})
        client = OllamaClient("http://host:11434/api/generate", "m", options={"num_ctx": 100}, think=False, default_timeout=42)
        schema = {"type": "object"}
        with _with_requests(fake):
            reply = client.generate("user", system="SYS ", json_schema=schema, timeout=9)
            client.generate("plain")

        first, second = fake.posts
        self.assertEqual(first["url"], "http://host:11434/api/generate")
        self.assertEqual(first["json"], {
            "model": "m", "prompt": "SYS user", "stream": False, "format": schema,
            "options": {"num_ctx": 100}, "think": False,
        })
        self.assertEqual(first["timeout"], 9)
        self.assertEqual(second["timeout"], 42)
        self.assertEqual(second["json"]["prompt"], "plain")
        self.assertNotIn("format", second["json"])
        self.assertIsInstance(reply, ModelReply)
        self.assertEqual((reply.done_reason, reply.eval_count, reply.prompt_eval_count), ("stop", 5, 9))
        self.assertEqual((reply.provider, reply.model, reply.metadata["eval_duration"]), ("ollama", "m", 7))
        self.assertIsInstance(client, ModelClient)

    def test_no_options_or_think_sends_neither(self):
        fake = FakeRequests({"response": "ok"})
        with _with_requests(fake):
            OllamaClient("http://host:11434", "m").generate("p")
        self.assertEqual(fake.posts[0]["json"], {"model": "m", "prompt": "p", "stream": False})

    def test_empty_response_raises_but_length_returns_empty_reply(self):
        with _with_requests(FakeRequests({"response": "  "})):
            with self.assertRaises(EmptyModelResponse):
                OllamaClient("http://h", "m").generate("p")
        with _with_requests(FakeRequests({"response": "", "done_reason": "length", "thinking": "x"})):
            reply = OllamaClient("http://h", "m").generate("p")
        self.assertEqual((str(reply), reply.done_reason), ("", "length"))

    def test_embed_and_embed_many(self):
        fake = FakeRequests({"embeddings": [[0.5, 1.0]]}, {"embeddings": [[1.0], [2.0]]})
        client = OllamaClient("http://h:1/api/generate", "emb", default_timeout=3)
        with _with_requests(fake):
            self.assertEqual(client.embed("a"), [0.5, 1.0])
            self.assertEqual(client.embed_many(["a", "b"]), [[1.0], [2.0]])
            self.assertEqual(client.embed_many([]), [])
        self.assertEqual([post["url"] for post in fake.posts], ["http://h:1/api/embed"] * 2)
        self.assertEqual(fake.posts[1]["json"], {"model": "emb", "input": ["a", "b"]})

    def test_legacy_functions_keep_their_behavior(self):
        fake = FakeRequests({"response": "  text  "}, {"embedding": [1, 2]}, {"embeddings": [[3]]})
        with _with_requests(fake):
            self.assertEqual(generate_text("p", "http://h", "m", timeout=4), "text")
            self.assertEqual(embed_text("t", "http://h", "e"), [1.0, 2.0])
            self.assertEqual(embed_texts(["t"], "http://h", "e"), [[3.0]])
        self.assertEqual(fake.posts[0]["json"], {"model": "m", "prompt": "p", "stream": False})
        self.assertEqual(fake.posts[0]["timeout"], 4)
        for body in ({"response": ""}, {"response": "", "done_reason": "length"}):
            with _with_requests(FakeRequests(body)):
                with self.assertRaisesRegex(ValueError, "did not contain text"):
                    generate_text("p", "http://h", "m")

    def test_email_wrapper_returns_failure_text_on_error(self):
        import email_weekly_report as weekly

        with _with_requests(FakeRequests({"response": ""})), patch.object(weekly, "log_status"):
            self.assertEqual(weekly.query_ollama("p", "http://h", "m"), "Failed to generate report.")
        with _with_requests(FakeRequests({"response": " report "})), patch.object(weekly, "log_status"):
            self.assertEqual(weekly.query_ollama("p", "http://h", "m"), "report")


class TestAnthropicClient(unittest.TestCase):
    def test_system_is_cached_and_prompt_is_the_user_message(self):
        fake = FakeAnthropic(_message("hello"))
        reply = AnthropicClient("claude-sonnet-5-5", client=fake).generate("per-call", system="stable", timeout=12)

        call = fake.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-5-5")
        self.assertEqual(call["max_tokens"], 16000)
        self.assertEqual(call["system"], [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(call["messages"], [{"role": "user", "content": "per-call"}])
        for forbidden in ("thinking", "temperature", "top_p", "top_k", "output_config"):
            self.assertNotIn(forbidden, call)
        self.assertEqual(fake.timeouts, [12])
        self.assertEqual(str(reply), "hello")
        self.assertEqual((reply.done_reason, reply.provider, reply.model), ("stop", "anthropic", "claude-sonnet-5-5"))

    def test_no_system_block_without_system_and_cache_can_be_disabled(self):
        fake = FakeAnthropic(_message())
        AnthropicClient("claude-haiku-4-5", client=fake).generate("p")
        AnthropicClient("claude-haiku-4-5", client=fake, cache_system=False, default_timeout=77).generate("p", system="s")
        self.assertNotIn("system", fake.calls[0])
        self.assertEqual(fake.calls[1]["system"], [{"type": "text", "text": "s"}])
        self.assertEqual(fake.timeouts, [600, 77])

    def test_output_config_carries_json_schema_and_effort(self):
        fake = FakeAnthropic(_message("{}"))
        client = AnthropicClient("claude-opus-5-5", effort="medium", max_tokens=8000, client=fake)
        client.generate("p", json_schema=VALIDATOR_RESPONSE_SCHEMA)
        client.generate("p")

        self.assertEqual(fake.calls[0]["output_config"], {
            "format": {"type": "json_schema", "schema": VALIDATOR_RESPONSE_SCHEMA},
            "effort": "medium",
        })
        self.assertEqual(fake.calls[0]["max_tokens"], 8000)
        self.assertEqual(fake.calls[1]["output_config"], {"effort": "medium"})

    def test_refusal_returns_empty_reply_with_stop_details(self):
        details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
        fake = FakeAnthropic(_message("partial", stop_reason="refusal", stop_details=details))
        reply = AnthropicClient("claude-sonnet-5-5", client=fake).generate("p")
        self.assertEqual((str(reply), reply.done_reason), ("", "refusal"))
        self.assertEqual(reply.metadata["stop_details"], details)

    def test_max_tokens_maps_to_length_and_keeps_partial_text(self):
        fake = FakeAnthropic(_message("<<<REPORT_HTML>>> cut", stop_reason="max_tokens"))
        reply = AnthropicClient("claude-sonnet-5-5", client=fake).generate("p")
        self.assertEqual((str(reply), reply.done_reason), ("<<<REPORT_HTML>>> cut", "length"))

    def test_usage_including_cache_tokens_is_recorded(self):
        usage = SimpleNamespace(input_tokens=50, output_tokens=40, cache_read_input_tokens=9000, cache_creation_input_tokens=0)
        reply = AnthropicClient("claude-sonnet-5-5", client=FakeAnthropic(_message(usage=usage))).generate("p", system="s")
        self.assertEqual(reply.usage, {
            "input_tokens": 50, "output_tokens": 40, "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 0,
        })
        self.assertEqual((reply.eval_count, reply.prompt_eval_count), (40, 50))

    def test_empty_text_is_an_empty_model_response(self):
        with self.assertRaises(EmptyModelResponse):
            AnthropicClient("claude-sonnet-5-5", client=FakeAnthropic(_message(""))).generate("p")

    def test_empty_user_prompt_gets_placeholder_and_ollama_options_are_ignored(self):
        fake = FakeAnthropic(_message())
        AnthropicClient("claude-sonnet-5-5", client=fake, options={"num_ctx": 1}, think=False).generate("", system="all")
        self.assertTrue(fake.calls[0]["messages"][0]["content"].strip())
        self.assertNotIn("options", fake.calls[0])

    def test_transient_sdk_errors_are_model_call_failures(self):
        for exc in (SDK["APITimeoutError"]("t"), SDK["APIConnectionError"]("c"),
                    _status_error("RateLimitError", 429), _status_error("InternalServerError", 500),
                    _status_error("APIStatusError", 529)):
            self.assertTrue(_is_model_call_failure(exc), type(exc).__name__)
        self.assertTrue(_is_timeout(SDK["APITimeoutError"]("t")))
        self.assertFalse(_is_timeout(SDK["APIConnectionError"]("c")))
        self.assertFalse(_is_model_call_failure(_status_error("BadRequestError", 400)))
        # Same class names outside the anthropic package are not treated as SDK errors.
        self.assertFalse(_is_model_call_failure(type("RateLimitError", (Exception,), {})("x")))

    @unittest.skipIf(anthropic is None, "anthropic SDK not installed")
    def test_real_sdk_exception_classes_are_classified(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

        def status(cls, code):
            return cls("err", response=httpx.Response(code, request=request), body=None)

        self.assertTrue(_is_timeout(anthropic.APITimeoutError(request=request)))
        self.assertTrue(_is_model_call_failure(anthropic.APIConnectionError(request=request)))
        self.assertTrue(_is_model_call_failure(status(anthropic.RateLimitError, 429)))
        self.assertTrue(_is_model_call_failure(status(anthropic.InternalServerError, 500)))
        self.assertFalse(_is_model_call_failure(status(anthropic.BadRequestError, 400)))

    def test_harness_retries_timeout_once_then_stops(self):
        fake = FakeAnthropic(SDK["APITimeoutError"]("timed out"))
        harness = ReportHarness(author_client=AnthropicClient("claude-sonnet-5-5", client=fake))
        with self.assertRaises(ModelUnavailableError):
            harness._author_call("prompt")
        self.assertEqual(len(fake.calls), 2)
        failed = [event for event in harness.trace.events if event["event"] == "model_call_failed"]
        self.assertEqual([event["error_type"] for event in failed], ["APITimeoutError"] * 2)

    def test_harness_does_not_retry_bad_request(self):
        fake = FakeAnthropic(_status_error("BadRequestError", 400))
        harness = ReportHarness(author_client=AnthropicClient("claude-sonnet-5-5", client=fake))
        with self.assertRaises(SDK["BadRequestError"]):
            harness._author_call("prompt")
        self.assertEqual(len(fake.calls), 1)
        self.assertFalse(any(event["event"] == "model_call_retry" for event in harness.trace.events))


class TestHarnessPromptSplit(unittest.TestCase):
    def _run(self, **harness_kwargs):
        harness = ReportHarness(limits=HarnessLimits(max_validation_cycles=1), **harness_kwargs)
        harness.run("objective", _snapshot(), "author ctx", {}, validator_context="validator ctx",
                    report_requirements=["Heading A"])
        return harness

    def test_ollama_receives_the_same_single_prompt_as_the_query_path(self):
        queried = []

        def fake_query(prompt, url, model, **kwargs):
            queried.append((prompt, model, kwargs))
            return [MARKED_REPORT, PASS][len(queried) - 1]

        self._run(ollama_url="http://h", author_model="a", validator_model="v", query=fake_query,
                  model_options={"author": {"think": False}})
        fake = FakeRequests({"response": MARKED_REPORT}, {"response": PASS})
        with _with_requests(fake):
            harness = self._run(
                author_client=OllamaClient("http://h", "a", think=False),
                validator_client=OllamaClient("http://h", "v"),
            )

        self.assertEqual([post["json"]["prompt"] for post in fake.posts], [prompt for prompt, _, _ in queried])
        trace_prompts = [event["prompt"] for event in harness.trace.events if event["event"] == "model_call"]
        self.assertEqual(trace_prompts, [prompt for prompt, _, _ in queried])
        expected_author = author_base_prompt("author ctx", "objective", json.dumps(_snapshot().to_dict(), indent=2), ["Heading A"], "{}")
        self.assertEqual(queried[0][0], expected_author)
        self.assertTrue(queried[1][0].startswith("validator ctx\n\nVALIDATE THIS REPORT:"))
        self.assertIs(fake.posts[0]["json"]["think"], False)
        self.assertEqual(fake.posts[1]["json"]["format"], VALIDATOR_RESPONSE_SCHEMA)

    def test_anthropic_gets_instructions_as_cached_system_and_content_as_user(self):
        author = FakeAnthropic(_message(MARKED_REPORT))
        validator = FakeAnthropic(_message('{"decision":"fail",'), _message(PASS))
        harness = self._run(
            author_client=AnthropicClient("claude-sonnet-5-5", client=author),
            validator_client=AnthropicClient("claude-opus-5-5", effort="medium", client=validator),
        )

        base = author_base_prompt("author ctx", "objective", json.dumps(_snapshot().to_dict(), indent=2), ["Heading A"], "{}")
        self.assertEqual(author.calls[0]["system"][0]["text"], base)
        self.assertEqual(author.calls[0]["system"][0]["cache_control"], {"type": "ephemeral"})
        first, retry = validator.calls
        self.assertEqual(first["system"][0]["text"], "validator ctx")
        self.assertTrue(first["messages"][0]["content"].startswith("\n\nVALIDATE THIS REPORT:"))
        self.assertEqual(retry["system"], first["system"])  # format retry keeps the cached prefix
        self.assertIn("FORMAT RETRY", retry["messages"][0]["content"])
        self.assertEqual(first["output_config"]["format"]["schema"], VALIDATOR_RESPONSE_SCHEMA)
        calls = [event for event in harness.trace.events if event["event"] == "model_call"]
        self.assertEqual(calls[1]["prompt"], "validator ctx" + first["messages"][0]["content"])
        self.assertEqual([event["provider"] for event in calls], ["anthropic"] * 3)
        started = harness.trace.events[0]
        self.assertEqual((started["author_provider"], started["validator_model"]), ("anthropic", "claude-opus-5-5"))
        response = next(event for event in harness.trace.events if event["event"] == "model_response")
        self.assertEqual(response["usage"]["input_tokens"], 120)

    def test_prompt_parts_text_is_the_concatenation(self):
        self.assertEqual(PromptParts("a", "b").text, "ab")


class TestValidatorSchema(unittest.TestCase):
    def test_nullable_field_uses_any_of_and_objects_are_closed(self):
        finding = VALIDATOR_RESPONSE_SCHEMA["properties"]["findings"]["items"]
        self.assertEqual(finding["properties"]["revision_instruction"], {"anyOf": [{"type": "string"}, {"type": "null"}]})

        def objects(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    yield node
                for value in node.values():
                    yield from objects(value)
            elif isinstance(node, list):
                for value in node:
                    yield from objects(value)

        found = list(objects(VALIDATOR_RESPONSE_SCHEMA))
        self.assertGreaterEqual(len(found), 9)
        self.assertTrue(all(node.get("additionalProperties") is False for node in found))
        self.assertNotIn('"type": [', json.dumps(VALIDATOR_RESPONSE_SCHEMA))


class TestModelRoles(unittest.TestCase):
    LEGACY = {
        "model_server_url": "http://host:11434/api/generate",
        "model_name": "gemma4:e4b",
        "validator_model_name": "gemma4:12b",
        "ollama_timeout": 900,
        "ollama_options": {"author": {"num_predict": 12000}, "validator": {"think": False, "temperature": 0.1, "top_k": None}},
    }

    def test_legacy_keys_are_the_fallback(self):
        author = resolve_model_role(self.LEGACY, "author")
        validator = resolve_model_role(self.LEGACY, "validator")
        self.assertEqual((author["provider"], author["model"], author["options"], author["think"], author["timeout"]),
                         ("ollama", "gemma4:e4b", {"num_predict": 12000}, None, 900))
        self.assertEqual((validator["model"], validator["options"], validator["think"]), ("gemma4:12b", {"temperature": 0.1}, False))
        self.assertEqual(resolve_model_role({"model_name": "x"}, "validator")["model"], "x")
        self.assertEqual(resolve_model_role(self.LEGACY, "author")["base_url"], self.LEGACY["model_server_url"])

    def test_eval_role_falls_back_to_the_validator(self):
        spec = resolve_model_role(self.LEGACY, "eval")
        self.assertEqual((spec["role"], spec["provider"], spec["model"], spec["think"]), ("eval", "ollama", "gemma4:12b", False))
        client = build_model_client(self.LEGACY, "eval")
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual((client.model, client.think, client.options), ("gemma4:12b", False, {"temperature": 0.1}))

    def test_model_roles_override_per_role(self):
        config = {**self.LEGACY, "model_roles": {
            "author": {"provider": "ollama", "model": "gemma4:12b", "options": {"num_ctx": 4096}},
            "eval": {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium", "max_tokens": 16000},
        }}
        author = build_model_client(config, "author")
        validator = build_model_client(config, "validator")
        evaluator = build_model_client(config, "eval", anthropic_client=FakeAnthropic(_message()))
        self.assertEqual((author.model, author.options, author.think), ("gemma4:12b", {"num_ctx": 4096}, None))
        self.assertEqual((validator.model, validator.think), ("gemma4:12b", False))  # legacy validator keys
        self.assertIsInstance(evaluator, AnthropicClient)
        self.assertEqual((evaluator.model, evaluator.effort, evaluator.max_tokens, evaluator.default_timeout),
                         ("claude-sonnet-5-5", "medium", 16000, 900))

    def test_ollama_role_entry_without_options_uses_legacy_options(self):
        config = {**self.LEGACY, "model_roles": {"validator": {"provider": "ollama", "model": "gemma4:12b"}}}
        self.assertIs(build_model_client(config, "validator").think, False)

    def test_config_example_resolves_to_the_recommended_roles(self):
        config = json.loads((Path(__file__).resolve().parent.parent / "config.example.json").read_text(encoding="utf-8"))
        author, validator, evaluator = (resolve_model_role(config, role) for role in ("author", "validator", "eval"))
        self.assertEqual((author["provider"], author["model"]), ("ollama", "gemma4:e4b"))
        self.assertEqual((validator["provider"], validator["model"], validator["think"]), ("ollama", "gemma4:12b", False))
        self.assertEqual((evaluator["provider"], evaluator["model"], evaluator["effort"]), ("anthropic", "claude-sonnet-5-5", "medium"))

    def test_invalid_roles_and_providers_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model_role(self.LEGACY, "critic")
        with self.assertRaises(ValueError):
            resolve_model_role({**self.LEGACY, "model_roles": {"author": {"provider": "other", "model": "m"}}}, "author")
        with self.assertRaises(ValueError):
            build_model_client({"model_name": "m"}, "author")  # Ollama without a server URL

    def test_report_generation_builds_role_clients_and_records_providers(self):
        rows = [
            {"transaction_month": date(2026, 2, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 450},
            {"transaction_month": date(2026, 1, 1), "category": "Groceries", "parent_group": "Needs", "net_real_expense": 400},
        ]
        captured = {}

        class CapturingHarness:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.trace = RunTrace()

            def run(self, **kwargs):
                return {"report_html": "<p>ok</p>", "plain_text": "ok"}, ValidationDecision("pass", [], "ok"), self.trace

        config = {"Monthly_budget": 8500, "Fixed_costs": 1000, "ollama_options": self.LEGACY["ollama_options"],
                  "model_roles": {"validator": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}}
        with patch.object(reporting, "ReportHarness", CapturingHarness):
            result = reporting.generate_finance_report(
                config=config, project_name="p", ollama_url="http://ollama/api/generate", model_name="gemma4:e4b",
                rows=rows, selected_report_month=date(2026, 2, 1),
            )

        self.assertIsInstance(captured["author_client"], OllamaClient)
        self.assertEqual(captured["author_client"].options, {"num_predict": 12000})
        self.assertIsInstance(captured["validator_client"], AnthropicClient)
        self.assertEqual(
            {key: result.metrics[key] for key in ("model", "validator_model", "author_provider", "validator_provider")},
            {"model": "gemma4:e4b", "validator_model": "claude-sonnet-5-5", "author_provider": "ollama", "validator_provider": "anthropic"},
        )

    def test_progress_records_provider_per_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ProgressRecorder(Path(tmp) / "r", "r", "gemma4:e4b", "claude-sonnet-5-5", echo=False,
                                        validator_provider="anthropic")
            progress = json.loads((Path(tmp) / "r" / "progress.json").read_text(encoding="utf-8"))
            log = (Path(tmp) / "r" / "progress.log").read_text(encoding="utf-8")
        self.assertEqual((progress["author_provider"], progress["validator_provider"]), ("ollama", "anthropic"))
        self.assertIn("validator=anthropic:claude-sonnet-5-5", log)
        self.assertIsNotNone(recorder)


# ----- html_to_text parity with the three extractors it replaced (copied verbatim as references) -----


class _LegacyReportTextExtractor(HTMLParser):
    block_tags = {"address", "article", "br", "div", "h1", "h2", "h3", "li", "p", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "style":
            self.in_style = True
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() == "style":
            self.in_style = False
        if tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.in_style:
            self.parts.append(data)


class _LegacyHistoryTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"style", "script", "head", "title"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"style", "script", "head", "title"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


class _LegacyQaTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def _legacy_report(markup):
    parser = _LegacyReportTextExtractor()
    parser.feed(markup)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


def _legacy_history(markup):
    parser = _LegacyHistoryTextExtractor()
    parser.feed(markup)
    return " ".join(parser.parts)


def _legacy_qa(markup):
    parser = _LegacyQaTextExtractor()
    parser.feed(markup or "")
    return "\n".join(parser.parts)


HTML_SAMPLES = [
    "",
    "plain text only",
    "<html><head><title>T</title><style>body{color:#123}</style></head><body><h1>Hello</h1><p>World &amp; more</p></body></html>",
    "<div>One<br>Two<br/>Three</div><ul><li> a </li><li>b</li></ul>",
    "<table><tr><td>Cat</td><td>$1,234.00</td></tr><tr><td>Dog</td><td>-$5</td></tr></table>",
    "<STYLE>.x{}</STYLE><P>Upper case</P><Script>var a = '<p>no</p>';</Script><section>   </section>",
    "<p>a</p>\n\n\n<p>b</p>   <div>\n \n</div><h2>c&nbsp;d &#127843;</h2>",
    "<head><meta charset='utf-8'><title>x</title></head><body>Body <b>bold</b> <i>it</i></body>",
    "</style>stray close<style/>after self-closing<p>end",
    "<article><address>Addr</address></article> trailing &amp text",
]


class TestHtmlToTextParity(unittest.TestCase):
    def test_report_plain_text_matches_previous_extractor(self):
        for sample in HTML_SAMPLES:
            self.assertEqual(_plain_text_from_html(sample), _legacy_report(sample), sample)

    def test_history_text_matches_previous_extractor(self):
        for sample in HTML_SAMPLES:
            self.assertEqual(
                html_to_text(sample, skip_tags=reporting.HISTORY_SKIP_TAGS, separator=" "), _legacy_history(sample), sample
            )
            entry = {"response_text": sample} if sample.strip() else {"response_text": "", "response_html": "<p>x</p>"}
            expected = re.sub(r"\s+", " ", _legacy_history(entry.get("response_html") or sample)).strip()
            if sample.strip() and not re.search(r"<\s*(html|body|style|div|table)\b", sample, re.I):
                expected = re.sub(r"\s+", " ", sample).strip()
            self.assertEqual(reporting._history_text(entry), expected, sample)

    def test_qa_text_matches_previous_extractor(self):
        for sample in HTML_SAMPLES + [None]:
            self.assertEqual(qa._html_to_text(sample), _legacy_qa(sample), sample)


if __name__ == "__main__":
    unittest.main()
