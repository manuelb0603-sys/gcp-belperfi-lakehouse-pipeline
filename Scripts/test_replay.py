"""Validator replay tests: temp run folders and fake clients only (no Ollama or paid API calls)."""

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from finance import replay
from finance.harness.prompts import validator_prompt_parts, validator_format_retry_suffix
from finance.harness.protocol import VALIDATOR_RESPONSE_SCHEMA
from finance.model_clients import ModelReply
from finance.run_archive import update_index

SYSTEM_V1 = "OLD VALIDATOR INSTRUCTIONS"
CONFIG = {
    "model_server_url": "http://localhost:11434/api/generate",
    "model_name": "gemma4:e4b",
    "model_roles": {
        "validator": {"provider": "ollama", "model": "gemma4:12b", "think": False, "options": {"num_ctx": 32768}},
    },
}
GATES = ("financial_math", "goal_math", "required_content", "visual_design", "readability_accessibility", "html_safety")


def _payload(decision="fail", findings=None):
    return {
        "decision": decision,
        "summary": "Reviewed.",
        "audit": {gate: {"status": "pass", "evidence": "ok"} for gate in GATES},
        "findings": findings if findings is not None else [{
            "severity": "error", "category": "arithmetic", "message": "Wrong total",
            "evidence": "Report says $10 " + "x" * 400, "revision_instruction": "Use $12",
        }],
    }


def _reply(payload=None, done_reason="stop", usage=None, text=None):
    body = text if text is not None else json.dumps(payload or _payload())
    return ModelReply(body, done_reason=done_reason, provider="anthropic", model="claude-haiku-4-5", usage=usage or {
        "input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 8000,
    })


class FakeClient:
    def __init__(self, *replies, provider="anthropic", model="claude-haiku-4-5"):
        self.replies = list(replies)
        self.calls = []
        self.provider, self.model = provider, model

    def generate(self, prompt, *, system=None, json_schema=None, timeout=None):
        self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema, "timeout": timeout})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _prompt(cycle):
    return validator_prompt_parts(SYSTEM_V1, f"<html>report {cycle}</html>", "{}", None, [], None, [], [])


def _write_run(root: Path, run_id="20261005T133631_gemma4_e4b"):
    folder = root / run_id
    folder.mkdir(parents=True)
    events = [{"event": "run_started", "timestamp": 1.0}]
    for cycle in (1, 2):
        parts = _prompt(cycle)
        events.append({"event": "model_call", "timestamp": 2.0, "role": "validator", "model": "gemma4:12b",
                       "provider": "ollama", "prompt": parts.text, "validation_cycle": cycle, "attempt": 1})
        if cycle == 2:  # a format retry must never be chosen
            retry = parts.text + validator_format_retry_suffix("bad json", False)
            events.append({"event": "model_call", "timestamp": 3.0, "role": "validator", "model": "gemma4:12b",
                           "provider": "ollama", "prompt": retry, "validation_cycle": 2, "attempt": 2})
        events.append({"event": "validation", "timestamp": 4.0, "cycle": cycle, "decision": "fail" if cycle == 1 else "pass"})
    (folder / "trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps({"model": "gemma4:e4b", "status": "completed"}), encoding="utf-8")
    return folder


class ReplayTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.run_dir = _write_run(self.root)
        self.prompt_dir = Path(self.tmp.name) / "prompts"
        self.prompt_dir.mkdir()
        (self.prompt_dir / "validator.system.md").write_text("CURRENT VALIDATOR INSTRUCTIONS", encoding="utf-8")
        self.factory_calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def factory(self, client):
        def build(config, spec):
            self.factory_calls.append((config, spec))
            return client
        return build

    def run_main(self, argv, client):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = replay.main(argv, config=CONFIG, runs_root=self.root, prompt_dir=self.prompt_dir,
                               client_factory=self.factory(client))
        return code, out.getvalue(), err.getvalue()


class TestPromptRebuild(ReplayTestCase):
    def test_rebuilds_system_and_user_parts_for_a_cycle(self):
        call, parts, decision = replay.build_replay_prompt(self.run_dir, 1)
        self.assertEqual((call.cycle, call.provider, call.model, decision), (1, "ollama", "gemma4:12b", "fail"))
        self.assertEqual(parts, _prompt(1))
        last, parts, decision = replay.build_replay_prompt(self.run_dir, "last")
        self.assertEqual((last.cycle, decision), (2, "pass"))
        self.assertEqual(parts, _prompt(2))  # attempt 1, not the format retry
        self.assertNotIn("FORMAT RETRY", parts.user)

    def test_current_prompt_swaps_only_the_system_part(self):
        _, parts, _ = replay.build_replay_prompt(self.run_dir, 2, current_prompt=True, prompt_dir=self.prompt_dir)
        self.assertEqual(parts.system, "CURRENT VALIDATOR INSTRUCTIONS")
        self.assertEqual(parts.user, _prompt(2).user)

    def test_unknown_cycle_and_prompt_without_marker_are_errors(self):
        with self.assertRaisesRegex(replay.ReplayError, "cycles 1, 2"):
            replay.build_replay_prompt(self.run_dir, 7)
        with self.assertRaises(replay.ReplayError):
            replay.split_validator_prompt("no marker here")
        with self.assertRaises(replay.ReplayError):
            replay.resolve_run_dir("replays", self.root)

    def test_old_traces_without_cycle_fields_are_numbered_in_order(self):
        folder = self.root / "old_run"
        folder.mkdir()
        events = [{"event": "model_call", "role": "validator", "model": "m", "prompt": _prompt(n).text} for n in (1, 2)]
        events.insert(1, {"event": "model_call", "role": "validator", "prompt": _prompt(1).text + "\n\nFORMAT RETRY: x"})
        (folder / "trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        calls, _ = replay.load_validator_calls(folder)
        self.assertEqual([(call.cycle, call.prompt) for call in calls], [(1, _prompt(1).text), (2, _prompt(2).text)])


class TestReplayRole(unittest.TestCase):
    def test_default_is_the_configured_validator(self):
        _, spec = replay.replay_role(CONFIG)
        self.assertEqual((spec["provider"], spec["model"], spec["think"], spec["options"]), ("ollama", "gemma4:12b", False, {"num_ctx": 32768}))

    def test_role_model_and_flags_override_for_this_replay_only(self):
        config, spec = replay.replay_role(CONFIG, "anthropic:claude-sonnet-5-5", effort="low", think=True, max_tokens=32000)
        self.assertEqual((spec["provider"], spec["model"], spec["effort"], spec["thinking"], spec["max_tokens"]),
                         ("anthropic", "claude-sonnet-5-5", "low", "adaptive", 32000))
        self.assertNotIn("options", config["model_roles"]["validator"])  # Ollama settings are not carried over
        self.assertEqual(CONFIG["model_roles"]["validator"]["model"], "gemma4:12b")  # the input is untouched

        _, spec = replay.replay_role(CONFIG, "ollama:gemma4:e4b", think=True, max_tokens=4000)
        self.assertEqual((spec["model"], spec["think"], spec["options"]), ("gemma4:e4b", True, {"num_ctx": 32768, "num_predict": 4000}))

    def test_invalid_flags_are_rejected(self):
        for kwargs in (
            {"role_model": "claude-haiku-4-5"},
            {"role_model": "openai:gpt"},
            {"role_model": "anthropic:claude-opus-5-5", "think": False},
            {"effort": "low"},  # the configured validator is Ollama
            {"role_model": "anthropic:claude-opus-5-5", "max_tokens": 64001},
        ):
            with self.assertRaises(replay.ReplayError, msg=str(kwargs)):
                replay.replay_role(CONFIG, **kwargs)


class TestCost(unittest.TestCase):
    def test_cost_estimate_includes_cache_tokens(self):
        usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 10000, "cache_creation_input_tokens": 8000}
        # haiku $1/$5: 1000*1 + 10000*0.1 + 8000*1.25 + 500*5 = 14500 per million
        self.assertEqual(replay.estimate_cost("anthropic", "claude-haiku-4-5", usage), 0.0145)
        self.assertEqual(replay.estimate_cost("anthropic", "claude-opus-5-5", {"input_tokens": 10**6, "output_tokens": 10**6}), 24.0)
        self.assertIsNone(replay.estimate_cost("ollama", "gemma4:12b", usage))
        # The budget guard's worst case: ceil(40000 / 3.5) = 11429 input tokens at list price plus max_tokens output.
        self.assertEqual(replay.worst_case_cost("anthropic", "claude-sonnet-5-5", 40000, 16000), round((11429 * 2 + 16000 * 10) / 1e6, 4))


class TestReplayCli(ReplayTestCase):
    def test_replay_calls_with_schema_prints_verdict_and_saves_json(self):
        client = FakeClient(_reply())
        code, out, err = self.run_main(
            ["20261005T133631_gemma4_e4b", "--cycle", "1", "--role-model", "anthropic:claude-haiku-4-5", "--effort", "low"], client)
        self.assertEqual(code, 0, err)
        call = client.calls[0]
        self.assertEqual((call["system"], call["prompt"], call["json_schema"]), (SYSTEM_V1, _prompt(1).user, VALIDATOR_RESPONSE_SCHEMA))
        spec = self.factory_calls[0][1]
        self.assertEqual((spec["provider"], spec["model"], spec["effort"]), ("anthropic", "claude-haiku-4-5", "low"))
        self.assertEqual(call["timeout"], spec["timeout"])
        self.assertIn("Decision: FAIL (run recorded: fail)", out)
        self.assertIn("financial_math=pass", out)
        self.assertIn("[arithmetic] Wrong total", out)
        self.assertIn("…", out)  # long evidence is truncated
        self.assertIn("Tokens in/out: 1000/500; cache read 0, cache write 8000", out)
        self.assertIn("Estimated cost: $0.0135", out)  # 1000*1 + 8000*1.25 + 500*5
        saved = list((self.root / "replays").glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertRegex(saved[0].name, r"^\d{8}T\d{6}_20261005T133631_gemma4_e4b_claude-haiku-4-5\.json$")
        result = json.loads(saved[0].read_text(encoding="utf-8"))
        self.assertEqual((result["status"], result["decision"], result["cycle"], result["role"]["effort"]), ("ok", "fail", 1, "low"))
        self.assertEqual(result["payload"]["findings"][0]["category"], "arithmetic")

    def test_repeat_prints_a_tally_and_saves_each_result(self):
        client = FakeClient(_reply(_payload("pass", [])), _reply(), _reply(_payload("pass", [])))
        code, out, _ = self.run_main(["20261005T133631_gemma4_e4b", "--repeat", "3", "--current-prompt"], client)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(call["system"] == "CURRENT VALIDATOR INSTRUCTIONS" for call in client.calls))
        self.assertIn("Tally over 3 replay(s): pass 2, fail 1, errors 0.", out)
        names = sorted(path.name for path in (self.root / "replays").glob("*.json"))
        self.assertEqual([name.rsplit("_", 1)[-1] for name in names], ["r1.json", "r2.json", "r3.json"])

    def test_dry_run_makes_no_call_and_prints_sizes_and_worst_case_cost(self):
        client = FakeClient(_reply())
        code, out, _ = self.run_main(
            ["20261005T133631_gemma4_e4b", "--role-model", "anthropic:claude-opus-5-5", "--max-tokens", "32000", "--dry-run-prompt"], client)
        self.assertEqual(code, 0)
        self.assertEqual((client.calls, self.factory_calls), ([], []))
        self.assertIn(f"System: {len(SYSTEM_V1):,} chars", out)
        self.assertIn("Worst-case cost $", out)
        self.assertIn("No model call was made.", out)
        self.assertFalse((self.root / "replays").exists())

    def test_refusal_malformed_json_and_failures_exit_non_zero(self):
        cases = (
            (ModelReply("", done_reason="refusal", metadata={"stop_details": {"category": "x"}}), "refusal"),
            (_reply(text="{not json"), "malformed_json"),
            (_reply(text=json.dumps({"decision": "maybe", "audit": {}, "findings": []})), "malformed_json"),
            (ConnectionError("down"), "error"),
        )
        for reply, status in cases:
            with self.subTest(status=status):
                code, out, err = self.run_main(["20261005T133631_gemma4_e4b", "--role-model", "anthropic:claude-haiku-4-5"], FakeClient(reply))
                self.assertEqual(code, 1)
                self.assertIn(f"FAILED ({status})", out)
                self.assertIn("replay call(s) failed", err)

    def test_setup_errors_exit_2_without_a_call(self):
        client = FakeClient(_reply())
        for argv in (["missing_run"], ["20261005T133631_gemma4_e4b", "--cycle", "9"],
                     ["20261005T133631_gemma4_e4b", "--role-model", "nope"]):
            code, _, err = self.run_main(argv, client)
            self.assertEqual(code, 2, argv)
            self.assertIn("Error:", err)
        self.assertEqual(client.calls, [])


class TestIndexIgnoresReplays(ReplayTestCase):
    def test_replays_dir_is_not_a_run_and_replays_are_listed(self):
        replay.replay_validator("20261005T133631_gemma4_e4b", config=CONFIG, runs_root=self.root, prompt_dir=self.prompt_dir,
                                client_factory=self.factory(FakeClient(_reply())), echo=lambda _line: None,
                                now=lambda: datetime(2026, 10, 5, 12, tzinfo=timezone.utc))
        (self.root / "replays" / "metrics.json").write_text(json.dumps({"model": "bogus"}), encoding="utf-8")
        page = update_index(self.root).read_text(encoding="utf-8")
        self.assertIn('value="20261005T133631_gemma4_e4b"', page)
        self.assertNotIn('value="replays"', page)
        self.assertIn("Validator replays", page)
        self.assertIn("replays/", page)


if __name__ == "__main__":
    unittest.main()
