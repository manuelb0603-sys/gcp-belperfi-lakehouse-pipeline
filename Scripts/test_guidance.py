"""guidance: strict | guided — strict prompts stay byte-identical; guided relaxes wording, structure, and plans."""

import json
import unittest

from finance.harness import HarnessLimits, ReportHarness, ToolSpec
from finance.harness.feedback import focused_feedback
from finance.harness.prompts import (
    GUIDED_AUTHOR_ADDENDUM,
    GUIDED_REVISION_INSTRUCTION,
    GUIDED_VALIDATOR_ADDENDUM,
    GUIDED_VALIDATOR_CHECKLIST_INSTRUCTION,
    STRICT_REVISION_INSTRUCTION,
    VALIDATOR_CHECKLIST_INSTRUCTION,
    revision_block,
)
from finance.model_clients import ModelReply, resolve_model_role
from finance.report_models import ValidationDecision, ValidationFinding
from finance.report_requirements import build_report_requirements
from finance.settings import DEFAULTS, ReportSettings, SettingsError
from guidance_test_fixtures import (
    FIVE_ERRORS,
    GOLDEN_PATH,
    PASS,
    REPORT,
    ScriptedClient,
    capture_strict_prompts,
    cuts_insufficient_rows,
    run_scripted,
    scenarios,
)
from guidance_test_fixtures import finding as _finding

# GOLDEN_PATH holds the full strict prompts captured from HEAD 17093ab (before the guidance setting
# existed) with guidance_test_fixtures.capture_strict_prompts(); strict must reproduce them byte for byte.


class StrictGoldenTests(unittest.TestCase):
    def test_strict_prompts_match_pre_guidance_golden_capture(self):
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        for kwargs in ({}, {"guidance": "strict"}):
            current = capture_strict_prompts(**kwargs)
            self.assertEqual(current["requirements"], golden["requirements"])
            for name in ("cuts_insufficient",):
                self.assertEqual(current[name]["decision"], golden[name]["decision"])
                self.assertEqual(len(current[name]["calls"]), len(golden[name]["calls"]))
                for index, (now, before) in enumerate(zip(current[name]["calls"], golden[name]["calls"])):
                    self.assertEqual(now, before, f"{name} call {index + 1} ({before['role']}) changed")

    def test_strict_prompts_contain_no_guided_text(self):
        snapshot = scenarios()["cuts_insufficient"]
        calls, _, _, _ = run_scripted(snapshot, build_report_requirements(snapshot))
        for call in calls:
            text = f"{call['system']}{call['user']}"
            for guided_text in (GUIDED_AUTHOR_ADDENDUM, GUIDED_VALIDATOR_ADDENDUM, GUIDED_VALIDATOR_CHECKLIST_INSTRUCTION,
                                GUIDED_REVISION_INSTRUCTION, "Mogumogu's alternative plan"):
                self.assertNotIn(guided_text, text)


class GuidedChecklistTests(unittest.TestCase):
    def test_cuts_insufficient_month_lists_required_facts_in_any_wording(self):
        snapshot = scenarios()["cuts_insufficient"]
        lines = build_report_requirements(snapshot, "guided")
        body = "\n".join(lines)
        self.assertLessEqual(len(lines), 20)
        # Sections in order, alternative headings allowed when recognizable.
        self.assertIn('"Monthly Snapshot" -> "Spending Trends & Analysis"', lines[0])
        self.assertIn("alternative heading is allowed only if each required section stays clearly recognizable", lines[0])
        self.assertNotIn("Keep the wording exactly as given", body)
        # Any wording; rounding only when labeled approximate.
        self.assertIn("any wording", body)
        self.assertIn("rounded where the text clearly labels it approximate", body)
        # The disclosure is a required fact, not a required sentence.
        self.assertIn(
            "Required fact (any wording): discretionary cuts alone cannot reach the $2,000.00 savings target; "
            "even cutting all discretionary spend leaves a remaining gap of $2,166.67.", body
        )
        self.assertNotIn("State explicitly", body)
        # Facts stay exact: totals, MoM, disposable cash, shortfall, eligible total, per-category spend.
        for fact in ("August 2026 total net expense $3,269.84", "July 2026 total net expense $3,170.50",
                     "month-over-month change 3.13% increase", "disposable cash -$1,769.84",
                     "shortfall $3,769.84", "eligible discretionary spend $1,603.17",
                     "Needs: Groceries $1,100.00, Health -$33.33 (label as refund/credit), Transport $600.00.",
                     "1. Food & Drink: current $812.40; previous (July 2026) $700.00; MoM 16.06% increase; "
                     "advisor-suggested cap $0.00"):
            self.assertIn(fact, body)
        # Code-computed caps stay the baseline; the optional alternative plan must reconcile.
        self.assertIn("code-computed baseline plan", body)
        self.assertIn("never a configured goal", body)
        self.assertIn("any remaining gap it claims must equal the $3,769.84 shortfall minus those cuts", body)

    def test_no_target_month_forbids_a_gap_in_the_alternative_plan(self):
        body = "\n".join(build_report_requirements(scenarios()["no_target"], "guided"))
        self.assertIn("no numeric savings target is configured", body)
        self.assertIn("do not claim a savings target or gap", body)

    def test_unknown_guidance_is_rejected(self):
        with self.assertRaises(ValueError):
            build_report_requirements(scenarios()["no_target"], "loose")


class GuidedPromptTests(unittest.TestCase):
    def _guided_run(self, limits=None):
        snapshot = scenarios()["cuts_insufficient"]
        return run_scripted(snapshot, build_report_requirements(snapshot, "guided"), limits=limits, guidance="guided")

    def test_addenda_appear_only_in_guided_mode_in_the_cached_system_parts(self):
        calls, decision, _, _ = self._guided_run()
        self.assertTrue(decision.passed)
        author = [call for call in calls if call["role"] == "author"]
        validator = [call for call in calls if call["role"] == "validator"]
        for call in author:
            self.assertTrue(call["system"].endswith(f"\n\n{GUIDED_AUTHOR_ADDENDUM}"))
            self.assertNotIn(GUIDED_VALIDATOR_ADDENDUM, call["system"] + call["user"])
        for call in validator:
            self.assertEqual(call["system"], f"VALIDATOR CONTEXT\n\n{GUIDED_VALIDATOR_ADDENDUM}")
            self.assertIn(GUIDED_VALIDATOR_CHECKLIST_INSTRUCTION, call["user"])
            self.assertNotIn(VALIDATOR_CHECKLIST_INSTRUCTION, call["user"])
            self.assertNotIn(GUIDED_AUTHOR_ADDENDUM, call["system"] + call["user"])
        # Strict: neither addendum.
        snapshot = scenarios()["cuts_insufficient"]
        strict_calls, _, _, _ = run_scripted(snapshot, build_report_requirements(snapshot))
        self.assertEqual([c["system"] for c in strict_calls if c["role"] == "validator"], ["VALIDATOR CONTEXT"] * 3)

    def test_guided_addenda_keep_the_hard_rules(self):
        self.assertIn("Facts stay authoritative", GUIDED_AUTHOR_ADDENDUM)
        self.assertIn("never as a configured goal", GUIDED_AUTHOR_ADDENDUM)
        self.assertIn("snapshot.savings_shortfall minus those cuts", GUIDED_AUTHOR_ADDENDUM)
        self.assertIn("evidence rule is unchanged", GUIDED_VALIDATOR_ADDENDUM)
        self.assertIn("visual-design hard requirements", GUIDED_VALIDATOR_ADDENDUM)
        self.assertIn("presented as a configured goal", GUIDED_VALIDATOR_ADDENDUM)

    def test_revision_feedback_has_all_errors_when_guided_and_top_three_when_strict(self):
        calls, _, _, _ = self._guided_run()
        first_revision = [call for call in calls if call["role"] == "author"][2]["user"]
        self.assertIn("FIXES FROM THE REVIEW (every error finding, highest priority first)", first_revision)
        for _, message in FIVE_ERRORS:
            self.assertIn(message, first_revision)
        self.assertNotIn("remaining_lower_priority_errors", first_revision)
        self.assertNotIn("Minor spacing", first_revision)  # warnings are still not sent
        self.assertIn(GUIDED_REVISION_INSTRUCTION, first_revision)
        self.assertNotIn("minimal edits", first_revision)

        snapshot = scenarios()["cuts_insufficient"]
        strict_calls, _, _, _ = run_scripted(snapshot, build_report_requirements(snapshot))
        strict_revision = [call for call in strict_calls if call["role"] == "author"][2]["user"]
        self.assertIn("PRIORITY FIXES FROM THE REVIEW:", strict_revision)
        self.assertIn('"remaining_lower_priority_errors": 2', strict_revision)
        for message in ("Shortfall wrong for Shopping", "Missing remaining gap", "Unsupported coffee claim"):
            self.assertIn(message, strict_revision)
        for message in ("Bars not pink", "Low contrast"):
            self.assertNotIn(message, strict_revision)
        self.assertIn(STRICT_REVISION_INSTRUCTION, strict_revision)
        self.assertNotIn(GUIDED_REVISION_INSTRUCTION, strict_revision)

    def test_focused_feedback_orders_all_errors_by_priority_and_keeps_snapshot_values(self):
        snapshot = scenarios()["cuts_insufficient"]
        decision = ValidationDecision("fail", [ValidationFinding(**_finding(c, m)) for c, m in FIVE_ERRORS], "s")
        feedback = focused_feedback(decision, snapshot, None)
        self.assertEqual([fix["category"] for fix in feedback["fix_these"]],
                         ["arithmetic", "completeness", "evidence", "usability", "visual_design"])
        self.assertIn("authoritative_values", feedback["fix_these"][0])
        self.assertEqual(len(focused_feedback(decision, snapshot)["fix_these"]), 3)

    def test_minimal_edit_text_only_in_strict_revision_block(self):
        feedback = {"fix_these": []}
        self.assertIn("minimal edits", revision_block("<p>x</p>", feedback, []))
        self.assertIn("minimal edits", revision_block("<p>x</p>", feedback, [], "strict"))
        guided = revision_block("<p>x</p>", feedback, [], "guided")
        self.assertNotIn("minimal edits", guided)
        self.assertIn("you may restructure sections if that produces a better report, keeping all required facts", guided)

    def test_guided_doubles_tool_limits_only(self):
        limits = HarnessLimits(max_agent_rounds=4, max_tool_calls=12, max_calls_per_tool=3, max_validation_cycles=3,
                               max_report_bytes=1000, max_run_seconds=900)
        _, _, trace, harness = self._guided_run(limits)
        self.assertEqual((harness.limits.max_tool_calls, harness.limits.max_calls_per_tool), (24, 6))
        self.assertEqual(
            (harness.limits.max_agent_rounds, harness.limits.max_validation_cycles, harness.limits.max_report_bytes,
             harness.limits.max_run_seconds),
            (4, 3, 1000, 900),
        )
        self.assertIs(harness.configured_limits, limits)
        started = next(event for event in trace.events if event["event"] == "run_started")
        self.assertEqual(started["guidance"], "guided")
        self.assertEqual((started["limits"]["max_tool_calls"], started["limits"]["max_calls_per_tool"]), (24, 6))
        self.assertEqual((started["configured_limits"]["max_tool_calls"], started["configured_limits"]["max_calls_per_tool"]), (12, 3))

        strict = ReportHarness(limits=limits, author_client=ScriptedClient("a", [], []))
        self.assertIs(strict.limits, limits)

    def test_guided_tool_budget_allows_more_calls_per_tool(self):
        snapshot = scenarios()["no_target"]
        replies = [json.dumps({"tool_calls": [{"name": "t", "arguments": {}}] * 2}), REPORT]
        calls = []
        harness = ReportHarness(
            limits=HarnessLimits(max_tool_calls=2, max_calls_per_tool=1, max_validation_cycles=1),
            author_client=ScriptedClient("author", replies, calls),
            validator_client=ScriptedClient("validator", [PASS], calls),
            guidance="guided",
        )
        _, _, trace = harness.run("o", snapshot, "c", {"t": ToolSpec("t", "d", lambda _: {"ok": True})})
        self.assertEqual(sum(event["event"] == "tool_call" for event in trace.events), 2)
        self.assertFalse(any(event["event"] == "tool_error" for event in trace.events))

    def test_harness_rejects_unknown_guidance(self):
        with self.assertRaises(ValueError):
            ReportHarness(author_client=ScriptedClient("a", [], []), guidance="loose")


class GuidanceSettingsTests(unittest.TestCase):
    def test_default_is_strict(self):
        self.assertEqual(DEFAULTS["guidance"], "strict")
        self.assertEqual(ReportSettings.from_config({}).guidance, "strict")

    def test_top_level_and_case(self):
        self.assertEqual(ReportSettings.from_config({"guidance": "Guided"}).guidance, "guided")
        self.assertEqual(ReportSettings.from_config({"guidance": None}).guidance, "strict")

    def test_author_role_override_wins_over_top_level(self):
        roles = {"author": {"provider": "ollama", "model": "m", "guidance": "guided"}}
        settings = ReportSettings.from_config({"guidance": "strict", "model_roles": roles})
        self.assertEqual(settings.guidance, "guided")
        self.assertEqual(settings.public_role("author")["guidance"], "guided")
        roles = {"author": {"provider": "ollama", "model": "m", "guidance": "strict"}}
        self.assertEqual(ReportSettings.from_config({"guidance": "guided", "model_roles": roles}).guidance, "strict")
        # The override also applies when the author entry has no model (legacy model_name author).
        legacy = ReportSettings.from_config({"model_name": "m", "model_roles": {"author": {"guidance": "guided"}}})
        self.assertEqual(legacy.guidance, "guided")
        self.assertEqual(legacy.role("author")["model"], "m")
        # Without an override, the top-level value applies and the role carries no guidance key.
        plain = ReportSettings.from_config({"guidance": "guided", "model_roles": {"author": {"provider": "ollama", "model": "m"}}})
        self.assertEqual(plain.guidance, "guided")
        self.assertNotIn("guidance", plain.role("author"))
        self.assertNotIn("guidance", resolve_model_role({"model_name": "m"}, "author"))

    def test_invalid_values_raise_settings_error(self):
        for config in (
            {"guidance": "loose"},
            {"guidance": 1},
            {"guidance": True},
            {"model_roles": {"author": {"provider": "ollama", "model": "m", "guidance": "free"}}},
            {"model_roles": {"validator": {"provider": "ollama", "model": "m", "guidance": "guided"}}},
        ):
            with self.subTest(config=config), self.assertRaises(SettingsError) as raised:
                ReportSettings.from_config(config)
            self.assertIn("guidance", str(raised.exception))

    def test_run_metadata_records_guidance_and_effective_limits(self):
        from finance.run_archive import collect_run_metadata

        strict = collect_run_metadata(ReportSettings.from_config({}), code_version={"commit": "a", "dirty": False})
        self.assertEqual(strict["guidance"], "strict")
        self.assertNotIn("effective_harness_limits", strict)
        guided = collect_run_metadata(ReportSettings.from_config({"guidance": "guided"}), code_version={"commit": "a", "dirty": False})
        self.assertEqual(guided["guidance"], "guided")
        self.assertEqual(guided["harness_limits"]["max_tool_calls"], 12)
        self.assertEqual(guided["effective_harness_limits"]["max_tool_calls"], 24)
        self.assertEqual(guided["effective_harness_limits"]["max_calls_per_tool"], 6)


class GuidanceReportingTests(unittest.TestCase):
    """generate_finance_report wiring: checklist variant, harness guidance, and metrics.json guidance."""

    def _generate(self, config):
        from unittest.mock import patch

        from finance import reporting
        from finance.model_clients import OllamaClient

        prompts = []
        replies = [REPORT, PASS]

        def fake_generate(self, prompt, *, system=None, json_schema=None, timeout=None, stream=False):
            prompts.append(f"{system or ''}{prompt}")
            return ModelReply(replies.pop(0), done_reason="stop")

        base = {"model_server_url": "http://fake:11434", "model_name": "m", "Monthly_budget": 5000,
                "Fixed_costs": 3500, "Savings_goal_target": 2000}
        with patch.object(OllamaClient, "generate", fake_generate):
            result = reporting.generate_finance_report(
                {**base, **config}, "proj", "http://fake:11434", "m", rows=cuts_insufficient_rows()
            )
        return result, prompts

    def test_guided_config_reaches_the_prompts_trace_and_metrics(self):
        result, prompts = self._generate({"guidance": "guided"})
        self.assertEqual(result.metrics["guidance"], "guided")
        started = next(event for event in result.trace.events if event["event"] == "run_started")
        self.assertEqual(started["guidance"], "guided")
        self.assertIn(GUIDED_AUTHOR_ADDENDUM, prompts[0])
        self.assertIn("Required fact (any wording): discretionary cuts alone cannot reach", prompts[0])
        self.assertIn(GUIDED_VALIDATOR_ADDENDUM, prompts[1])

    def test_strict_config_records_strict(self):
        result, prompts = self._generate({})
        self.assertEqual(result.metrics["guidance"], "strict")
        self.assertNotIn(GUIDED_AUTHOR_ADDENDUM, prompts[0])
        self.assertIn("State explicitly that discretionary cuts alone cannot reach", prompts[0])


class GuidanceReplayAndCompareTests(unittest.TestCase):
    def _run_dir(self, root, guidance):
        from finance.harness.prompts import validator_prompt_parts

        folder = root / f"run_{guidance}"
        folder.mkdir(parents=True)
        parts = validator_prompt_parts("OLD VALIDATOR", "<html>r</html>", "{}", ["Fact."], [], None, [], [], guidance)
        events = [
            {"event": "run_started", "timestamp": 1.0, "guidance": guidance},
            {"event": "model_call", "timestamp": 2.0, "role": "validator", "model": "m", "provider": "ollama",
             "prompt": parts.text, "validation_cycle": 1, "attempt": 1},
            {"event": "validation", "timestamp": 3.0, "cycle": 1, "decision": "pass"},
        ]
        (folder / "trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        (folder / "metrics.json").write_text(json.dumps({"model": "m", "status": "validated", "guidance": guidance}), encoding="utf-8")
        return folder

    def test_replay_current_prompt_keeps_the_guided_addendum_only_for_guided_runs(self):
        import tempfile
        from pathlib import Path

        from finance import replay

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "prompts").mkdir()
            (root / "prompts" / "validator.system.md").write_text("NEW VALIDATOR", encoding="utf-8")
            for guidance, expected in (("guided", f"NEW VALIDATOR\n\n{GUIDED_VALIDATOR_ADDENDUM}"), ("strict", "NEW VALIDATOR")):
                run_dir = self._run_dir(root / "runs", guidance)
                _, original, _ = replay.build_replay_prompt(run_dir, 1)
                self.assertEqual(GUIDED_VALIDATOR_ADDENDUM in original.system, guidance == "guided")
                _, parts, _ = replay.build_replay_prompt(run_dir, 1, current_prompt=True, prompt_dir=root / "prompts")
                self.assertEqual(parts.system, expected)
                self.assertEqual(parts.user, original.user)

    def test_run_comparison_summary_shows_guidance(self):
        import tempfile
        from pathlib import Path

        from finance import run_compare

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guided = run_compare.gather_run(self._run_dir(root, "guided"))
            old = self._run_dir(root, "strict")
            (old / "metrics.json").write_text(json.dumps({"model": "m", "status": "validated"}), encoding="utf-8")
            (old / "trace.jsonl").write_text(json.dumps({"event": "run_started", "timestamp": 1.0}) + "\n", encoding="utf-8")
            older = run_compare.gather_run(old)
        self.assertEqual(guided["guidance"], "guided")
        self.assertEqual(older["guidance"], "not recorded")
        self.assertIn("guidance", run_compare.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
