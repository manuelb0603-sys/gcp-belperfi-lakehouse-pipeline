import json
import re
import unittest

from finance import settings
from finance.harness import HarnessLimits
from finance.model_clients import DEFAULT_TIMEOUT, build_model_client
from finance.settings import DEFAULTS, ReportSettings, SettingsError

README = settings.PROJECT_ROOT / "README.md"
EXAMPLE = settings.PROJECT_ROOT / "config.example.json"


def _readme_defaults() -> dict:
    """Rows of the README "Code defaults" table: | `key` | `json value` | ..."""
    text = README.read_text(encoding="utf-8")
    section = text.split("##### Code defaults", 1)[1].split("\n#", 1)[0]
    rows = re.findall(r"^\|\s*`([A-Za-z_]+)`\s*\|\s*`([^`]*)`\s*\|", section, re.M)
    return {key: json.loads(value) for key, value in rows}


class TestDefaults(unittest.TestCase):
    def test_readme_defaults_table_matches_code(self):
        self.assertEqual(_readme_defaults(), DEFAULTS)

    def test_empty_config_uses_defaults(self):
        s = ReportSettings.from_config({})
        self.assertEqual(s.report_mode, "legacy")
        self.assertFalse(s.dry_run)
        self.assertEqual(s.ollama_timeout, 600)
        self.assertEqual(s.limits(), {
            "max_agent_rounds": 6, "max_tool_calls": 12, "max_calls_per_tool": 3,
            "max_validation_cycles": 3, "max_report_bytes": 250000, "max_run_seconds": 2700,
        })
        self.assertEqual(s.days_back, 60)
        self.assertEqual(s.embedding_model_name, "nomic-embed-text")
        self.assertEqual(s.memory_schema, "memory")
        self.assertEqual((s.qa_report_context_chunks, s.qa_long_fact_limit), (5, 5))
        self.assertEqual(s.category_goals, {})
        self.assertEqual(s.email_listings, [])
        self.assertIsNone(s.savings_goal_target)
        self.assertEqual(s.savings_goal_target_raw, "not configured")

    def test_other_modules_take_their_defaults_from_settings(self):
        limits = HarnessLimits()
        self.assertEqual(
            {key: getattr(limits, key) for key in ReportSettings.from_config({}).limits()},
            ReportSettings.from_config({}).limits(),
        )
        self.assertEqual(DEFAULT_TIMEOUT, DEFAULTS["ollama_timeout"])

    def test_example_config_is_valid_and_shows_recommended_roles(self):
        example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.assertNotIn("validator_model_name", example)
        self.assertNotIn("ollama_options", example)
        s = ReportSettings.from_config(example)
        self.assertEqual(s.author_role["model"], "gemma4:e4b")
        self.assertEqual(s.validator_role["model"], "gemma4:12b")
        self.assertIs(s.validator_role["think"], False)
        self.assertEqual(s.validator_role["options"]["num_predict"], 4000)
        self.assertEqual(s.author_role["options"]["num_ctx"], 32768)
        self.assertEqual(s.validator_role["options"]["num_ctx"], 32768)
        self.assertEqual(s.max_validation_cycles, 5)
        self.assertEqual(s.max_run_seconds, 2700)


class TestParsing(unittest.TestCase):
    def test_values_are_typed_and_money_is_parsed(self):
        s = ReportSettings.from_config({
            "report_mode": "AGENT", "dry_run": 1, "ollama_timeout": "1800", "max_run_seconds": 0,
            "Monthly_budget": "5k", "Fixed_costs": "Rent: $2000, Car: $500", "Savings_goal_target": 1000,
            "category_goals": {"Dining": 200}, "email_listings": ["a@x"], "memory_schema": "finance_v2",
        })
        self.assertEqual(s.report_mode, "agent")
        self.assertIs(s.dry_run, True)
        self.assertEqual(s.ollama_timeout, 1800)
        self.assertIsNone(s.max_run_seconds)
        self.assertEqual((s.monthly_budget, s.fixed_costs, s.savings_goal_target), (5000.0, 2500.0, 1000.0))
        self.assertEqual(s.monthly_budget_raw, "5k")
        self.assertEqual(s.category_goals, {"Dining": 200})
        self.assertEqual(s.memory_db_path.name, "finance_v2.sqlite3")

    def test_unparseable_budget_is_none_until_required(self):
        self.assertIsNone(ReportSettings.from_config({"Monthly_budget": "your-monthly-budget"}).monthly_budget)

    def test_validation_errors_name_the_key(self):
        cases = [
            ({"report_mode": "turbo"}, "report_mode"),
            ({"max_agent_rounds": 0}, "max_agent_rounds"),
            ({"max_tool_calls": -1}, "max_tool_calls"),
            ({"max_calls_per_tool": "many"}, "max_calls_per_tool"),
            ({"max_validation_cycles": 2.5}, "max_validation_cycles"),
            ({"max_report_bytes": True}, "max_report_bytes"),
            ({"max_run_seconds": -5}, "max_run_seconds"),
            ({"max_run_seconds": "soon"}, "max_run_seconds"),
            ({"ollama_timeout": 0}, "ollama_timeout"),
            ({"days_back": 0}, "days_back"),
            ({"memory_schema": "../escape"}, "memory_schema"),
            ({"Savings_goal_target": "a lot"}, "Savings_goal_target"),
            ({"Savings_goal_target": True}, "Savings_goal_target"),
            ({"category_goals": ["Dining"]}, "category_goals"),
            ({"email_listings": "a@x"}, "email_listings"),
            ({"model_roles": {"author": {"provider": "other", "model": "m"}}}, "model_roles"),
        ]
        for config, key in cases:
            with self.subTest(key=key, config=config):
                with self.assertRaisesRegex(SettingsError, key):
                    ReportSettings.from_config(config)
        self.assertTrue(issubclass(SettingsError, ValueError))

    def test_repr_masks_credentials_and_hides_raw_config(self):
        s = ReportSettings.from_config({"Gmail_app_credentials": "abcd efgh ijkl mnop", "email_sender": "s@x"})
        self.assertEqual(s.gmail_app_credentials, "abcd efgh ijkl mnop")
        for text in (repr(s), str(s)):
            self.assertNotIn("abcd efgh", text)
            self.assertIn("gmail_app_credentials='***'", text)
            self.assertNotRegex(text, r"\braw=")

    def test_settings_are_frozen(self):
        s = ReportSettings.from_config({})
        with self.assertRaises(Exception):
            s.max_tool_calls = 99  # type: ignore[misc]

    def test_coerce_accepts_dict_or_settings(self):
        s = ReportSettings.from_config({"days_back": 30})
        self.assertIs(ReportSettings.coerce(s), s)
        self.assertEqual(ReportSettings.coerce({"days_back": 30}).days_back, 30)
        self.assertEqual(ReportSettings.coerce(None).days_back, 60)


class TestModelRoles(unittest.TestCase):
    def test_model_roles_are_authoritative_over_legacy_keys(self):
        s = ReportSettings.from_config({
            "model_server_url": "http://ollama:11434",
            "model_name": "legacy-author",
            "validator_model_name": "legacy-validator",
            "ollama_options": {"validator": {"num_predict": 99}},
            "model_roles": {
                "author": {"provider": "ollama", "model": "gemma4:e4b"},
                "validator": {"provider": "ollama", "model": "gemma4:12b", "think": False, "options": {"num_predict": 4000}},
            },
        })
        self.assertEqual(s.author_role["model"], "gemma4:e4b")
        self.assertEqual(s.validator_role["model"], "gemma4:12b")
        self.assertEqual(s.validator_role["options"], {"num_predict": 4000})
        self.assertEqual(s.eval_role["model"], "gemma4:12b")  # eval falls back to the validator
        self.assertEqual(build_model_client(s, "validator").model, "gemma4:12b")

    def test_legacy_keys_are_the_fallback(self):
        s = ReportSettings.from_config({
            "model_server_url": "http://ollama:11434", "model_name": "a", "validator_model_name": "v",
            "ollama_options": {"validator": {"think": False, "num_predict": 4000}},
        })
        self.assertEqual((s.author_role["model"], s.validator_role["model"]), ("a", "v"))
        self.assertIs(s.validator_role["think"], False)

    def test_caller_overrides_fill_legacy_keys(self):
        s = ReportSettings.from_config({"model_name": "a"}).with_model_overrides("http://other:11434", "b")
        self.assertEqual(s.author_role["model"], "b")
        self.assertEqual(s.author_role["base_url"], "http://other:11434")

    def test_public_role_has_no_urls_or_secrets(self):
        s = ReportSettings.from_config({
            "model_server_url": "http://secret-host:11434", "Gmail_app_credentials": "pw",
            "model_roles": {"author": {"provider": "ollama", "model": "m", "options": {"num_ctx": 32768}, "think": False}},
        })
        public = s.public_role("author")
        self.assertEqual(public["model"], "m")
        self.assertEqual(public["options"], {"num_ctx": 32768})
        self.assertIs(public["think"], False)
        self.assertNotIn("base_url", public)
        self.assertNotIn("secret-host", json.dumps(public))


if __name__ == "__main__":
    unittest.main()
