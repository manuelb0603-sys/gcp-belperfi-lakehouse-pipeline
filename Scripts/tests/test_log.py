"""finance.log: one console handler, the [timestamp] message format, redirect-safe, encoding-safe."""

import io
import re
import unittest
from contextlib import redirect_stdout

from finance.log import get_logger
from finance.status_log import log_status

LINE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] (.*)$")


class Cp1252Stream(io.StringIO):
    """A console stream that cannot encode emoji, like a cp1252 Windows console."""

    encoding = "cp1252"

    def write(self, text):
        text.encode(self.encoding)  # raises UnicodeEncodeError for characters cp1252 lacks
        return super().write(text)


class TestFinanceLog(unittest.TestCase):
    def test_log_status_keeps_the_timestamped_line_format(self):
        with redirect_stdout(io.StringIO()) as out:
            log_status("Starting weekly report run")
        self.assertRegex(out.getvalue(), r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Starting weekly report run\n$")

    def test_elapsed_suffix_and_one_line_per_call(self):
        with redirect_stdout(io.StringIO()) as out:
            log_status("done", start_time=0.0)
            log_status("again")
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(LINE.match(lines[0]).group(1), r"^done \| elapsed=\d+\.\ds$")

    def test_preformatted_records_are_written_unchanged(self):
        with redirect_stdout(io.StringIO()) as out:
            get_logger("finance.run_archive").info("[2026-01-01 00:00:00] already stamped", extra={"preformatted": True})
        self.assertEqual(out.getvalue(), "[2026-01-01 00:00:00] already stamped\n")

    def test_names_outside_the_namespace_are_nested_under_finance(self):
        self.assertEqual(get_logger("reports.weekly").name, "finance.reports.weekly")
        self.assertEqual(get_logger("finance.qa").name, "finance.qa")

    def test_emoji_on_a_console_that_cannot_encode_them_does_not_raise(self):
        stream = Cp1252Stream()
        with redirect_stdout(stream):
            log_status("Report sent \U0001f363")
        self.assertTrue(stream.getvalue().rstrip("\n").endswith("Report sent ?"))

    def test_one_handler_however_many_loggers(self):
        for name in ("finance.a", "finance.b", "finance.a"):
            get_logger(name)
        self.assertEqual(len(get_logger("finance").handlers), 1)
        self.assertFalse(get_logger("finance").propagate)


if __name__ == "__main__":
    unittest.main()
