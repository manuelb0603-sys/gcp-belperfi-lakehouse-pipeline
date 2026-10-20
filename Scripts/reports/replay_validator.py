"""Replay an archived validator call against the configured validator or another model.

Usage (from the project root):
    ..\\.venv\\Scripts\\python.exe Scripts\\reports\\replay_validator.py RUN_ID [--cycle N|last] [--role-model PROVIDER:MODEL]
        [--effort low|medium|high] [--think true|false] [--max-tokens N] [--repeat K]
        [--dry-run-prompt] [--current-prompt] [--config PATH]

Results are saved under agent/runs/replays/. Logic lives in finance/replay.py.
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])  # Scripts/, so `import finance` works from this subfolder
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from finance.replay import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
