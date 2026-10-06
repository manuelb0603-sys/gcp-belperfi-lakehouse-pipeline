"""Compare archived agent runs with the eval model and save an HTML comparison page.

Usage (from the project root):
    ..\\.venv\\Scripts\\python.exe Scripts\\reports\\compare_runs.py RUN_ID RUN_ID [...] [--config PATH] [--dry-run-prompt]
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])  # Scripts/, so `import finance` works from this subfolder
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from finance.run_compare import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
