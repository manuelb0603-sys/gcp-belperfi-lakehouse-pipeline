"""Compare archived agent runs with the eval model and save an HTML comparison page.

Usage (from the project root):
    ..\\.venv\\Scripts\\python.exe Scripts\\compare_runs.py RUN_ID RUN_ID [...] [--config PATH] [--dry-run-prompt]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finance.run_compare import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
