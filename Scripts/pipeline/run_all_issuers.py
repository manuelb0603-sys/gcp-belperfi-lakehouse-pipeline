import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])  # Scripts/, so `import finance` works from this subfolder
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from finance.settings import DEFAULT_CONFIG_PATH, PROJECT_ROOT  # noqa: E402

project_root = PROJECT_ROOT
config_path = DEFAULT_CONFIG_PATH

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

issuers = list(config.get("issuers", {}).keys())

if not issuers:
    raise SystemExit("No issuers found in config.json")

for issuer in issuers:
    print(f"Running upload for issuer: {issuer}")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "upload_csv.py"), "--config", "config.json", issuer],
        cwd=project_root,
        check=True,
    )
