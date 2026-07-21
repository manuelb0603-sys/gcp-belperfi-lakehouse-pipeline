import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

issuers = list(config.get("issuers", {}).keys())

if not issuers:
    raise SystemExit("No issuers found in config.json")

for issuer in issuers:
    print(f"Running upload for issuer: {issuer}")
    subprocess.run(
        [sys.executable, str(project_root / "Scripts" / "upload_csv.py"), "--config", "config.json", issuer],
        cwd=project_root,
        check=True,
    )
