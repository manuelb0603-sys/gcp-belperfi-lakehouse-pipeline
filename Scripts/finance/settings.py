"""Shared project configuration and finance-setting normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load project settings from the requested path or the project config."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def parse_money(value: Any) -> float:
    """Normalize numbers or amount strings (including k/m suffixes) to a float."""
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?\s*[kKmM]?", str(value or ""))
    if not matches:
        raise ValueError(f"Could not parse a financial amount from: {value!r}")
    total = 0.0
    for match in matches:
        normalized = match.replace(" ", "")
        multiplier = 1
        if normalized[-1:].lower() == "k":
            multiplier, normalized = 1_000, normalized[:-1]
        elif normalized[-1:].lower() == "m":
            multiplier, normalized = 1_000_000, normalized[:-1]
        total += float(normalized) * multiplier
    return total
