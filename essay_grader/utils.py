from __future__ import annotations

import json
import os
from typing import Any, Dict


def safe_json_loads(s: str) -> Dict[str, Any]:
    """Safely parse a JSON string into a Python dict.

    Ensures the result is a dictionary and raises ValueError otherwise.
    """
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by LLM: {e}\nRaw: {s[:500]}")
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object/dict.")
    return data


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_json_file(path: str) -> Dict[str, Any]:
    """Load and return JSON file as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data: Dict[str, Any]) -> None:
    """Save dict as JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
