"""Write and update pipeline/metrics.json."""

import json
from pathlib import Path

from pipeline.config import PROJECT_ROOT

_METRICS_PATH: Path = PROJECT_ROOT / "pipeline" / "metrics.json"


def write(data: dict) -> None:
    """Merge data into metrics.json, creating the file if it doesn't exist."""
    existing: dict = {}
    if _METRICS_PATH.exists():
        try:
            existing = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(data)
    _METRICS_PATH.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
