import json
from pathlib import Path
from typing import Any


def load_seed(path: str | Path) -> list[dict[str, Any]]:
    seed_path = Path(path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Seed file must contain a non-empty 'items' list")

    required_keys = {"id", "category", "intent", "question", "answer"}
    for item in items:
        missing = required_keys.difference(item.keys())
        if missing:
            raise ValueError(f"Seed item missing keys: {missing}")
    return items
