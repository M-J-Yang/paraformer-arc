from __future__ import annotations

import json

from models.history_entry import HistoryEntry
from utils.config import PROJECT_ROOT

HISTORY_FILE = PROJECT_ROOT / ".atc_history.json"
MAX_HISTORY_ENTRIES = 200


def load_history_entries() -> list[HistoryEntry]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return []
    entries = [HistoryEntry.from_dict(item) for item in data if isinstance(item, dict)]
    entries.sort(key=lambda item: item.created_at, reverse=True)
    return entries


def append_history_entry(entry: HistoryEntry) -> None:
    entries = load_history_entries()
    entries.insert(0, entry)
    trimmed_entries = entries[:MAX_HISTORY_ENTRIES]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump([item.to_dict() for item in trimmed_entries], file, ensure_ascii=False, indent=2)
        file.write("\n")


def clear_history_entries() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
