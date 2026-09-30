"""Audit log: satu baris JSON per event/request, ditulis ke
app/output/audit.jsonl (JSON Lines -- satu objek JSON per baris).

Dipakai dari call_agent() (audit request lengkap tiap giliran chat) dan dari
conversation.py (event compaction) -- juga dari call_agent() untuk event
pergantian provider (Gemini -> Anthropic).
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).resolve().parents[2] / "app" / "output" / "audit.jsonl"

_write_lock = threading.Lock()


def log(record: dict) -> None:
    """Tambahkan satu baris JSON ke audit.jsonl. `timestamp` (UTC, ISO 8601)
    ditambahkan otomatis kalau belum ada di `record`."""
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_last(n: int = 20) -> list[dict]:
    """Baca n baris terakhir dari audit.jsonl, terlama -> terbaru. Return
    list kosong kalau file belum ada."""
    if not AUDIT_LOG_PATH.exists():
        return []

    with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    last_lines = [line for line in lines[-n:] if line.strip()]
    return [json.loads(line) for line in last_lines]
