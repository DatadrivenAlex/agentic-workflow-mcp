"""Local memory. Short-term is per thread. Long-term survives later runs."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    namespace: str
    content: str
    created_at: str


class ShortTermMemory:
    """Bounded JSON scratchpad keyed by thread id."""

    def __init__(self, window: int, path: Path | None = None) -> None:
        if window < 1:
            raise ValueError("short-term window must be at least 1")
        self.window = window
        self.path = path
        self._items: dict[str, list[dict[str, str]]] = {}
        if path is not None and path.exists():
            self._items = self._load(path)

    def add(self, thread_id: str, role: str, content: str) -> None:
        bucket = self._items.setdefault(thread_id, [])
        bucket.append(
            {
                "role": role,
                "content": content,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(bucket) > self.window:
            self._items[thread_id] = bucket[-self.window :]
        self._flush()

    def recent(self, thread_id: str) -> list[dict[str, str]]:
        return list(self._items.get(thread_id, []))

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, indent=2), encoding="utf-8")

    @staticmethod
    def _load(path: Path) -> dict[str, list[dict[str, str]]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        cleaned: dict[str, list[dict[str, str]]] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, list):
                cleaned[key] = [item for item in value if isinstance(item, dict)]
        return cleaned


class LongTermMemory:
    """SQLite fact store shared across threads and processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, content: str, namespace: str = "default") -> MemoryRecord:
        text = content.strip()
        if not text:
            raise ValueError("memory content is empty")
        space = namespace.strip() or "default"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories (namespace, content, created_at) VALUES (?, ?, ?)",
                (space, text, created_at),
            )
            memory_id = int(cursor.lastrowid)
        return MemoryRecord(id=memory_id, namespace=space, content=text, created_at=created_at)

    def list_recent(self, limit: int = 20) -> list[MemoryRecord]:
        limit = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, namespace, content, created_at
                FROM memories
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row(row) for row in rows]

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        tokens = _tokens(query)
        if not tokens:
            return []
        limit = max(1, min(limit, 20))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, namespace, content, created_at FROM memories ORDER BY id DESC"
            ).fetchall()
        scored: list[tuple[int, int, MemoryRecord]] = []
        for row in rows:
            record = _row(row)
            haystack = set(_tokens(record.content))
            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((score, record.id, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def _row(row: tuple) -> MemoryRecord:
    return MemoryRecord(id=int(row[0]), namespace=str(row[1]), content=str(row[2]), created_at=str(row[3]))


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS and len(token) > 1]
