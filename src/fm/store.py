from __future__ import annotations

import json
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

from fm.models import Tip

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tips (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    purpose TEXT,
    steps TEXT,
    trigger TEXT,
    negative_example TEXT,
    priority TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_project TEXT,
    task_context TEXT,
    subtask_id TEXT,
    subtask_description TEXT,
    embedding BLOB,
    embedding_provider TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_sessions (
    session_id TEXT PRIMARY KEY,
    jsonl_path TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    tip_count INTEGER
);

CREATE TABLE IF NOT EXISTS retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tip_id TEXT NOT NULL,
    query_snippet TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    retrieved_at TEXT NOT NULL,
    FOREIGN KEY (tip_id) REFERENCES tips(id)
);
"""


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


class TipStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self.migrate_add_subtask_columns()

    def migrate_add_subtask_columns(self) -> None:
        """Add subtask_id and subtask_description columns if they don't exist (idempotent)."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(tips)").fetchall()}
        if "subtask_id" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN subtask_id TEXT")
        if "subtask_description" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN subtask_description TEXT")
        self._conn.commit()

    def get_embedding_key(self, tip: Tip) -> str:
        """Return the text to embed for retrieval — subtask description when available, else content+trigger."""
        if tip.subtask_description:
            return tip.subtask_description
        return f"{tip.content} {tip.trigger}"

    def add_tip(
        self,
        tip: Tip,
        *,
        embedding: list[float] | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        embedding_blob = _pack_embedding(embedding) if embedding else None
        self._conn.execute(
            """INSERT INTO tips (
                id, category, content, purpose, steps, trigger,
                negative_example, priority, source_session_id,
                source_project, task_context, subtask_id, subtask_description,
                embedding, embedding_provider, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tip.id,
                tip.category,
                tip.content,
                tip.purpose,
                json.dumps(tip.steps),
                tip.trigger,
                tip.negative_example,
                tip.priority,
                tip.source_session_id,
                tip.source_project,
                tip.task_context,
                tip.subtask_id,
                tip.subtask_description,
                embedding_blob,
                embedding_provider,
                tip.created_at,
            ),
        )
        self._conn.commit()

    def get_tip(self, tip_id: str) -> Tip | None:
        row = self._conn.execute(
            "SELECT * FROM tips WHERE id = ?", (tip_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_tip(row)

    def get_tip_with_embedding(self, tip_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tips WHERE id = ?", (tip_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result["embedding"]:
            result["embedding"] = _unpack_embedding(result["embedding"])
        return result

    def get_tips_with_embeddings(self, provider: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM tips WHERE embedding IS NOT NULL AND embedding_provider = ?",
            (provider,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["embedding"] = _unpack_embedding(d["embedding"])
            results.append(d)
        return results

    def list_tips(self, *, category: str | None = None) -> list[Tip]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM tips WHERE category = ? ORDER BY created_at DESC",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tips ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_tip(row) for row in rows]

    def mark_session_processed(
        self, session_id: str, jsonl_path: str, tip_count: int
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO processed_sessions
            (session_id, jsonl_path, processed_at, tip_count)
            VALUES (?, ?, ?, ?)""",
            (
                session_id,
                jsonl_path,
                datetime.now(timezone.utc).isoformat(),
                tip_count,
            ),
        )
        self._conn.commit()

    def log_retrieval(
        self, tip_id: str, query: str, similarity_score: float
    ) -> None:
        self._conn.execute(
            """INSERT INTO retrievals (tip_id, query_snippet, similarity_score, retrieved_at)
            VALUES (?, ?, ?, ?)""",
            (
                tip_id,
                query[:200],
                similarity_score,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_retrieval_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
        top_tips = self._conn.execute(
            """SELECT t.content, t.category, t.priority, COUNT(r.id) as hits,
               AVG(r.similarity_score) as avg_score
               FROM retrievals r JOIN tips t ON r.tip_id = t.id
               GROUP BY r.tip_id ORDER BY hits DESC LIMIT 10"""
        ).fetchall()
        never_retrieved = self._conn.execute(
            """SELECT COUNT(*) FROM tips WHERE id NOT IN
               (SELECT DISTINCT tip_id FROM retrievals)"""
        ).fetchone()[0]
        recent = self._conn.execute(
            """SELECT t.content, r.similarity_score, r.retrieved_at, r.query_snippet
               FROM retrievals r JOIN tips t ON r.tip_id = t.id
               ORDER BY r.retrieved_at DESC LIMIT 5"""
        ).fetchall()
        return {
            "total_retrievals": total,
            "never_retrieved": never_retrieved,
            "top_tips": [dict(r) for r in top_tips],
            "recent": [dict(r) for r in recent],
        }

    def is_session_processed(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    def _row_to_tip(self, row: sqlite3.Row) -> Tip:
        return Tip(
            id=row["id"],
            category=row["category"],
            content=row["content"],
            purpose=row["purpose"] or "",
            steps=json.loads(row["steps"]) if row["steps"] else [],
            trigger=row["trigger"] or "",
            negative_example=row["negative_example"],
            priority=row["priority"],
            source_session_id=row["source_session_id"],
            source_project=row["source_project"] or "",
            task_context=row["task_context"],
            subtask_id=row["subtask_id"],
            subtask_description=row["subtask_description"],
            created_at=row["created_at"],
        )
