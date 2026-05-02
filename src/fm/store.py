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
    session_id TEXT,
    query_snippet TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    retrieved_at TEXT NOT NULL,
    FOREIGN KEY (tip_id) REFERENCES tips(id)
);

CREATE TABLE IF NOT EXISTS session_injections (
    session_id TEXT NOT NULL,
    tip_id TEXT NOT NULL,
    injected_at TEXT NOT NULL,
    PRIMARY KEY (session_id, tip_id)
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
        self.migrate_add_consolidation_columns()
        self.migrate_add_watermark_columns()
        self.migrate_add_title_column()
        self.migrate_add_embedding_abstracted_column()

    def migrate_add_subtask_columns(self) -> None:
        """Add subtask_id and subtask_description columns if they don't exist (idempotent)."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(tips)").fetchall()}
        if "subtask_id" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN subtask_id TEXT")
        if "subtask_description" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN subtask_description TEXT")
        self._conn.commit()

    def migrate_add_watermark_columns(self) -> None:
        """Add last_turn_count to processed_sessions and session_id to retrievals (idempotent)."""
        existing_ps = {row[1] for row in self._conn.execute("PRAGMA table_info(processed_sessions)").fetchall()}
        if "last_turn_count" not in existing_ps:
            self._conn.execute("ALTER TABLE processed_sessions ADD COLUMN last_turn_count INTEGER NOT NULL DEFAULT 0")
        existing_ret = {row[1] for row in self._conn.execute("PRAGMA table_info(retrievals)").fetchall()}
        if "session_id" not in existing_ret:
            self._conn.execute("ALTER TABLE retrievals ADD COLUMN session_id TEXT")
        self._conn.commit()

    def migrate_add_consolidation_columns(self) -> None:
        """Add status and merged_into columns if they don't exist (idempotent)."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(tips)").fetchall()}
        if "status" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "merged_into" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN merged_into TEXT REFERENCES tips(id)")
        self._conn.commit()

    def migrate_add_title_column(self) -> None:
        """Add title column to tips if it doesn't exist (idempotent)."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(tips)").fetchall()}
        if "title" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN title TEXT NOT NULL DEFAULT ''")
        self._conn.commit()

    def migrate_add_embedding_abstracted_column(self) -> None:
        """Add embedding_abstracted flag so --force re-embed runs are resumable (idempotent)."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(tips)").fetchall()}
        if "embedding_abstracted" not in existing:
            self._conn.execute("ALTER TABLE tips ADD COLUMN embedding_abstracted INTEGER NOT NULL DEFAULT 0")
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
                id, category, content, title, purpose, steps, trigger,
                negative_example, priority, source_session_id,
                source_project, task_context, subtask_id, subtask_description,
                embedding, embedding_provider, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tip.id,
                tip.category,
                tip.content,
                tip.title,
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
            "SELECT * FROM tips WHERE status = 'active' AND embedding IS NOT NULL AND embedding_provider = ?",
            (provider,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["embedding"] = _unpack_embedding(d["embedding"])
            results.append(d)
        return results

    def list_tips(self, *, category: str | None = None, include_merged: bool = False) -> list[Tip]:
        status_filter = "" if include_merged else "AND status = 'active'"
        if category:
            rows = self._conn.execute(
                f"SELECT * FROM tips WHERE category = ? {status_filter} ORDER BY created_at DESC",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT * FROM tips WHERE 1=1 {status_filter} ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_tip(row) for row in rows]

    def get_injected_tip_ids(self, session_id: str) -> set[str]:
        """Return tip IDs already injected in this session window."""
        rows = self._conn.execute(
            "SELECT tip_id FROM session_injections WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return {row[0] for row in rows}

    def record_injections(self, session_id: str, tip_ids: list[str]) -> None:
        """Record tips as injected for this session."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "INSERT OR IGNORE INTO session_injections (session_id, tip_id, injected_at) VALUES (?, ?, ?)",
            [(session_id, tip_id, now) for tip_id in tip_ids],
        )
        self._conn.commit()

    def clear_session_injections(self, session_id: str) -> None:
        """Clear injection history for a session (called on PreCompact)."""
        self._conn.execute(
            "DELETE FROM session_injections WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def get_last_turn_count(self, session_id: str) -> int:
        """Return the turn count at last extraction, or 0 if never extracted."""
        row = self._conn.execute(
            "SELECT last_turn_count FROM processed_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def mark_merged(self, tip_ids: list[str], canonical_id: str) -> None:
        """Soft-delete tips that were merged into a canonical tip."""
        for tip_id in tip_ids:
            self._conn.execute(
                "UPDATE tips SET status = 'merged', merged_into = ? WHERE id = ?",
                (canonical_id, tip_id),
            )
        self._conn.commit()

    def mark_session_processed(
        self, session_id: str, jsonl_path: str, tip_count: int, last_turn_count: int = 0
    ) -> None:
        self._conn.execute(
            """INSERT INTO processed_sessions
               (session_id, jsonl_path, processed_at, tip_count, last_turn_count)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   processed_at = excluded.processed_at,
                   tip_count = tip_count + excluded.tip_count,
                   last_turn_count = excluded.last_turn_count""",
            (
                session_id,
                jsonl_path,
                datetime.now(timezone.utc).isoformat(),
                tip_count,
                last_turn_count,
            ),
        )
        self._conn.commit()

    def log_retrieval(
        self, tip_id: str, query: str, similarity_score: float, session_id: str | None = None
    ) -> None:
        self._conn.execute(
            """INSERT INTO retrievals (tip_id, session_id, query_snippet, similarity_score, retrieved_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                tip_id,
                session_id,
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

    def get_dashboard_stats(self) -> dict:
        """Aggregate stats for the dashboard command."""
        # Corpus
        active = self._conn.execute("SELECT COUNT(*) FROM tips WHERE status = 'active'").fetchone()[0]
        merged = self._conn.execute("SELECT COUNT(*) FROM tips WHERE status = 'merged'").fetchone()[0]
        by_category = {
            row[0]: row[1]
            for row in self._conn.execute(
                "SELECT category, COUNT(*) FROM tips WHERE status = 'active' GROUP BY category"
            ).fetchall()
        }
        by_priority = {
            row[0]: row[1]
            for row in self._conn.execute(
                "SELECT priority, COUNT(*) FROM tips WHERE status = 'active' GROUP BY priority"
            ).fetchall()
        }
        embedded = self._conn.execute(
            "SELECT COUNT(*) FROM tips WHERE status = 'active' AND embedding IS NOT NULL"
        ).fetchone()[0]

        # Sessions
        sessions_total = self._conn.execute("SELECT COUNT(*) FROM processed_sessions").fetchone()[0]
        sessions_with_tips = self._conn.execute(
            "SELECT COUNT(*) FROM processed_sessions WHERE tip_count > 0"
        ).fetchone()[0]
        recent_sessions = self._conn.execute(
            """SELECT jsonl_path, tip_count, processed_at
               FROM processed_sessions
               WHERE tip_count > 0
               ORDER BY processed_at DESC LIMIT 5"""
        ).fetchall()

        # Retrieval
        total_retrievals = self._conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
        tips_with_hits = self._conn.execute(
            "SELECT COUNT(DISTINCT tip_id) FROM retrievals"
        ).fetchone()[0]
        top_tips = self._conn.execute(
            """SELECT t.content, t.title, t.trigger, t.category, t.priority,
               COUNT(r.id) as hits, AVG(r.similarity_score) as avg_score
               FROM retrievals r JOIN tips t ON r.tip_id = t.id
               WHERE t.status = 'active'
               GROUP BY r.tip_id ORDER BY hits DESC LIMIT 5"""
        ).fetchall()
        never_hit = self._conn.execute(
            """SELECT COUNT(*) FROM tips
               WHERE status = 'active'
               AND id NOT IN (SELECT DISTINCT tip_id FROM retrievals)"""
        ).fetchone()[0]

        return {
            "corpus": {
                "active": active,
                "merged": merged,
                "by_category": by_category,
                "by_priority": by_priority,
                "embedded": embedded,
            },
            "sessions": {
                "total": sessions_total,
                "with_tips": sessions_with_tips,
                "recent": [dict(r) for r in recent_sessions],
            },
            "retrieval": {
                "total": total_retrievals,
                "tips_with_hits": tips_with_hits,
                "never_hit": never_hit,
                "top_tips": [dict(r) for r in top_tips],
            },
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
            title=row["title"] or "",
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
