import json
from pathlib import Path

import pytest

from fm.models import Tip
from fm.store import TipStore


def _make_tip(**overrides) -> Tip:
    defaults = dict(
        category="strategy",
        content="Always verify prerequisites",
        purpose="Prevents failures",
        steps=["Step 1", "Step 2"],
        trigger="When task involves multi-step operations",
        priority="high",
        source_session_id="session-001",
        source_project="test-project",
    )
    defaults.update(overrides)
    return Tip(**defaults)


class TestTipStore:
    def test_init_creates_tables(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        assert tmp_db.exists()

    def test_add_and_get_tip(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip()
        store.add_tip(tip)
        retrieved = store.get_tip(tip.id)
        assert retrieved is not None
        assert retrieved.content == tip.content
        assert retrieved.category == tip.category

    def test_get_nonexistent_returns_none(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        assert store.get_tip("nonexistent") is None

    def test_list_tips(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.add_tip(_make_tip(content="Tip 1"))
        store.add_tip(_make_tip(content="Tip 2"))
        tips = store.list_tips()
        assert len(tips) == 2

    def test_list_tips_filter_by_category(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.add_tip(_make_tip(category="strategy"))
        store.add_tip(_make_tip(category="recovery"))
        tips = store.list_tips(category="strategy")
        assert len(tips) == 1
        assert tips[0].category == "strategy"

    def test_add_tip_with_embedding(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip()
        embedding = [0.1, 0.2, 0.3]
        store.add_tip(tip, embedding=embedding, embedding_provider="voyage")
        raw = store.get_tip_with_embedding(tip.id)
        assert raw is not None
        assert raw["embedding_provider"] == "voyage"
        assert len(raw["embedding"]) == 3

    def test_get_all_embeddings(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.add_tip(_make_tip(content="Tip 1"), embedding=[0.1, 0.2], embedding_provider="voyage")
        store.add_tip(_make_tip(content="Tip 2"), embedding=[0.3, 0.4], embedding_provider="voyage")
        store.add_tip(_make_tip(content="Tip 3"))  # No embedding
        results = store.get_tips_with_embeddings(provider="voyage")
        assert len(results) == 2

    def test_mark_session_processed(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.mark_session_processed("session-001", "/path/to/file.jsonl", tip_count=3)
        assert store.is_session_processed("session-001")

    def test_unprocessed_session(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        assert not store.is_session_processed("session-999")

    def test_steps_stored_as_json(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip(steps=["Do A", "Do B", "Do C"])
        store.add_tip(tip)
        retrieved = store.get_tip(tip.id)
        assert retrieved is not None
        assert retrieved.steps == ["Do A", "Do B", "Do C"]
