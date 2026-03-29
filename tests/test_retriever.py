from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fm.embeddings import EmbeddingResult
from fm.models import Tip
from fm.retriever import format_tips, retrieve_tips
from fm.store import TipStore


def _make_tip(content: str, trigger: str = "general", **kw) -> Tip:
    defaults = dict(
        category="strategy",
        content=content,
        purpose="Test purpose",
        steps=["Step 1"],
        trigger=trigger,
        priority="high",
        source_session_id="s1",
        source_project="proj",
    )
    defaults.update(kw)
    return Tip(**defaults)


class TestRetrieveTips:
    def test_returns_matching_tips(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip1 = _make_tip("Check prerequisites before deployment")
        tip2 = _make_tip("Use bulk operations for batch deletes")
        store.add_tip(tip1, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")
        store.add_tip(tip2, embedding=[0.0, 1.0, 0.0], embedding_provider="voyage")

        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[0.95, 0.05, 0.0], provider="voyage"
            )
            results = retrieve_tips("deploy the app", store)

        assert len(results) >= 1
        assert results[0].content == "Check prerequisites before deployment"

    def test_filters_below_threshold(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip("Unrelated tip")
        store.add_tip(tip, embedding=[0.0, 0.0, 1.0], embedding_provider="voyage")

        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[1.0, 0.0, 0.0], provider="voyage"
            )
            results = retrieve_tips("deploy the app", store, threshold=0.6)

        assert len(results) == 0

    def test_returns_top_k(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        for i in range(10):
            tip = _make_tip(f"Tip number {i}")
            vec = [0.9, 0.1 * (i / 10), 0.0]
            store.add_tip(tip, embedding=vec, embedding_provider="voyage")

        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[1.0, 0.0, 0.0], provider="voyage"
            )
            results = retrieve_tips("query", store, top_k=5)

        assert len(results) <= 5

    def test_returns_empty_when_no_embeddings(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = None
            results = retrieve_tips("query", store)
        assert results == []


class TestFormatTips:
    def test_formats_single_tip(self) -> None:
        tip = _make_tip(
            content="Always check prerequisites",
            trigger="When deploying",
            priority="high",
            category="strategy",
        )
        tip.source_session_id = "abc123"
        tip.created_at = "2026-03-29T01:00:00+00:00"
        output = format_tips([tip])
        assert "[PRIORITY: HIGH]" in output
        assert "Strategy Tip" in output
        assert "Always check prerequisites" in output
        assert "When deploying" in output
        assert "abc123" in output

    def test_formats_empty_list(self) -> None:
        output = format_tips([])
        assert output == ""
