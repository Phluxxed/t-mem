"""Tests for the tip consolidation pipeline."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fm.consolidator import (
    Cluster,
    MergeResult,
    _UnionFind,
    _cosine_similarity,
    apply_merge,
    decide_merge,
    find_clusters,
)
from fm.models import Tip
from fm.store import TipStore


def _make_tip(**overrides) -> Tip:
    defaults = dict(
        category="strategy",
        content="Always check file existence before reading",
        purpose="Prevents failures on missing files",
        steps=["Check path exists", "Then read"],
        trigger="When reading files",
        priority="high",
        source_session_id="session-001",
        source_project="test-project",
    )
    defaults.update(overrides)
    return Tip(**defaults)


class TestUnionFind:
    def test_initially_separate(self) -> None:
        uf = _UnionFind(["a", "b", "c"])
        assert uf.find("a") != uf.find("b")

    def test_union_joins(self) -> None:
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.find("a") != uf.find("c")

    def test_transitive(self) -> None:
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_clusters_only_multimember(self) -> None:
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        clusters = uf.clusters()
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a", "b"}


class TestCosineSimilarity:
    def test_identical(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestFindClusters:
    def test_no_clusters_when_dissimilar(self, tmp_path: Path) -> None:
        store = TipStore(tmp_path / "tips.db")
        t1 = _make_tip(content="Check file exists before reading")
        t2 = _make_tip(content="Always run tests after code changes")
        store.add_tip(t1, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")
        store.add_tip(t2, embedding=[0.0, 1.0, 0.0], embedding_provider="voyage")

        with patch("fm.consolidator.get_available_provider", return_value="voyage"):
            clusters = find_clusters(store, threshold=0.88)

        assert clusters == []

    def test_finds_similar_cluster(self, tmp_path: Path) -> None:
        store = TipStore(tmp_path / "tips.db")
        t1 = _make_tip(content="Check file exists before reading")
        t2 = _make_tip(content="Verify file path exists before attempting to read it")
        # nearly identical vectors
        store.add_tip(t1, embedding=[0.99, 0.14, 0.0], embedding_provider="voyage")
        store.add_tip(t2, embedding=[0.99, 0.13, 0.0], embedding_provider="voyage")

        with patch("fm.consolidator.get_available_provider", return_value="voyage"):
            clusters = find_clusters(store, threshold=0.88)

        assert len(clusters) == 1
        assert len(clusters[0].tips) == 2

    def test_transitive_cluster(self, tmp_path: Path) -> None:
        store = TipStore(tmp_path / "tips.db")
        t1 = _make_tip(content="tip A")
        t2 = _make_tip(content="tip B")
        t3 = _make_tip(content="tip C")
        # A~B and B~C but not A~C directly above threshold
        store.add_tip(t1, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")
        store.add_tip(t2, embedding=[0.95, 0.31, 0.0], embedding_provider="voyage")
        store.add_tip(t3, embedding=[0.90, 0.44, 0.0], embedding_provider="voyage")

        with patch("fm.consolidator.get_available_provider", return_value="voyage"):
            clusters = find_clusters(store, threshold=0.88)

        # all three should end up in one cluster via transitivity
        assert len(clusters) == 1
        assert len(clusters[0].tips) == 3

    def test_merged_tips_excluded(self, tmp_path: Path) -> None:
        store = TipStore(tmp_path / "tips.db")
        t1 = _make_tip(content="tip A")
        t2 = _make_tip(content="tip B")
        store.add_tip(t1, embedding=[0.99, 0.14, 0.0], embedding_provider="voyage")
        store.add_tip(t2, embedding=[0.99, 0.13, 0.0], embedding_provider="voyage")
        # soft-delete t2
        store.mark_merged([t2.id], t1.id)

        with patch("fm.consolidator.get_available_provider", return_value="voyage"):
            clusters = find_clusters(store, threshold=0.88)

        assert clusters == []


class TestDecideMerge:
    def _cluster(self) -> Cluster:
        t1 = _make_tip(content="Check file exists before reading")
        t2 = _make_tip(content="Verify path exists before reading file")
        return Cluster(tips=[t1, t2], max_similarity=0.95)

    def test_merge_decision(self) -> None:
        response = json.dumps({
            "action": "merge",
            "reasoning": "Both tips give identical advice about file existence checks",
            "category": "strategy",
            "priority": "high",
            "content": "Before reading a file, verify the path exists",
            "purpose": "Prevents failures on missing files",
            "trigger": "When reading files that may not exist",
            "steps": ["Check path exists", "Read file"],
            "negative_example": None,
        })
        with patch("fm.consolidator.call_claude", return_value=response):
            result = decide_merge(self._cluster())

        assert result is not None
        assert result.action == "merge"
        assert result.canonical_tip is not None
        assert result.canonical_tip.priority == "high"

    def test_keep_decision(self) -> None:
        response = json.dumps({
            "action": "keep",
            "reasoning": "These tips apply in different contexts",
        })
        with patch("fm.consolidator.call_claude", return_value=response):
            result = decide_merge(self._cluster())

        assert result is not None
        assert result.action == "keep"
        assert result.canonical_tip is None

    def test_llm_failure_returns_none(self) -> None:
        with patch("fm.consolidator.call_claude", return_value=None):
            result = decide_merge(self._cluster())
        assert result is None

    def test_priority_takes_highest(self) -> None:
        t1 = _make_tip(priority="medium")
        t2 = _make_tip(priority="critical")
        cluster = Cluster(tips=[t1, t2], max_similarity=0.95)
        response = json.dumps({
            "action": "merge",
            "reasoning": "same advice",
            "category": "strategy",
            "priority": "medium",  # LLM says medium, but highest wins
            "content": "Canonical tip",
            "purpose": "Purpose",
            "trigger": "Trigger",
            "steps": [],
            "negative_example": None,
        })
        with patch("fm.consolidator.call_claude", return_value=response):
            result = decide_merge(cluster)

        assert result is not None
        assert result.canonical_tip is not None
        assert result.canonical_tip.priority == "critical"


class TestApplyMerge:
    def test_stores_canonical_and_marks_merged(self, tmp_path: Path) -> None:
        store = TipStore(tmp_path / "tips.db")
        t1 = _make_tip(content="tip A")
        t2 = _make_tip(content="tip B")
        store.add_tip(t1, embedding=[0.9, 0.1], embedding_provider="voyage")
        store.add_tip(t2, embedding=[0.9, 0.1], embedding_provider="voyage")

        canonical = _make_tip(content="canonical tip")
        cluster = Cluster(tips=[t1, t2], max_similarity=0.95)
        result = MergeResult(action="merge", reasoning="same", canonical_tip=canonical)

        with patch("fm.consolidator.get_available_provider", return_value="voyage"):
            with patch("fm.consolidator.embed_texts_batch", return_value=[None]):
                apply_merge(result, cluster, store)

        active = store.list_tips()
        assert any(t.id == canonical.id for t in active)
        assert not any(t.id == t1.id for t in active)
        assert not any(t.id == t2.id for t in active)

        # originals still in DB with merged status
        all_tips = store.list_tips(include_merged=True)
        merged = [t for t in all_tips if t.id in {t1.id, t2.id}]
        assert len(merged) == 2
