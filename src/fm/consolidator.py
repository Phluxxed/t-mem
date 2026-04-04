from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from fm.embeddings import embed_texts_batch, get_available_provider
from fm.llm import call_claude
from fm.models import Tip
from fm.prompts.consolidate import build_consolidation_prompt
from fm.store import TipStore

_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# --- Union-Find ---

class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self._parent[self.find(a)] = self.find(b)

    def clusters(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return [g for g in groups.values() if len(g) >= 2]


# --- Similarity ---

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Cluster finding ---

@dataclass
class Cluster:
    tips: list[Tip]
    max_similarity: float


def find_clusters(store: TipStore, threshold: float = 0.88) -> list[Cluster]:
    """Find groups of tips with pairwise cosine similarity >= threshold."""
    provider = get_available_provider()
    if not provider:
        print("Warning: no embedding provider available, cannot cluster tips.", file=sys.stderr)
        return []

    rows = store.get_tips_with_embeddings(provider)
    if len(rows) < 2:
        return []

    tip_map = {row["id"]: store.get_tip(row["id"]) for row in rows}
    embeddings = {row["id"]: row["embedding"] for row in rows}
    ids = list(embeddings.keys())

    uf = _UnionFind(ids)
    max_sim: dict[tuple[str, str], float] = {}

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = _cosine_similarity(embeddings[ids[i]], embeddings[ids[j]])
            if sim >= threshold:
                uf.union(ids[i], ids[j])
                key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                max_sim[key] = max(max_sim.get(key, 0.0), sim)

    clusters = []
    for group in uf.clusters():
        tips = [tip_map[tip_id] for tip_id in group if tip_map.get(tip_id)]
        if len(tips) < 2:
            continue
        # max similarity within this cluster
        sims = [
            max_sim.get((min(a, b), max(a, b)), 0.0)
            for i, a in enumerate(group)
            for b in group[i + 1:]
        ]
        clusters.append(Cluster(tips=tips, max_similarity=max(sims) if sims else 0.0))

    return sorted(clusters, key=lambda c: c.max_similarity, reverse=True)


# --- LLM merge decision ---

@dataclass
class MergeResult:
    action: str  # "merge" | "keep"
    reasoning: str
    canonical_tip: Tip | None = None  # set when action == "merge"


def _highest_priority(tips: list[Tip]) -> str:
    return min((t.priority for t in tips), key=lambda p: _PRIORITY_RANK.get(p, 99))


def _parse_merge_response(raw: str, source_tips: list[Tip]) -> MergeResult | None:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    action = data.get("action")
    reasoning = data.get("reasoning", "")

    if action == "keep":
        return MergeResult(action="keep", reasoning=reasoning)

    if action == "merge":
        try:
            canonical = Tip(
                category=data["category"],
                content=data["content"],
                purpose=data.get("purpose", ""),
                steps=data.get("steps", []),
                trigger=data.get("trigger", ""),
                negative_example=data.get("negative_example"),
                priority=_highest_priority(source_tips),
                source_session_id=",".join({t.source_session_id for t in source_tips}),
                source_project=source_tips[0].source_project,
            )
        except (KeyError, ValueError) as e:
            print(f"consolidator: malformed merge response: {e}", file=sys.stderr)
            return None
        return MergeResult(action="merge", reasoning=reasoning, canonical_tip=canonical)

    return None


def decide_merge(cluster: Cluster, *, model: str = "sonnet") -> MergeResult | None:
    prompt = build_consolidation_prompt(cluster.tips)
    raw = call_claude(prompt, model=model)
    if raw is None:
        return None
    return _parse_merge_response(raw, cluster.tips)


# --- Apply merge ---

def apply_merge(result: MergeResult, cluster: Cluster, store: TipStore) -> None:
    """Persist the canonical tip and soft-delete the originals."""
    assert result.action == "merge" and result.canonical_tip is not None

    provider = get_available_provider()
    embedding_key = store.get_embedding_key(result.canonical_tip)
    emb_results = embed_texts_batch([embedding_key], provider=provider)
    emb = emb_results[0] if emb_results else None

    if emb:
        store.add_tip(result.canonical_tip, embedding=emb.vector, embedding_provider=emb.provider)
    else:
        store.add_tip(result.canonical_tip)
        print(f"Warning: could not embed canonical tip {result.canonical_tip.id[:8]}", file=sys.stderr)

    store.mark_merged([t.id for t in cluster.tips], result.canonical_tip.id)
