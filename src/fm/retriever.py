from __future__ import annotations

import numpy as np

from fm.embeddings import embed_text
from fm.models import Tip
from fm.store import TipStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve_tips(
    query: str,
    store: TipStore,
    *,
    threshold: float = 0.6,
    top_k: int = 5,
    provider: str | None = None,
    session_id: str | None = None,
) -> list[Tip]:
    """Retrieve relevant tips for a query using cosine similarity."""
    if provider is None:
        from fm.embeddings import get_available_provider
        provider = get_available_provider()

    query_result = embed_text(query, provider=provider)
    if query_result is None:
        return []

    stored = store.get_tips_with_embeddings(provider=query_result.provider)
    if not stored:
        return []

    already_injected = store.get_injected_tip_ids(session_id) if session_id else set()

    scored: list[tuple[float, dict]] = []
    for tip_row in stored:
        if tip_row["id"] in already_injected:
            continue
        score = _cosine_similarity(query_result.vector, tip_row["embedding"])
        if score >= threshold:
            scored.append((score, tip_row))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_k]

    tips = []
    for score, row in scored:
        store.log_retrieval(row["id"], query, score, session_id=session_id)
        tips.append(store.get_tip(row["id"]))
    return [t for t in tips if t is not None]


def format_tips(tips: list[Tip]) -> str:
    """Format tips for prompt injection."""
    if not tips:
        return ""

    parts = []
    for tip in tips:
        category_label = f"{tip.category.title()} Tip"
        priority_label = tip.priority.upper()
        date = tip.created_at[:10] if tip.created_at else "unknown"

        section = f"[PRIORITY: {priority_label}] {category_label}:\n"
        section += f"{tip.content}\n\n"
        section += f"Apply when: {tip.trigger}\n"

        if tip.steps:
            section += "Steps:\n"
            for i, step in enumerate(tip.steps, 1):
                section += f"{i}. {step}\n"

        if tip.negative_example:
            section += f"\nAvoid: {tip.negative_example}\n"

        section += f"\nSource: session {tip.source_session_id} ({date})"
        parts.append(section)

    return "\n---\n".join(parts)
