from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass
class EmbeddingResult:
    vector: list[float]
    provider: str  # "voyage" | "huggingface"


def get_available_provider() -> str | None:
    """Return the best available embedding provider, or None."""
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    try:
        resp = requests.head(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            timeout=2,
        )
        if resp.status_code < 500:
            return "huggingface"
    except requests.RequestException:
        pass
    return None


def embed_text(text: str, *, provider: str | None = None) -> EmbeddingResult | None:
    """Embed a text string, falling back through providers on failure."""
    if provider is None:
        return None

    # Build the fallback chain starting from the requested provider
    chain = []
    if provider == "voyage":
        chain = [("voyage", _embed_voyage), ("huggingface", _embed_huggingface)]
    elif provider == "huggingface":
        chain = [("huggingface", _embed_huggingface)]
    else:
        return None

    for name, fn in chain:
        vector = fn(text)
        if vector is not None:
            return EmbeddingResult(vector=vector, provider=name)

    return None


def _embed_voyage(text: str) -> list[float] | None:
    """Embed text using Voyage AI API."""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None
    try:
        import voyageai

        client = voyageai.Client(api_key=api_key)
        result = client.embed([text], model="voyage-3-lite")
        return result.embeddings[0]
    except Exception:
        return None


def _embed_huggingface(text: str) -> list[float] | None:
    """Embed text using HuggingFace Inference API (free tier)."""
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    try:
        resp = requests.post(
            api_url,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=10,
        )
        resp.raise_for_status()
        vector = resp.json()
        if isinstance(vector, list) and isinstance(vector[0], float):
            return vector
        if isinstance(vector, list) and isinstance(vector[0], list):
            import numpy as np

            return list(np.mean(vector, axis=0).astype(float))
        return None
    except Exception:
        return None
