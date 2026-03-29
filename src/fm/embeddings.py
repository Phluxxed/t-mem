from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import requests
import urllib3


@dataclass
class EmbeddingResult:
    vector: list[float]
    provider: str  # "voyage" | "huggingface"


_ssl_verify_cache: bool | None = None


def _ssl_verify() -> bool:
    """Return SSL verification setting for requests. Result is cached."""
    global _ssl_verify_cache
    if _ssl_verify_cache is not None:
        return _ssl_verify_cache

    # Allow explicit override
    if os.environ.get("FM_SSL_VERIFY", "").lower() in ("0", "false", "no"):
        _ssl_verify_cache = False
        return False
    # Auto-detect: try a quick TLS handshake with default verification
    try:
        requests.head("https://api.voyageai.com", timeout=2)
        _ssl_verify_cache = True
    except requests.exceptions.SSLError:
        _ssl_verify_cache = False
    except requests.RequestException:
        _ssl_verify_cache = True  # Non-SSL error, verification itself is fine
    return _ssl_verify_cache


def get_available_provider() -> str | None:
    """Return the best available embedding provider, or None."""
    verify = _ssl_verify()
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    try:
        resp = requests.head(
            "https://router.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            timeout=2,
            verify=verify,
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
    """Embed text using Voyage AI API directly via requests."""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None
    try:
        verify = _ssl_verify()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=urllib3.exceptions.InsecureRequestWarning)
            resp = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                json={"input": [text], "model": "voyage-4-lite"},
                headers={"Authorization": f"Bearer {api_key}"},
                verify=verify,
                timeout=15,
            )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        import sys
        print(f"Voyage embed error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _embed_huggingface(text: str) -> list[float] | None:
    """Embed text using HuggingFace Inference API (free tier)."""
    api_url = "https://router.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    verify = _ssl_verify()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=urllib3.exceptions.InsecureRequestWarning)
            resp = requests.post(
                api_url,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=10,
                verify=verify,
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
