from __future__ import annotations

import os
import time
import threading
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
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    if os.environ.get("HF_TOKEN"):
        return "huggingface"
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


def embed_texts_batch(
    texts: list[str], *, provider: str | None = None
) -> list[EmbeddingResult | None]:
    """Embed multiple texts in a single API call where possible."""
    if provider is None or not texts:
        return [None] * len(texts)

    if provider == "voyage":
        vectors = _embed_voyage_batch(texts)
        if vectors:
            return [
                EmbeddingResult(vector=v, provider="voyage") if v else None
                for v in vectors
            ]
    # Fall back to individual calls
    return [embed_text(t, provider=provider) for t in texts]


# Thread-safe rate limiter for Voyage API
_voyage_lock = threading.Lock()
_last_voyage_call: float = 0.0
_MIN_INTERVAL = 0.5  # seconds between calls
_MAX_RETRIES = 6
_BASE_BACKOFF = 2.0  # seconds, doubles each retry


def _voyage_post(payload: dict, *, timeout: int) -> requests.Response:
    """POST to Voyage embeddings API with rate limiting and exponential backoff retry."""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not set")

    verify = _ssl_verify()
    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://api.voyageai.com/v1/embeddings"

    resp: requests.Response | None = None
    for attempt in range(_MAX_RETRIES):
        with _voyage_lock:
            global _last_voyage_call
            elapsed = time.monotonic() - _last_voyage_call
            if elapsed < _MIN_INTERVAL:
                time.sleep(_MIN_INTERVAL - elapsed)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=urllib3.exceptions.InsecureRequestWarning)
                resp = requests.post(url, json=payload, headers=headers, verify=verify, timeout=timeout)
            _last_voyage_call = time.monotonic()

        if resp.status_code != 429:
            return resp

        # 429: back off and retry
        retry_after = float(resp.headers.get("Retry-After", _BASE_BACKOFF * (2 ** attempt)))
        if attempt < _MAX_RETRIES - 1:
            time.sleep(retry_after)

    # All retries exhausted — return last response so caller can raise_for_status
    if resp is None:
        raise RuntimeError("No attempts made (MAX_RETRIES must be > 0)")
    return resp


def _embed_voyage_batch(texts: list[str]) -> list[list[float] | None] | None:
    """Embed multiple texts in a single Voyage API call."""
    import sys

    if not os.environ.get("VOYAGE_API_KEY"):
        return None

    try:
        resp = _voyage_post({"input": texts, "model": "voyage-4-lite"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]
    except Exception as e:
        print(f"Voyage batch embed error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _embed_voyage(text: str) -> list[float] | None:
    """Embed text using Voyage AI API directly via requests."""
    import sys

    if not os.environ.get("VOYAGE_API_KEY"):
        return None

    try:
        resp = _voyage_post({"input": [text], "model": "voyage-4-lite"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Voyage embed error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _embed_huggingface(text: str) -> list[float] | None:
    """Embed text using HuggingFace Inference API (requires HF_TOKEN)."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None
    api_url = "https://router.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    verify = _ssl_verify()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=urllib3.exceptions.InsecureRequestWarning)
            resp = requests.post(
                api_url,
                json={"inputs": text, "options": {"wait_for_model": True}},
                headers={"Authorization": f"Bearer {hf_token}"},
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
    except Exception as e:
        import sys
        print(f"HuggingFace embed error: {type(e).__name__}: {e}", file=sys.stderr)
        return None
