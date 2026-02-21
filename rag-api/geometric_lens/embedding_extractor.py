"""Extract embeddings from llama-server's /embedding endpoint."""

import json
import logging
import os
from typing import List
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _get_embed_url() -> str:
    """Return the URL for the embedding server.

    Uses LLAMA_EMBED_URL if set, otherwise falls back to LLAMA_URL.
    """
    return os.environ.get(
        "LLAMA_EMBED_URL",
        os.environ.get("LLAMA_URL", "http://llama-service:8000"),
    )


def extract_embedding(text: str) -> List[float]:
    """Extract an embedding vector from llama-server.

    Handles both pooled responses (flat list) from models like nomic-embed
    and per-token responses (nested list) that need mean pooling.

    Returns:
        List of floats with model-native dimensionality.
    """
    url = f"{_get_embed_url()}/embedding"
    payload = json.dumps({"content": text}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    # Response: [{"index": 0, "embedding": <flat list or nested list>}]
    raw = data[0]["embedding"]

    # Pooled: flat list of floats (e.g. nomic-embed-text)
    if not isinstance(raw[0], list):
        return raw

    # Per-token: mean-pool across tokens
    per_token = raw
    n_tokens = len(per_token)

    if n_tokens == 0:
        raise ValueError("No token embeddings returned")

    dim = len(per_token[0])

    pooled = [0.0] * dim
    for tok_emb in per_token:
        for i, v in enumerate(tok_emb):
            pooled[i] += v
    for i in range(dim):
        pooled[i] /= n_tokens

    return pooled


def extract_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Extract embeddings for multiple texts sequentially."""
    results = []
    for text in texts:
        results.append(extract_embedding(text))
    return results
