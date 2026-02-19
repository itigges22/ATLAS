"""Extract embeddings from llama-server's /embedding endpoint."""

import json
import logging
import os
from typing import List
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 5120  # Qwen3-14B verified dimension


def _get_llama_url() -> str:
    return os.environ.get("LLAMA_URL", "http://llama-service:8000")


def extract_embedding(text: str) -> List[float]:
    """Extract a mean-pooled embedding vector from llama-server.

    Uses the /embedding endpoint which returns per-token embeddings
    (pooling=none), then applies mean pooling to get a single vector.

    Returns:
        List of floats with length EMBEDDING_DIM (5120).
    """
    url = f"{_get_llama_url()}/embedding"
    payload = json.dumps({"content": text}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    # Response: [{"index": 0, "embedding": [[tok1_dim0, ...], [tok2_dim0, ...], ...]}]
    per_token = data[0]["embedding"]
    n_tokens = len(per_token)

    if n_tokens == 0:
        raise ValueError("No token embeddings returned")

    dim = len(per_token[0])
    if dim != EMBEDDING_DIM:
        logger.warning(f"Expected dim {EMBEDDING_DIM}, got {dim}")

    # Mean pooling across tokens
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
