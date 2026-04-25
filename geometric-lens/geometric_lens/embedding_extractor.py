"""Extract embeddings from vLLM's /v1/embeddings endpoint (OpenAI-compatible)."""

import json
import logging
import os
from typing import List
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _get_embed_url() -> str:
    """Return the base URL for the embedding server.

    Resolves in this order:
      LLAMA_EMBED_URL  → dedicated vLLM embed instance (port 8001 by convention)
      LLAMA_URL        → legacy single-server fallback
      default          → http://vllm-embed:8001 (matches docker-compose service name)
    """
    return os.environ.get(
        "LLAMA_EMBED_URL",
        os.environ.get("LLAMA_URL", "http://vllm-embed:8001"),
    )


def _get_embed_model() -> str:
    """Model name to send in the vLLM request. Must match --served-model-name
    on the embed instance (default: qwen3.5-9b-embed)."""
    return os.environ.get("LLAMA_EMBED_MODEL", "qwen3.5-9b-embed")


def extract_embedding(text: str) -> List[float]:
    """Extract an embedding vector from a vLLM embed instance.

    vLLM's /v1/embeddings is OpenAI-compatible and returns pooled sentence
    embeddings. The embed instance must be started with `--runner pooling
    --convert embed` (the current API; the deprecated `--task embed` was
    removed in vLLM 0.17+). For Qwen3.5-9B that yields a 4096-dim float
    vector matching the hidden state dimension.

    Returns:
        List of floats (4096-dim for Qwen3.5-9B).
    """
    url = f"{_get_embed_url()}/v1/embeddings"
    payload = json.dumps({
        "model": _get_embed_model(),
        "input": text,
    }).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    # OpenAI format: {"data": [{"embedding": [...], "index": 0, "object": "embedding"}], ...}
    raw = data["data"][0]["embedding"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("Empty embedding returned from vLLM")
    return raw


def extract_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Extract embeddings for multiple texts in a single batched request.

    vLLM's embeddings endpoint accepts a list of inputs and returns one
    embedding per input — much faster than one HTTP round-trip per text.
    """
    if not texts:
        return []

    url = f"{_get_embed_url()}/v1/embeddings"
    payload = json.dumps({
        "model": _get_embed_model(),
        "input": texts,
    }).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    items = data["data"]
    # vLLM returns items with explicit `index` field — sort to be safe.
    items.sort(key=lambda x: x.get("index", 0))
    return [it["embedding"] for it in items]
