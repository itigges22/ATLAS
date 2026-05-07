"""Extract embeddings from llama-server's /embedding endpoint."""

import base64
import json
import logging
import os
import struct
from typing import Dict, List, Optional, Tuple
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


def _post_embedding(text: str, layers: Optional[List[int]] = None, timeout: float = 120) -> dict:
    """POST to /embedding and return the parsed first item.

    Sends the optional PC-202 `layers` extension when provided. Returns the
    raw response dict so callers can read both `embedding` and (if layers
    were requested) `hidden_states` + the shape metadata.
    """
    url = f"{_get_embed_url()}/embedding"
    body: Dict = {"content": text}
    if layers:
        body["layers"] = layers
    payload = json.dumps(body).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected /embedding response shape: {type(data).__name__}")
    return data[0]


def extract_embedding(text: str) -> List[float]:
    """Extract an embedding vector from llama-server.

    Handles both pooled responses (flat list) from self-embedding endpoints
    and per-token responses (nested list) that need mean pooling.

    Returns:
        List of floats with model-native dimensionality.
    """
    item = _post_embedding(text)
    raw = item["embedding"]

    # Pooled: flat list of floats (e.g. llama-server self-embeddings)
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


def extract_per_token(text: str) -> Tuple[List[List[float]], int]:
    """Extract per-token last-layer hidden states from /embedding.

    Used by PC-207 lens-as-PRM to score each generation step instead of only
    pooled completed text. Works against vanilla llama-server (no PC-202 patch
    required) because Qwen3.5 returns per-token by default; if the server is
    configured with pooling != none we still detect the pooled-flat shape and
    raise rather than silently degrade.

    Returns:
        (per_token_vectors, hidden_dim) — outer list is one entry per input
        token, inner list is the hidden_dim float vector at the last layer.
    """
    item = _post_embedding(text)
    raw = item["embedding"]
    if not isinstance(raw[0], list):
        raise ValueError(
            "extract_per_token needs per-token embeddings; "
            "llama-server appears to be pooling. Start it with --pooling none "
            "(Qwen3.5 default) or use a model whose default pooling is none."
        )
    return raw, len(raw[0])


def extract_per_layer_per_token(text: str, layers: List[int]) -> Tuple[Dict[int, List[List[float]]], int, int]:
    """Extract per-token residual hidden states at the requested layers.

    Uses the PC-202 `/embedding` extension (`layers: [int]`). The server
    must have been built with `inference/patches/expose-hidden-states.patch`
    applied; on an unpatched server the `layers` field is silently ignored
    and only the standard `embedding` field comes back, which we detect
    and raise on.

    Returns:
        (per_layer_dict, n_tokens, hidden_dim). Each layer's value is a
        list of per-token vectors (length n_tokens, each of len hidden_dim).
    """
    if not layers:
        raise ValueError("layers must be a non-empty list of layer indices")
    item = _post_embedding(text, layers=layers)
    if "hidden_states" not in item:
        raise RuntimeError(
            "/embedding response missing `hidden_states`; "
            "llama-server is likely missing the PC-202 hidden-states patch. "
            "Rebuild atlas-llama-server with inference/Dockerfile.v31."
        )
    n_tokens = int(item["hidden_states_n_tokens"])
    hidden_dim = int(item["hidden_states_dim"])
    out: Dict[int, List[List[float]]] = {}
    for layer_str, b64 in item["hidden_states"].items():
        raw = base64.b64decode(b64)
        n_floats = len(raw) // 4
        if n_floats != n_tokens * hidden_dim:
            raise ValueError(
                f"layer {layer_str}: decoded {n_floats} floats, "
                f"expected {n_tokens}*{hidden_dim}={n_tokens*hidden_dim}"
            )
        flat = struct.unpack(f"<{n_floats}f", raw)
        # reshape [n_tokens, hidden_dim]
        rows: List[List[float]] = [
            list(flat[i * hidden_dim : (i + 1) * hidden_dim]) for i in range(n_tokens)
        ]
        out[int(layer_str)] = rows
    return out, n_tokens, hidden_dim
