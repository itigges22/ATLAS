"""Extract embeddings from llama-server's /embedding endpoint."""

import base64
import json
import logging
import math
import os
import struct
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class EmbeddingContractError(RuntimeError):
    """The embed server's response violates the loaded artifact's
    embedding contract (wrong response shape or vector norm). Raised
    instead of silently adapting — the two conventions produce vectors
    from different distributions and only one matches the trained
    weights."""


# Contract of the currently loaded artifacts, installed by
# service._verify_model_identity() at weight load. None = no contract
# declared (legacy artifacts): both response shapes are accepted with a
# warning, matching pre-contract behavior.
_EMBEDDING_CONTRACT: Optional[dict] = None

# One-time-warning latch for the legacy per-token path. A mutable holder
# rather than a rebound global so callers reset it in place.
_legacy_shape_warn = {"warned": False}


def set_embedding_contract(contract: Optional[dict]) -> None:
    """Install (or clear) the embedding contract enforced by
    extract_embedding(). Called on weight load/reload."""
    global _EMBEDDING_CONTRACT
    _EMBEDDING_CONTRACT = dict(contract) if contract else None
    _legacy_shape_warn["warned"] = False


def get_embedding_contract() -> Optional[dict]:
    return dict(_EMBEDDING_CONTRACT) if _EMBEDDING_CONTRACT else None


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _classify_response(raw) -> str:
    """Classify an /embedding response's shape.

    llama-server encodes pooled embeddings either as a flat list or as a
    single-row nested list ([[...]] — observed on the pinned build with
    --pooling mean), so one nested row still means "pooled". Only a
    multi-row nested response is genuinely per-token. (A one-token input
    under --pooling none is indistinguishable from a pooled row; lens
    inputs are code snippets, never a single token.)
    """
    if bool(raw) and isinstance(raw[0], list) and len(raw) > 1:
        return "per_token"
    return "flat"


def _unwrap(raw) -> List[float]:
    """Return the pooled vector from either encoding (flat or [1][dim])."""
    if bool(raw) and isinstance(raw[0], list):
        return raw[0]
    return raw


def _get_embed_url() -> str:
    """Return the URL for the embedding server.

    Uses LLAMA_EMBED_URL if set, otherwise falls back to LLAMA_URL.
    """
    return os.environ.get(
        "LLAMA_EMBED_URL",
        os.environ.get("LLAMA_URL", "http://llama-server:8080"),
    )


def _post_embedding(text: str, layers: Optional[List[int]] = None,
                    timeout: float = 120,
                    embd_normalize: Optional[int] = None) -> dict:
    """POST to /embedding and return the parsed first item.

    Sends the optional PC-202 `layers` extension when provided, and the
    per-request `embd_normalize` mode when set (llama-server has no
    `--embd-normalize` server flag; normalization is a request field on
    the native endpoint). Returns the raw response dict so callers can
    read both `embedding` and (if layers were requested) `hidden_states`
    + the shape metadata.
    """
    url = f"{_get_embed_url()}/embedding"
    body: Dict = {"content": text}
    if layers:
        body["layers"] = layers
    if embd_normalize is not None:
        body["embd_normalize"] = embd_normalize
    payload = json.dumps(body).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected /embedding response shape: {type(data).__name__}")
    return data[0]


def extract_embedding(text: str) -> List[float]:
    """Extract an embedding vector from llama-server.

    When the loaded artifacts declare an embedding_contract
    (model_identity.json), the response is validated against it: a shape
    or norm mismatch raises EmbeddingContractError instead of silently
    adapting. Without a contract (legacy artifacts) both pooled (flat)
    and per-token (nested, mean-pooled here) responses are accepted, with
    a one-time warning on the per-token path.

    Returns:
        List of floats with model-native dimensionality.
    """
    contract = _EMBEDDING_CONTRACT
    # Under a normalized contract, request L2 normalization explicitly —
    # server defaults vary across llama.cpp revisions.
    item = _post_embedding(
        text,
        embd_normalize=2 if contract and contract.get("normalized") else None)
    raw = item["embedding"]
    got = _classify_response(raw)
    is_per_token = got == "per_token"

    if contract:
        expected = contract.get("response_shape", "flat")
        if got != expected:
            raise EmbeddingContractError(
                f"embed server returned a {got} /embedding response but the "
                f"loaded lens artifacts were trained on {expected}. "
                f"Scores computed from the wrong convention are invalid. "
                f"Start the embed server with `--pooling mean` (see "
                f"inference/entrypoint-v3.1.sh) or reload artifacts "
                f"trained for this convention."
            )

    if is_per_token:
        # Per-token: mean-pool across tokens
        per_token = raw
        n_tokens = len(per_token)

        if n_tokens == 0:
            raise ValueError("No token embeddings returned")

        if contract is None and not _legacy_shape_warn["warned"]:
            _legacy_shape_warn["warned"] = True
            logger.warning(
                "per-token /embedding response mean-pooled without a "
                "declared embedding_contract — if the loaded artifacts "
                "were trained on pooled+normalized embeddings, every "
                "score is off-distribution. Retrain or add "
                "embedding_contract to model_identity.json."
            )

        dim = len(per_token[0])

        pooled = [0.0] * dim
        for tok_emb in per_token:
            for i, v in enumerate(tok_emb):
                pooled[i] += v
        for i in range(dim):
            pooled[i] /= n_tokens
        vec = pooled
    else:
        vec = _unwrap(raw)

    if contract and contract.get("normalized"):
        norm = _l2_norm(vec)
        tol = float(contract.get("norm_tolerance", 0.05))
        if abs(norm - 1.0) > tol:
            raise EmbeddingContractError(
                f"embedding norm ‖v‖={norm:.3f} but the loaded lens "
                f"artifacts expect L2-normalized vectors (1±{tol:g}). "
                f"The embed server ignored the requested "
                f"`embd_normalize: 2` — scores from unnormalized "
                f"embeddings are invalid at this scale."
            )

    return vec


def observe_embedding_convention(
        text: str = "def add(a, b):\n    return a + b") -> dict:
    """Probe the embed server once and describe the convention it serves.

    Used at training time to record what the training embeddings were
    actually built from, so the resulting artifacts carry an accurate
    embedding_contract.
    """
    item = _post_embedding(text)
    raw = item["embedding"]
    if _classify_response(raw) == "per_token":
        return {"pooling": "none", "response_shape": "per_token",
                "normalized": False, "norm_tolerance": 0.05}
    norm = _l2_norm(_unwrap(raw))
    return {"pooling": "mean", "response_shape": "flat",
            "normalized": abs(norm - 1.0) <= 0.05,
            "norm_tolerance": 0.05}


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
    required) when the server uses `--pooling none`; if it is configured with
    another pooling mode we detect the pooled-flat shape and raise rather than
    silently degrade.

    Returns:
        (per_token_vectors, hidden_dim) — outer list is one entry per input
        token, inner list is the hidden_dim float vector at the last layer.
    """
    item = _post_embedding(text)
    raw = item["embedding"]
    # A flat response is pooled; so is a single nested row (the pinned
    # build's pooled encoding under --pooling mean). Genuine per-token
    # output for a code snippet always has multiple rows.
    if _classify_response(raw) != "per_token":
        raise ValueError(
            "extract_per_token needs per-token embeddings; "
            "llama-server appears to be pooling. Start it with --pooling none."
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
