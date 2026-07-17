"""Embedding-contract enforcement in extract_embedding().

Regression for the 2026-07-15 bench incident: a rebuilt embed server
returned per-token unnormalized embeddings, extract_embedding() silently
mean-pooled them (‖v‖≈60 vs the trained ~1), and C(x) served ~600
against a calibrated ~20-30 with every health check green. The contract
makes the wrong convention a hard error instead of a silent one.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometric_lens import embedding_extractor as ee
from geometric_lens.identity import (
    validate_embedding_contract,
    validate_model_identity,
)


@pytest.fixture(autouse=True)
def _clear_contract():
    """Every test starts with no contract installed and restores after."""
    ee.set_embedding_contract(None)
    yield
    ee.set_embedding_contract(None)


def _flat(dim=8, norm=1.0):
    """A flat embedding with a chosen L2 norm."""
    import math
    base = [1.0] * dim
    scale = norm / math.sqrt(dim)
    return [v * scale for v in base]


def _install_response(monkeypatch, embedding):
    monkeypatch.setattr(ee, "_post_embedding",
                        lambda *a, **kw: {"embedding": embedding})


# --- contract validation ---------------------------------------------------

def test_validate_embedding_contract_defaults_and_bounds():
    c = validate_embedding_contract({})
    assert c == {"pooling": "mean", "response_shape": "flat",
                 "normalized": False, "norm_tolerance": 0.05}
    for bad in ({"pooling": "sum"}, {"response_shape": "matrix"},
                {"normalized": "yes"}, {"norm_tolerance": 0},
                {"norm_tolerance": 1.5}):
        with pytest.raises(ValueError):
            validate_embedding_contract(bad)


def test_model_identity_carries_optional_contract():
    ident = validate_model_identity({
        "model": "m", "embedding_dim": 8,
        "embedding_contract": {"normalized": True, "response_shape": "flat"},
    })
    assert ident["embedding_contract"]["normalized"] is True
    # Absent contract stays absent (backward compatible).
    assert "embedding_contract" not in validate_model_identity(
        {"model": "m", "embedding_dim": 8})


# --- enforcement -----------------------------------------------------------

def test_flat_normalized_response_passes_contract(monkeypatch):
    ee.set_embedding_contract({"pooling": "mean", "response_shape": "flat",
                               "normalized": True, "norm_tolerance": 0.05})
    _install_response(monkeypatch, _flat(norm=1.0))
    out = ee.extract_embedding("x")
    assert len(out) == 8


def test_per_token_response_under_flat_contract_raises(monkeypatch):
    """The exact incident shape: server returns per-token, artifacts
    expect flat. Must raise, not silently pool."""
    ee.set_embedding_contract({"pooling": "mean", "response_shape": "flat",
                               "normalized": True, "norm_tolerance": 0.05})
    _install_response(monkeypatch, [_flat(norm=7.0), _flat(norm=7.0)])
    with pytest.raises(ee.EmbeddingContractError) as exc:
        ee.extract_embedding("x")
    assert "per_token" in str(exc.value)


def test_unnormalized_flat_response_under_normalized_contract_raises(monkeypatch):
    ee.set_embedding_contract({"pooling": "mean", "response_shape": "flat",
                               "normalized": True, "norm_tolerance": 0.05})
    _install_response(monkeypatch, _flat(norm=60.0))
    with pytest.raises(ee.EmbeddingContractError) as exc:
        ee.extract_embedding("x")
    assert "60" in str(exc.value) or "norm" in str(exc.value).lower()


def test_no_contract_accepts_both_shapes(monkeypatch):
    """Legacy artifacts (no contract) keep the permissive behavior."""
    _install_response(monkeypatch, _flat(norm=50.0))
    assert len(ee.extract_embedding("x")) == 8
    _install_response(monkeypatch, [[1.0, 2.0], [3.0, 4.0]])
    assert ee.extract_embedding("x") == [2.0, 3.0]  # mean-pooled


def test_single_row_nested_is_pooled_not_per_token(monkeypatch):
    """The pinned llama-server encodes pooled embeddings as [[...]] (one
    nested row) under --pooling mean. That must satisfy a flat contract
    and unwrap to the vector — not raise as per-token."""
    ee.set_embedding_contract({"pooling": "mean", "response_shape": "flat",
                               "normalized": True, "norm_tolerance": 0.05})
    _install_response(monkeypatch, [_flat(norm=1.0)])
    out = ee.extract_embedding("x")
    assert len(out) == 8 and not isinstance(out[0], list)


def test_normalized_contract_requests_embd_normalize(monkeypatch):
    """Under a normalized contract the request must carry embd_normalize=2
    (server defaults vary across llama.cpp revisions)."""
    seen = {}

    def _spy(text, layers=None, timeout=120, embd_normalize=None):
        seen["embd_normalize"] = embd_normalize
        return {"embedding": _flat(norm=1.0)}

    monkeypatch.setattr(ee, "_post_embedding", _spy)
    ee.set_embedding_contract({"pooling": "mean", "response_shape": "flat",
                               "normalized": True, "norm_tolerance": 0.05})
    ee.extract_embedding("x")
    assert seen["embd_normalize"] == 2

    ee.set_embedding_contract(None)
    ee.extract_embedding("x")
    assert seen["embd_normalize"] is None


def test_extract_per_token_rejects_pooled_encodings(monkeypatch):
    _install_response(monkeypatch, _flat(norm=1.0))
    with pytest.raises(ValueError):
        ee.extract_per_token("x")
    _install_response(monkeypatch, [_flat(norm=1.0)])  # single nested row
    with pytest.raises(ValueError):
        ee.extract_per_token("x")
    _install_response(monkeypatch, [[1.0, 2.0], [3.0, 4.0]])
    vecs, dim = ee.extract_per_token("x")
    assert len(vecs) == 2 and dim == 2


def test_observe_convention_reports_flat_normalized(monkeypatch):
    _install_response(monkeypatch, _flat(norm=1.0))
    c = ee.observe_embedding_convention("x")
    assert c["response_shape"] == "flat" and c["normalized"] is True

    _install_response(monkeypatch, [_flat(norm=5.0), _flat(norm=5.0)])
    c = ee.observe_embedding_convention("x")
    assert c["response_shape"] == "per_token" and c["pooling"] == "none"
