"""Geometric Lens service interface — main entry point for geometric-lens integration.

Provides:
- evaluate(embedding) -> energy scalar (C(x))
- evaluate_correctability(embedding) -> correctability scalar (G(x))
- is_enabled() -> bool
- get_geometric_energy(query) -> float in [0,1] for router signal
"""

import logging
import os
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded models (CPU only)
_cost_field = None
_metric_tensor = None  # PCAMetricTensor wrapper (legacy)
_gx_xgboost = None        # XGBoost G(x) classifier
_gx_pca_components = None  # PCA projection matrix, numpy (128, 4096)
_gx_pca_mean = None        # PCA mean vector, numpy (4096,)
_gx_top_dims = None        # Top contributing PCA dimensions
_models_loaded = False
_load_attempted = False


def is_enabled() -> bool:
    """Check if Geometric Lens is enabled (GEOMETRIC_LENS_ENABLED env var)."""
    return os.environ.get("GEOMETRIC_LENS_ENABLED", "false").lower() in ("true", "1", "yes")


def _ensure_models_loaded():
    """Lazy-load C(x) cost field and G(x) metric tensor on first use."""
    global _cost_field, _metric_tensor, _models_loaded, _load_attempted
    global _gx_xgboost, _gx_pca_components, _gx_pca_mean, _gx_top_dims

    if _models_loaded or _load_attempted:
        return _models_loaded

    _load_attempted = True

    try:
        import torch
        from geometric_lens.cost_field import CostField

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        cost_path = os.path.join(models_dir, "cost_field.pt")

        if not os.path.exists(cost_path):
            logger.warning(f"Geometric Lens model files not found in {models_dir}")
            return False

        sd = torch.load(cost_path, map_location="cpu", weights_only=True)
        dim = sd["net.0.weight"].shape[1]
        _cost_field = CostField(input_dim=dim)
        _cost_field.load_state_dict(sd)
        _cost_field.set_eval_mode()

        logger.info(f"Geometric Lens C(x) model loaded successfully (CPU, dim={dim})")

        # Load G(x) metric tensor (optional — service works without it)
        gx_path = os.path.join(models_dir, "metric_tensor.pt")
        if os.path.exists(gx_path):
            try:
                from geometric_lens.metric_tensor import load_metric_tensor
                _metric_tensor = load_metric_tensor(gx_path)
                if _metric_tensor is not None:
                    logger.info(f"Geometric Lens G(x) loaded (PCA {_metric_tensor.original_dim}→{_metric_tensor.pca_dim})")
                else:
                    logger.warning("G(x) load returned None — correctability unavailable")
            except Exception as e:
                logger.warning(f"G(x) load failed (non-fatal): {e}")
                _metric_tensor = None
        else:
            logger.info("No G(x) model found — correctability unavailable")

        # Load G(x) XGBoost model (preferred over metric tensor when available)
        if _metric_tensor is None:
            xgb_path = os.path.join(models_dir, "gx_xgboost.pkl")
            weights_path = os.path.join(models_dir, "gx_weights.json")
            if os.path.exists(xgb_path) and os.path.exists(weights_path):
                try:
                    import pickle
                    import json as json_mod
                    import numpy as np

                    with open(xgb_path, 'rb') as f:
                        _gx_xgboost = pickle.load(f)

                    with open(weights_path, 'r') as f:
                        weights = json_mod.load(f)

                    _gx_pca_components = np.array(weights['pca_components'], dtype=np.float32)
                    _gx_pca_mean = np.array(weights['pca_mean'], dtype=np.float32)
                    _gx_top_dims = weights.get('top_dims', [])

                    logger.info(
                        f"G(x) XGBoost loaded (AUC={weights.get('cv_auc_mean', 0):.4f}, "
                        f"PCA {weights.get('original_dim', '?')}→{weights.get('pca_dim', '?')})"
                    )
                except ImportError:
                    logger.warning("G(x) XGBoost model found but xgboost package not installed")
                    _gx_xgboost = None
                except Exception as e:
                    logger.warning(f"G(x) XGBoost load failed (non-fatal): {e}")
                    _gx_xgboost = None

        _models_loaded = True
        return True

    except Exception as e:
        logger.error(f"Failed to load Geometric Lens models: {e}")
        return False


def reload_weights(model_dir: str = None) -> dict:
    """Reload C(x) and G(x) weights from disk without restarting the process.

    Used after retraining to hot-swap model weights.
    """
    global _cost_field, _metric_tensor, _gx_xgboost, _gx_pca_components
    global _gx_pca_mean, _gx_top_dims, _models_loaded, _load_attempted

    _models_loaded = False
    _load_attempted = False
    _cost_field = None
    _metric_tensor = None
    _gx_xgboost = None
    _gx_pca_components = None
    _gx_pca_mean = None
    _gx_top_dims = None

    if model_dir:
        try:
            from geometric_lens.training import load_cost_field
            _cost_field = load_cost_field(model_dir)
            _models_loaded = True
            _load_attempted = True
            logger.info(f"Geometric Lens C(x) reloaded from {model_dir}")
            return {"status": "reloaded", "model_dir": model_dir}
        except Exception as e:
            logger.error(f"Failed to reload models from {model_dir}: {e}")
            _load_attempted = True
            return {"status": "error", "message": str(e)}
    else:
        success = _ensure_models_loaded()
        return {
            "status": "reloaded" if success else "error",
            "gx_loaded": _metric_tensor is not None,
        }


def get_geometric_energy(query: str) -> float:
    """Compute normalized geometric energy for a query.

    Extracts embedding from llama-server, evaluates through C(x),
    and returns a normalized energy in [0, 1].

    Used as the 4th signal in the Confidence Router.

    Returns 0.0 if lens is disabled or models aren't loaded.
    """
    if not is_enabled():
        return 0.0

    if not _ensure_models_loaded():
        return 0.0

    try:
        import torch
        from geometric_lens.embedding_extractor import extract_embedding

        start = time.monotonic()

        # Extract embedding
        emb = extract_embedding(query)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        # Evaluate C(x)
        with torch.no_grad():
            energy = _cost_field(x).item()

        elapsed_ms = (time.monotonic() - start) * 1000

        # Normalize energy to [0, 1] using sigmoid-like scaling
        # Fox 9B retrained: PASS ~13.2, FAIL ~24.9, midpoint ~19.0
        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 19.0) / 2.0))

        logger.debug(
            f"Geometric energy: raw={energy:.2f} normalized={normalized:.3f} "
            f"latency={elapsed_ms:.1f}ms"
        )

        return min(1.0, max(0.0, normalized))

    except Exception as e:
        logger.error(f"Geometric energy computation failed: {e}")
        return 0.0


def evaluate_energy(query: str) -> Tuple[float, float]:
    """Evaluate raw and normalized energy for a query.

    Returns (raw_energy, normalized_energy).
    Returns (0.0, 0.0) if lens is disabled or models aren't loaded.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return (0.0, 0.0)

    try:
        import torch
        from geometric_lens.embedding_extractor import extract_embedding

        emb = extract_embedding(query)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            energy = _cost_field(x).item()

        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 19.0) / 2.0))
        normalized = min(1.0, max(0.0, normalized))

        return (energy, normalized)

    except Exception as e:
        logger.error(f"Geometric lens evaluation failed: {e}")
        return (0.0, 0.0)


def get_model_info() -> dict:
    """Get info about loaded models for health/status endpoints."""
    if not _models_loaded:
        return {"loaded": False, "enabled": is_enabled()}

    cost_params = sum(p.numel() for p in _cost_field.parameters())

    info = {
        "loaded": True,
        "enabled": is_enabled(),
        "cost_field_params": cost_params,
        "device": "cpu",
        "gx_loaded": _metric_tensor is not None or _gx_xgboost is not None,
        "gx_type": "xgboost" if _gx_xgboost is not None else (
            "metric_tensor" if _metric_tensor is not None else "none"
        ),
    }

    if _gx_xgboost is not None:
        info["gx_pca_dim"] = _gx_pca_components.shape[0] if _gx_pca_components is not None else 0
        info["gx_top_dims"] = _gx_top_dims[:10] if _gx_top_dims else []
        info["total_params"] = cost_params
    elif _metric_tensor is not None:
        gx_params = sum(p.numel() for p in _metric_tensor.parameters())
        info["metric_tensor_params"] = gx_params
        info["gx_pca_dim"] = _metric_tensor.pca_dim
        info["total_params"] = cost_params + gx_params
    else:
        info["total_params"] = cost_params

    return info


def evaluate_correctability(query: str) -> Tuple[float, float, float]:
    """Compute correctability score using G(x) metric tensor.

    Returns (correctability, raw_energy, normalized_energy).
    Returns (0.0, 0.0, 0.0) if G(x) is not loaded.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return (0.0, 0.0, 0.0)

    if _metric_tensor is None:
        # G(x) not available — return energy only
        raw, norm = evaluate_energy(query)
        return (0.0, raw, norm)

    try:
        import torch
        from geometric_lens.embedding_extractor import extract_embedding
        from geometric_lens.correction import compute_correctability

        start = time.monotonic()

        emb = extract_embedding(query)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        # C(x) energy
        with torch.no_grad():
            energy = _cost_field(x).item()
        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 19.0) / 2.0))
        normalized = min(1.0, max(0.0, normalized))

        # G(x) correctability
        corr = compute_correctability(x, _cost_field, _metric_tensor)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"Correctability: {corr:.4f}, energy: raw={energy:.2f} "
            f"norm={normalized:.3f}, latency={elapsed_ms:.1f}ms"
        )

        return (corr, energy, normalized)

    except Exception as e:
        logger.error(f"Correctability computation failed: {e}")
        return (0.0, 0.0, 0.0)


def evaluate_gx(query: str) -> dict:
    """Score code quality using XGBoost G(x) classifier.

    Returns dict with gx_score (0-1 probability of PASS), verdict, and metadata.
    Falls back gracefully if XGBoost model is not available.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return {"gx_score": 0.5, "verdict": "unavailable", "gx_available": False}

    if _gx_xgboost is None:
        return {"gx_score": 0.5, "verdict": "unavailable", "gx_available": False}

    try:
        import numpy as np
        from geometric_lens.embedding_extractor import extract_embedding

        start = time.monotonic()

        emb = extract_embedding(query)
        emb_np = np.array(emb, dtype=np.float32).reshape(1, -1)

        # PCA transform
        x_pca = (emb_np - _gx_pca_mean) @ _gx_pca_components.T

        # XGBoost prediction
        proba = _gx_xgboost.predict_proba(x_pca)[0]
        gx_score = float(proba[1])  # probability of PASS class

        if gx_score >= 0.7:
            verdict = "likely_correct"
        elif gx_score >= 0.3:
            verdict = "uncertain"
        else:
            verdict = "likely_incorrect"

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(f"G(x) score: {gx_score:.4f} ({verdict}), latency={elapsed_ms:.1f}ms")

        return {
            "gx_score": gx_score,
            "verdict": verdict,
            "top_dims": _gx_top_dims[:10] if _gx_top_dims else [],
            "gx_available": True,
            "latency_ms": round(elapsed_ms, 1),
        }

    except Exception as e:
        logger.error(f"G(x) evaluation failed: {e}")
        return {"gx_score": 0.5, "verdict": "error", "gx_available": False, "error": str(e)}


def evaluate_combined(query: str) -> dict:
    """Combined C(x) + G(x) evaluation using a single embedding extraction.

    Returns dict with C(x) energy, G(x) quality score, and verdict.
    Most efficient way to get both scores — avoids duplicate embedding calls.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "gx_score": 0.5, "verdict": "unavailable",
            "enabled": False, "gx_available": False,
        }

    try:
        import torch
        import numpy as np
        from geometric_lens.embedding_extractor import extract_embedding

        start = time.monotonic()

        # Single embedding extraction (shared between C(x) and G(x))
        emb = extract_embedding(query)

        # C(x) evaluation
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            energy = _cost_field(x).item()
        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 19.0) / 2.0))
        normalized = min(1.0, max(0.0, normalized))

        # G(x) evaluation (if available)
        gx_score = 0.5
        verdict = "unavailable"
        gx_available = False

        if _gx_xgboost is not None:
            emb_np = np.array(emb, dtype=np.float32).reshape(1, -1)
            x_pca = (emb_np - _gx_pca_mean) @ _gx_pca_components.T
            proba = _gx_xgboost.predict_proba(x_pca)[0]
            gx_score = float(proba[1])
            gx_available = True

            if gx_score >= 0.7:
                verdict = "likely_correct"
            elif gx_score >= 0.3:
                verdict = "uncertain"
            else:
                verdict = "likely_incorrect"

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"Combined: C(x)={energy:.2f}({normalized:.3f}) G(x)={gx_score:.4f} "
            f"({verdict}) latency={elapsed_ms:.1f}ms"
        )

        return {
            "cx_energy": energy,
            "cx_normalized": normalized,
            "gx_score": gx_score,
            "verdict": verdict,
            "gx_available": gx_available,
            "enabled": True,
            "latency_ms": round(elapsed_ms, 1),
        }

    except Exception as e:
        logger.error(f"Combined evaluation failed: {e}")
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "gx_score": 0.5, "verdict": "error",
            "enabled": True, "gx_available": False, "error": str(e),
        }
