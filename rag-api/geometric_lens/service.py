"""Geometric Lens service interface — main entry point for rag-api integration.

Provides:
- evaluate(embedding) → energy scalar
- correct(embedding) → corrected embedding
- is_enabled() → bool
- get_geometric_energy(query) → float in [0,1] for router signal
"""

import logging
import os
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded models (CPU only)
_cost_field = None
_metric_tensor = None
_models_loaded = False
_load_attempted = False


def is_enabled() -> bool:
    """Check if Geometric Lens is enabled (GEOMETRIC_LENS_ENABLED env var)."""
    return os.environ.get("GEOMETRIC_LENS_ENABLED", "false").lower() in ("true", "1", "yes")


def _ensure_models_loaded():
    """Lazy-load C(x) and G(x) models on first use."""
    global _cost_field, _metric_tensor, _models_loaded, _load_attempted

    if _models_loaded or _load_attempted:
        return _models_loaded

    _load_attempted = True

    try:
        import torch
        from geometric_lens.training import load_models

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        cost_path = os.path.join(models_dir, "cost_field.pt")

        if not os.path.exists(cost_path):
            logger.warning(f"Geometric Lens model files not found in {models_dir}")
            return False

        _cost_field, _metric_tensor = load_models(models_dir)
        _models_loaded = True
        logger.info("Geometric Lens models loaded successfully (CPU)")
        return True

    except Exception as e:
        logger.error(f"Failed to load Geometric Lens models: {e}")
        return False


def reload_weights(model_dir: str = None) -> dict:
    """Reload C(x) and G(x) weights from disk without restarting the process.

    Used after retraining to hot-swap model weights.

    Args:
        model_dir: Directory containing cost_field.pt and metric_tensor.pt.
                   Defaults to the models/ subdirectory.

    Returns:
        Dict with status and reload details.
    """
    global _cost_field, _metric_tensor, _models_loaded, _load_attempted

    # Reset flags so _ensure_models_loaded() will try again
    _models_loaded = False
    _load_attempted = False
    _cost_field = None
    _metric_tensor = None

    if model_dir:
        try:
            import torch
            from geometric_lens.training import load_models
            _cost_field, _metric_tensor = load_models(model_dir)
            _models_loaded = True
            _load_attempted = True
            logger.info(f"Geometric Lens models reloaded from {model_dir}")
            return {"status": "reloaded", "model_dir": model_dir}
        except Exception as e:
            logger.error(f"Failed to reload models from {model_dir}: {e}")
            _load_attempted = True
            return {"status": "error", "message": str(e)}
    else:
        success = _ensure_models_loaded()
        return {"status": "reloaded" if success else "error"}


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
        # Measured outputs: PASS ~5.00, FAIL ~14.04, midpoint ~9.5
        # Training targets were 2.0/25.0; model converged to 5.00/14.04
        normalized = 1.0 / (1.0 + 2.718 ** (-(energy - 9.5) / 3.0))

        logger.debug(
            f"Geometric energy: raw={energy:.2f} normalized={normalized:.3f} "
            f"latency={elapsed_ms:.1f}ms"
        )

        return min(1.0, max(0.0, normalized))

    except Exception as e:
        logger.error(f"Geometric energy computation failed: {e}")
        return 0.0


def evaluate_and_correct(
    query: str, alpha: float = 0.05
) -> Tuple[float, float, Optional[List[float]]]:
    """Full geometric lens pipeline: evaluate energy and optionally correct.

    Returns (energy_before, energy_after, corrected_embedding).
    If correction isn't needed (low energy), returns (energy, energy, None).
    """
    if not is_enabled() or not _ensure_models_loaded():
        return (0.0, 0.0, None)

    try:
        import torch
        from geometric_lens.embedding_extractor import extract_embedding
        from geometric_lens.correction import compute_correction, evaluate_energy

        start = time.monotonic()

        # Extract embedding
        emb = extract_embedding(query)
        x = torch.tensor(emb, dtype=torch.float32)

        # Evaluate energy before
        energy_before = evaluate_energy(x, _cost_field)

        # Only correct if energy is above threshold (high = bug-prone)
        # Threshold: midpoint between PASS mean (2.0) and FAIL mean (24.4)
        energy_threshold = 10.0
        if energy_before > energy_threshold:
            corrected = compute_correction(x, _cost_field, _metric_tensor, alpha=alpha)
            energy_after = evaluate_energy(corrected, _cost_field)
            corrected_list = corrected.tolist()
        else:
            energy_after = energy_before
            corrected_list = None

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            f"Geometric lens: energy={energy_before:.2f}->{energy_after:.2f} "
            f"corrected={corrected_list is not None} latency={elapsed_ms:.1f}ms"
        )

        return (energy_before, energy_after, corrected_list)

    except Exception as e:
        logger.error(f"Geometric lens evaluation failed: {e}")
        return (0.0, 0.0, None)


def get_model_info() -> dict:
    """Get info about loaded models for health/status endpoints."""
    if not _models_loaded:
        return {"loaded": False, "enabled": is_enabled()}

    import torch

    cost_params = sum(p.numel() for p in _cost_field.parameters())
    metric_params = sum(p.numel() for p in _metric_tensor.parameters())

    return {
        "loaded": True,
        "enabled": is_enabled(),
        "cost_field_params": cost_params,
        "metric_tensor_params": metric_params,
        "total_params": cost_params + metric_params,
        "device": "cpu",
    }
