"""Geometric Lens service interface — main entry point for geometric-lens integration.

Provides:
- evaluate(embedding) -> energy scalar (C(x))
- evaluate_gx(query) -> XGBoost G(x) verdict dict
- is_enabled() -> bool
- get_geometric_energy(query) -> float in [0,1] for router signal
"""

import logging
import os
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Guards the mutable model globals below. reload_weights()/_ensure_models_loaded
# mutate them as a set; scoring paths read them as a set. Both run concurrently
# in FastAPI's threadpool (and the v3 ThreadingHTTPServer), so a hot reload can
# otherwise null a global mid-scoring. Reentrant because the load path nests
# (_ensure_models_loaded → reload_weights → _ensure_models_loaded). Held only
# for the global read/swap — never across the torch/xgboost forward passes or
# the embedding HTTP call.
_weights_lock = threading.RLock()

# Lazy-loaded models (CPU only)
_cost_field = None
_gx_xgboost = None        # XGBoost G(x) classifier
_gx_pca_components = None  # PCA projection matrix, numpy (pca_dim, hidden_dim)
_gx_pca_mean = None        # PCA mean vector, numpy (hidden_dim,)
_gx_top_dims = None        # Top contributing PCA dimensions
_models_loaded = False
_load_attempted = False
_artifact_model_identity = None
_model_identity_error = ""
# Directory the current C(x)/G(x) artifacts were loaded from; the drift
# fingerprint check reads drift_fingerprint.json from here.
_active_models_dir = None

# Cached llama-server /v1/models probe. The lens artifact must match the
# model the server is actually serving, not just whatever ATLAS_MODEL_NAME
# was exported at container start. Reset by reload_weights().
_served_model_id = None
_served_model_probed = False

# Once-per-load-cycle warning flag for the router's energy signal — a
# disabled lens must not silently read as "low energy". Reset by
# reload_weights().
_energy_disabled_logged = False

# C(x) energy scales are learned, not universal. The selected model's Lens
# artifact must provide its own sigmoid calibration. Without one, raw energy is
# still useful telemetry but the normalized value stays neutral (0.5).
_cx_normalization = None

# Per-model lens operating thresholds. These travel WITH the lens artifact
# (gx_thresholds.json in the model's lens dir) because the G(x) score scale is
# model-specific: a 0.3 off-rails cutoff that fits one model's distribution is
# wrong for another. Missing calibration leaves threshold-based interventions
# disabled; raw scores remain available as telemetry.
#   off_rails — per-token gx below this marks the first "stop generating" idx
#   low       — aggregate gx_min below this is a low-quality write (proxy)
#   severe    — aggregate gx_min below this is bad enough to act on one sample
_gx_thresholds = None


def _probe_served_model() -> str:
    """Return the model id llama-server is actually serving ("" if unknown).

    Cheap by design: short timeout, one probe per load cycle (the result is
    cached until reload_weights() resets it).
    """
    global _served_model_id, _served_model_probed
    if _served_model_probed:
        return _served_model_id or ""
    _served_model_probed = True
    _served_model_id = ""
    base = os.environ.get("LLAMA_URL", "http://llama-server:8080").rstrip("/")
    try:
        import json as _json
        from urllib.request import urlopen
        with urlopen(f"{base}/v1/models", timeout=2.0) as resp:
            data = _json.loads(resp.read())
        models = data.get("data", []) if isinstance(data, dict) else []
        if models and models[0].get("id"):
            _served_model_id = str(models[0]["id"])
    except Exception as exc:
        logger.warning(
            "Served-model probe (%s/v1/models) failed: %s — "
            "falling back to ATLAS_MODEL_NAME", base, exc)
    return _served_model_id or ""


def _verify_model_identity(models_dir: str, embedding_dim: int = 0) -> bool:
    """Require Lens artifacts to identify the actually-served runtime model.

    The reference identity is the model id llama-server reports on
    /v1/models; ATLAS_MODEL_NAME is the fallback when the probe fails.
    When embedding_dim is provided (the cost field's input dim), it must
    match the artifact's declared embedding_dim as well.
    """
    global _artifact_model_identity, _model_identity_error
    _artifact_model_identity = None
    _model_identity_error = ""
    selected_model = _probe_served_model() or os.environ.get(
        "ATLAS_MODEL_NAME", "").strip()
    if not selected_model:
        _model_identity_error = ("served-model probe failed and "
                                 "ATLAS_MODEL_NAME is unset")
        logger.error("Lens artifact identity check failed: %s",
                     _model_identity_error)
        return False
    try:
        from geometric_lens.identity import (
            identity_matches,
            load_model_identity,
        )
        identity = load_model_identity(models_dir)
        if not identity_matches(identity, selected_model):
            raise ValueError(
                f"artifacts are for {identity['model']!r}, selected model is "
                f"{selected_model!r}"
            )
        if embedding_dim and identity["embedding_dim"] != int(embedding_dim):
            raise ValueError(
                f"cost_field.pt input dim is {int(embedding_dim)}, artifact "
                f"identity declares embedding_dim {identity['embedding_dim']}"
            )
        _artifact_model_identity = identity
        # Install the artifact's embedding contract so extract_embedding()
        # rejects wrong-convention responses instead of silently adapting.
        from geometric_lens.embedding_extractor import set_embedding_contract
        contract = identity.get("embedding_contract")
        set_embedding_contract(contract)
        if contract:
            logger.info("Embedding contract active: %s", contract)
        else:
            logger.warning(
                "model_identity.json declares no embedding_contract — "
                "extract_embedding() accepts any response convention; a "
                "server-side pooling/normalization change will shift scores "
                "silently. Retrain to record one.")
        return True
    except Exception as exc:
        logger.error("Lens artifact identity check failed: %s", exc,
                     exc_info=True)
        from geometric_lens.embedding_extractor import set_embedding_contract
        set_embedding_contract(None)
        # get_model_info() surfaces this string over HTTP — keep the
        # exception type but not its text (full detail is in the log above).
        _model_identity_error = (
            f"{type(exc).__name__}: artifact identity check failed "
            "(see service log)"
        )
        return False


def active_models_dir():
    """Directory the current artifacts were loaded from (None before the
    first successful load). The drift-fingerprint check reads
    drift_fingerprint.json from here."""
    return _active_models_dir


def _load_cx_normalization(models_dir: str) -> None:
    """Load the selected model's C(x) energy calibration."""
    global _cx_normalization
    _cx_normalization = None
    import json as _json
    from geometric_lens.calibration import validate_cx_normalization
    path = os.path.join(models_dir, "cx_normalization.json")
    if not os.path.exists(path):
        logger.warning("No cx_normalization.json — C(x) normalized scores are neutral")
        return
    try:
        with open(path) as fh:
            _cx_normalization = validate_cx_normalization(_json.load(fh))
        logger.info("Loaded per-model C(x) calibration from %s: %s",
                    path, _cx_normalization)
    except Exception as e:
        _cx_normalization = None
        logger.warning("cx_normalization.json load failed (%s) — C(x) normalized scores are neutral", e)


def _normalize_cx_energy(energy: float, cx_cfg=None) -> float:
    from geometric_lens.calibration import normalize_cx_energy
    cfg = _cx_normalization if cx_cfg is None else cx_cfg
    return normalize_cx_energy(energy, cfg)


def _snapshot_weights():
    """Read the mutable model globals into locals as one consistent set.

    Scoring paths call this once, then compute against the returned
    references so a concurrent reload_weights() cannot null a global (or
    mix generations) mid-forward-pass. Returns a tuple in a fixed order.
    """
    with _weights_lock:
        return (
            _cost_field, _gx_xgboost,
            _gx_pca_components, _gx_pca_mean, _gx_top_dims,
            _cx_normalization, _gx_thresholds,
        )


def _gx_verdict(score: float, thresholds=None) -> str:
    """Classify a G(x) score only when this model has calibration.

    thresholds defaults to the module global; scoring paths pass a snapshot
    taken under _weights_lock so a concurrent reload can't swap it mid-call.
    """
    t = _gx_thresholds if thresholds is None else thresholds
    if t is None:
        return "uncalibrated"
    if score < t["severe"]:
        return "likely_incorrect"
    if score < t["low"]:
        return "uncertain"
    return "likely_correct"


def _load_gx_thresholds(models_dir: str) -> None:
    """Load calibrated operating thresholds for the selected model.

    Missing or invalid calibration keeps scoring available but disables
    threshold-based verdicts/interventions. Borrowing another model's cutoffs
    would be a silent correctness failure.
    """
    global _gx_thresholds
    _gx_thresholds = None
    import json as _json
    from geometric_lens.thresholds import validate_gx_thresholds
    path = os.path.join(models_dir, "gx_thresholds.json")
    if not os.path.exists(path):
        logger.warning("No gx_thresholds.json — Lens scores are uncalibrated; threshold interventions disabled")
        return
    try:
        with open(path) as fh:
            loaded = _json.load(fh)
        _gx_thresholds = validate_gx_thresholds(loaded)
        logger.info("Loaded per-model lens thresholds from %s: %s", path, _gx_thresholds)
    except Exception as e:
        _gx_thresholds = None
        logger.warning("gx_thresholds.json load failed (%s) — threshold interventions disabled", e)


def is_enabled() -> bool:
    """Check if Geometric Lens is enabled (GEOMETRIC_LENS_ENABLED env var)."""
    return os.environ.get("GEOMETRIC_LENS_ENABLED", "false").lower() in ("true", "1", "yes")


class _BoosterClassifier:
    """Minimal predict_proba shim around an xgboost.Booster.

    The legacy code path loaded an xgboost.sklearn.XGBClassifier from pickle
    and called .predict_proba(x). The native-JSON path (PC-031) avoids the
    pickle compat warning and the sklearn runtime dep — but raw Booster only
    exposes .predict(). For binary:logistic objectives, that returns the
    positive-class probability directly. This shim shapes it back into the
    [P(neg), P(pos)] layout the callers in this module expect, so the
    downstream `proba[1]` indexing keeps working unchanged.
    """

    def __init__(self, booster):
        self._booster = booster

    def predict_proba(self, x):
        import numpy as np
        import xgboost as xgb
        dmatrix = xgb.DMatrix(np.asarray(x, dtype=np.float32))
        pos = self._booster.predict(dmatrix)
        return np.column_stack([1.0 - pos, pos])


def _load_gx_models(models_dir: str) -> None:
    """Load G(x) models from `models_dir` (XGBoost preferred, metric tensor legacy).

    Shared by _ensure_models_loaded and the reload_weights(model_dir=...)
    path so per-directory reloads yield a complete lens. Non-fatal: any
    failure leaves the corresponding G(x) slot None and scoring degrades
    gracefully.
    """
    global _gx_xgboost, _gx_pca_components, _gx_pca_mean, _gx_top_dims

    _gx_xgboost = None
    _gx_pca_components = None
    _gx_pca_mean = None
    _gx_top_dims = None

    # G(x) XGBoost model (preferred). Prefer the native JSON dump
    # (gx_xgboost.json) — version-stable, no pickle-compat warning, no
    # sklearn dep. Fall back to gx_xgboost.pkl for users who haven't
    # refreshed their model dir yet. See ISSUES.md PC-031.
    xgb_json = os.path.join(models_dir, "gx_xgboost.json")
    xgb_pkl = os.path.join(models_dir, "gx_xgboost.pkl")
    weights_path = os.path.join(models_dir, "gx_weights.json")
    if os.path.exists(weights_path) and (os.path.exists(xgb_json) or os.path.exists(xgb_pkl)):
        try:
            import json as json_mod
            import numpy as np
            import xgboost as xgb

            if os.path.exists(xgb_json):
                booster = xgb.Booster()
                booster.load_model(xgb_json)
                _gx_xgboost = _BoosterClassifier(booster)
                load_path = "json"
            elif os.environ.get("ATLAS_ALLOW_PICKLE_GX") == "1":
                # Legacy pickle fallback, opt-in only: unpickling executes
                # arbitrary code, and model dirs can contain downloaded
                # artifacts. Users on the old format set
                # ATLAS_ALLOW_PICKLE_GX=1 once to load it, then re-export.
                import pickle
                with open(xgb_pkl, 'rb') as f:
                    _gx_xgboost = pickle.load(f)
                load_path = "pickle (deprecated — re-export to gx_xgboost.json)"
            else:
                logger.warning(
                    "G(x) found only as legacy gx_xgboost.pkl; refusing to "
                    "unpickle by default. Set ATLAS_ALLOW_PICKLE_GX=1 to load "
                    "it once and re-export to gx_xgboost.json."
                )
                _gx_xgboost = None

            if _gx_xgboost is not None:
                with open(weights_path, 'r') as f:
                    weights = json_mod.load(f)

                _gx_pca_components = np.array(weights['pca_components'], dtype=np.float32)
                _gx_pca_mean = np.array(weights['pca_mean'], dtype=np.float32)
                _gx_top_dims = weights.get('top_dims', [])

                logger.info(
                    f"G(x) XGBoost loaded ({load_path}, AUC={weights.get('cv_auc_mean', 0):.4f}, "
                    f"PCA {weights.get('original_dim', '?')}→{weights.get('pca_dim', '?')})"
                )
        except ImportError:
            logger.warning("G(x) XGBoost model found but xgboost package not installed")
            _gx_xgboost = None
        except Exception as e:
            logger.warning(f"G(x) XGBoost load failed (non-fatal): {e}")
            _gx_xgboost = None

    if _gx_xgboost is None:
        logger.info("No G(x) model found — G(x) verdicts unavailable")


def _ensure_models_loaded():
    """Lazy-load C(x) cost field and G(x) models (XGBoost preferred, metric tensor legacy) on first use."""
    global _load_attempted

    if _models_loaded or _load_attempted:
        return _models_loaded

    # Serialize the one-time load so two concurrent first-requests can't both
    # run it (duplicate load / half-populated globals). Re-check under the lock.
    with _weights_lock:
        if _models_loaded or _load_attempted:
            return _models_loaded

        _load_attempted = True

        return _do_load_models()


def _do_load_models() -> bool:
    """Load C(x)/G(x) artifacts. Callers hold _weights_lock."""
    global _cost_field, _models_loaded

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

        # Identity check includes the checkpoint's input dim so a wrong-dim
        # artifact disables cleanly instead of failing per-request in torch.
        if not _verify_model_identity(models_dir, embedding_dim=dim):
            return False

        global _active_models_dir
        _active_models_dir = models_dir

        # Per-model calibration ships alongside the lens artifact.
        _load_cx_normalization(models_dir)
        _load_gx_thresholds(models_dir)

        _cost_field = CostField(input_dim=dim)
        _cost_field.load_state_dict(sd)
        _cost_field.set_eval_mode()

        logger.info(f"Geometric Lens C(x) model loaded successfully (CPU, dim={dim})")

        _load_gx_models(models_dir)

        _models_loaded = True
        return True

    except Exception as e:
        logger.error(f"Failed to load Geometric Lens models: {e}")
        return False


def reload_weights(model_dir: str = None) -> dict:
    """Reload C(x) and G(x) weights from disk without restarting the process.

    Used after retraining to hot-swap model weights.

    All global mutation happens in _reload_weights_locked(), which declares
    the globals it assigns; this wrapper only takes the lock.
    """
    # Hold the lock across the whole reset+load so scoring never observes the
    # nulled-then-repopulated globals of an in-progress swap. This is the write
    # critical section; the artifact loads here are not scoring forward passes.
    with _weights_lock:
        return _reload_weights_locked(model_dir)


def _reload_weights_locked(model_dir: str = None) -> dict:
    """Body of reload_weights(); callers hold _weights_lock."""
    global _cost_field, _gx_xgboost, _gx_pca_components
    global _gx_pca_mean, _gx_top_dims, _models_loaded, _load_attempted
    global _cx_normalization, _gx_thresholds
    global _artifact_model_identity, _model_identity_error
    global _served_model_id, _served_model_probed, _active_models_dir

    _models_loaded = False
    _load_attempted = False
    _cost_field = None
    _gx_xgboost = None
    _gx_pca_components = None
    _gx_pca_mean = None
    _gx_top_dims = None
    _cx_normalization = None
    _gx_thresholds = None
    _artifact_model_identity = None
    _model_identity_error = ""
    _active_models_dir = None
    from geometric_lens.embedding_extractor import set_embedding_contract
    set_embedding_contract(None)
    # Re-probe llama-server on reload — the served model may have changed.
    _served_model_id = None
    _served_model_probed = False
    global _energy_disabled_logged
    _energy_disabled_logged = False

    if model_dir:
        try:
            from geometric_lens.training import load_cost_field
            cost_field = load_cost_field(model_dir)
            dim = next(cost_field.parameters()).shape[1]
            if not _verify_model_identity(model_dir, embedding_dim=int(dim)):
                raise ValueError(_model_identity_error)
            _cost_field = cost_field
            _active_models_dir = model_dir
            _load_cx_normalization(model_dir)
            _load_gx_thresholds(model_dir)
            _load_gx_models(model_dir)
            _models_loaded = True
            _load_attempted = True
            logger.info(f"Geometric Lens C(x) reloaded from {model_dir}")
            return {
                "status": "reloaded",
                "model_dir": model_dir,
                "gx_loaded": _gx_xgboost is not None,
            }
        except Exception as e:
            logger.error(f"Failed to reload models from {model_dir}: {e}",
                         exc_info=True)
            _load_attempted = True
            # The message reaches /internal/lens/reload and /retrain HTTP
            # responses — full detail stays in the log above.
            return {"status": "error",
                    "message": f"{type(e).__name__}: reload failed "
                               "(see service log)"}
    else:
        success = _ensure_models_loaded()
        return {
            "status": "reloaded" if success else "error",
            "gx_loaded": _gx_xgboost is not None,
        }


def get_geometric_energy(query: str) -> float:
    """Compute normalized geometric energy for a query.

    Extracts embedding from llama-server, evaluates through C(x),
    and returns a normalized energy in [0, 1].

    Used as the 4th signal in the Confidence Router.

    Returns 0.0 if lens is disabled or models aren't loaded — logged once
    per load cycle so a disabled lens can't silently masquerade as
    "genuinely low energy" in the router's signal mix.
    """
    global _energy_disabled_logged
    if not is_enabled() or not _ensure_models_loaded():
        if not _energy_disabled_logged:
            _energy_disabled_logged = True
            logger.warning(
                "get_geometric_energy: lens disabled or weights not "
                "loaded — returning 0.0 to the confidence router for "
                "every query until weights load")
        return 0.0

    try:
        import torch
        from geometric_lens.embedding_extractor import extract_embedding

        cost_field, _, _, _, _, cx_cfg, _ = _snapshot_weights()

        start = time.monotonic()

        # Extract embedding
        emb = extract_embedding(query)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        # Evaluate C(x)
        with torch.no_grad():
            energy = cost_field(x).item()

        elapsed_ms = (time.monotonic() - start) * 1000

        normalized = _normalize_cx_energy(energy, cx_cfg)

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

        cost_field, _, _, _, _, cx_cfg, _ = _snapshot_weights()

        emb = extract_embedding(query)
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            energy = cost_field(x).item()

        normalized = _normalize_cx_energy(energy, cx_cfg)

        return (energy, normalized)

    except Exception as e:
        logger.error(f"Geometric lens evaluation failed: {e}")
        return (0.0, 0.0)


def get_model_info() -> dict:
    """Get info about loaded models for health/status endpoints."""
    if not _models_loaded:
        return {
            "loaded": False,
            "enabled": is_enabled(),
            "artifact_model": (_artifact_model_identity or {}).get("model"),
            "error": _model_identity_error or None,
        }

    cost_params = sum(p.numel() for p in _cost_field.parameters())

    info = {
        "loaded": True,
        "enabled": is_enabled(),
        "cost_field_params": cost_params,
        "device": "cpu",
        "cx_calibrated": _cx_normalization is not None,
        "gx_loaded": _gx_xgboost is not None,
        "gx_calibrated": _gx_thresholds is not None,
        "gx_thresholds": (dict(_gx_thresholds)
                          if _gx_thresholds is not None else None),
        "artifact_model": (_artifact_model_identity or {}).get("model"),
        "gx_type": "xgboost" if _gx_xgboost is not None else "none",
    }

    if _gx_xgboost is not None:
        info["gx_pca_dim"] = _gx_pca_components.shape[0] if _gx_pca_components is not None else 0
        info["gx_top_dims"] = _gx_top_dims[:10] if _gx_top_dims else []
    info["total_params"] = cost_params

    return info


def evaluate_gx(query: str) -> dict:
    """Score code quality using XGBoost G(x) classifier.

    Returns dict with gx_score (0-1 probability of PASS), verdict, and metadata.
    Falls back gracefully if XGBoost model is not available.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return {"gx_score": 0.5, "verdict": "unavailable", "gx_available": False}

    (_, gx_xgboost, gx_pca_components, gx_pca_mean,
     gx_top_dims, _, gx_thresholds) = _snapshot_weights()

    if gx_xgboost is None:
        return {"gx_score": 0.5, "verdict": "unavailable", "gx_available": False}

    try:
        import numpy as np
        from geometric_lens.embedding_extractor import extract_embedding

        start = time.monotonic()

        emb = extract_embedding(query)
        emb_np = np.array(emb, dtype=np.float32).reshape(1, -1)

        # PCA transform
        x_pca = (emb_np - gx_pca_mean) @ gx_pca_components.T

        # XGBoost prediction
        proba = gx_xgboost.predict_proba(x_pca)[0]
        gx_score = float(proba[1])  # probability of PASS class

        verdict = _gx_verdict(gx_score, gx_thresholds)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(f"G(x) score: {gx_score:.4f} ({verdict}), latency={elapsed_ms:.1f}ms")

        return {
            "gx_score": gx_score,
            "verdict": verdict,
            "top_dims": gx_top_dims[:10] if gx_top_dims else [],
            "gx_available": True,
            "latency_ms": round(elapsed_ms, 1),
        }

    except Exception as e:
        logger.error(f"G(x) evaluation failed: {e}", exc_info=True)
        return {"gx_score": 0.5, "verdict": "error", "gx_available": False,
                "error": f"{type(e).__name__}: G(x) evaluation failed "
                         "(see service log)"}


def evaluate_combined(query: str) -> dict:
    """Combined C(x) + G(x) evaluation using a single embedding extraction.

    Returns dict with C(x) energy, G(x) quality score, and verdict.
    Most efficient way to get both scores — avoids duplicate embedding calls.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "cx_calibrated": False,
            "gx_score": 0.5, "verdict": "unavailable",
            "enabled": False, "gx_available": False,
        }

    try:
        import torch
        import numpy as np
        from geometric_lens.embedding_extractor import extract_embedding

        (cost_field, gx_xgboost, gx_pca_components, gx_pca_mean,
         _, cx_cfg, gx_thresholds) = _snapshot_weights()

        start = time.monotonic()

        # Single embedding extraction (shared between C(x) and G(x))
        emb = extract_embedding(query)

        # C(x) evaluation
        x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            energy = cost_field(x).item()
        normalized = _normalize_cx_energy(energy, cx_cfg)

        # G(x) evaluation (if available)
        gx_score = 0.5
        verdict = "unavailable"
        gx_available = False

        if gx_xgboost is not None:
            emb_np = np.array(emb, dtype=np.float32).reshape(1, -1)
            x_pca = (emb_np - gx_pca_mean) @ gx_pca_components.T
            proba = gx_xgboost.predict_proba(x_pca)[0]
            gx_score = float(proba[1])
            gx_available = True

            verdict = _gx_verdict(gx_score, gx_thresholds)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"Combined: C(x)={energy:.2f}({normalized:.3f}) G(x)={gx_score:.4f} "
            f"({verdict}) latency={elapsed_ms:.1f}ms"
        )

        return {
            "cx_energy": energy,
            "cx_normalized": normalized,
            "cx_calibrated": cx_cfg is not None,
            "gx_score": gx_score,
            "verdict": verdict,
            "gx_available": gx_available,
            "enabled": True,
            "latency_ms": round(elapsed_ms, 1),
        }

    except Exception as e:
        logger.error(f"Combined evaluation failed: {e}", exc_info=True)
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "cx_calibrated": False,
            "gx_score": 0.5, "verdict": "error",
            "enabled": True, "gx_available": False,
            "error": f"{type(e).__name__}: combined evaluation failed "
                     "(see service log)",
        }


def evaluate_per_step(query: str, layer: Optional[int] = None) -> dict:
    """PC-207 lens-as-PRM: score every token in `query` instead of pooling first.

    For each input token, applies C(x) and (when available) G(x) to that
    token's hidden-state vector. This turns the lens from an ORM (scores
    completed text) into a PRM (scores each generation step), which lets
    callers detect off-rails generation early — e.g. catch the May 6 53-min
    repetition loop at token ~80 instead of after the full 8K-token decode.

    Args:
        query: text to score per token.
        layer: optional transformer-block index. None (default) uses the
            last-layer hidden state via vanilla `/embedding` (no PC-202 patch
            required). When set, uses the PC-202 `layers` extension to score
            the residual stream at that specific layer — useful for PC-204
            multi-layer experiments.

    Returns:
        Dict with `per_step` (list of per-token dicts), `aggregate` (min/
        max/mean across tokens), `n_tokens`, `hidden_dim`, `layer`, and
        `latency_ms`. On error, `enabled=False` or `error` keys are set.
    """
    if not is_enabled() or not _ensure_models_loaded():
        return {
            "enabled": False, "gx_available": False,
            "per_step": [], "aggregate": {}, "n_tokens": 0,
        }

    try:
        import numpy as np
        import torch
        from geometric_lens.embedding_extractor import (
            extract_per_layer_per_token,
            extract_per_token,
        )

        (cost_field, gx_xgboost, gx_pca_components, gx_pca_mean,
         gx_top_dims, cx_cfg, gx_thresholds) = _snapshot_weights()

        start = time.monotonic()

        # Pull per-token hidden states from llama-server
        if layer is None:
            per_token_vecs, hidden_dim = extract_per_token(query)
            tap_label = "last"
        else:
            per_layer, _, hidden_dim = extract_per_layer_per_token(query, [int(layer)])
            per_token_vecs = per_layer[int(layer)]
            tap_label = str(layer)

        n_tokens = len(per_token_vecs)
        if n_tokens == 0:
            return {
                "enabled": True, "gx_available": gx_xgboost is not None,
                "per_step": [], "aggregate": {}, "n_tokens": 0,
                "layer": tap_label,
                "error": "empty token list",
            }

        # Batched C(x): one MLP forward over [n_tokens, hidden_dim]
        x = torch.tensor(per_token_vecs, dtype=torch.float32)
        with torch.no_grad():
            cx_raw = cost_field(x).squeeze(-1).cpu().numpy()  # (n_tokens,)
        if cx_cfg is None:
            cx_norm = np.full(n_tokens, 0.5, dtype=float)
        else:
            midpoint = cx_cfg["midpoint"]
            steepness = cx_cfg["steepness"]
            z = np.clip(steepness * (cx_raw - midpoint), -709.0, 709.0)
            cx_norm = 1.0 / (1.0 + np.exp(-z))

        # Batched G(x) when XGBoost is loaded
        gx_available = gx_xgboost is not None and gx_pca_components is not None
        if gx_available:
            emb_np = np.asarray(per_token_vecs, dtype=np.float32)
            x_pca = (emb_np - gx_pca_mean) @ gx_pca_components.T
            proba = gx_xgboost.predict_proba(x_pca)
            gx_scores = proba[:, 1].astype(float)
        else:
            gx_scores = np.full(n_tokens, 0.5, dtype=float)

        per_step = []
        for i in range(n_tokens):
            score = float(gx_scores[i])
            if gx_available and gx_thresholds is not None:
                if score < gx_thresholds["severe"]:
                    verdict = "likely_incorrect"
                elif score < gx_thresholds["low"]:
                    verdict = "uncertain"
                else:
                    verdict = "likely_correct"
            elif gx_available:
                verdict = "uncalibrated"
            else:
                verdict = "unavailable"
            per_step.append({
                "token_idx":     i,
                "cx_energy":     float(cx_raw[i]),
                "cx_normalized": float(cx_norm[i]),
                "gx_score":      score,
                "gx_verdict":    verdict,
            })

        aggregate = {
            "cx_energy_min":  float(cx_raw.min()),
            "cx_energy_max":  float(cx_raw.max()),
            "cx_energy_mean": float(cx_raw.mean()),
            "cx_norm_min":    float(cx_norm.min()),
            "cx_norm_max":    float(cx_norm.max()),
            "cx_norm_mean":   float(cx_norm.mean()),
            "gx_score_min":   float(gx_scores.min()),
            "gx_score_max":   float(gx_scores.max()),
            "gx_score_mean":  float(gx_scores.mean()),
            # token index where the lens first sees a low-quality state —
            # the natural "stop generating" signal for PC-207 callers.
            "first_off_rails_idx": int(np.argmax(gx_scores < gx_thresholds["off_rails"])) if gx_available and gx_thresholds is not None and (gx_scores < gx_thresholds["off_rails"]).any() else -1,
        }

        elapsed_ms = (time.monotonic() - start) * 1000
        # !r on tap_label — user-controllable string (layer id from
        # request) — defends against py/log-injection via embedded CRLF.
        logger.debug(
            f"per-step lens: n={n_tokens} layer={tap_label!r} "
            f"cx_norm[mean,max]=({aggregate['cx_norm_mean']:.3f},{aggregate['cx_norm_max']:.3f}) "
            f"gx[min,mean]=({aggregate['gx_score_min']:.3f},{aggregate['gx_score_mean']:.3f}) "
            f"latency={elapsed_ms:.1f}ms"
        )

        return {
            "enabled":      True,
            "gx_available": gx_available,
            "cx_calibrated": cx_cfg is not None,
            "per_step":     per_step,
            "aggregate":    aggregate,
            "n_tokens":     n_tokens,
            "hidden_dim":   hidden_dim,
            "layer":        tap_label,
            "latency_ms":   round(elapsed_ms, 1),
            # The per-model thresholds this score was judged against. The proxy
            # uses these for its run-of-N / severe regression checks instead of
            # its own hardcoded constants, so the whole intervention chain is
            # calibrated to the loaded model's score scale.
            "thresholds":   dict(gx_thresholds) if gx_thresholds is not None else None,
        }

    except Exception as e:
        logger.error(f"per-step evaluation failed: {e}", exc_info=True)
        return {
            "enabled": True, "gx_available": False,
            "per_step": [], "aggregate": {}, "n_tokens": 0,
            "error": f"{type(e).__name__}: per-step evaluation failed "
                     "(see service log)",
        }
