"""atlas model recommendations — back-compat shim (PC-055.2 → PC-056).

Originally introduced in PC-055.2 as the bridge that held per-tier
default-model lookups separate from `tier.py`. PC-056 absorbed this
surface into `model_registry.py` (richer schema with `lens_status`,
`download_url`, `sha256`, etc.) but the public API stays identical so
existing callers (doctor.check_tier_match, tier.py rendering) don't
churn.

This module is intentionally trivial — it just re-exports the same
three names that PC-055.2 callers import:

    from atlas.cli.commands import model_recommendations
    rec = model_recommendations.for_tier("medium")
    tname = model_recommendations.tier_for_model("Qwen3.5-9B-Q6_K.gguf")
    # ModelRecommendation is now an alias for Model (superset of fields)

If you're adding a new caller, prefer importing `model_registry`
directly — this shim will be removed in a future Phase-1 cleanup once
the in-tree call sites have migrated.
"""

from .model_registry import (
    Model as ModelRecommendation,
    for_tier,
    tier_for_model,
    as_dict,
)

__all__ = ["ModelRecommendation", "for_tier", "tier_for_model", "as_dict"]
