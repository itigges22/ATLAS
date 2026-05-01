"""atlas model recommendations — tier -> default model mapping (PC-055.2).

Bridge module that holds the per-tier *default* model recommendation, kept
deliberately separate from `tier.py` (hardware capability) so PC-056's
full model registry can absorb this surface without touching `TierProfile`
or any of its callers.

Design intent (the split between this and tier.py):

  tier.py                 — pure hardware capability:
                            "what can this host run?" (VRAM, RAM, cores)
  model_recommendations   — capability -> recommendation:
                            "given that tier, what should you run?"
  PC-056 model registry   — full inventory: list, install, remove,
                            verify, multi-model variants per tier;
                            will replace this module's body but keep
                            the `for_tier()` API stable.

Why split now: TierProfile previously carried `model_file`, `model_display`,
`model_size_gb`. That conflated two concerns. PC-056 will introduce
multiple models per tier (e.g., reasoning vs. coding variants) and a
download/verify lifecycle, none of which belongs on a hardware tier
record. Splitting now means PC-056 lands as an upgrade in place rather
than a refactor of every caller.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelRecommendation:
    """The default model ATLAS recommends for a given hardware tier.

    PC-056 will extend this with `download_url`, `sha256`, `license`,
    and possibly a list of alternates per tier. Today's callers
    (doctor, tier card display, future PC-054 wizard) only need the
    file/display/size triple, so that's all we expose for now.
    """
    tier: str            # 'cpu' | 'small' | 'medium' | 'large' | 'xlarge'
    model_file: str      # gguf filename — matches ATLAS_MODEL_FILE in .env
    model_display: str   # human-friendly name for UI / doctor messages
    model_size_gb: float # on-disk size; informs disk-space messaging

    def env_vars(self) -> Dict[str, str]:
        """Render as the .env keys the wizard / installer would write."""
        return {
            "ATLAS_MODEL_FILE": self.model_file,
            "ATLAS_MODEL_NAME": self.model_file.rsplit(".", 1)[0],
        }


# Single source of truth for tier -> default-model. PC-056 will replace
# the body of this module with a richer registry, but `for_tier()` and
# `ModelRecommendation` will remain the stable public API.
RECOMMENDATIONS: Dict[str, ModelRecommendation] = {
    "cpu": ModelRecommendation(
        tier="cpu",
        model_file="N/A",
        model_display="N/A — install a CUDA GPU",
        model_size_gb=0.0,
    ),
    "small": ModelRecommendation(
        tier="small",
        model_file="Qwen3.5-7B-Q4_K_M.gguf",
        model_display="Qwen3.5 7B (Q4_K_M)",
        model_size_gb=4.4,
    ),
    "medium": ModelRecommendation(
        tier="medium",
        model_file="Qwen3.5-9B-Q6_K.gguf",
        model_display="Qwen3.5 9B (Q6_K)",
        model_size_gb=6.9,
    ),
    "large": ModelRecommendation(
        tier="large",
        model_file="Qwen3.5-14B-Q5_K_M.gguf",
        model_display="Qwen3.5 14B (Q5_K_M)",
        model_size_gb=10.5,
    ),
    "xlarge": ModelRecommendation(
        tier="xlarge",
        model_file="Qwen3.5-32B-Q5_K_M.gguf",
        model_display="Qwen3.5 32B (Q5_K_M)",
        model_size_gb=23.0,
    ),
}


def for_tier(tier_name: str) -> Optional[ModelRecommendation]:
    """Look up the default-model recommendation for a tier name.

    Stable public API: PC-056's registry will preserve this signature.
    Returns None for unknown tier names rather than raising — callers
    are CLI/diagnostic tools where a missing recommendation is a
    rendering issue, not a fatal error.
    """
    return RECOMMENDATIONS.get(tier_name)


def tier_for_model(model_file: str) -> Optional[str]:
    """Reverse lookup: which tier owns a given model filename?

    Used by doctor.check_tier_match to figure out which tier the
    user's currently-configured model belongs to, so it can compare
    against the host's recommended tier and warn on overshoot.
    Returns the tier name or None if the model isn't a known preset.
    """
    for name, rec in RECOMMENDATIONS.items():
        if rec.model_file == model_file:
            return name
    return None


def as_dict(rec: ModelRecommendation) -> Dict:
    """Serializer for JSON output — kept here so callers don't import
    dataclasses just to render us."""
    return asdict(rec)
