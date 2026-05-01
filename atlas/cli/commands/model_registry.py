"""atlas model registry — known models + their lens artifact status (PC-056).

Single source of truth for "which models can ATLAS run end-to-end?"

This module is the upgrade-in-place of PC-055.2's
`model_recommendations.py` stub. It preserves the stable public API
(`for_tier`, `tier_for_model`, the Model record's name-compat fields)
that doctor + tier callers depend on, while extending each entry with
download metadata + the critical `lens_status` field that captures the
key truth surfaced during PC-056 scoping:

    Most "supported" tier presets in PC-055 are aspirational. Only the
    9B model has actual Lens artifacts (metric tensor + embeddings).
    Other entries can be downloaded as raw GGUFs but G(x) silently
    no-ops on them — half of what makes ATLAS *ATLAS* is missing.

`lens_status` makes that visible. The CLI surfaces it; doctor warns on
overshoot; users who want a no-artifacts model must pass `--no-lens`.

Bringing more models to `lens_status: supported` is the work of
PC-058 (`atlas lens build`). PC-057 (`atlas lens check`) is the cheap
pre-flight that says "is this model Lens-compatible at all?" before
you invest hours in PC-058's training pipeline. PC-059 / PC-060 are
the contribution flow that takes locally-trained artifacts and
publishes them back to the registry.

Schema notes:
- `model_file` / `model_display` / `model_size_gb` are kept as field
  names (not renamed to `file`/`display`/`size`) so the PC-055.2
  `ModelRecommendation` API alias works without code changes.
- `download_url` is None for gated/missing upstreams. CLI must check
  before invoking `atlas model install`.
- `sha256` is the HuggingFace `x-linked-etag` value (the content-addressed
  storage hash) where available. Verifies download integrity, not
  provenance. Stronger provenance verification is PC-060 territory.
- `lens_status` values:
    "supported"   — metric tensor + embeddings present in repo
    "no-artifacts" — model exists but Lens won't score it
    "unverified"  — has artifacts but never validated end-to-end
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Model:
    """A known model entry. Field names preserve the PC-055.2
    ModelRecommendation contract (model_file/model_display/model_size_gb)
    so the back-compat shim is a trivial re-export.
    """
    name: str                      # registry key, what `atlas model install` takes
    tier: str                      # 'cpu'|'small'|'medium'|'large'|'xlarge'
    model_file: str                # gguf filename — matches ATLAS_MODEL_FILE
    model_display: str             # human-friendly UI name
    model_size_gb: float           # on-disk size — informs disk-space messaging
    lens_status: str               # 'supported' | 'no-artifacts' | 'unverified'
    download_url: Optional[str] = None   # None = gated upstream, install will fail clean
    sha256: Optional[str] = None         # content hash (HF x-linked-etag) when known
    license: Optional[str] = None        # SPDX-ish identifier (Apache-2.0, etc.)
    notes: str = ""

    def env_vars(self) -> Dict[str, str]:
        """The .env keys the wizard / installer would write for this model."""
        return {
            "ATLAS_MODEL_FILE": self.model_file,
            "ATLAS_MODEL_NAME": self.model_file.rsplit(".", 1)[0],
        }

    @property
    def can_install(self) -> bool:
        """True when `atlas model install` can actually fetch this model.
        False for gated upstreams (download_url is None)."""
        return self.download_url is not None


# Single source of truth. Order: by tier (cpu → xlarge), then by name.
#
# Truthful state today:
#   - 9B: SUPPORTED. Actual unsloth/Qwen3.5-9B-GGUF on HF, public, has
#     trained metric tensor + embeddings in the ATLAS repo.
#   - 7B / 14B / 32B: NO-ARTIFACTS. unsloth's repos are gated (HTTP 401
#     on HEAD), no Lens artifacts trained. Listed so users know which
#     tier they map to and what's missing.
#
# Adding more models = a PC-058 build run + PC-059 PR.
REGISTRY: List[Model] = [
    Model(
        name="Qwen3.5-7B-Q4_K_M",
        tier="small",
        model_file="Qwen3.5-7B-Q4_K_M.gguf",
        model_display="Qwen3.5 7B (Q4_K_M)",
        model_size_gb=4.4,
        lens_status="no-artifacts",
        download_url=None,  # unsloth/Qwen3.5-7B-GGUF returns HTTP 401
        sha256=None,
        license="Apache-2.0",
        notes="Upstream repo unsloth/Qwen3.5-7B-GGUF is gated "
              "(HTTP 401). No Lens artifacts trained for this model. "
              "Will install as raw llama.cpp model only — G(x) "
              "verification will silently no-op. See PC-058 roadmap.",
    ),
    Model(
        name="Qwen3.5-9B-Q6_K",
        tier="medium",
        model_file="Qwen3.5-9B-Q6_K.gguf",
        model_display="Qwen3.5 9B (Q6_K)",
        # Verified 2026-05-01: HF Content-Length = 7458301152 bytes.
        model_size_gb=6.94,
        lens_status="supported",
        download_url=("https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/"
                      "resolve/main/Qwen3.5-9B-Q6_K.gguf"),
        # HuggingFace x-linked-etag (content-addressed storage SHA256).
        # Verifies download integrity, not provenance.
        sha256="91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59",
        license="Apache-2.0",
        notes="ATLAS development target. Lens artifacts trained and "
              "shipped in the repo (cost_field.pt + metric_tensor.pt). "
              "End-to-end supported.",
    ),
    Model(
        name="Qwen3.5-14B-Q5_K_M",
        tier="large",
        model_file="Qwen3.5-14B-Q5_K_M.gguf",
        model_display="Qwen3.5 14B (Q5_K_M)",
        model_size_gb=10.5,
        lens_status="no-artifacts",
        download_url=None,  # unsloth/Qwen3.5-14B-GGUF returns HTTP 401
        sha256=None,
        license="Apache-2.0",
        notes="Upstream repo unsloth/Qwen3.5-14B-GGUF is gated "
              "(HTTP 401). Tested in past ATLAS work but the trained "
              "Lens artifacts have been removed from the repo. Will "
              "install as raw llama.cpp model only — G(x) verification "
              "will silently no-op. See PC-058 roadmap to retrain.",
    ),
    Model(
        name="Qwen3.5-32B-Q5_K_M",
        tier="xlarge",
        model_file="Qwen3.5-32B-Q5_K_M.gguf",
        model_display="Qwen3.5 32B (Q5_K_M)",
        model_size_gb=23.0,
        lens_status="no-artifacts",
        download_url=None,  # unsloth/Qwen3.5-32B-GGUF returns HTTP 401
        sha256=None,
        license="Apache-2.0",
        notes="Upstream repo unsloth/Qwen3.5-32B-GGUF is gated "
              "(HTTP 401). No Lens artifacts trained for this model. "
              "Will install as raw llama.cpp model only — G(x) "
              "verification will silently no-op. See PC-058 roadmap.",
    ),
]


# ---------------------------------------------------------------------------
# Lookups — preserves PC-055.2 model_recommendations public API
# ---------------------------------------------------------------------------

def for_tier(tier_name: str) -> Optional[Model]:
    """Return the default model recommendation for a tier name.

    "Default" = the supported model if any tier-matched entry has
    `lens_status == "supported"`, otherwise the first tier-matched
    entry (which by definition is `no-artifacts`). Callers can inspect
    `lens_status` to render the warning.

    Returns None for tier names not in the registry (e.g., 'cpu',
    or unknown tiers). Caller decides how to render that.
    """
    matches = [m for m in REGISTRY if m.tier == tier_name]
    if not matches:
        return None
    supported = [m for m in matches if m.lens_status == "supported"]
    return supported[0] if supported else matches[0]


def tier_for_model(model_file: str) -> Optional[str]:
    """Reverse lookup: which tier owns a given gguf filename?

    Used by doctor.check_tier_match for the "you're running a
    larger-than-recommended model" warning.
    """
    for m in REGISTRY:
        if m.model_file == model_file:
            return m.tier
    return None


def by_name(name: str) -> Optional[Model]:
    """Look up a model by its registry name (the key
    `atlas model install` takes)."""
    for m in REGISTRY:
        if m.name == name:
            return m
    return None


def all_models() -> List[Model]:
    """Return all known models (defensive copy of REGISTRY)."""
    return list(REGISTRY)


def models_for_tier(tier_name: str) -> List[Model]:
    """All models registered against a tier (not just the default)."""
    return [m for m in REGISTRY if m.tier == tier_name]


def supported_models() -> List[Model]:
    """Models with end-to-end Lens support — i.e., what `atlas model
    install` will install without `--no-lens`."""
    return [m for m in REGISTRY if m.lens_status == "supported"]


# ---------------------------------------------------------------------------
# Install-state probe
# ---------------------------------------------------------------------------

def is_installed(model: Model, models_dir: str) -> bool:
    """Return True if the model's gguf file is present in models_dir
    AND larger than 100 MB (sanity threshold — guards against the
    "left an empty file from an aborted download" failure mode).

    SHA verification is intentionally NOT done here — it'd require
    reading the whole file every time `atlas model list` runs. Doctor's
    check_model_file can opt into SHA verification when registry has
    a sha256 to compare against (PC-056.1 / PC-058 follow-up).
    """
    path = os.path.join(models_dir, model.model_file)
    try:
        st = os.stat(path)
    except (FileNotFoundError, OSError):
        return False
    return st.st_size > 100 * 1024 * 1024


def installed_size_gb(model: Model, models_dir: str) -> Optional[float]:
    """Return the on-disk size in GB if installed, else None.
    Useful for doctor's storage diagnostics."""
    path = os.path.join(models_dir, model.model_file)
    try:
        return os.stat(path).st_size / (1024 ** 3)
    except (FileNotFoundError, OSError):
        return None


def as_dict(model: Model) -> Dict:
    """Serializer for JSON output."""
    return asdict(model)
