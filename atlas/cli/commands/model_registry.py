"""atlas model registry — known models + their lens artifact status
(PC-056, hardened in PC-056.1).

Single source of truth for "which models can ATLAS run end-to-end?"

This module is the upgrade-in-place of PC-055.2's
`model_recommendations.py` stub. It preserves the stable public API
(`for_tier`, `tier_for_model`, the Model record's name-compat fields)
that doctor + tier callers depend on, while extending each entry with
download metadata + the critical `lens_status` field that captures the
key truth surfaced during PC-056 scoping:

    Most "supported" tier presets in PC-055 are aspirational. Only the
    9B Q6_K model has actual Lens artifacts (metric tensor + embeddings).
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
  before invoking `atlas model install`. **PC-056.1: URLs are pinned
  to a specific HF commit hash** (e.g.,
  `resolve/3885219b…/Qwen3.5-9B-Q6_K.gguf`) instead of `resolve/main/`,
  so the registered SHA256 stays valid even if upstream re-uploads
  with the same filename.
- `sha256` is the HuggingFace `x-linked-etag` value (the content-addressed
  storage hash) where available. **PC-056.1: actually verified during
  install** (was print-only in PC-056). Verifies download integrity,
  not provenance. Stronger provenance verification is PC-060 territory.
- `lens_status` values:
    "supported"   — metric tensor + embeddings present in repo, has
                    been validated end-to-end against this exact quant
    "no-artifacts" — model exists but Lens won't score it (no tensor
                    trained at all)
    "unverified"  — has artifacts that should structurally apply (e.g.,
                    different quant of the same model family) but the
                    exact combination hasn't been validated
- `lens_artifact_dir` (PC-056.1): Optional override for where this
  model's Lens artifacts live. None = use the global
  ATLAS_LENS_MODELS dir (current single-model layout). When PC-058's
  per-model training pipeline lands, each entry will populate this
  with `geometric-lens/geometric_lens/models/<model_name>/` or similar.
- `lens_artifact_files` (PC-056.1): list of files that must exist in
  the artifact dir for `lens_status: supported` to be honest. Doctor's
  tier_match cross-checks this — if a model claims supported but
  artifacts are missing, that's a config drift that should warn.

PC-056.1 install hardening (relative to PC-056):
- SHA256 enforced at download time (delete file + exit 1 on mismatch)
- Resume from .part file (Range: bytes=N-, append-write)
- HF_TOKEN env var honored (Authorization: Bearer header)
- New `atlas model verify` command for post-install integrity check
"""

import hashlib
import os
from dataclasses import dataclass, asdict, field
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
    download_url: Optional[str] = None   # None = no known URL at all
    sha256: Optional[str] = None         # content hash (HF x-linked-etag) when known
    license: Optional[str] = None        # SPDX-ish identifier (Apache-2.0, etc.)
    # PC-056.1: True for upstream URLs that 401 anonymously. install
    # path checks HF_TOKEN before attempting. URL is still populated
    # so authenticated users can install; sha256 will typically be None
    # for these since we can't HEAD them anonymously to capture it.
    requires_hf_token: bool = False
    # PC-056.1: where this model's Lens artifacts live. None = use the
    # global ATLAS_LENS_MODELS dir (current single-model layout). PC-058
    # per-model training will populate this with model-specific subdirs.
    lens_artifact_dir: Optional[str] = None
    # PC-056.1: files that must exist in the artifact dir for the
    # `supported` claim to be honest. doctor.check_tier_match
    # cross-checks this — registered "supported" + missing files = warn.
    lens_artifact_files: List[str] = field(default_factory=list)
    notes: str = ""

    def env_vars(self) -> Dict[str, str]:
        """The .env keys the wizard / installer would write for this model."""
        return {
            "ATLAS_MODEL_FILE": self.model_file,
            "ATLAS_MODEL_NAME": self.model_file.rsplit(".", 1)[0],
        }

    @property
    def can_install(self) -> bool:
        """True when we have *some* URL to attempt. Gated upstreams will
        still 401 without HF_TOKEN — that's a separate gate at runtime
        (`requires_hf_token` + `_hf_token()` check), not a registry
        property. Returns False only for entries with no URL at all
        (e.g., a model whose upstream has been deleted)."""
        return self.download_url is not None

    @property
    def is_gated(self) -> bool:
        """True for entries whose upstream returns 401 without auth.
        Now backed by an explicit field rather than 'no URL' so we
        can populate URLs for authenticated users while still steering
        anonymous users to the helpful message."""
        return self.requires_hf_token


# PC-056.1: pin all unsloth/Qwen3.5-* URLs to this commit so a future
# upstream re-upload can't silently invalidate our recorded SHA256s.
# Verified 2026-05-01 via HF HEAD x-repo-commit header.
_UNSLOTH_QWEN35_COMMIT = "3885219b6810b007914f3a7950a8d1b469d598a5"

def _unsloth_qwen35_url(repo: str, file: str) -> str:
    return (f"https://huggingface.co/unsloth/{repo}/resolve/"
            f"{_UNSLOTH_QWEN35_COMMIT}/{file}")


# Single source of truth. Order: by tier (cpu → xlarge), then by quant.
#
# Truthful state today:
#   - 9B Q6_K: SUPPORTED. Public unsloth/Qwen3.5-9B-GGUF, trained metric
#     tensor + embeddings in geometric-lens/geometric_lens/models/.
#   - 9B Q4_K_M / Q8_0: UNVERIFIED. Same model family, different quant —
#     embedding space is structurally similar so the metric tensor
#     should transfer, but the exact (model, quant) combo hasn't been
#     validated against pass/fail labels. PC-058 will close this.
#   - 7B / 14B / 32B: NO-ARTIFACTS. Upstream repos return HTTP 401
#     (gated). Setting HF_TOKEN may unlock the download path; even then,
#     no Lens artifacts trained. Listed so users know what's missing.
#
# Adding more models / variants = a PC-058 build run + PC-059 PR.
REGISTRY: List[Model] = [
    Model(
        name="Qwen3.5-7B-Q4_K_M",
        tier="small",
        model_file="Qwen3.5-7B-Q4_K_M.gguf",
        model_display="Qwen3.5 7B (Q4_K_M)",
        model_size_gb=4.4,
        lens_status="no-artifacts",
        # PC-056.1: URL populated so HF_TOKEN-authenticated users can
        # install; sha256 is None (we can't HEAD anonymously to capture).
        download_url=_unsloth_qwen35_url("Qwen3.5-7B-GGUF",
                                          "Qwen3.5-7B-Q4_K_M.gguf"),
        sha256=None,
        license="Apache-2.0",
        requires_hf_token=True,
        notes="Upstream repo unsloth/Qwen3.5-7B-GGUF is gated "
              "(HTTP 401 anonymous). Set HF_TOKEN in env to authenticate "
              "and unlock the download path. Even with auth, no Lens "
              "artifacts trained for this model — will install as raw "
              "llama.cpp model only and G(x) verification will silently "
              "no-op (--no-lens to acknowledge). See PC-058 roadmap.",
    ),
    Model(
        name="Qwen3.5-9B-Q4_K_M",
        tier="medium",
        model_file="Qwen3.5-9B-Q4_K_M.gguf",
        model_display="Qwen3.5 9B (Q4_K_M)",
        # PC-056.1 verified 2026-05-01: Content-Length = 5680522464 bytes.
        model_size_gb=5.29,
        lens_status="unverified",
        download_url=_unsloth_qwen35_url("Qwen3.5-9B-GGUF",
                                          "Qwen3.5-9B-Q4_K_M.gguf"),
        sha256="03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        license="Apache-2.0",
        notes="Smaller-than-Q6 9B variant. Uses the same Lens artifacts "
              "as the Q6 (different quant of the same model family — "
              "embedding space is structurally similar). Quality is "
              "lower than Q6_K; should be measurably degraded but "
              "still functional. Marked `unverified` because the "
              "exact (Q4_K_M, Lens) combo hasn't been validated end-"
              "to-end. PC-058 will close this.",
    ),
    Model(
        name="Qwen3.5-9B-Q6_K",
        tier="medium",
        model_file="Qwen3.5-9B-Q6_K.gguf",
        model_display="Qwen3.5 9B (Q6_K)",
        # PC-056 verified 2026-05-01: Content-Length = 7458301152 bytes.
        model_size_gb=6.94,
        lens_status="supported",
        # PC-056.1: pinned to commit hash (was resolve/main/...).
        download_url=_unsloth_qwen35_url("Qwen3.5-9B-GGUF",
                                          "Qwen3.5-9B-Q6_K.gguf"),
        # HuggingFace x-linked-etag (content-addressed storage SHA256).
        # Verifies download integrity, not provenance.
        sha256="91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59",
        license="Apache-2.0",
        # PC-056.1: declare the artifact files so doctor can cross-check.
        # lens_artifact_dir=None means "use the global ATLAS_LENS_MODELS
        # dir" — current single-model layout.
        lens_artifact_files=["cost_field.pt", "metric_tensor.pt"],
        notes="ATLAS development target. Lens artifacts trained and "
              "shipped in the repo (cost_field.pt + metric_tensor.pt). "
              "End-to-end supported.",
    ),
    Model(
        name="Qwen3.5-9B-Q8_0",
        tier="medium",
        model_file="Qwen3.5-9B-Q8_0.gguf",
        model_display="Qwen3.5 9B (Q8_0)",
        # PC-056.1 verified 2026-05-01: Content-Length = 9527502048 bytes.
        model_size_gb=8.87,
        lens_status="unverified",
        download_url=_unsloth_qwen35_url("Qwen3.5-9B-GGUF",
                                          "Qwen3.5-9B-Q8_0.gguf"),
        sha256="809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4",
        license="Apache-2.0",
        notes="Higher-quality 9B variant for hosts with 24+ GB VRAM. "
              "Uses the same Lens artifacts as Q6_K (different quant "
              "of the same model family). Quality is higher than Q6_K "
              "but the exact (Q8_0, Lens) combo hasn't been validated "
              "end-to-end. Marked `unverified` until PC-058 closes that.",
    ),
    Model(
        name="Qwen3.5-14B-Q5_K_M",
        tier="large",
        model_file="Qwen3.5-14B-Q5_K_M.gguf",
        model_display="Qwen3.5 14B (Q5_K_M)",
        model_size_gb=10.5,
        lens_status="no-artifacts",
        download_url=_unsloth_qwen35_url("Qwen3.5-14B-GGUF",
                                          "Qwen3.5-14B-Q5_K_M.gguf"),
        sha256=None,
        license="Apache-2.0",
        requires_hf_token=True,
        notes="Upstream repo unsloth/Qwen3.5-14B-GGUF is gated "
              "(HTTP 401 anonymous). Set HF_TOKEN in env to authenticate "
              "and unlock the download path. Tested in past ATLAS work "
              "but the trained Lens artifacts have been removed from the "
              "repo. Even with auth, will install as raw llama.cpp model "
              "only — G(x) verification will silently no-op (--no-lens "
              "to acknowledge). See PC-058 roadmap to retrain.",
    ),
    Model(
        name="Qwen3.5-32B-Q5_K_M",
        tier="xlarge",
        model_file="Qwen3.5-32B-Q5_K_M.gguf",
        model_display="Qwen3.5 32B (Q5_K_M)",
        model_size_gb=23.0,
        lens_status="no-artifacts",
        download_url=_unsloth_qwen35_url("Qwen3.5-32B-GGUF",
                                          "Qwen3.5-32B-Q5_K_M.gguf"),
        sha256=None,
        license="Apache-2.0",
        requires_hf_token=True,
        notes="Upstream repo unsloth/Qwen3.5-32B-GGUF is gated "
              "(HTTP 401 anonymous). Set HF_TOKEN in env to authenticate "
              "and unlock the download path. No Lens artifacts trained "
              "for this model. Even with auth, will install as raw "
              "llama.cpp model only — G(x) verification will silently "
              "no-op (--no-lens to acknowledge). See PC-058 roadmap.",
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


# ---------------------------------------------------------------------------
# SHA verification (PC-056.1)
# ---------------------------------------------------------------------------

def compute_sha256(path: str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Stream-hash a file and return the lowercase-hex SHA256.

    4 MiB chunks balance throughput vs. memory — for a 7 GB model file
    this reads ~1750 chunks total. Used by both install (verify after
    download) and `atlas model verify` (post-install integrity check).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def verify_installed(model: Model, models_dir: str) -> Dict:
    """Compute the installed file's SHA256 and compare to the registered
    one. Returns a dict suitable for JSON / human display:

        {
          "installed": bool,                # is the file even there?
          "actual_sha256": str | None,      # what we computed
          "expected_sha256": str | None,    # from registry
          "match": "ok" | "mismatch" | "no-expected" | "missing",
          "actual_size_gb": float | None,
        }

    `match` semantics:
      - "ok"          : file exists, expected SHA known, hashes match
      - "mismatch"    : file exists, expected SHA known, hashes differ
      - "no-expected" : file exists but registry has no SHA to compare
      - "missing"     : no file on disk
    """
    path = os.path.join(models_dir, model.model_file)
    out: Dict = {"installed": False, "actual_sha256": None,
                 "expected_sha256": model.sha256, "match": "missing",
                 "actual_size_gb": None}
    if not os.path.exists(path):
        return out
    out["installed"] = True
    try:
        out["actual_size_gb"] = os.stat(path).st_size / (1024 ** 3)
    except OSError:
        return out
    if model.sha256 is None:
        out["match"] = "no-expected"
        return out
    out["actual_sha256"] = compute_sha256(path)
    out["match"] = "ok" if out["actual_sha256"] == model.sha256 else "mismatch"
    return out


# ---------------------------------------------------------------------------
# Lens artifact path resolution (PC-056.1)
# ---------------------------------------------------------------------------

def lens_artifact_dir_for(model: Model, atlas_root: str) -> Optional[str]:
    """Resolve where this model's Lens artifacts should live.

    Resolution:
      1. model.lens_artifact_dir if set (PC-058 future per-model layout)
      2. else: ATLAS_LENS_MODELS env var if set
      3. else: <atlas_root>/geometric-lens/geometric_lens/models/

    Returns the resolved absolute path, or None for models that don't
    claim Lens support (lens_status != "supported"). Callers should
    treat None as "no expectation, don't check."
    """
    if model.lens_status != "supported":
        return None
    if model.lens_artifact_dir:
        if os.path.isabs(model.lens_artifact_dir):
            return model.lens_artifact_dir
        return os.path.normpath(os.path.join(atlas_root,
                                              model.lens_artifact_dir))
    env = os.environ.get("ATLAS_LENS_MODELS")
    if env:
        return env if os.path.isabs(env) else \
            os.path.normpath(os.path.join(atlas_root, env))
    return os.path.normpath(os.path.join(
        atlas_root, "geometric-lens", "geometric_lens", "models"))


def lens_artifacts_present(model: Model, atlas_root: str) -> Dict:
    """Check whether the Lens artifact files claimed by `model` actually
    exist in the resolved artifact dir. Returns:

        {
          "expected_dir": str | None,       # None for non-supported models
          "expected_files": List[str],
          "missing_files": List[str],       # subset of expected_files
          "ok": bool,                       # True iff all expected exist
        }

    For models where lens_status != "supported" (or no expected files),
    returns ok=True with an empty missing_files (nothing was claimed).
    """
    out: Dict = {"expected_dir": None, "expected_files": [],
                 "missing_files": [], "ok": True}
    if model.lens_status != "supported" or not model.lens_artifact_files:
        return out
    out["expected_dir"] = lens_artifact_dir_for(model, atlas_root)
    out["expected_files"] = list(model.lens_artifact_files)
    if out["expected_dir"] is None:
        return out
    out["missing_files"] = [f for f in out["expected_files"]
                             if not os.path.exists(
                                 os.path.join(out["expected_dir"], f))]
    out["ok"] = len(out["missing_files"]) == 0
    return out
