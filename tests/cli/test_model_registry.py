"""Tests for atlas.cli.commands.model_registry (PC-056) + the
PC-055.2 model_recommendations back-compat shim.

Covers:
  - REGISTRY shape + lens_status truth (only 9B is supported today)
  - for_tier / tier_for_model / by_name lookups
  - is_installed / installed_size_gb file-system probe
  - can_install reflects download_url
  - supported_models / models_for_tier filters
  - PC-055.2 shim still resolves the same names + ModelRecommendation
    is now an alias for Model
"""

from atlas.cli.commands import model_registry, model_recommendations


# ---------------------------------------------------------------------------
# Registry shape — locks in the Phase 0 truth
# ---------------------------------------------------------------------------

def test_registry_has_known_qwen_entries():
    """PC-056 shipped with 4 tier presets; PC-056.1 added Q4_K_M and
    Q8_0 variants of the 9B for a total of 6. Adding more is a
    deliberate scope change and should be a separate ticket — flag
    it loudly here."""
    assert len(model_registry.REGISTRY) == 6
    names = {m.name for m in model_registry.REGISTRY}
    assert names == {"Qwen3.5-7B-Q4_K_M",
                     "Qwen3.5-9B-Q4_K_M", "Qwen3.5-9B-Q6_K", "Qwen3.5-9B-Q8_0",
                     "Qwen3.5-14B-Q5_K_M", "Qwen3.5-32B-Q5_K_M"}


def test_only_9b_is_supported_today():
    """The PC-056 architectural conversation surfaced that only the 9B
    has Lens artifacts. If this changes (e.g., PC-058 adds 14B back),
    update here AND the doctor's tier_match check + the docs."""
    supported = model_registry.supported_models()
    assert len(supported) == 1
    assert supported[0].name == "Qwen3.5-9B-Q6_K"


def test_only_9b_quants_are_publicly_installable():
    """PC-056.1: gated entries got download_urls populated (so
    HF_TOKEN-authenticated users CAN install them) but they're flagged
    requires_hf_token. The "publicly installable without auth" set is
    just the three 9B quants."""
    public = [m for m in model_registry.REGISTRY
               if m.can_install and not m.requires_hf_token]
    public_names = {m.name for m in public}
    assert public_names == {"Qwen3.5-9B-Q4_K_M", "Qwen3.5-9B-Q6_K",
                              "Qwen3.5-9B-Q8_0"}


def test_9b_has_verified_download_metadata():
    """Sanity: the 9B entry has a working download URL, a sensible size,
    and the SHA256 we captured from HF's x-linked-etag."""
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    assert m is not None
    assert m.download_url and m.download_url.startswith("https://huggingface.co/")
    assert 6.0 < m.model_size_gb < 8.0  # 6.94 GB observed
    assert m.sha256 is not None and len(m.sha256) == 64


def test_no_artifacts_models_have_explanatory_notes():
    """If lens_status == 'no-artifacts', the user needs to understand
    why. Blank notes would be a UX failure."""
    for m in model_registry.REGISTRY:
        if m.lens_status == "no-artifacts":
            assert m.notes, f"{m.name} has no notes explaining no-artifacts"
            assert "no-artifacts" in m.notes.lower() or \
                   "no lens artifacts" in m.notes.lower() or \
                   "g(x)" in m.notes.lower() or \
                   "silently no-op" in m.notes.lower()


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def test_for_tier_prefers_supported():
    """If a tier has multiple registered models, for_tier picks the
    supported one. Today every tier has only one entry; this guards
    against future breakage when PC-058 adds variants."""
    m = model_registry.for_tier("medium")
    assert m is not None
    assert m.lens_status == "supported"


def test_for_tier_falls_back_to_first_when_no_supported():
    """When no tier-matched entry is supported, return the first match
    (caller renders the warning)."""
    m = model_registry.for_tier("xlarge")
    assert m is not None
    assert m.tier == "xlarge"
    assert m.lens_status == "no-artifacts"


def test_for_tier_unknown_returns_none():
    assert model_registry.for_tier("colossal") is None
    assert model_registry.for_tier("cpu") is None  # no model registered for cpu


def test_tier_for_model_roundtrips():
    """Every registered model's file resolves back to its tier."""
    for m in model_registry.REGISTRY:
        assert model_registry.tier_for_model(m.model_file) == m.tier


def test_tier_for_model_unknown_returns_none():
    assert model_registry.tier_for_model("not-a-real-file.gguf") is None


def test_by_name_roundtrips():
    for m in model_registry.REGISTRY:
        assert model_registry.by_name(m.name) is m


def test_by_name_unknown_returns_none():
    assert model_registry.by_name("Llama-Made-Up") is None


def test_models_for_tier_returns_only_matches():
    """PC-056.1: medium tier now has 3 entries (Q4_K_M, Q6_K, Q8_0)."""
    medium = model_registry.models_for_tier("medium")
    assert all(m.tier == "medium" for m in medium)
    assert len(medium) == 3


# ---------------------------------------------------------------------------
# Install-state probe
# ---------------------------------------------------------------------------

def test_is_installed_false_for_missing_file(tmp_path):
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    assert model_registry.is_installed(m, str(tmp_path)) is False


def test_is_installed_false_for_too_small_file(tmp_path):
    """100 MB sanity threshold guards against aborted-download zero-byte
    files showing as installed."""
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    p = tmp_path / m.model_file
    p.write_bytes(b"x" * 1024)  # 1 KB — way under the 100 MB threshold
    assert model_registry.is_installed(m, str(tmp_path)) is False


def test_is_installed_true_for_large_enough_file(tmp_path):
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    p = tmp_path / m.model_file
    # Use sparse-file trick: seek + write 1 byte → file appears as
    # whatever offset we picked, without actually consuming disk.
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)  # just past 100 MB
        f.write(b"\0")
    assert model_registry.is_installed(m, str(tmp_path)) is True


def test_installed_size_gb_returns_size_when_present(tmp_path):
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    p = tmp_path / m.model_file
    with open(p, "wb") as f:
        f.seek(2 * 1024 ** 3 - 1)  # 2 GB - 1 byte
        f.write(b"\0")
    sz = model_registry.installed_size_gb(m, str(tmp_path))
    assert sz is not None
    assert 1.9 < sz < 2.1


def test_installed_size_gb_returns_none_when_absent(tmp_path):
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    assert model_registry.installed_size_gb(m, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# PC-055.2 back-compat shim
# ---------------------------------------------------------------------------

def test_shim_for_tier_resolves_same_as_registry():
    for tier_name in ("small", "medium", "large", "xlarge"):
        a = model_recommendations.for_tier(tier_name)
        b = model_registry.for_tier(tier_name)
        assert a is b, f"shim and registry disagree on tier {tier_name!r}"


def test_shim_tier_for_model_works():
    assert model_recommendations.tier_for_model("Qwen3.5-9B-Q6_K.gguf") == "medium"


def test_shim_modelrecommendation_is_model_alias():
    """The shim must keep ModelRecommendation pointing at Model so
    isinstance() checks in PC-055.2-era code keep working."""
    assert model_recommendations.ModelRecommendation is model_registry.Model


def test_shim_callers_can_access_old_field_names():
    """PC-055.2 callers do `rec.model_file`, `rec.model_display`,
    `rec.model_size_gb`. The PC-056 Model preserves those exact field
    names so the shim is transparent."""
    rec = model_recommendations.for_tier("medium")
    assert hasattr(rec, "model_file")
    assert hasattr(rec, "model_display")
    assert hasattr(rec, "model_size_gb")
    assert hasattr(rec, "tier")
    # And new fields are also visible (callers can opt in)
    assert hasattr(rec, "lens_status")
    assert hasattr(rec, "download_url")


# ---------------------------------------------------------------------------
# PC-056.1 schema additions: 9B variants, commit-pinned URLs, requires_hf_token
# ---------------------------------------------------------------------------

def test_pc0561_registry_now_has_six_entries():
    """PC-056.1 added Q4_K_M and Q8_0 variants of the 9B. Adding more is
    a deliberate scope change — flag it loudly here."""
    assert len(model_registry.REGISTRY) == 6


def test_pc0561_three_quants_for_9b():
    """All three 9B quants present — Q4_K_M (smaller), Q6_K (default
    supported), Q8_0 (higher quality)."""
    nine_quants = {m.name for m in model_registry.REGISTRY
                    if m.tier == "medium"}
    assert nine_quants == {"Qwen3.5-9B-Q4_K_M", "Qwen3.5-9B-Q6_K",
                            "Qwen3.5-9B-Q8_0"}


def test_pc0561_only_q6k_is_supported_others_unverified():
    """Lens metric tensor was trained on Q6_K specifically. Other quants
    of the same model should mark `unverified` — Lens should structurally
    transfer but the exact (quant, Lens) combo isn't validated."""
    assert model_registry.by_name("Qwen3.5-9B-Q6_K").lens_status == "supported"
    assert model_registry.by_name("Qwen3.5-9B-Q4_K_M").lens_status == "unverified"
    assert model_registry.by_name("Qwen3.5-9B-Q8_0").lens_status == "unverified"


def test_pc0561_for_tier_medium_still_picks_supported_q6k():
    """With multiple medium-tier entries (Q4 unverified, Q6 supported,
    Q8 unverified), for_tier must pick the SUPPORTED one — that's why
    the 'prefer supported' rule was important."""
    assert model_registry.for_tier("medium").name == "Qwen3.5-9B-Q6_K"


def test_pc0561_urls_pinned_to_commit_hash():
    """download_urls must NOT include `/main/` (PC-056 mistake) — they
    must be pinned to a specific commit so SHA256 stays valid even if
    upstream re-uploads with the same filename."""
    for m in model_registry.REGISTRY:
        if m.download_url is None:
            continue
        assert "/main/" not in m.download_url, (
            f"{m.name} URL not commit-pinned: {m.download_url}")
        # Should contain the unsloth Qwen3.5 commit hash
        assert "/3885219b" in m.download_url, (
            f"{m.name} not pinned to expected commit: {m.download_url}")


def test_pc0561_gated_entries_have_url_and_flag():
    """PC-056.1: gated entries got download_url populated (so HF_TOKEN
    users can install) AND requires_hf_token=True (so anonymous users
    get the helpful message before even trying)."""
    for name in ("Qwen3.5-7B-Q4_K_M", "Qwen3.5-14B-Q5_K_M",
                  "Qwen3.5-32B-Q5_K_M"):
        m = model_registry.by_name(name)
        assert m.download_url is not None, (
            f"{name} should now have a populated URL (PC-056.1)")
        assert m.requires_hf_token is True, (
            f"{name} should be flagged requires_hf_token=True")
        assert m.is_gated is True
        # gated still means can_install is True (URL is present); the
        # gate is at install-time, not in the registry property.
        assert m.can_install is True


def test_pc0561_supported_entry_does_not_require_hf_token():
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    assert m.requires_hf_token is False
    assert m.is_gated is False


def test_pc0561_supported_model_has_lens_artifact_files():
    """The supported 9B-Q6_K must declare which files prove its
    'supported' claim — doctor cross-checks against this list."""
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    assert "cost_field.pt" in m.lens_artifact_files
    assert "metric_tensor.pt" in m.lens_artifact_files


# ---------------------------------------------------------------------------
# PC-056.1 SHA verification helpers
# ---------------------------------------------------------------------------

def test_compute_sha256_matches_known_value(tmp_path):
    """Computed hash must equal the known SHA of 'hello world'."""
    p = tmp_path / "h.txt"
    p.write_bytes(b"hello world")
    expected = ("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7a"
                "ce2efcde9")
    assert model_registry.compute_sha256(str(p)) == expected


def test_verify_installed_missing_returns_missing(tmp_path):
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    r = model_registry.verify_installed(m, str(tmp_path))
    assert r["match"] == "missing"
    assert r["installed"] is False
    assert r["actual_sha256"] is None


def test_verify_installed_no_expected_returns_no_expected(tmp_path):
    """Models with sha256=None (e.g., gated entries we couldn't HEAD)
    can't be verified end-to-end. Status should be 'no-expected'."""
    m = model_registry.by_name("Qwen3.5-7B-Q4_K_M")
    p = tmp_path / m.model_file
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    r = model_registry.verify_installed(m, str(tmp_path))
    assert r["match"] == "no-expected"
    assert r["installed"] is True


def test_verify_installed_mismatch_detects_corruption(tmp_path):
    """A file with unexpected contents must report mismatch."""
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    p = tmp_path / m.model_file
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    r = model_registry.verify_installed(m, str(tmp_path))
    assert r["match"] == "mismatch"
    assert r["actual_sha256"] is not None
    assert r["actual_sha256"] != r["expected_sha256"]


# ---------------------------------------------------------------------------
# PC-056.1 Lens artifact resolution + presence
# ---------------------------------------------------------------------------

def test_lens_artifact_dir_for_uses_default_when_unset(tmp_path):
    """Default resolution: <atlas_root>/geometric-lens/geometric_lens/models"""
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    d = model_registry.lens_artifact_dir_for(m, str(tmp_path))
    assert d.endswith("geometric-lens/geometric_lens/models")


def test_lens_artifact_dir_for_returns_none_for_unsupported_model():
    """No expectation = no path — caller should treat None as 'don't check'."""
    m = model_registry.by_name("Qwen3.5-7B-Q4_K_M")  # no-artifacts
    assert model_registry.lens_artifact_dir_for(m, "/some/root") is None


def test_lens_artifact_dir_for_honors_atlas_lens_models_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    m = model_registry.by_name("Qwen3.5-9B-Q6_K")
    d = model_registry.lens_artifact_dir_for(m, "/unused")
    assert d == str(tmp_path)


def test_lens_artifacts_present_ok_when_files_exist(tmp_path):
    """Set up a fake artifact dir with the required .pt files, point
    ATLAS_LENS_MODELS at it, expect ok=True."""
    art_dir = tmp_path / "lens-models"
    art_dir.mkdir()
    (art_dir / "cost_field.pt").write_bytes(b"x")
    (art_dir / "metric_tensor.pt").write_bytes(b"y")
    import os
    os.environ["ATLAS_LENS_MODELS"] = str(art_dir)
    try:
        m = model_registry.by_name("Qwen3.5-9B-Q6_K")
        state = model_registry.lens_artifacts_present(m, str(tmp_path))
        assert state["ok"] is True
        assert state["missing_files"] == []
    finally:
        del os.environ["ATLAS_LENS_MODELS"]


def test_lens_artifacts_present_missing_files_listed(tmp_path):
    """No .pt files at the expected path → both should be in missing_files."""
    import os
    os.environ["ATLAS_LENS_MODELS"] = str(tmp_path / "nonexistent")
    try:
        m = model_registry.by_name("Qwen3.5-9B-Q6_K")
        state = model_registry.lens_artifacts_present(m, str(tmp_path))
        assert state["ok"] is False
        assert "cost_field.pt" in state["missing_files"]
        assert "metric_tensor.pt" in state["missing_files"]
    finally:
        del os.environ["ATLAS_LENS_MODELS"]


def test_lens_artifacts_present_skips_unsupported_models(tmp_path):
    """Models without lens_status='supported' should always return ok=True
    (nothing was claimed; nothing to verify)."""
    m = model_registry.by_name("Qwen3.5-14B-Q5_K_M")  # no-artifacts
    state = model_registry.lens_artifacts_present(m, str(tmp_path))
    assert state["ok"] is True
    assert state["expected_files"] == []
