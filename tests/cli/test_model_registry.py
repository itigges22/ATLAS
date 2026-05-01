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

def test_registry_has_four_qwen_entries():
    """PC-056 ships with the four Qwen3.5 tier presets known from PC-055.
    Adding a fifth or removing one is a deliberate scope change and
    should be a separate ticket — flag it loudly here."""
    assert len(model_registry.REGISTRY) == 4
    names = {m.name for m in model_registry.REGISTRY}
    assert names == {"Qwen3.5-7B-Q4_K_M", "Qwen3.5-9B-Q6_K",
                     "Qwen3.5-14B-Q5_K_M", "Qwen3.5-32B-Q5_K_M"}


def test_only_9b_is_supported_today():
    """The PC-056 architectural conversation surfaced that only the 9B
    has Lens artifacts. If this changes (e.g., PC-058 adds 14B back),
    update here AND the doctor's tier_match check + the docs."""
    supported = model_registry.supported_models()
    assert len(supported) == 1
    assert supported[0].name == "Qwen3.5-9B-Q6_K"


def test_only_9b_is_installable():
    """unsloth's 7B / 14B / 32B repos return HTTP 401. Until that
    changes upstream OR we mirror the GGUFs OR PC-058 produces our own,
    only the 9B has a public download URL."""
    installable = [m for m in model_registry.REGISTRY if m.can_install]
    assert len(installable) == 1
    assert installable[0].name == "Qwen3.5-9B-Q6_K"


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
    medium = model_registry.models_for_tier("medium")
    assert all(m.tier == "medium" for m in medium)
    assert len(medium) == 1


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
