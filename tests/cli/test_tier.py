"""Tests for atlas.cli.commands.tier (PC-055 + PC-055.1 + PC-055.2).

These tests cover the bugs we caught across paranoid passes — keeping
them as regression coverage means the next bug is caught by `pytest`
instead of by re-reading the module. Specifically:

  - multi-GPU pick-max (caught in PC-055 second paranoid pass)
  - vram=0 + has_gpu=True classified as cpu (PC-055 second paranoid pass)
  - install_dir plumbed to disk probe (PC-055.1 first paranoid pass)
  - RAM warn (within 15%) vs fail (past threshold) — semantics check
  - cpu tier returned for no-GPU hosts
  - PC-055.2 split: model fields removed from TierProfile, present on
    ModelRecommendation; reverse lookup tier_for_model works.

Tests are intentionally small + use monkeypatch rather than mock
frameworks. The probe surface is deliberately narrow (4 _read_* helpers)
specifically so it's testable without nvidia-smi or real disks.
"""

import tempfile

import pytest

from atlas.cli.commands import tier, model_recommendations


# ---------------------------------------------------------------------------
# Helpers — build a Probe without touching the host
# ---------------------------------------------------------------------------

def make_probe(**overrides) -> tier.Probe:
    """Default to a comfortable medium-tier host; override per test."""
    defaults = dict(
        has_gpu=True, gpu_name="Test GPU", vram_gb=16.0, gpu_count=1,
        system_ram_gb=32.0, cpu_cores=8, disk_free_gb=100.0,
        platform="linux",
    )
    defaults.update(overrides)
    return tier.Probe(**defaults)


# ---------------------------------------------------------------------------
# classify() — tier selection
# ---------------------------------------------------------------------------

def test_classify_no_gpu_returns_cpu():
    p = make_probe(has_gpu=False, gpu_name=None, vram_gb=0.0, gpu_count=0)
    assert tier.classify(p).tier == "cpu"


def test_classify_has_gpu_but_zero_vram_returns_cpu():
    """Rare nvidia-smi failure mode — device enumerates but mem query fails.
    Without the `or vram_gb <= 0` guard we'd hand full small-tier settings
    to a non-functional GPU. (PC-055 second paranoid pass.)"""
    p = make_probe(has_gpu=True, vram_gb=0.0, gpu_count=1)
    assert tier.classify(p).tier == "cpu"


@pytest.mark.parametrize("vram_gb,expected", [
    (8.0,  "small"),
    (11.9, "small"),
    (12.0, "medium"),
    (16.0, "medium"),
    (19.9, "medium"),
    (20.0, "large"),
    (24.0, "large"),
    (31.9, "large"),
    (32.0, "xlarge"),
    (48.0, "xlarge"),
    (80.0, "xlarge"),
])
def test_classify_band_breakpoints(vram_gb, expected):
    """Each tier band is half-open [min, max). Exact-breakpoint values
    are the highest-risk for off-by-one bugs."""
    p = make_probe(vram_gb=vram_gb)
    assert tier.classify(p).tier == expected


def test_classify_below_smallest_band_returns_warned_small():
    """4 GB GPU is below the small tier's 8 GB minimum. We return a
    cloned small tier with a warning baked into `notes` so the user
    sees what to upgrade to. (Documented behavior, not a bug.)"""
    p = make_probe(vram_gb=4.0)
    t = tier.classify(p)
    assert t.tier == "small"
    assert "below" in t.notes.lower()
    assert "OOM" in t.notes


# ---------------------------------------------------------------------------
# Multi-GPU pick-max (regression test for PC-055 second paranoid pass)
# ---------------------------------------------------------------------------

def test_read_nvidia_smi_picks_max_vram(monkeypatch):
    """Hybrid-graphics workstation: iGPU 8 GB enumerated first + A6000 48 GB
    second. _read_nvidia_smi must return the A6000 (max VRAM), not the
    iGPU (first). Pre-fix: classified as small (took iGPU).
    Post-fix: classified as xlarge (took A6000)."""
    fake_stdout = "Intel iGPU, 8192\nNVIDIA RTX A6000, 49152\n"

    class FakeRun:
        returncode = 0
        stdout = fake_stdout

    monkeypatch.setattr(tier.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(tier.subprocess, "run",
                        lambda *a, **kw: FakeRun())
    has_gpu, name, vram, count = tier._read_nvidia_smi()
    assert has_gpu is True
    assert name == "NVIDIA RTX A6000"
    assert vram == pytest.approx(48.0, rel=0.01)
    assert count == 2


def test_read_nvidia_smi_no_smi_binary(monkeypatch):
    monkeypatch.setattr(tier.shutil, "which", lambda _: None)
    has_gpu, name, vram, count = tier._read_nvidia_smi()
    assert (has_gpu, name, vram, count) == (False, None, 0.0, 0)


# ---------------------------------------------------------------------------
# evaluate_constraints — RAM warn vs fail semantics
# ---------------------------------------------------------------------------

def _check(checks, name):
    return next(c for c in checks if c.name == name)


def test_constraints_all_pass_on_comfortable_host():
    p = make_probe(vram_gb=16.0, system_ram_gb=32.0, cpu_cores=8,
                   disk_free_gb=100.0)
    t = tier.by_name("medium")
    checks = tier.evaluate_constraints(p, t)
    assert tier.overall_status(checks) == "pass"
    assert all(c.status == "pass" for c in checks)


def test_constraints_ram_warn_within_15pct():
    """medium tier needs 16 GB RAM. 14 GB = 87.5% of 16 → within 15% → warn."""
    p = make_probe(system_ram_gb=14.0)
    t = tier.by_name("medium")
    c = _check(tier.evaluate_constraints(p, t), "system_ram")
    assert c.status == "warn"


def test_constraints_ram_fail_past_threshold():
    """medium tier needs 16 GB RAM. 8 GB = 50% → past 15% threshold → fail."""
    p = make_probe(system_ram_gb=8.0)
    t = tier.by_name("medium")
    c = _check(tier.evaluate_constraints(p, t), "system_ram")
    assert c.status == "fail"


def test_constraints_cpu_shortage_is_warn_not_fail():
    """CPU shortage = slow, not OOM. Always warn (never fail).
    Documented status semantics (PC-055.1)."""
    p = make_probe(cpu_cores=2)  # medium needs 4
    t = tier.by_name("medium")
    c = _check(tier.evaluate_constraints(p, t), "cpu_cores")
    assert c.status == "warn"
    # Even a brutal CPU shortage stays warn — confirms intent, not threshold luck.
    p2 = make_probe(cpu_cores=1)
    t2 = tier.by_name("xlarge")  # needs 16
    c2 = _check(tier.evaluate_constraints(p2, t2), "cpu_cores")
    assert c2.status == "warn"


def test_overall_status_picks_worst_case():
    """RAM=warn + disk=fail must reduce to fail (not warn)."""
    p = make_probe(system_ram_gb=14.0, disk_free_gb=5.0)
    t = tier.by_name("medium")
    assert tier.overall_status(tier.evaluate_constraints(p, t)) == "fail"


# ---------------------------------------------------------------------------
# install_dir plumbing (regression test for PC-055.1 paranoid pass)
# ---------------------------------------------------------------------------

def test_probe_install_dir_used_for_disk_check(monkeypatch):
    """If install_dir is provided + exists, disk_free is measured against
    it (not /). Pre-fix: doctor's tier_constraints called probe() without
    install_dir, so users with /opt/atlas on /data mount got misleading
    disk_free against /."""
    captured = {}

    def fake_disk_usage(path):
        captured["path"] = path
        # Return any plausible value — we only care about the path.
        class S: free = 50 * (1024 ** 3)
        return S()

    monkeypatch.setattr(tier.shutil, "disk_usage", fake_disk_usage)
    with tempfile.TemporaryDirectory() as td:
        tier.probe(install_dir=td)
        assert captured["path"] == td


def test_probe_install_dir_falls_back_when_path_missing(monkeypatch):
    """A non-existent install_dir must fall back to / rather than crash."""
    captured = {}

    def fake_disk_usage(path):
        captured["path"] = path
        class S: free = 50 * (1024 ** 3)
        return S()

    monkeypatch.setattr(tier.shutil, "disk_usage", fake_disk_usage)
    tier.probe(install_dir="/nonexistent/path/does/not/exist")
    assert captured["path"] == "/"


# ---------------------------------------------------------------------------
# PC-055.2 split — TierProfile is hardware-only, model lives elsewhere
# ---------------------------------------------------------------------------

def test_tier_profile_has_no_model_fields():
    """The split removed model_file/model_display/model_size_gb from
    TierProfile. If they reappear, someone's regressing the layering."""
    t = tier.by_name("medium")
    for field in ("model_file", "model_display", "model_size_gb"):
        assert not hasattr(t, field), (
            f"TierProfile.{field} should live in model_recommendations, "
            "not on the tier (PC-055.2 split).")


def test_tier_env_vars_excludes_model_keys():
    """Tier renders runtime knobs only; model env vars come from
    ModelRecommendation.env_vars()."""
    env = tier.by_name("medium").env_vars()
    assert "ATLAS_MODEL_FILE" not in env
    assert "ATLAS_MODEL_NAME" not in env
    assert "ATLAS_CTX_SIZE" in env
    assert "PARALLEL_SLOTS" in env


def test_model_recommendations_for_each_tier():
    """Every tier should have a corresponding model recommendation —
    rendering breaks otherwise."""
    for t in tier.TIERS:
        rec = model_recommendations.for_tier(t.tier)
        assert rec is not None, f"missing recommendation for tier {t.tier!r}"
        assert rec.tier == t.tier


def test_model_recommendation_reverse_lookup_roundtrip():
    """Forward + reverse lookups must agree — used by doctor.check_tier_match."""
    for tier_name in ("small", "medium", "large", "xlarge"):
        rec = model_recommendations.for_tier(tier_name)
        assert model_recommendations.tier_for_model(rec.model_file) == tier_name


def test_model_recommendation_unknown_returns_none():
    assert model_recommendations.for_tier("nonexistent") is None
    assert model_recommendations.tier_for_model("not-a-real.gguf") is None
