"""Tests for atlas.cli.commands.model (PC-056) — the CLI command.

Network-touching paths (the actual urllib download in install) are
out of scope here — those are integration tests run against a fresh
VM. This file covers everything that runs without a network or with
mocked filesystem state:

  - list filters (--tier, --installed, --lens-supported)
  - list JSON shape
  - recommend on a host classified to a supported tier
  - install --dry-run renders correct preview
  - install refuses no-artifacts without --no-lens (the safety gate)
  - install refuses gated upstream (download_url is None)
  - install on unknown name returns 1
  - install refuses overwrite without --yes
  - remove without --yes refuses; with --yes deletes
  - remove on missing model is a no-op success
"""

import json

from atlas.cli.commands import model, tier


# ---------------------------------------------------------------------------
# `atlas model list`
# ---------------------------------------------------------------------------

def test_list_default_shows_all(capsys):
    rc = model.main(["list", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in ("Qwen3.5-7B-Q4_K_M", "Qwen3.5-9B-Q6_K",
                 "Qwen3.5-14B-Q5_K_M", "Qwen3.5-32B-Q5_K_M"):
        assert name in out


def test_list_tier_filter(capsys):
    rc = model.main(["list", "--tier", "medium", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Qwen3.5-9B-Q6_K" in out
    assert "Qwen3.5-7B-Q4_K_M" not in out
    assert "Qwen3.5-32B-Q5_K_M" not in out


def test_list_lens_supported_filter_returns_only_9b(capsys):
    rc = model.main(["list", "--lens-supported", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Qwen3.5-9B-Q6_K" in out
    assert "Qwen3.5-14B-Q5_K_M" not in out


def test_list_installed_filter_returns_empty_on_missing_dir(tmp_path, capsys):
    rc = model.main(["list", "--installed", "--models-dir", str(tmp_path),
                     "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no models match these filters" in out


def test_list_installed_picks_up_present_file(tmp_path, capsys):
    """is_installed sees the gguf file → list --installed shows it."""
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf"
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)  # > 100 MB sanity threshold
        f.write(b"\0")
    rc = model.main(["list", "--installed", "--models-dir", str(tmp_path),
                     "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Qwen3.5-9B-Q6_K" in out


def test_list_json_structure(tmp_path, capsys):
    rc = model.main(["list", "--json", "--models-dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_dir"] == str(tmp_path)
    assert isinstance(payload["models"], list)
    # PC-056.1: 6 entries (added Q4_K_M and Q8_0 9B variants).
    assert len(payload["models"]) == 6
    nine = next(m for m in payload["models"] if m["name"] == "Qwen3.5-9B-Q6_K")
    assert nine["lens_status"] == "supported"
    assert nine["installed"] is False
    assert nine["installed_size_gb"] is None
    assert nine["download_url"].startswith("https://huggingface.co/")


# ---------------------------------------------------------------------------
# `atlas model recommend`
# ---------------------------------------------------------------------------

def test_recommend_on_medium_tier_returns_supported(monkeypatch, capsys):
    """Mock the tier classifier so the test doesn't depend on the host's
    actual GPU. Classify as medium → recommend should print 9B as the
    Lens-supported tier-default."""
    fake = tier.Probe(has_gpu=True, gpu_name="Test GPU", vram_gb=16.0,
                      gpu_count=1, system_ram_gb=32.0, cpu_cores=8,
                      disk_free_gb=100.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: fake)

    rc = model.main(["recommend", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Detected tier: medium" in out
    assert "Qwen3.5-9B-Q6_K" in out
    assert "Lens supported" in out


def test_recommend_on_xlarge_surfaces_fallback_to_9b(monkeypatch, capsys):
    """On xlarge hardware, the tier-default (32B) is no-artifacts, so
    recommend should surface the 9B as the supported fallback."""
    fake = tier.Probe(has_gpu=True, gpu_name="A100", vram_gb=80.0,
                      gpu_count=1, system_ram_gb=128.0, cpu_cores=32,
                      disk_free_gb=500.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: fake)

    rc = model.main(["recommend", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Detected tier: xlarge" in out
    assert "Qwen3.5-32B-Q5_K_M" in out
    assert "no Lens artifacts" in out
    assert "Recommended fallback" in out
    assert "Qwen3.5-9B-Q6_K" in out


def test_recommend_json_includes_fallback_when_default_unsupported(monkeypatch, capsys):
    fake = tier.Probe(has_gpu=True, gpu_name="A100", vram_gb=80.0,
                      gpu_count=1, system_ram_gb=128.0, cpu_cores=32,
                      disk_free_gb=500.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: fake)

    rc = model.main(["recommend", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host_tier"] == "xlarge"
    assert payload["recommendation"]["name"] == "Qwen3.5-32B-Q5_K_M"
    assert payload["fallback"] is not None
    assert payload["fallback"]["name"] == "Qwen3.5-9B-Q6_K"


def test_recommend_json_no_fallback_when_default_supported(monkeypatch, capsys):
    fake = tier.Probe(has_gpu=True, gpu_name="Test", vram_gb=16.0,
                      gpu_count=1, system_ram_gb=32.0, cpu_cores=8,
                      disk_free_gb=100.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: fake)

    rc = model.main(["recommend", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host_tier"] == "medium"
    assert payload["recommendation"]["name"] == "Qwen3.5-9B-Q6_K"
    assert payload["fallback"] is None


# ---------------------------------------------------------------------------
# `atlas model install` — safety gates (no network paths covered)
# ---------------------------------------------------------------------------

def test_install_unknown_name_returns_1(capsys):
    rc = model.main(["install", "Llama-Made-Up", "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Unknown model" in out


def test_install_no_artifacts_refused_without_no_lens_flag(tmp_path, capsys):
    """Safety gate: refuse to install a model with no Lens artifacts
    unless the user explicitly passes --no-lens to acknowledge G(x)
    will silently no-op."""
    rc = model.main(["install", "Qwen3.5-14B-Q5_K_M", "--dry-run",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Refusing" in out
    assert "no-artifacts" in out
    assert "--no-lens" in out


def test_install_no_artifacts_with_no_lens_then_blocked_by_hf_token(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    """User passes --no-lens for the gated 14B but the upstream
    requires HF_TOKEN — install must still refuse, with the helpful
    HF_TOKEN message (PC-056.1)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    rc = model.main(["install", "Qwen3.5-14B-Q5_K_M", "--no-lens",
                     "--dry-run", "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "HF_TOKEN" in out
    assert "huggingface.co/settings/tokens" in out


def test_install_dry_run_for_supported_model_prints_url(tmp_path, capsys):
    rc = model.main(["install", "Qwen3.5-9B-Q6_K", "--dry-run",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "huggingface.co/unsloth/Qwen3.5-9B-GGUF" in out
    assert str(tmp_path) in out
    assert "SHA256" in out  # 9B has a verified sha256


def test_install_refuses_overwrite_without_yes(tmp_path, capsys):
    """If the target file already exists, refuse without --yes."""
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf"
    with open(p, "wb") as f:
        f.seek(7 * 1024 ** 3)  # 7 GB sparse file
        f.write(b"\0")
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "already exists" in out
    assert "--yes" in out


# ---------------------------------------------------------------------------
# `atlas model remove`
# ---------------------------------------------------------------------------

def test_remove_unknown_name_returns_1(capsys):
    rc = model.main(["remove", "Llama-Made-Up", "--no-color"])
    assert rc == 1


def test_remove_missing_file_is_idempotent_zero(tmp_path, capsys):
    """If the model isn't installed, remove is a no-op success."""
    rc = model.main(["remove", "Qwen3.5-9B-Q6_K", "--yes",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0


def test_remove_without_yes_refuses(tmp_path, capsys):
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf"
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    rc = model.main(["remove", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    assert p.exists(), "remove without --yes must not delete the file"


def test_remove_with_yes_deletes(tmp_path, capsys):
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf"
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    rc = model.main(["remove", "Qwen3.5-9B-Q6_K", "--yes",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0
    assert not p.exists(), "remove --yes must delete the file"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_models_dir_env_var_honored(tmp_path, monkeypatch, capsys):
    """ATLAS_MODELS_DIR env var should take precedence when --models-dir
    isn't given."""
    monkeypatch.setenv("ATLAS_MODELS_DIR", str(tmp_path))
    rc = model.main(["list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_dir"] == str(tmp_path)


def test_models_dir_flag_overrides_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ATLAS_MODELS_DIR", "/some/other/place")
    rc = model.main(["list", "--json", "--models-dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_dir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# PC-056.1 — install hardening: HF_TOKEN gate, list rendering with auth
# ---------------------------------------------------------------------------

def test_install_gated_without_hf_token_refuses_with_helpful_msg(tmp_path,
                                                                   monkeypatch,
                                                                   capsys):
    """Gated entries (requires_hf_token=True) refuse early when HF_TOKEN
    is not in the env, with the helpful 'set HF_TOKEN' message."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    rc = model.main(["install", "Qwen3.5-14B-Q5_K_M", "--no-lens",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "requires HuggingFace authentication" in out
    assert "HF_TOKEN" in out
    assert "huggingface.co/settings/tokens" in out


def test_install_gated_with_hf_token_proceeds_to_dry_run(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """With HF_TOKEN set, the gated check passes — dry-run prints the URL."""
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_test_token")
    rc = model.main(["install", "Qwen3.5-14B-Q5_K_M", "--no-lens",
                     "--dry-run", "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "huggingface.co/unsloth/Qwen3.5-14B-GGUF" in out


def test_install_alt_token_env_var_also_honored(tmp_path, monkeypatch, capsys):
    """HUGGING_FACE_HUB_TOKEN (HF SDK alt spelling) should work too."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_dummy_alt")
    rc = model.main(["install", "Qwen3.5-14B-Q5_K_M", "--no-lens",
                     "--dry-run", "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0


def test_list_renders_requires_hf_token_marker_without_token(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    rc = model.main(["list", "--tier", "small", "--models-dir", str(tmp_path),
                     "--no-color"])
    assert rc == 0
    assert "requires HF_TOKEN" in capsys.readouterr().out


def test_list_renders_token_present_marker_with_token(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    rc = model.main(["list", "--tier", "small", "--models-dir", str(tmp_path),
                     "--no-color"])
    assert rc == 0
    assert "HF_TOKEN present" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# PC-056.1 — atlas model verify subcommand
# ---------------------------------------------------------------------------

def test_verify_no_installed_models_returns_0(tmp_path, capsys):
    rc = model.main(["verify", "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0
    assert "No installed models" in capsys.readouterr().out


def test_verify_unknown_name_returns_1(capsys):
    rc = model.main(["verify", "Llama-Made-Up", "--no-color"])
    assert rc == 1


def test_verify_corrupted_file_detects_mismatch(tmp_path, capsys):
    """Sparse file with wrong contents → SHA mismatch → exit 1 + helpful
    message about re-installing."""
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf"
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    rc = model.main(["verify", "Qwen3.5-9B-Q6_K", "--models-dir", str(tmp_path),
                     "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert "atlas model install" in out


def test_verify_no_expected_sha_reports_skipped(tmp_path, capsys):
    """A file that's installed but registry has no expected SHA → status
    'no-expected', exit 0 (we can't tell if it's corrupt or not)."""
    p = tmp_path / "Qwen3.5-7B-Q4_K_M.gguf"  # registry sha256=None
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    rc = model.main(["verify", "Qwen3.5-7B-Q4_K_M", "--models-dir",
                     str(tmp_path), "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no expected SHA256" in out


def test_verify_json_includes_per_model_results(tmp_path, capsys):
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf"
    with open(p, "wb") as f:
        f.seek(101 * 1024 * 1024)
        f.write(b"\0")
    rc = model.main(["verify", "Qwen3.5-9B-Q6_K", "--models-dir", str(tmp_path),
                     "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_dir"] == str(tmp_path)
    assert payload["any_mismatch"] is True  # corrupted file
    assert len(payload["results"]) == 1
    r = payload["results"][0]
    assert r["name"] == "Qwen3.5-9B-Q6_K"
    assert r["match"] == "mismatch"
    assert rc == 1
