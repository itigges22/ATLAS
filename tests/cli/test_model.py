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
from typing import Optional

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


# ---------------------------------------------------------------------------
# PC-056.2 — concurrent install lock (item A)
# ---------------------------------------------------------------------------

def test_install_lock_acquire_and_release(tmp_path):
    """First acquire succeeds; second on the same target fails until
    the first releases."""
    target = str(tmp_path / "foo.gguf.part")
    lock1 = model._acquire_install_lock(target, color=False)
    assert lock1 is not None
    lock2 = model._acquire_install_lock(target, color=False)
    assert lock2 is None, "concurrent acquire should refuse"
    model._release_install_lock(lock1)
    lock3 = model._acquire_install_lock(target, color=False)
    assert lock3 is not None, "acquire after release should succeed"
    model._release_install_lock(lock3)


def test_install_lock_stale_pid_reclaimed(tmp_path):
    """A lock file owned by a dead PID is reclaimed automatically."""
    target = str(tmp_path / "foo.gguf.part")
    lock_path = target + ".lock"
    # PID 99999999 almost certainly doesn't exist on this system
    with open(lock_path, "w") as f:
        f.write("99999999\n0\n")
    lock = model._acquire_install_lock(target, color=False)
    assert lock is not None, "stale lock should be reclaimed"
    model._release_install_lock(lock)


def test_install_lock_unparseable_lock_reclaimed(tmp_path):
    """A lock file with garbage contents should also be reclaimed —
    something else corrupted it; not a real holder."""
    target = str(tmp_path / "foo.gguf.part")
    lock_path = target + ".lock"
    with open(lock_path, "w") as f:
        f.write("not-a-pid")
    lock = model._acquire_install_lock(target, color=False)
    assert lock is not None
    model._release_install_lock(lock)


def test_install_lock_live_pid_refused(tmp_path, capsys):
    """A lock file owned by THIS process (definitely alive) should
    refuse the second acquire and print a helpful message."""
    target = str(tmp_path / "foo.gguf.part")
    lock_path = target + ".lock"
    import os as _os
    with open(lock_path, "w") as f:
        f.write(f"{_os.getpid()}\n0\n")
    lock = model._acquire_install_lock(target, color=False)
    assert lock is None
    out = capsys.readouterr().out
    assert "Another install is already in progress" in out
    assert str(_os.getpid()) in out


def test_release_install_lock_handles_none():
    """Defensive: release(None) is a no-op, never raises."""
    model._release_install_lock(None)


def test_release_install_lock_handles_missing_file(tmp_path):
    """If the lock file was already removed by something else, release
    must not crash."""
    model._release_install_lock(str(tmp_path / "nonexistent.lock"))


# ---------------------------------------------------------------------------
# PC-056.2 — oversized .part detection (item B)
# ---------------------------------------------------------------------------

def test_install_oversized_part_refused(tmp_path, capsys):
    """If existing .part is wildly larger than the registered model
    size, install must refuse rather than send Range: bytes=N- with
    N > total."""
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf.part"
    # Registered size is 6.94 GB. Make a sparse .part of 20 GB —
    # well over the 5% slack threshold.
    with open(p, "wb") as f:
        f.seek(20 * 1024 ** 3)
        f.write(b"\0")
    rc = model.main(["install", "Qwen3.5-9B-Q6_K", "--models-dir",
                     str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "larger than the expected model size" in out
    assert "rm " in out  # remediation includes manual delete
    assert "--no-resume" in out


def test_install_oversized_part_within_slack_does_not_refuse(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    """A .part within 5% slack of the expected size should be accepted
    by the oversized check (other gates may still refuse). We just
    verify the oversized message doesn't fire."""
    # Disable network — we just want to reach the oversized check, not
    # actually download. Use --dry-run-style by failing before download
    # via a stubbed urlopen that immediately raises.
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf.part"
    # ~7.0 GB — within 5% of the registered 6.94 GB
    with open(p, "wb") as f:
        f.seek(int(7.0 * 1024 ** 3))
        f.write(b"\0")

    def fake_urlopen(*a, **kw):
        raise urllib.error.URLError("network disabled in test")

    import atlas.cli.commands.model as model_mod
    monkeypatch.setattr(model_mod.urllib.request, "urlopen", fake_urlopen)
    rc = model.main(["install", "Qwen3.5-9B-Q6_K", "--models-dir",
                     str(tmp_path), "--no-color"])
    # We expect failure (network stubbed) but NOT the oversized message.
    assert rc == 1
    out = capsys.readouterr().out
    assert "larger than the expected model size" not in out


# ---------------------------------------------------------------------------
# PC-056.2 — mocked urlopen download path tests (item C)
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal response mock that mimics what urllib's urlopen returns:
    a context manager exposing .headers, .getcode(), and .read(n)."""
    def __init__(self, body: bytes, status: int = 200,
                 headers: Optional[dict] = None):
        self._buf = body
        self._pos = 0
        self._status = status
        # Use a plain dict — model.py only calls .headers.get(name).
        self.headers = headers or {"Content-Length": str(len(body))}

    # urllib uses .getcode() in some versions, .status in others.
    def getcode(self):
        return self._status

    def read(self, n=-1):
        if n is None or n < 0:
            chunk = self._buf[self._pos:]
            self._pos = len(self._buf)
        else:
            chunk = self._buf[self._pos:self._pos + n]
            self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_with_fake_urlopen(monkeypatch, body: bytes, status: int = 200,
                                expect_range_offset: Optional[int] = None,
                                content_range_total: Optional[int] = None,
                                captured: Optional[list] = None,
                                raise_for: Optional[Exception] = None):
    """Wire up a fake urlopen that returns the given body. If captured is
    a list, each Request is appended to it for header inspection."""
    import atlas.cli.commands.model as model_mod

    def fake(req, timeout=60):
        if captured is not None:
            captured.append(req)
        if raise_for is not None:
            raise raise_for
        if expect_range_offset is not None and status == 206:
            # Server is ack'ing a Range request — return only the
            # remaining slice of body (model.py concats with range_start
            # bytes already on disk to produce the full file).
            slice_body = body[expect_range_offset:]
            headers = {"Content-Length": str(len(slice_body))}
            if content_range_total is not None:
                headers["Content-Range"] = (
                    f"bytes {expect_range_offset}-{content_range_total - 1}"
                    f"/{content_range_total}")
            return _FakeHTTPResponse(slice_body, status=206, headers=headers)
        return _FakeHTTPResponse(body, status=status,
                                  headers={"Content-Length": str(len(body))})

    monkeypatch.setattr(model_mod.urllib.request, "urlopen", fake)


def test_download_fresh_install_succeeds_with_correct_sha(tmp_path, monkeypatch,
                                                            capsys):
    """End-to-end happy path: 200 GB body of zeros (mocked as 101 MB to
    pass the size sanity threshold), SHA matches, file is renamed into
    place."""
    body = b"\0" * (101 * 1024 * 1024)  # 101 MB
    expected_sha = hashlib.sha256(body).hexdigest()

    # Patch the registry temporarily so the 9B's expected sha matches our body.
    import atlas.cli.commands.model_registry as reg
    original = reg.by_name("Qwen3.5-9B-Q6_K")
    patched = type(original)(
        name=original.name, tier=original.tier,
        model_file=original.model_file, model_display=original.model_display,
        model_size_gb=0.11,  # 101 MB — keeps the disk-space check happy
        lens_status=original.lens_status,
        download_url=original.download_url, sha256=expected_sha,
        license=original.license, requires_hf_token=False,
        lens_artifact_dir=original.lens_artifact_dir,
        lens_artifact_files=original.lens_artifact_files,
        notes=original.notes,
    )
    monkeypatch.setattr(reg, "by_name",
                         lambda n: patched if n == original.name else None)
    monkeypatch.setattr(reg, "for_tier",
                         lambda t: patched if t == "medium" else None)

    _install_with_fake_urlopen(monkeypatch, body=body, status=200)
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "SHA256 verified" in out
    assert (tmp_path / "Qwen3.5-9B-Q6_K.gguf").exists()
    assert not (tmp_path / "Qwen3.5-9B-Q6_K.gguf.part").exists()


def test_download_sha_mismatch_deletes_part(tmp_path, monkeypatch, capsys):
    """If the body's SHA doesn't match registry — delete .part, exit 1."""
    body = b"\0" * (101 * 1024 * 1024)
    wrong_sha = "f" * 64  # definitely not the SHA of zero bytes

    import atlas.cli.commands.model_registry as reg
    original = reg.by_name("Qwen3.5-9B-Q6_K")
    patched = type(original)(
        name=original.name, tier=original.tier,
        model_file=original.model_file, model_display=original.model_display,
        model_size_gb=0.11,
        lens_status=original.lens_status,
        download_url=original.download_url, sha256=wrong_sha,
        license=original.license, requires_hf_token=False,
        lens_artifact_dir=original.lens_artifact_dir,
        lens_artifact_files=original.lens_artifact_files,
        notes=original.notes,
    )
    monkeypatch.setattr(reg, "by_name",
                         lambda n: patched if n == original.name else None)
    monkeypatch.setattr(reg, "for_tier",
                         lambda t: patched if t == "medium" else None)

    _install_with_fake_urlopen(monkeypatch, body=body, status=200)
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "mismatch" in out.lower()
    assert not (tmp_path / "Qwen3.5-9B-Q6_K.gguf.part").exists()
    assert not (tmp_path / "Qwen3.5-9B-Q6_K.gguf").exists()


def test_download_resume_continues_hash_correctly(tmp_path, monkeypatch, capsys):
    """Existing .part contains the first half of the body; mocked server
    returns the second half via 206. Final SHA should match the WHOLE body."""
    full_body = b"x" * (101 * 1024 * 1024)
    half = len(full_body) // 2
    expected_sha = hashlib.sha256(full_body).hexdigest()

    # Pre-populate .part with the first half
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf.part"
    with open(p, "wb") as f:
        f.write(full_body[:half])

    import atlas.cli.commands.model_registry as reg
    original = reg.by_name("Qwen3.5-9B-Q6_K")
    patched = type(original)(
        name=original.name, tier=original.tier,
        model_file=original.model_file, model_display=original.model_display,
        model_size_gb=0.11,
        lens_status=original.lens_status,
        download_url=original.download_url, sha256=expected_sha,
        license=original.license, requires_hf_token=False,
        lens_artifact_dir=original.lens_artifact_dir,
        lens_artifact_files=original.lens_artifact_files,
        notes=original.notes,
    )
    monkeypatch.setattr(reg, "by_name",
                         lambda n: patched if n == original.name else None)
    monkeypatch.setattr(reg, "for_tier",
                         lambda t: patched if t == "medium" else None)

    _install_with_fake_urlopen(monkeypatch, body=full_body, status=206,
                                 expect_range_offset=half,
                                 content_range_total=len(full_body))
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "Resuming from byte" in out
    assert "SHA256 verified" in out


def test_download_server_ignores_range_restarts_cleanly(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """If we send Range but server returns 200 (ignoring the range), we
    should restart from byte 0, NOT corrupt the hash by counting the
    .part bytes twice."""
    full_body = b"y" * (101 * 1024 * 1024)
    expected_sha = hashlib.sha256(full_body).hexdigest()

    # Pre-populate .part with junk that would make the hash wrong if
    # we mistakenly fold it into the digest after the server-ignores-range
    # restart.
    p = tmp_path / "Qwen3.5-9B-Q6_K.gguf.part"
    with open(p, "wb") as f:
        f.write(b"GARBAGE" * 10000)  # ~70 KB of junk

    import atlas.cli.commands.model_registry as reg
    original = reg.by_name("Qwen3.5-9B-Q6_K")
    patched = type(original)(
        name=original.name, tier=original.tier,
        model_file=original.model_file, model_display=original.model_display,
        model_size_gb=0.11,
        lens_status=original.lens_status,
        download_url=original.download_url, sha256=expected_sha,
        license=original.license, requires_hf_token=False,
        lens_artifact_dir=original.lens_artifact_dir,
        lens_artifact_files=original.lens_artifact_files,
        notes=original.notes,
    )
    monkeypatch.setattr(reg, "by_name",
                         lambda n: patched if n == original.name else None)
    monkeypatch.setattr(reg, "for_tier",
                         lambda t: patched if t == "medium" else None)

    # Server returns 200 (full body) even though we sent Range —
    # model.py's restart-from-zero logic should reset the hash.
    _install_with_fake_urlopen(monkeypatch, body=full_body, status=200)
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "Server ignored Range" in out
    assert "SHA256 verified" in out


def test_download_sends_authorization_header_with_hf_token(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """When HF_TOKEN is set, the Request must carry an Authorization header
    with Bearer <token>."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_xyz")
    body = b"z" * (101 * 1024 * 1024)
    expected_sha = hashlib.sha256(body).hexdigest()

    import atlas.cli.commands.model_registry as reg
    original = reg.by_name("Qwen3.5-9B-Q6_K")
    patched = type(original)(
        name=original.name, tier=original.tier,
        model_file=original.model_file, model_display=original.model_display,
        model_size_gb=0.11,
        lens_status=original.lens_status,
        download_url=original.download_url, sha256=expected_sha,
        license=original.license, requires_hf_token=False,
        lens_artifact_dir=original.lens_artifact_dir,
        lens_artifact_files=original.lens_artifact_files,
        notes=original.notes,
    )
    monkeypatch.setattr(reg, "by_name",
                         lambda n: patched if n == original.name else None)
    monkeypatch.setattr(reg, "for_tier",
                         lambda t: patched if t == "medium" else None)

    captured: list = []
    _install_with_fake_urlopen(monkeypatch, body=body, status=200,
                                 captured=captured)
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 0, capsys.readouterr().out
    assert len(captured) == 1
    req = captured[0]
    # Request stores headers with capitalized first letter.
    assert req.has_header("Authorization")
    assert req.get_header("Authorization") == "Bearer hf_test_xyz"


def test_download_401_keeps_part_for_retry(tmp_path, monkeypatch, capsys):
    """401 from the server should NOT delete .part — user may set
    HF_TOKEN and retry. Helpful message should fire."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    err = urllib.error.HTTPError(
        url="https://example/x", code=401, msg="Unauthorized",
        hdrs=None, fp=None)
    _install_with_fake_urlopen(monkeypatch, body=b"", raise_for=err)
    rc = model.main(["install", "Qwen3.5-9B-Q6_K",
                     "--models-dir", str(tmp_path), "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "HTTP 401" in out or "401" in out
    assert "HF_TOKEN" in out


# Imports needed for the mocked-urlopen fixtures
import hashlib
import urllib.error
import urllib.request
