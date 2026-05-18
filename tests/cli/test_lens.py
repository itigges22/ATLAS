"""Tests for atlas/cli/commands/lens.py (PC-057 + PC-058).

Coverage strategy:
- `_check_model` and `_emit_check` are tested by patching `probe_llama` to
  return synthetic LlamaProbe records (no HTTP, no llama-server required).
- `_read_saved_cost_field_dim` is exercised against a real torch.save'd
  state dict so the pickle peek path is real, not mocked.
- `_load_training_samples` is tested against tmp_path-written JSON/JSONL.
- `_emit_build`'s training step is exercised in --dry-run mode to keep
  tests fast (no actual CostField training).
"""

import json
import os
import sys

import pytest

from atlas.cli.commands import lens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(reachable=True, embedding_dim=4096, n_layers=32,
           model_name="Qwen3.5-9B-Q6_K.gguf", patch=True, error=""):
    return lens.LlamaProbe(
        reachable=reachable,
        url="http://test-llama:8080",
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        model_name=model_name,
        has_hidden_states_patch=patch,
        error=error,
    )


# ---------------------------------------------------------------------------
# _read_saved_cost_field_dim — real torch round-trip
# ---------------------------------------------------------------------------

def test_read_saved_cost_field_dim_inferred(tmp_path):
    """A genuine save_cost_field artifact's input dim is recoverable."""
    torch = pytest.importorskip("torch")
    # Build a minimal state dict matching the CostField layout:
    # net.0 is the first Linear (out=512, in=DIM).
    DIM = 768  # something non-canonical to prove the inference is real
    state = {
        "net.0.weight": torch.zeros(512, DIM),
        "net.0.bias":   torch.zeros(512),
        "net.2.weight": torch.zeros(128, 512),
        "net.2.bias":   torch.zeros(128),
        "net.4.weight": torch.zeros(1, 128),
        "net.4.bias":   torch.zeros(1),
    }
    torch.save(state, tmp_path / "cost_field.pt")
    assert lens._read_saved_cost_field_dim(str(tmp_path)) == DIM


def test_read_saved_cost_field_dim_missing_file(tmp_path):
    """No artifact file → returns None, not an exception."""
    assert lens._read_saved_cost_field_dim(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# _check_model verdicts
# ---------------------------------------------------------------------------

def test_check_unreachable_server_is_incompatible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        lens, "probe_llama",
        lambda *a, **kw: _probe(reachable=False, error="not reachable"))
    v = lens._check_model(None, str(tmp_path))
    assert v.verdict == "incompatible"
    assert v.exit_code == 2
    assert "not reachable" in v.reason


def test_check_zero_dim_is_incompatible(monkeypatch, tmp_path):
    """Server up but /embedding returned nothing -> incompatible."""
    monkeypatch.setattr(
        lens, "probe_llama",
        lambda *a, **kw: _probe(embedding_dim=0,
                                 error="embedding endpoint silent"))
    v = lens._check_model(None, str(tmp_path))
    assert v.verdict == "incompatible"


def test_check_missing_artifact_is_needs_build(monkeypatch, tmp_path):
    """Probe OK, no cost_field.pt exists -> needs-build (exit 1)."""
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(lens, "probe_llama", lambda *a, **kw: _probe())
    v = lens._check_model(None, str(tmp_path))
    assert v.verdict == "needs-build"
    assert v.exit_code == 1
    assert "no cost_field.pt" in v.reason


def test_check_dim_mismatch_is_needs_build(monkeypatch, tmp_path):
    """Artifact exists but its input dim != model's embedding dim."""
    torch = pytest.importorskip("torch")
    state = {"net.0.weight": torch.zeros(512, 2048)}  # 2048-dim artifact
    torch.save(state, tmp_path / "cost_field.pt")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(lens, "probe_llama",
                        lambda *a, **kw: _probe(embedding_dim=4096))
    v = lens._check_model(None, str(tmp_path))
    assert v.verdict == "needs-build"
    assert "Dim mismatch" in v.reason
    assert v.artifact_dim == 2048
    assert v.probe.embedding_dim == 4096


def test_check_dim_match_is_compat(monkeypatch, tmp_path):
    """Artifact dim matches model embedding dim -> compat (exit 0)."""
    torch = pytest.importorskip("torch")
    state = {"net.0.weight": torch.zeros(512, 4096)}
    torch.save(state, tmp_path / "cost_field.pt")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(lens, "probe_llama",
                        lambda *a, **kw: _probe(embedding_dim=4096))
    v = lens._check_model(None, str(tmp_path))
    assert v.verdict == "compat"
    assert v.exit_code == 0
    assert v.artifact_dim == 4096


def test_check_compat_warns_when_pc202_patch_missing(monkeypatch, tmp_path):
    """Compat verdict but no PC-202 patch -> reason mentions G(x) limitation."""
    torch = pytest.importorskip("torch")
    state = {"net.0.weight": torch.zeros(512, 4096)}
    torch.save(state, tmp_path / "cost_field.pt")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(lens, "probe_llama",
                        lambda *a, **kw: _probe(patch=False))
    v = lens._check_model(None, str(tmp_path))
    assert v.verdict == "compat"
    assert "PC-202" in v.reason
    assert "G(x)" in v.reason


# ---------------------------------------------------------------------------
# CLI JSON output shape — what scripts will key off
# ---------------------------------------------------------------------------

def test_check_json_output_has_stable_shape(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lens, "probe_llama",
                        lambda *a, **kw: _probe(reachable=False, error="oops"))
    rc = lens.main(["check", "--json"])
    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    for key in ("verdict", "reason", "probe", "exit_code"):
        assert key in payload
    assert payload["exit_code"] == 2
    assert payload["verdict"] == "incompatible"
    # Probe nested object preserves the dataclass fields scripts may want
    for key in ("reachable", "url", "embedding_dim", "n_layers"):
        assert key in payload["probe"]


# ---------------------------------------------------------------------------
# Training sample loader — JSON + JSONL
# ---------------------------------------------------------------------------

def test_load_training_samples_json_array(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([
        {"text": "ok", "label": 1},
        {"text": "bad", "label": 0},
    ]))
    samples = lens._load_training_samples(str(path))
    assert len(samples) == 2
    assert samples[0]["label"] == 1


def test_load_training_samples_jsonl(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"text": "a", "label": 1}\n'
        '{"text": "b", "label": 0}\n'
        '\n'  # blank line tolerated
        '{"text": "c", "label": 1}\n'
    )
    samples = lens._load_training_samples(str(path))
    assert len(samples) == 3
    assert [s["text"] for s in samples] == ["a", "b", "c"]


def test_load_training_samples_missing_file(tmp_path):
    """Missing file returns empty list, not an exception."""
    assert lens._load_training_samples(str(tmp_path / "nope.json")) == []
    assert lens._load_training_samples(None) == []


# ---------------------------------------------------------------------------
# Build subcommand — early-exit + dry-run paths
# ---------------------------------------------------------------------------

def test_build_refuses_without_samples_flag(monkeypatch, tmp_path, capsys):
    """Build with no --samples and no artifacts -> usage error (rc=1)."""
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(lens, "probe_llama", lambda *a, **kw: _probe())
    rc = lens.main(["build", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "--samples" in out
    assert "huggingface" in out.lower()


def test_build_refuses_on_unreachable_server(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        lens, "probe_llama",
        lambda *a, **kw: _probe(reachable=False, error="server down"))
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps([
        {"text": "x", "label": 1}, {"text": "y", "label": 0},
    ]))
    rc = lens.main(["build", "--samples", str(samples_path), "--no-color"])
    assert rc == 2  # incompatible -> hard fail


def test_build_refuses_when_too_few_samples(monkeypatch, tmp_path, capsys):
    """<50 samples -> refuse (would produce a bad C(x))."""
    monkeypatch.setattr(lens, "probe_llama", lambda *a, **kw: _probe())
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps([
        {"text": f"sample {i}", "label": i % 2} for i in range(20)
    ]))
    rc = lens.main(["build", "--samples", str(samples_path),
                    "--force", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "samples loaded" in out.lower() or "samples." in out.lower() or "20" in out


def test_build_refuses_when_one_class_missing(monkeypatch, tmp_path, capsys):
    """All-pass or all-fail samples -> refuse (contrastive needs both)."""
    monkeypatch.setattr(lens, "probe_llama", lambda *a, **kw: _probe())
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    samples_path = tmp_path / "samples.json"
    # 60 PASS, 0 FAIL — passes count threshold, fails class-balance
    samples_path.write_text(json.dumps([
        {"text": f"sample {i}", "label": 1} for i in range(60)
    ]))
    # Stub the embedding extractor so we don't try to hit a real server
    monkeypatch.setattr(lens, "_extract_training_embeddings",
                        lambda *a, **kw: {"embeddings": [], "labels": []})
    rc = lens.main(["build", "--samples", str(samples_path),
                    "--force", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "both" in out.lower() or "contrastive" in out.lower()


def test_build_compat_with_no_force_is_noop(monkeypatch, tmp_path, capsys):
    """If artifacts already exist for the current dim, build exits 0 without
    requiring --samples. Forcing a retrain is opt-in via --force."""
    torch = pytest.importorskip("torch")
    state = {"net.0.weight": torch.zeros(512, 4096)}
    torch.save(state, tmp_path / "cost_field.pt")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(lens, "probe_llama",
                        lambda *a, **kw: _probe(embedding_dim=4096))
    rc = lens.main(["build", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--force" in out


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def test_main_with_no_subcommand_shows_help(capsys):
    rc = lens.main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "check" in out and "build" in out and "publish" in out


# ---------------------------------------------------------------------------
# Publish subcommand (PC-059) — paths that don't require an HF token
# ---------------------------------------------------------------------------

def test_publish_refuses_when_no_artifact(monkeypatch, tmp_path, capsys):
    """No cost_field.pt in the artifact dir -> usage error (rc=1)."""
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K", "--dry-run", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No cost_field.pt" in out
    assert "atlas lens build" in out


def test_publish_dry_run_prints_pr_body(monkeypatch, tmp_path, capsys):
    """Dry-run hashes the artifact, renders the PR body, prints it."""
    torch = pytest.importorskip("torch")
    state = {"net.0.weight": torch.zeros(512, 4096),
             "net.0.bias": torch.zeros(512)}
    torch.save(state, tmp_path / "cost_field.pt")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K", "--dry-run", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    # PR body markdown should be present
    assert "Verification checklist" in out
    assert "Suggested registry diff" in out
    # SHA256 should appear (computed from the fake .pt above)
    assert "SHA256" in out or "sha256" in out
    # Should mention `atlas lens publish` (the tool that auto-generated it)
    assert "auto-generated" in out


def test_publish_dry_run_works_without_torch(monkeypatch, tmp_path, capsys):
    """Even without torch on the host, publish --dry-run should still
    print a usable PR body — just with dim shown as 'unverified'."""
    # Create a non-empty cost_field.pt that isn't a valid torch state dict.
    # The SHA-256 + size + path inspection paths don't need torch; only
    # the dim introspection does.
    (tmp_path / "cost_field.pt").write_bytes(b"fake pt content for sha256")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    # Force the dim-inspection to fail by stubbing _inspect_cost_field.
    monkeypatch.setattr(
        lens, "_inspect_cost_field",
        lambda d: lens.ArtifactInspection(present=True, dim=None,
                                           torch_available=False,
                                           error="torch missing"))
    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K", "--dry-run", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "unverified" in out  # dim_label fallback


def test_publish_requires_repo_unless_dry_run(monkeypatch, tmp_path, capsys):
    """No --repo + no --dry-run -> usage error explaining the requirement.

    We set HF_TOKEN here so the publish_preflight passes — we want to
    isolate the test to the --repo check, not the auth gate (covered by
    test_preflight_blocks_non_dryrun_when_token_missing)."""
    if not lens._huggingface_hub_available():
        pytest.skip("huggingface_hub not installed on this host")
    (tmp_path / "cost_field.pt").write_bytes(b"fake")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_for_test")
    monkeypatch.setattr(
        lens, "_inspect_cost_field",
        lambda d: lens.ArtifactInspection(present=True, dim=4096))
    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "--repo" in out


def test_publish_requires_hf_token_when_uploading(monkeypatch, tmp_path, capsys):
    """--repo given but HF_TOKEN missing -> clean error pointing at the
    token settings page."""
    (tmp_path / "cost_field.pt").write_bytes(b"fake")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(
        lens, "_inspect_cost_field",
        lambda d: lens.ArtifactInspection(present=True, dim=4096))
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K",
                    "--repo", "alice/atlas-lens-test", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "HF_TOKEN" in out
    assert "huggingface.co/settings/tokens" in out


def test_publish_skip_pr_writes_body_without_gh(monkeypatch, tmp_path, capsys):
    """--skip-pr in dry-run mode should print the PR body unconditionally
    (no `gh` invocation, no upload). Sanity-check the exit code is 0."""
    (tmp_path / "cost_field.pt").write_bytes(b"fake")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setattr(
        lens, "_inspect_cost_field",
        lambda d: lens.ArtifactInspection(present=True, dim=4096))
    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K",
                    "--dry-run", "--skip-pr", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Verification checklist" in out


def test_render_registry_pr_body_includes_required_fields():
    """Direct unit test on the renderer — guards against accidental
    field drops in the markdown template."""
    body = lens._render_registry_pr_body(
        model_name="TestModel-9B",
        hf_repo="alice/atlas-lens-test",
        base_model="TestModel 9B (Q6_K)",
        dim=4096,
        sha256="a" * 64,
        license_id="apache-2.0",
    )
    assert "TestModel-9B" in body
    assert "alice/atlas-lens-test" in body
    assert "apache-2.0" in body
    assert "a" * 64 in body
    # The maintainer needs the Python diff with `lens_status="supported"`
    assert 'lens_status="supported"' in body


def test_sha256_file_is_deterministic_and_correct(tmp_path):
    """_sha256_file should match what `sha256sum` would compute."""
    import hashlib
    content = b"deterministic content for sha test" * 100
    (tmp_path / "f.bin").write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert lens._sha256_file(str(tmp_path / "f.bin")) == expected


# ---------------------------------------------------------------------------
# publish pre-flight panel (PC-059.1 / publishing UX polish)
# ---------------------------------------------------------------------------

def test_preflight_dry_run_skips_auth_gates(monkeypatch, capsys):
    """--dry-run should always return True regardless of missing creds —
    the whole point is letting users preview without setting anything up."""
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    ok = lens.publish_preflight("lens", dry_run=True, color=False)
    out = capsys.readouterr().out
    assert ok is True, "dry-run preflight should always pass"
    assert "submission pre-flight" in out
    assert "PUBLISHING.md" in out, "should reference the docs walkthrough"
    assert "--dry-run" in out, "should call out that auth gates are skipped"


def test_preflight_blocks_non_dryrun_when_token_missing(monkeypatch, capsys):
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    ok = lens.publish_preflight("lens", dry_run=False, color=False)
    out = capsys.readouterr().out
    assert ok is False, "missing HF_TOKEN in real run should block"
    assert "HF_TOKEN" in out
    assert "huggingface.co/settings/tokens" in out


def test_preflight_passes_when_token_and_pkg_present(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_for_test")
    # huggingface_hub may or may not be installed in CI; only assert pass
    # when the pkg is genuinely available (otherwise the test would be
    # flaky depending on host env).
    if not lens._huggingface_hub_available():
        pytest.skip("huggingface_hub not installed on this host")
    ok = lens.publish_preflight("lens", dry_run=False, color=False)
    out = capsys.readouterr().out
    assert ok is True
    assert "HF_TOKEN env var" in out


def test_preflight_mentions_both_lens_and_asa_kinds(monkeypatch, capsys):
    """The panel header should reflect whichever publish flow invoked it
    so the user isn't confused about what they're shipping."""
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    lens.publish_preflight("asa", dry_run=True, color=False)
    out = capsys.readouterr().out
    assert "atlas asa publish" in out
    lens.publish_preflight("lens", dry_run=True, color=False)
    out = capsys.readouterr().out
    assert "atlas lens publish" in out


def test_preflight_gh_missing_is_optional_warning(monkeypatch, capsys):
    """Without gh installed the preflight should still pass — it just
    notes the user will paste the PR body manually."""
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    monkeypatch.setattr(lens, "_gh_available", lambda: False)
    if not lens._huggingface_hub_available():
        pytest.skip("huggingface_hub not installed on this host")
    ok = lens.publish_preflight("lens", dry_run=False, color=False)
    out = capsys.readouterr().out
    assert ok is True, "missing gh should not block — paste fallback is valid"
    assert "gh CLI" in out
    assert "paste" in out.lower() or "compare" in out.lower()


def test_preflight_dry_run_doesnt_flag_missing_token_as_failure(
    monkeypatch, capsys
):
    """Regression: with --dry-run, missing HF_TOKEN must NOT render with
    a red ✗ or print the alarming 'required — get a token at ...' hint.
    Dry-run is the path users take BEFORE setting up creds; making it
    look like they did something wrong is the opposite of what we want."""
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    ok = lens.publish_preflight("lens", dry_run=True, color=False)
    out = capsys.readouterr().out
    assert ok is True
    # The red ✗ marker must not appear next to HF_TOKEN in dry-run mode.
    # (We render the row with a neutral ○ instead.)
    assert "✗ HF_TOKEN" not in out, (
        "dry-run preflight should not show ✗ for missing token; "
        "got:\n" + out
    )
    # The alarming "Cannot continue" footer must not fire either.
    assert "Cannot continue" not in out
    # The dry-run footer should still be present and tell the user
    # nothing is being enforced.
    assert "nothing will leave the host" in out


def test_publish_uses_atlas_publish_branch_only_when_set(
    monkeypatch, tmp_path, capsys
):
    """Regression for the `gh pr create --head ""` bug: when
    ATLAS_PUBLISH_BRANCH is unset, we must NOT pass --head to gh
    (gh rejects empty refs instead of inferring the current branch).
    We assert on the subprocess args via a captured spy."""
    if not lens._huggingface_hub_available():
        pytest.skip("huggingface_hub not installed on this host")
    (tmp_path / "cost_field.pt").write_bytes(b"fake")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_for_test")
    monkeypatch.delenv("ATLAS_PUBLISH_BRANCH", raising=False)
    monkeypatch.setattr(
        lens, "_inspect_cost_field",
        lambda d: lens.ArtifactInspection(present=True, dim=4096))

    # Capture subprocess.run invocations. Return success so the publish
    # flow continues past the gh call as it would in the green path.
    captured = []

    class _FakeResult:
        returncode = 0
        stdout = "https://github.com/itigges22/ATLAS/pull/42"
        stderr = ""

    def _fake_run(cmd, *a, **kw):
        captured.append(cmd)
        return _FakeResult()

    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", _fake_run)
    # Skip the actual HF upload — we only care about the gh args here.
    from huggingface_hub import HfApi
    monkeypatch.setattr(HfApi, "create_repo", lambda self, **kw: None)
    monkeypatch.setattr(HfApi, "upload_file", lambda self, **kw: None)

    rc = lens.main(["publish", "Qwen3.5-9B-Q6_K",
                    "--repo", "alice/atlas-lens-test", "--no-color"])
    assert rc == 0
    gh_calls = [c for c in captured if c and c[0] == "gh"]
    assert gh_calls, "expected at least one gh invocation"
    gh_cmd = gh_calls[0]
    # Critical: no empty --head value.
    if "--head" in gh_cmd:
        idx = gh_cmd.index("--head")
        head_val = gh_cmd[idx + 1] if idx + 1 < len(gh_cmd) else ""
        assert head_val, (
            "gh pr create must not be invoked with an empty --head; "
            f"saw: {gh_cmd}"
        )


def test_publish_passes_atlas_publish_branch_when_set(
    monkeypatch, tmp_path, capsys
):
    """Inverse of the regression test: when ATLAS_PUBLISH_BRANCH IS set,
    we should pass --head <branch> through to gh."""
    if not lens._huggingface_hub_available():
        pytest.skip("huggingface_hub not installed on this host")
    (tmp_path / "cost_field.pt").write_bytes(b"fake")
    monkeypatch.setenv("ATLAS_LENS_MODELS", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_for_test")
    monkeypatch.setenv("ATLAS_PUBLISH_BRANCH", "registry/qwen-9b-lens")
    monkeypatch.setattr(
        lens, "_inspect_cost_field",
        lambda d: lens.ArtifactInspection(present=True, dim=4096))

    captured = []

    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, *a, **kw):
        captured.append(cmd)
        return _FakeResult()

    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", _fake_run)
    from huggingface_hub import HfApi
    monkeypatch.setattr(HfApi, "create_repo", lambda self, **kw: None)
    monkeypatch.setattr(HfApi, "upload_file", lambda self, **kw: None)

    lens.main(["publish", "Qwen3.5-9B-Q6_K",
               "--repo", "alice/atlas-lens-test", "--no-color"])
    gh_calls = [c for c in captured if c and c[0] == "gh"]
    assert gh_calls, "expected a gh invocation"
    gh_cmd = gh_calls[0]
    assert "--head" in gh_cmd
    idx = gh_cmd.index("--head")
    assert gh_cmd[idx + 1] == "registry/qwen-9b-lens"
