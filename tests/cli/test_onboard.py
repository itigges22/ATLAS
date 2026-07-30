"""Model onboarding tests for backend-specific inference lifecycles."""

from atlas.commands import onboard


def test_metal_onboarding_never_starts_cuda_container(monkeypatch, tmp_path):
    monkeypatch.setattr(onboard, "_serving_this", lambda url, model: (False, False))

    def unexpected_run(*args, **kwargs):
        raise AssertionError("Metal onboarding must not invoke docker compose")

    monkeypatch.setattr(onboard, "_run", unexpected_run)
    ready, detail = onboard._arch_supported(
        str(tmp_path), {"ATLAS_BACKEND": "metal"}, "model.gguf",
        start=True, color=False,
    )
    assert ready is False
    assert "atlas-llama-macos.sh" in detail
    assert "will not start the CUDA container" in detail


def test_nonmetal_log_inspection_uses_backend_overlay(monkeypatch, tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "docker-compose.rocm.yml").write_text("services: {}\n")
    monkeypatch.setattr(onboard, "_serving_this", lambda url, model: (False, False))
    calls = []

    def capture(cmd, **kwargs):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(onboard, "_run", capture)
    ready, _ = onboard._arch_supported(
        str(tmp_path), {"ATLAS_BACKEND": "rocm"}, "model.gguf",
        start=False, color=False,
    )
    assert ready is False
    assert calls == [[
        "docker", "compose", "-f", "docker-compose.yml", "-f",
        "docker-compose.rocm.yml", "logs", "--tail=200", "llama-server",
    ]]


def test_url_flow_with_apply_writes_env_keys(monkeypatch, capsys):
    """`atlas onboard --url ... --apply` writes ATLAS_MODEL_FILE +
    ATLAS_MODEL_NAME into .env after the download instead of only telling
    the user to hand-edit."""
    from atlas.commands import model as model_cmd
    from atlas.commands import fit as fit_module

    monkeypatch.setattr(model_cmd, "main", lambda argv: 0)
    written = {}

    def fake_write_env(values):
        written.update(values)
        return "/fake/.env"

    monkeypatch.setattr(fit_module, "_write_env", fake_write_env)
    rc = onboard.main(["--url", "https://example.com/My-Model-Q4_K_M.gguf",
                       "--apply", "--no-color"])
    assert rc == 0
    assert written["ATLAS_MODEL_FILE"] == "My-Model-Q4_K_M.gguf"
    assert written["ATLAS_MODEL_NAME"] == "My-Model-Q4_K_M"
    out = capsys.readouterr().out
    assert "/fake/.env" in out
    assert "re-run" in out.lower()


def test_url_flow_without_apply_keeps_manual_instructions(monkeypatch,
                                                            capsys):
    """Declined/non-interactive: the printed hand-edit fallback stays."""
    from atlas.commands import model as model_cmd
    from atlas.commands import fit as fit_module

    monkeypatch.setattr(model_cmd, "main", lambda argv: 0)

    def unexpected_write(values):
        raise AssertionError(".env must not be written without consent")

    monkeypatch.setattr(fit_module, "_write_env", unexpected_write)
    rc = onboard.main(["--url", "https://example.com/My-Model-Q4_K_M.gguf",
                       "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ATLAS_MODEL_FILE + ATLAS_MODEL_NAME" in out
    assert "--apply" in out
