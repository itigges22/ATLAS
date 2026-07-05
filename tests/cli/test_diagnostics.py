"""atlas diagnostics collect — bundle safety.

Verifies the filtered-config logic drops the service token, masks
secret-ish values, and runs values through the private-value filter.
No services required (the collectors are unit-tested in isolation).
"""

import os

from atlas.cli.commands import diagnostics


def test_filtered_env_drops_token_and_masks_secrets(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "ATLAS_MODEL_FILE=gemma.gguf\n"
        "ATLAS_SERVICE_TOKEN=atlas-st-should-never-appear\n"
        "HF_TOKEN=hf_fakeExampleToken\n"
        "DB_PASSWORD=fake-example-pw\n"
        "ATLAS_CTX_SIZE=131072\n"
        "SOME_URL=postgres://demo:example@localhost/test\n")
    out = diagnostics._filtered_env(str(tmp_path))

    # Token dropped entirely (not even masked)
    assert "ATLAS_SERVICE_TOKEN" not in out
    # Secret-ish keys masked
    assert out["HF_TOKEN"] == "[MASKED]"
    assert out["DB_PASSWORD"] == "[MASKED]"
    # Non-secret values preserved
    assert out["ATLAS_MODEL_FILE"] == "gemma.gguf"
    assert out["ATLAS_CTX_SIZE"] == "131072"
    # URL password filtered by the shared private-value filter
    assert ":example@" not in out["SOME_URL"]
    # And the raw fixture token never appears anywhere
    assert "atlas-st-should-never-appear" not in str(out)


def test_collect_excludes_source_and_has_schema(tmp_path, monkeypatch):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / ".env").write_text("ATLAS_MODEL_FILE=m.gguf\n")

    # Stub the network/subprocess collectors so the test is hermetic.
    monkeypatch.setattr(diagnostics, "_service_health", lambda root: {})
    monkeypatch.setattr(diagnostics, "_recent_logs", lambda root, n: {})
    monkeypatch.setattr(diagnostics, "_doctor_json", lambda root: {})
    monkeypatch.setattr(diagnostics, "_run", lambda *a, **k: "")

    bundle = diagnostics._collect(str(tmp_path), 10)
    assert bundle["schema_version"] == 1
    assert "meta" in bundle and "config" in bundle
    # no source-code section
    assert "source" not in bundle and "files" not in bundle
