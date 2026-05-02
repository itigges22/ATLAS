"""Tests for atlas.cli.commands.init (PC-054) — the first-run install wizard.

The wizard is a thin composer over tier + model_registry + model.install,
so these tests focus on the wizard's own behavior:

  - --yes happy path writes .env + api-keys.json with expected values
  - --reconfigure backs up existing .env before overwriting
  - already-configured guard refuses without --reconfigure
  - --skip-download produces config without calling install
  - --dry-run touches no files
  - --json shape is stable for the bootstrap script
  - api-keys.json permissions are 0600 + parent 0700
  - reconfigure of api-keys.json backs up the existing one
"""

import json
import os
import stat

from atlas.cli.commands import init


def _make_atlas_root(tmp_path) -> str:
    """Create a fake atlas_root with a docker-compose.yml so _find_atlas_root
    walks up from tmp_path and lands here. Returns absolute path."""
    root = tmp_path / "atlas_root"
    root.mkdir()
    (root / "docker-compose.yml").write_text("# fake compose for tests\n")
    return str(root)


def _run(monkeypatch, atlas_root, argv):
    """Run init.main with CWD set to atlas_root so atlas_root resolution
    finds the fake docker-compose.yml."""
    monkeypatch.chdir(atlas_root)
    return init.main(argv)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_yes_skip_download_writes_env_and_keys(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0

    env_path = os.path.join(root, ".env")
    keys_path = os.path.join(root, "secrets", "api-keys.json")
    assert os.path.isfile(env_path)
    assert os.path.isfile(keys_path)

    body = open(env_path).read()
    # Wizard must write every key the compose stack reads at boot.
    for key in ("ATLAS_MODELS_DIR", "ATLAS_MODEL_FILE", "ATLAS_MODEL_NAME",
                "ATLAS_CTX_SIZE", "ATLAS_GHCR_OWNER", "ATLAS_IMAGE_TAG",
                "ATLAS_LLAMA_PORT", "PARALLEL_SLOTS"):
        assert f"{key}=" in body, f"missing {key} in .env"

    # Default models_dir is ./models when it equals atlas_root/models.
    assert "ATLAS_MODELS_DIR=./models" in body
    assert "ATLAS_IMAGE_TAG=latest" in body
    assert "ATLAS_GHCR_OWNER=itigges22" in body


def test_api_keys_file_has_strict_permissions(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0

    keys_dir = os.path.join(root, "secrets")
    keys_path = os.path.join(keys_dir, "api-keys.json")
    dir_mode = stat.S_IMODE(os.stat(keys_dir).st_mode)
    file_mode = stat.S_IMODE(os.stat(keys_path).st_mode)
    assert dir_mode == 0o700, f"secrets/ mode {oct(dir_mode)} != 0700"
    assert file_mode == 0o600, f"api-keys.json mode {oct(file_mode)} != 0600"


def test_api_keys_payload_shape(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0
    payload = json.loads(open(os.path.join(root, "secrets", "api-keys.json")).read())
    # Exactly one key, of the expected sk-atlas-* prefix, valued correctly.
    assert len(payload) == 1
    (key, value), = payload.items()
    assert key.startswith("sk-atlas-")
    assert len(key) > len("sk-atlas-") + 20  # token_urlsafe(32) is well over 20 chars
    assert value == {"user": "local", "created_by": "atlas init"}


# ---------------------------------------------------------------------------
# Already-configured guard + --reconfigure
# ---------------------------------------------------------------------------

def test_already_configured_refuses_without_reconfigure(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    # Pre-existing .env from an earlier setup.
    existing_env = os.path.join(root, ".env")
    open(existing_env, "w").write("ATLAS_MODEL_FILE=hand-edited.gguf\n")

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Already configured" in out
    assert "--reconfigure" in out
    # And critically — original .env was NOT modified.
    assert open(existing_env).read() == "ATLAS_MODEL_FILE=hand-edited.gguf\n"


def test_reconfigure_backs_up_existing_env(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    existing_env = os.path.join(root, ".env")
    open(existing_env, "w").write("ATLAS_MODEL_FILE=old.gguf\n")

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--reconfigure"])
    assert rc == 0
    backup = existing_env + ".bak"
    assert os.path.isfile(backup)
    assert open(backup).read() == "ATLAS_MODEL_FILE=old.gguf\n"
    # New .env is the wizard's render — has the structured comment header.
    new_body = open(existing_env).read()
    assert "generated by `atlas init`" in new_body


def test_reconfigure_without_existing_env_still_works(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--reconfigure"])
    assert rc == 0
    assert os.path.isfile(os.path.join(root, ".env"))
    # No backup file when there was nothing to back up.
    assert not os.path.isfile(os.path.join(root, ".env.bak"))


def test_reconfigure_backs_up_existing_api_keys(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    secrets_dir = os.path.join(root, "secrets")
    os.makedirs(secrets_dir, mode=0o700)
    keys_path = os.path.join(secrets_dir, "api-keys.json")
    open(keys_path, "w").write('{"sk-old-key": {"user": "alice"}}\n')
    os.chmod(keys_path, 0o600)

    # Need .env present too so the reconfigure flag is the actual gate.
    open(os.path.join(root, ".env"), "w").write("ATLAS_MODEL_FILE=foo.gguf\n")

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--reconfigure"])
    assert rc == 0
    bak = keys_path + ".bak"
    assert os.path.isfile(bak)
    assert "sk-old-key" in open(bak).read()
    # New file has a fresh sk-atlas-* key, not the old one.
    new = json.loads(open(keys_path).read())
    assert all(k.startswith("sk-atlas-") for k in new.keys())


# ---------------------------------------------------------------------------
# --skip-download + --dry-run
# ---------------------------------------------------------------------------

def test_skip_download_does_not_touch_models_dir(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Skipping download" in out
    # Models dir not auto-created when skipped — the user is responsible
    # for placing the gguf themselves.
    assert not os.path.exists(os.path.join(root, "models"))


def test_dry_run_touches_no_files(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--dry-run", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(dry-run)" in out
    # Nothing on disk except the seed compose file.
    assert os.listdir(root) == ["docker-compose.yml"]


# ---------------------------------------------------------------------------
# --json shape
# ---------------------------------------------------------------------------

def test_json_output_shape_is_stable(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # JSON object is at the very end of stdout — find its opening brace.
    blob = out[out.rindex("{"):]
    payload = json.loads(blob)
    expected_keys = {"atlas_root", "tier", "model", "models_dir",
                     "env_path", "env_backup", "api_keys_path",
                     "api_keys_backup", "api_key", "image_tag",
                     "ghcr_owner", "dry_run"}
    assert expected_keys.issubset(payload.keys())
    assert payload["atlas_root"] == root
    assert payload["dry_run"] is False
    # api_key in JSON matches the file we wrote.
    file_payload = json.loads(open(payload["api_keys_path"]).read())
    assert payload["api_key"] in file_payload


# ---------------------------------------------------------------------------
# --image-tag / --ghcr-owner overrides
# ---------------------------------------------------------------------------

def test_image_tag_override_lands_in_env(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color",
               "--image-tag", "v1.2.3", "--ghcr-owner", "myfork"])
    assert rc == 0
    body = open(os.path.join(root, ".env")).read()
    assert "ATLAS_IMAGE_TAG=v1.2.3" in body
    assert "ATLAS_GHCR_OWNER=myfork" in body


# ---------------------------------------------------------------------------
# --models-dir override
# ---------------------------------------------------------------------------

def test_models_dir_override_lands_in_env_as_absolute(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    custom = tmp_path / "custom_models"
    custom.mkdir()
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color",
               "--models-dir", str(custom)])
    assert rc == 0
    body = open(os.path.join(root, ".env")).read()
    # Non-default path is written verbatim (absolute), not as ./models.
    assert f"ATLAS_MODELS_DIR={custom}" in body
    assert "ATLAS_MODELS_DIR=./models" not in body
