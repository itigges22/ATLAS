"""Unit tests for the service-token module and init generation step.

All tokens are synthetic fixtures.
"""

import os
import stat

from atlas.cli import token as token_mod


def test_read_token_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_SERVICE_TOKEN_FILE",
                       str(tmp_path / "absent"))
    assert token_mod.read_token() == ""
    assert token_mod.auth_headers() == {}


def test_read_token_and_headers(tmp_path, monkeypatch):
    tok = tmp_path / "service-token"
    tok.write_text("atlas-st-unitfixture\n")
    monkeypatch.setenv("ATLAS_SERVICE_TOKEN_FILE", str(tok))
    assert token_mod.read_token() == "atlas-st-unitfixture"
    assert token_mod.auth_headers() == {
        "Authorization": "Bearer atlas-st-unitfixture"}


def test_permission_check_flags_loose_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    tok = secrets / "service-token"
    tok.write_text("atlas-st-unitfixture\n")
    tok.chmod(0o644)  # loose
    monkeypatch.setenv("ATLAS_SERVICE_TOKEN_FILE", str(tok))
    ok, detail = token_mod.check_file_permissions()
    assert not ok
    assert "0o644" in detail
    assert "atlas-st-unitfixture" not in detail  # never leak the value


def test_permission_check_passes_strict(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    tok = secrets / "service-token"
    tok.write_text("atlas-st-unitfixture\n")
    tok.chmod(0o600)
    monkeypatch.setenv("ATLAS_SERVICE_TOKEN_FILE", str(tok))
    ok, detail = token_mod.check_file_permissions()
    assert ok, detail


def test_init_step_generates_and_keeps_token(tmp_path):
    from atlas.cli.commands import init as init_mod

    class Args:
        dry_run = False
        rotate_token = False

    path = init_mod._step_write_service_token(str(tmp_path), Args(), False)
    assert os.path.isfile(path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    first = open(path).read().strip()
    assert first.startswith("atlas-st-")

    # Re-run keeps the existing token (live stack keeps working)
    init_mod._step_write_service_token(str(tmp_path), Args(), False)
    assert open(path).read().strip() == first

    # --rotate-token regenerates
    class Rotate(Args):
        rotate_token = True

    init_mod._step_write_service_token(str(tmp_path), Rotate(), False)
    second = open(path).read().strip()
    assert second != first
    assert second.startswith("atlas-st-")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
