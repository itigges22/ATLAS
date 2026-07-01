"""Compose lifecycle tests for the interactive CLI."""

import sys
from types import SimpleNamespace

import pytest

from atlas.cli import repl


def _metal_root(tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "docker-compose.macos.yml").write_text("services: {}\n")
    (tmp_path / ".env").write_text("ATLAS_BACKEND=metal\n")
    return str(tmp_path)


def test_workspace_recreate_keeps_macos_overlay(monkeypatch, tmp_path):
    root = _metal_root(tmp_path)
    calls = []

    def capture(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(repl.subprocess, "run", capture)
    monkeypatch.setattr(repl, "_check_url", lambda *args, **kwargs: True)
    assert repl._recreate_docker_proxy(root, str(tmp_path / "project")) is True
    assert calls[0][:6] == [
        "docker", "compose", "-f", "docker-compose.yml", "-f",
        "docker-compose.macos.yml",
    ]
    assert "llama-server" not in calls[0]


def test_compose_ownership_check_keeps_macos_overlay(monkeypatch, tmp_path):
    root = _metal_root(tmp_path)
    calls = []

    def capture(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="atlas-proxy\n", stderr="")

    monkeypatch.setattr(repl.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(repl.subprocess, "run", capture)
    assert repl._docker_compose_owns_proxy(root) is True
    assert "docker-compose.macos.yml" in calls[0]


def test_proxy_capability_parser_rejects_old_or_malformed_payload(monkeypatch):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return self.body

    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b'{"status":"ok"}'),
    )
    assert repl._proxy_capabilities("http://proxy") == set()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"not json"),
    )
    assert repl._proxy_capabilities("http://proxy") == set()


def test_proxy_capability_parser_accepts_demo_contract(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return b'{"capabilities":["demo_raw_completion_v1"]}'

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    assert repl._proxy_supports_capability(
        repl.DEMO_RAW_CAPABILITY, "http://proxy"
    )


def test_ensure_proxy_repairs_reachable_stale_proxy(monkeypatch, tmp_path):
    repaired = []
    aligned = []
    monkeypatch.setattr(repl, "_find_atlas_dir", lambda: str(tmp_path))
    monkeypatch.setattr(repl, "_check_url", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        repl,
        "_repair_proxy_capability",
        lambda root, capability: repaired.append((root, capability)) or True,
    )
    monkeypatch.setattr(repl, "_align_workspace", lambda root: aligned.append(root))

    assert repl._ensure_proxy(repl.DEMO_RAW_CAPABILITY)
    assert repaired == [(str(tmp_path), repl.DEMO_RAW_CAPABILITY)]
    assert aligned == [str(tmp_path)]


def test_compose_proxy_rebuild_uses_checkout_source_and_overlay(monkeypatch, tmp_path):
    root = _metal_root(tmp_path)
    calls = []

    def capture(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(repl.subprocess, "run", capture)
    monkeypatch.setattr(repl, "_proxy_supports_capability", lambda *args: True)

    assert repl._rebuild_docker_proxy_for_capability(
        root, str(tmp_path / "project"), repl.DEMO_RAW_CAPABILITY
    )
    command, kwargs = calls[0]
    assert command[:6] == [
        "docker", "compose", "-f", "docker-compose.yml", "-f",
        "docker-compose.macos.yml",
    ]
    assert "--build" in command
    assert command[-1] == "atlas-proxy"
    assert kwargs["env"]["ATLAS_PROJECT_DIR"] == str(tmp_path / "project")


# ---------------------------------------------------------------------------
# Subcommand dispatch: unknown names, --help, and `atlas compose`
# ---------------------------------------------------------------------------

def test_unknown_subcommand_prints_usage_and_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["atlas", "bogus-subcommand"])
    with pytest.raises(SystemExit) as exc:
        repl.run()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown subcommand" in err
    assert "doctor" in err and "compose" in err  # usage list


def test_help_flag_prints_usage_and_exits_0(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["atlas", "--help"])
    with pytest.raises(SystemExit) as exc:
        repl.run()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: atlas" in out
    for name in ("init", "doctor", "model", "onboard", "compose"):
        assert name in out


def test_compose_subcommand_passes_through_to_docker_compose(
        monkeypatch, tmp_path):
    root = _metal_root(tmp_path)
    calls = []

    monkeypatch.setattr(repl.compose_config, "find_atlas_root", lambda: root)
    monkeypatch.setattr(repl.subprocess, "call",
                        lambda cmd, **kwargs: calls.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["atlas", "compose", "ps"])
    with pytest.raises(SystemExit) as exc:
        repl.run()
    assert exc.value.code == 0
    assert calls[0][:2] == ["docker", "compose"]
    # The metal backend's overlay set is honored, args pass through.
    assert "docker-compose.macos.yml" in calls[0]
    assert calls[0][-1] == "ps"


def test_compose_subcommand_requires_checkout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(repl.compose_config, "find_atlas_root",
                        lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["atlas", "compose", "ps"])
    with pytest.raises(SystemExit) as exc:
        repl.run()
    assert exc.value.code == 1
    assert "docker-compose.yml" in capsys.readouterr().err
