"""Proxy runtime tests — compose lifecycle behind ensure_proxy()."""

from types import SimpleNamespace

from atlas import runtime


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

    monkeypatch.setattr(runtime.subprocess, "run", capture)
    monkeypatch.setattr(runtime, "_check_url", lambda *args, **kwargs: True)
    assert runtime._recreate_docker_proxy(root, str(tmp_path / "project")) is True
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

    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(runtime.subprocess, "run", capture)
    assert runtime._docker_compose_owns_proxy(root) is True
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
    assert runtime._proxy_capabilities("http://proxy") == set()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"not json"),
    )
    assert runtime._proxy_capabilities("http://proxy") == set()


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
    assert runtime._proxy_supports_capability(
        runtime.DEMO_RAW_CAPABILITY, "http://proxy"
    )


def test_ensure_proxy_repairs_reachable_stale_proxy(monkeypatch, tmp_path):
    repaired = []
    aligned = []
    monkeypatch.setattr(runtime, "_find_atlas_dir", lambda: str(tmp_path))
    monkeypatch.setattr(runtime, "_check_url", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        runtime,
        "_repair_proxy_capability",
        lambda root, capability: repaired.append((root, capability)) or True,
    )
    monkeypatch.setattr(runtime, "_align_workspace", lambda root: aligned.append(root))

    assert runtime.ensure_proxy(runtime.DEMO_RAW_CAPABILITY)
    assert repaired == [(str(tmp_path), runtime.DEMO_RAW_CAPABILITY)]
    assert aligned == [str(tmp_path)]


def test_compose_proxy_rebuild_uses_checkout_source_and_overlay(monkeypatch, tmp_path):
    root = _metal_root(tmp_path)
    calls = []

    def capture(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", capture)
    monkeypatch.setattr(runtime, "_proxy_supports_capability", lambda *args: True)

    assert runtime._rebuild_docker_proxy_for_capability(
        root, str(tmp_path / "project"), runtime.DEMO_RAW_CAPABILITY
    )
    command, kwargs = calls[0]
    assert command[:6] == [
        "docker", "compose", "-f", "docker-compose.yml", "-f",
        "docker-compose.macos.yml",
    ]
    assert "--build" in command
    assert command[-1] == "atlas-proxy"
    assert kwargs["env"]["ATLAS_PROJECT_DIR"] == str(tmp_path / "project")
