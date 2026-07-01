"""Public policy tests for backend-aware Compose command construction."""

import pathlib

import pytest

from atlas.cli import compose


def _root(tmp_path: pathlib.Path) -> str:
    for name in (
        "docker-compose.yml",
        "docker-compose.macos.yml",
        "docker-compose.rocm.yml",
        "docker-compose.vulkan.yml",
    ):
        (tmp_path / name).write_text("services: {}\n")
    return str(tmp_path)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cuda", []),
        ("metal", ["-f", "docker-compose.yml", "-f",
                   "docker-compose.macos.yml"]),
        ("rocm", ["-f", "docker-compose.yml", "-f",
                  "docker-compose.rocm.yml"]),
        ("vulkan", ["-f", "docker-compose.yml", "-f",
                    "docker-compose.vulkan.yml"]),
    ],
)
def test_backend_selects_expected_compose_files(tmp_path, backend, expected):
    assert compose.file_args(_root(tmp_path), backend=backend,
                             environ={}) == expected


def _root_with_cpu(tmp_path: pathlib.Path) -> str:
    """Root that also carries the CPU overlay, for the GPU-less path."""
    for name in (
        "docker-compose.yml",
        "docker-compose.vulkan.yml",
        "docker-compose.cpu.yml",
    ):
        (tmp_path / name).write_text("services: {}\n")
    return str(tmp_path)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("vulkan", ["-f", "docker-compose.yml", "-f",
                    "docker-compose.vulkan.yml"]),
        # GPU-less/headless CPU path stacks the Vulkan + CPU overlays, in
        # merge order, matching the bootstrap's compose_files_args.
        ("cpu", ["-f", "docker-compose.yml", "-f", "docker-compose.vulkan.yml",
                 "-f", "docker-compose.cpu.yml"]),
    ],
)
def test_gpuless_backends_select_overlay_stack(tmp_path, backend, expected):
    assert compose.file_args(_root_with_cpu(tmp_path), backend=backend,
                             environ={}) == expected


@pytest.mark.parametrize("backend", ["", None, "unknown", "cuda"])
def test_unknown_or_empty_backend_yields_base_only(tmp_path, backend):
    """cuda/nvidia and any unrecognized (or absent) backend keep Compose's
    default discovery — no explicit -f flags, base file only."""
    assert compose.file_args(_root_with_cpu(tmp_path), backend=backend,
                             environ={}) == []


def test_cpu_backend_reports_missing_second_overlay(tmp_path):
    """The CPU stack needs BOTH overlays; a missing one fails before docker."""
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "docker-compose.vulkan.yml").write_text("services: {}\n")
    with pytest.raises(FileNotFoundError, match="docker-compose.cpu.yml"):
        compose.file_args(str(tmp_path), backend="cpu", environ={})


def test_shell_environment_wins_over_dotenv_values(tmp_path):
    root = _root(tmp_path)
    (tmp_path / ".env").write_text("ATLAS_BACKEND=rocm\n")
    assert compose.resolve_backend(
        root,
        values={"ATLAS_BACKEND": "vulkan"},
        environ={"ATLAS_BACKEND": "metal"},
    ) == "metal"


def test_command_places_overlay_before_operation(tmp_path):
    root = _root(tmp_path)
    cmd = compose.command(root, ["up", "-d", "atlas-proxy"],
                          backend="metal", environ={})
    assert cmd == [
        "docker", "compose", "-f", "docker-compose.yml", "-f",
        "docker-compose.macos.yml", "up", "-d", "atlas-proxy",
    ]


def test_missing_required_overlay_fails_before_docker(tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    with pytest.raises(FileNotFoundError, match="docker-compose.macos.yml"):
        compose.command(str(tmp_path), ["up", "-d"], backend="metal",
                        environ={})


# ---------------------------------------------------------------------------
# .env parsing edge cases
# ---------------------------------------------------------------------------

def test_read_env_file_empty_value_with_inline_comment(tmp_path):
    """`KEY= # comment` parses as an empty value (compose semantics),
    not as the literal string "# comment"."""
    (tmp_path / ".env").write_text(
        "EMPTY_WITH_COMMENT= # explanation\n"
        "PORT=8080  # note\n"
        "HEX=#fff\n"
        "PLAIN=value\n"
    )
    values = compose.read_env_file(str(tmp_path))
    assert values["EMPTY_WITH_COMMENT"] == ""
    assert values["PORT"] == "8080"
    assert values["HEX"] == "#fff"  # no whitespace before '#': not a comment
    assert values["PLAIN"] == "value"


# ---------------------------------------------------------------------------
# Service URL / port resolution
# ---------------------------------------------------------------------------

def test_service_url_explicit_env_var_wins(tmp_path):
    url = compose.service_url(
        "llama", atlas_root=str(tmp_path),
        values={"ATLAS_LLAMA_PORT": "9001"},
        environ={"ATLAS_INFERENCE_URL": "http://inference.test:9000/"})
    assert url == "http://inference.test:9000"


def test_service_url_port_env_var_beats_dotenv(tmp_path):
    url = compose.service_url(
        "proxy", atlas_root=str(tmp_path),
        values={"ATLAS_PROXY_PORT": "7777"},
        environ={"ATLAS_PROXY_PORT": "8888"})
    assert url == "http://localhost:8888"


def test_service_url_reads_port_from_dotenv(tmp_path):
    (tmp_path / ".env").write_text("ATLAS_SANDBOX_PORT=31111\n")
    url = compose.service_url("sandbox", atlas_root=str(tmp_path), environ={})
    assert url == "http://localhost:31111"


def test_service_url_falls_back_to_default(tmp_path):
    assert compose.service_url("lens", atlas_root=str(tmp_path),
                               environ={}) == "http://localhost:8099"
    assert compose.service_url("v3", atlas_root=str(tmp_path),
                               environ={}) == "http://localhost:8070"


def test_service_port_resolution_order(tmp_path):
    (tmp_path / ".env").write_text("ATLAS_LLAMA_PORT=9002\n")
    assert compose.service_port("llama", atlas_root=str(tmp_path),
                                environ={}) == "9002"
    assert compose.service_port("llama", atlas_root=str(tmp_path),
                                environ={"ATLAS_LLAMA_PORT": "9003"}) == "9003"


# ---------------------------------------------------------------------------
# Compose container resolution
# ---------------------------------------------------------------------------

def test_container_id_uses_compose_ps_output(tmp_path, monkeypatch):
    root = _root(tmp_path)
    calls = []

    def capture(cmd, **kwargs):
        calls.append(cmd)
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout="abc123def\n", stderr="")

    monkeypatch.setattr(compose.subprocess, "run", capture)
    cid = compose.container_id(root, "geometric-lens", environ={},
                               fallback="atlas-geometric-lens-1")
    assert cid == "abc123def"
    assert calls[0][:2] == ["docker", "compose"]
    assert calls[0][-2:] == ["-q", "geometric-lens"]


def test_container_id_falls_back_when_compose_fails(tmp_path, monkeypatch):
    root = _root(tmp_path)

    def boom(cmd, **kwargs):
        raise FileNotFoundError("docker not installed")

    monkeypatch.setattr(compose.subprocess, "run", boom)
    cid = compose.container_id(root, "geometric-lens", environ={},
                               fallback="atlas-geometric-lens-1")
    assert cid == "atlas-geometric-lens-1"


def test_container_id_falls_back_when_service_down(tmp_path, monkeypatch):
    root = _root(tmp_path)

    def capture(cmd, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(compose.subprocess, "run", capture)
    cid = compose.container_id(root, "sandbox", environ={},
                               fallback="atlas-sandbox-1")
    assert cid == "atlas-sandbox-1"
