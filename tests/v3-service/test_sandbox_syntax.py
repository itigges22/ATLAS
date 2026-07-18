import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException


def _load_sandbox_module():
    module_path = Path(__file__).parents[2] / "sandbox" / "executor_server.py"
    spec = importlib.util.spec_from_file_location("atlas_sandbox_executor", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_syntax_check_accepts_valid_document(tmp_path):
    sandbox = _load_sandbox_module()

    assert sandbox._syntax_check_impl("json", '{"ready": true}', tmp_path) == []


def test_json_syntax_check_rejects_invalid_document(tmp_path):
    sandbox = _load_sandbox_module()

    errors = sandbox._syntax_check_impl("json", '{"ready": }', tmp_path)

    assert errors


def test_xml_syntax_check_rejects_invalid_document(tmp_path):
    sandbox = _load_sandbox_module()

    errors = sandbox._syntax_check_impl("xml", "<root>", tmp_path)

    assert errors


def test_xml_syntax_check_rejects_entity_expansion(tmp_path):
    sandbox = _load_sandbox_module()
    document = """<!DOCTYPE bomb [
      <!ENTITY a "1234567890">
      <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
    ]><root>&b;</root>"""

    errors = sandbox._syntax_check_impl("xml", document, tmp_path)

    assert errors


def test_overlay_write_rejects_symlink_leaf(tmp_path):
    sandbox = _load_sandbox_module()
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not overwrite")
    (root / "candidate.py").symlink_to(outside)

    with pytest.raises(HTTPException):
        sandbox._write_overlay_files(root, {"candidate.py": "attacker content"})

    assert outside.read_text() == "do not overwrite"


def test_overlay_write_rejects_symlink_parent(tmp_path):
    sandbox = _load_sandbox_module()
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "src").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException):
        sandbox._write_overlay_files(root, {"src/candidate.py": "attacker content"})

    assert not (outside / "candidate.py").exists()


def test_shell_overlay_runs_without_mutating_workspace(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = workspace / "app.py"
    original.write_text("raise RuntimeError('real workspace should not run')\n")

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        response = sandbox.run_shell(
            sandbox.ShellRequest(
                command="python3 -m py_compile app.py",
                cwd=str(workspace),
                files={"app.py": "print('candidate overlay')\n"},
            )
        )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base

    assert response.success is True
    assert original.read_text() == "raise RuntimeError('real workspace should not run')\n"


def test_shell_overlay_translates_absolute_workspace_paths(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = workspace / "app.py"
    original.write_text("def broken(:\n")

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        response = sandbox.run_shell(
            sandbox.ShellRequest(
                command=f"python3 -m py_compile {workspace}/app.py",
                cwd=str(workspace),
                files={"app.py": "print('candidate overlay')\n"},
            )
        )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base

    assert response.success is True
    assert original.read_text() == "def broken(:\n"


def test_shell_overlay_rejects_path_traversal(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        with pytest.raises(HTTPException):
            sandbox.run_shell(
                sandbox.ShellRequest(
                    command="true",
                    cwd=str(workspace),
                    files={"../escape.py": "print('nope')\n"},
                )
            )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base


def test_shell_snapshot_skips_external_symlinks(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (workspace / "link.txt").symlink_to(outside)

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        response = sandbox.run_shell(
            sandbox.ShellRequest(
                command="test ! -e link.txt",
                cwd=str(workspace),
                files={"candidate.py": "print('ok')\n"},
            )
        )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base

    assert response.success is True


def test_shell_snapshot_preserves_safe_internal_symlinks(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "real.txt").write_text("inside")
    (workspace / "link.txt").symlink_to("real.txt")

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        response = sandbox.run_shell(
            sandbox.ShellRequest(
                command="test -L link.txt && test \"$(cat link.txt)\" = inside",
                cwd=str(workspace),
                files={"candidate.py": "print('ok')\n"},
            )
        )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base

    assert response.success is True


def test_shell_snapshot_keeps_small_node_modules(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    package_dir = workspace / "node_modules" / "tiny"
    package_dir.mkdir(parents=True)
    (package_dir / "index.js").write_text("module.exports = 1;\n")

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        response = sandbox.run_shell(
            sandbox.ShellRequest(
                command="test -f node_modules/tiny/index.js",
                cwd=str(workspace),
                files={"candidate.js": "console.log('ok')\n"},
            )
        )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base

    assert response.success is True


def test_shell_snapshot_skips_large_artifacts(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "model.gguf").write_text("large model placeholder")

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    try:
        response = sandbox.run_shell(
            sandbox.ShellRequest(
                command="test ! -e model.gguf",
                cwd=str(workspace),
                files={"candidate.py": "print('ok')\n"},
            )
        )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base

    assert response.success is True


def test_shell_snapshot_fails_when_byte_limit_is_exceeded(tmp_path):
    sandbox = _load_sandbox_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "small.txt").write_text("too many bytes for this test")

    previous_root = sandbox.WORKSPACE_ROOT
    previous_base = sandbox.WORKSPACE_BASE
    previous_limit = sandbox.SHELL_SNAPSHOT_MAX_BYTES
    sandbox.WORKSPACE_ROOT = workspace
    sandbox.WORKSPACE_BASE = tmp_path
    sandbox.SHELL_SNAPSHOT_MAX_BYTES = 4
    try:
        with pytest.raises(HTTPException) as exc:
            sandbox.run_shell(
                sandbox.ShellRequest(
                    command="true",
                    cwd=str(workspace),
                    files={"candidate.py": "print('ok')\n"},
                )
            )
    finally:
        sandbox.WORKSPACE_ROOT = previous_root
        sandbox.WORKSPACE_BASE = previous_base
        sandbox.SHELL_SNAPSHOT_MAX_BYTES = previous_limit

    assert exc.value.status_code == 413
