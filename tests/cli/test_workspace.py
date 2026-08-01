"""`atlas workspace` — reports and moves the bind ATLAS operates on.

The command is a thin wrapper over runtime._align_workspace, so these tests
cover the wrapper's own decisions: when it reports aligned, when it refuses,
and that it never recreates one container without the other. The recreate
itself is runtime's and is tested in test_runtime.py.
"""
import os

import pytest

from atlas.commands import workspace


@pytest.fixture
def binds(monkeypatch):
    """Control what the proxy/sandbox appear to be bound to."""
    state = {"proxy": None, "sandbox": None, "recreated": []}

    monkeypatch.setattr(workspace.runtime, "_docker_proxy_workspace",
                        lambda: state["proxy"])
    monkeypatch.setattr(workspace.runtime, "_docker_sandbox_workspace",
                        lambda: state["sandbox"])

    def _recreate(atlas_dir, project_dir):
        state["recreated"].append(project_dir)
        state["proxy"] = state["sandbox"] = project_dir  # runtime moves BOTH
        return True

    monkeypatch.setattr(workspace.runtime, "_recreate_docker_proxy", _recreate)
    return state


def test_a_subdirectory_of_the_bind_counts_as_aligned(binds, tmp_path, capsys):
    """The rule that makes one broad mount serve many folders: cwd inside the
    bind needs no recreate. _align_workspace uses the same test."""
    binds["proxy"] = binds["sandbox"] = str(tmp_path)
    sub = tmp_path / "project"
    sub.mkdir()

    assert workspace.main(["show", "--dir", str(sub)]) == 0
    assert "aligned" in capsys.readouterr().out
    assert workspace.main(["align", "--dir", str(sub)]) == 0
    assert binds["recreated"] == [], "a covered directory must not recreate"


def test_an_unrelated_directory_reports_not_aligned(binds, tmp_path, capsys):
    other = tmp_path / "elsewhere"
    other.mkdir()
    binds["proxy"] = binds["sandbox"] = str(other)
    target = tmp_path / "here"
    target.mkdir()

    assert workspace.main(["show", "--dir", str(target)]) == 1
    assert "NOT aligned" in capsys.readouterr().out


def test_align_moves_both_binds_together(binds, tmp_path):
    """Recreating one alone is the split-brain bug: file tools read one tree
    while run_command uses the other, with every health check green."""
    old = tmp_path / "old"
    old.mkdir()
    binds["proxy"] = binds["sandbox"] = str(old)
    target = tmp_path / "new"
    target.mkdir()

    assert workspace.main(["align", "--dir", str(target)]) == 0
    assert binds["recreated"] == [os.path.realpath(str(target))]
    assert binds["proxy"] == binds["sandbox"]


def test_a_split_bind_is_reported_as_a_split(binds, tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    binds["proxy"], binds["sandbox"] = str(a), str(b)

    assert workspace.main(["show", "--dir", str(a)]) == 1
    assert "SPLIT" in capsys.readouterr().out


def test_no_docker_is_not_an_error(binds, tmp_path, capsys):
    """A local (non-Docker) proxy works in its own cwd — there is no bind to
    align, and saying so beats reporting a failure."""
    assert workspace.main(["show", "--dir", str(tmp_path)]) == 0
    assert workspace.main(["align", "--dir", str(tmp_path)]) == 0
    assert binds["recreated"] == []
    assert "not running under Docker" in capsys.readouterr().out


def test_a_failed_recreate_is_reported(binds, tmp_path, monkeypatch, capsys):
    old = tmp_path / "old"
    old.mkdir()
    binds["proxy"] = binds["sandbox"] = str(old)
    monkeypatch.setattr(workspace.runtime, "_recreate_docker_proxy",
                        lambda *a: False)
    target = tmp_path / "new"
    target.mkdir()

    assert workspace.main(["align", "--dir", str(target)]) == 1
    assert "alignment failed" in capsys.readouterr().err


def test_a_missing_directory_is_rejected(binds, tmp_path):
    assert workspace.main(["align", "--dir", str(tmp_path / "nope")]) == 2


def test_it_is_registered_as_a_subcommand():
    """Unregistered means `atlas workspace` prints 'unknown subcommand'."""
    from atlas.__main__ import _SUBCOMMAND_HELP
    assert "workspace" in {name for name, _ in _SUBCOMMAND_HELP}
