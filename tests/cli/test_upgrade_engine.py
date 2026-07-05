"""Upgrade/rollback engine — the restore-on-failure guarantee.

Uses a fake Steps that records calls and can be told to fail at a chosen
stage, plus a temp ATLAS root with a .env. No Docker, no registry, no
network — the orchestration and restore path are exercised directly.
"""

import json
import os

import pytest

from atlas.cli import upgrade_engine as eng


def _root(tmp_path, tag="v1.0.0"):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / ".env").write_text(
        f"ATLAS_MODEL_FILE=m.gguf\nATLAS_IMAGE_TAG={tag}\nOTHER=keep\n")
    return str(tmp_path)


class FakeSteps:
    """Records the sequence of side-effects; fails on demand."""

    def __init__(self, fail_on=None, ready=True, smoke=True):
        self.calls = []
        self.fail_on = fail_on          # "pull" | "up" | None
        self._ready = ready
        self._smoke = smoke
        self.tag_writes = []

    def as_steps(self, atlas_root):
        def set_tag(root, tag):
            self.calls.append(("set_env_tag", tag))
            self.tag_writes.append(tag)
            # mirror the real writer so read_env_tag reflects it
            _write_env_tag(root, tag)

        def pull(root):
            self.calls.append(("pull", eng.read_env_tag(root)))
            if self.fail_on == "pull":
                raise eng.UpgradeError("pull failed (simulated)")

        def up(root):
            self.calls.append(("up", eng.read_env_tag(root)))
            if self.fail_on == "up":
                raise eng.UpgradeError("up failed (simulated)")

        return eng.Steps(
            snapshot_digests=lambda root: {"atlas-proxy": "sha256:old"},
            set_env_tag=set_tag,
            pull=pull,
            up=up,
            readiness=lambda root: self._ready,
            smoke=lambda root: self._smoke,
            log=lambda m: self.calls.append(("log", m)),
        )


def _write_env_tag(root, tag):
    path = os.path.join(root, ".env")
    lines, found = [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATLAS_IMAGE_TAG="):
                lines.append(f"ATLAS_IMAGE_TAG={tag}\n")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"ATLAS_IMAGE_TAG={tag}\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


def test_successful_upgrade_finalizes_on_target(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps()
    res = eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp1")
    assert res["status"] == "upgraded"
    assert eng.read_env_tag(root) == "v2.0.0"
    # a restore point was recorded pointing back to v1.0.0
    point = eng.read_restore_point(root)
    assert point["previous_tag"] == "v1.0.0"
    assert point["target_tag"] == "v2.0.0"


def test_failed_pull_restores_previous_release(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps(fail_on="pull")
    with pytest.raises(eng.UpgradeError) as ei:
        eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp2")
    assert "Automatically restored" in str(ei.value)
    # .env is back on the previous tag
    assert eng.read_env_tag(root) == "v1.0.0"


def test_failed_readiness_restores(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps(ready=False)
    with pytest.raises(eng.UpgradeError):
        eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp3")
    assert eng.read_env_tag(root) == "v1.0.0"
    # the last up() during restore ran on the previous tag
    up_tags = [tag for (call, tag) in fake.calls if call == "up"]
    assert up_tags[-1] == "v1.0.0"


def test_failed_smoke_restores(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps(smoke=False)
    with pytest.raises(eng.UpgradeError) as ei:
        eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp4")
    assert "smoke" in str(ei.value)
    assert eng.read_env_tag(root) == "v1.0.0"


def test_env_backup_preserves_other_keys(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps(fail_on="up")
    with pytest.raises(eng.UpgradeError):
        eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp5")
    # restore brought back the full .env, not just the tag
    env = (tmp_path / ".env").read_text()
    assert "OTHER=keep" in env
    assert "ATLAS_IMAGE_TAG=v1.0.0" in env


def test_noop_when_already_on_target(tmp_path):
    root = _root(tmp_path, "v2.0.0")
    fake = FakeSteps()
    res = eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp6")
    assert res["status"] == "noop"
    # no pull/up attempted
    assert not [c for c in fake.calls if c[0] in ("pull", "up")]


def test_rollback_to_restore_point(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps()
    eng.run_upgrade(root, "v2.0.0", fake.as_steps(root), "stamp7")
    assert eng.read_env_tag(root) == "v2.0.0"
    # now roll back
    fake2 = FakeSteps()
    res = eng.run_rollback(root, fake2.as_steps(root))
    assert res["status"] == "rolled-back"
    assert res["target_tag"] == "v1.0.0"
    assert eng.read_env_tag(root) == "v1.0.0"


def test_rollback_to_explicit_tag(tmp_path):
    root = _root(tmp_path, "v2.0.0")
    fake = FakeSteps()
    res = eng.run_rollback(root, fake.as_steps(root), target_tag="v1.5.0")
    assert res["status"] == "rolled-back"
    assert eng.read_env_tag(root) == "v1.5.0"


def test_rollback_without_point_errors(tmp_path):
    root = _root(tmp_path, "v1.0.0")
    fake = FakeSteps()
    with pytest.raises(eng.UpgradeError) as ei:
        eng.run_rollback(root, fake.as_steps(root))
    assert "no restore point" in str(ei.value)
