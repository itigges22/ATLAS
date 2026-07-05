"""Upgrade / rollback engine for the Docker Compose deployment.

The design goal is that a *failed* upgrade always leaves a working
install: every upgrade first records a restore point (the current image
tag, the resolved image digests, and a copy of .env), then runs the
staged steps, and on ANY failure automatically restores the prior
release. `atlas rollback` re-applies a restore point on demand.

The Docker/health steps are injected as callables so the orchestration
(backup → stage → start → readiness → smoke → finalize, with
restore-on-failure) is deterministically testable without Docker or a
registry. The default callables shell out to docker compose + doctor.
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

RESTORE_DIR = ".atlas-upgrade"
RESTORE_POINT = "restore-point.json"


class UpgradeError(Exception):
    """A step failed; the engine restores the prior release and re-raises
    a message suitable for the CLI."""


@dataclass
class Steps:
    """Injectable side-effects. Defaults are wired in cli/commands.

    Each raises UpgradeError (or returns False for readiness/smoke) to
    signal failure; the engine converts that into an automatic restore.
    """
    snapshot_digests: Callable[[str], Dict[str, str]]
    set_env_tag: Callable[[str, str], None]
    pull: Callable[[str], None]
    up: Callable[[str], None]
    readiness: Callable[[str], bool]
    smoke: Callable[[str], bool]
    log: Callable[[str], None] = field(default=lambda m: None)


def restore_dir(atlas_root: str) -> str:
    return os.path.join(atlas_root, RESTORE_DIR)


def restore_point_path(atlas_root: str) -> str:
    return os.path.join(restore_dir(atlas_root), RESTORE_POINT)


def _env_path(atlas_root: str) -> str:
    return os.path.join(atlas_root, ".env")


def read_env_tag(atlas_root: str, default: str = "latest") -> str:
    """Current ATLAS_IMAGE_TAG from .env (the deployed release marker)."""
    path = _env_path(atlas_root)
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("ATLAS_IMAGE_TAG="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return val or default
    except OSError:
        pass
    return default


def write_restore_point(atlas_root: str, previous_tag: str, target_tag: str,
                        digests: Dict[str, str], stamp: str) -> str:
    """Record how to get back to the current release, and back up .env.

    stamp is passed in (the CLI provides it) so the engine stays free of
    wall-clock calls.
    """
    rdir = restore_dir(atlas_root)
    os.makedirs(rdir, exist_ok=True)
    env_backup = os.path.join(rdir, f"env-{stamp}.bak")
    src = _env_path(atlas_root)
    if os.path.isfile(src):
        shutil.copy2(src, env_backup)
    point = {
        "stamp": stamp,
        "previous_tag": previous_tag,
        "target_tag": target_tag,
        "previous_digests": digests,
        "env_backup": os.path.relpath(env_backup, atlas_root)
        if os.path.isfile(env_backup) else None,
    }
    # Unique temp in the same dir so concurrent writers don't race on a
    # shared .tmp name; os.replace is atomic on the final path.
    dest = restore_point_path(atlas_root)
    fd, tmp = tempfile.mkstemp(dir=rdir, prefix=".rp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(point, fh, indent=2)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return env_backup


def read_restore_point(atlas_root: str) -> Optional[dict]:
    try:
        with open(restore_point_path(atlas_root)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _restore(atlas_root: str, point: dict, steps: Steps) -> None:
    """Put the prior release back: restore .env (incl. its
    ATLAS_IMAGE_TAG) and bring the stack up on the previous tag."""
    steps.log(f"restoring previous release ({point['previous_tag']})…")
    env_backup = point.get("env_backup")
    if env_backup:
        backup_abs = os.path.join(atlas_root, env_backup)
        if os.path.isfile(backup_abs):
            shutil.copy2(backup_abs, _env_path(atlas_root))
    # Belt-and-suspenders: force the tag back even if the .env restore
    # was a no-op (e.g. .env absent when the restore point was written).
    steps.set_env_tag(atlas_root, point["previous_tag"])
    # The previous images are already present locally (they were running
    # before the upgrade), so a pull failure here must NOT abort the
    # restore — prefer bringing the stack back up on cached images.
    try:
        steps.pull(atlas_root)
    except UpgradeError as e:
        steps.log(f"restore pull skipped ({e}); using cached images")
    steps.up(atlas_root)


def run_upgrade(atlas_root: str, target_tag: str, steps: Steps,
                stamp: str, run_smoke: bool = True) -> dict:
    """Staged upgrade with automatic restore on failure.

    Returns a result dict {status, previous_tag, target_tag, ...}.
    Raises UpgradeError (after a completed restore) if any step fails.
    """
    previous_tag = read_env_tag(atlas_root)
    if previous_tag == target_tag:
        return {"status": "noop", "previous_tag": previous_tag,
                "target_tag": target_tag,
                "detail": f"already on {target_tag}"}

    steps.log(f"upgrade {previous_tag} → {target_tag}")
    digests = steps.snapshot_digests(atlas_root)
    write_restore_point(atlas_root, previous_tag, target_tag, digests, stamp)
    steps.log("restore point recorded")

    try:
        steps.set_env_tag(atlas_root, target_tag)
        steps.log("staging images (pull)…")
        steps.pull(atlas_root)
        steps.log("starting target release…")
        steps.up(atlas_root)
        steps.log("waiting for readiness…")
        if not steps.readiness(atlas_root):
            raise UpgradeError("services did not become ready")
        if run_smoke:
            steps.log("running smoke check…")
            if not steps.smoke(atlas_root):
                raise UpgradeError("post-upgrade smoke check failed")
    except UpgradeError as e:
        point = read_restore_point(atlas_root)
        if point:
            _restore(atlas_root, point, steps)
        raise UpgradeError(
            f"{e}. Automatically restored the previous release "
            f"({previous_tag}).")
    except Exception as e:  # unexpected: still attempt restore
        point = read_restore_point(atlas_root)
        if point:
            _restore(atlas_root, point, steps)
        raise UpgradeError(
            f"unexpected error: {e}. Restored the previous release "
            f"({previous_tag}).")

    steps.log("upgrade succeeded")
    return {"status": "upgraded", "previous_tag": previous_tag,
            "target_tag": target_tag,
            "rollback_hint": f"atlas rollback  # returns to {previous_tag}"}


def run_rollback(atlas_root: str, steps: Steps,
                 target_tag: Optional[str] = None) -> dict:
    """Roll back to the recorded restore point, or to an explicit tag.

    With no argument, restores the last upgrade's previous release. With
    --to TAG, points the deployment at that immutable tag directly.
    """
    if target_tag:
        previous = read_env_tag(atlas_root)
        steps.log(f"rolling back to {target_tag}…")
        steps.set_env_tag(atlas_root, target_tag)
        steps.pull(atlas_root)
        steps.up(atlas_root)
        if not steps.readiness(atlas_root):
            raise UpgradeError(
                f"services did not become ready on {target_tag}; the tag "
                f"may not exist. Previous deployment tag was {previous}.")
        return {"status": "rolled-back", "target_tag": target_tag}

    point = read_restore_point(atlas_root)
    if not point:
        raise UpgradeError(
            "no restore point found — nothing to roll back to. Use "
            "`atlas rollback --to <tag>` to target a specific release.")
    _restore(atlas_root, point, steps)
    if not steps.readiness(atlas_root):
        raise UpgradeError(
            "restored the previous release but it did not become ready.")
    return {"status": "rolled-back", "target_tag": point["previous_tag"]}
