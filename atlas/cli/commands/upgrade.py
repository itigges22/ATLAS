"""atlas upgrade — staged, restore-on-failure upgrade of the Compose
deployment.

Records a restore point, stages the target images, starts them, waits
for readiness, runs a smoke check, and finalizes — automatically
restoring the prior release if any step fails. See upgrade_engine for
the orchestration; this module supplies the real Docker/health steps.
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional

from atlas.cli import compose as compose_config
from atlas.cli import upgrade_engine as eng


def _compose(atlas_root: str, args: List[str], timeout: int = 600) -> None:
    cmd = compose_config.command(atlas_root, args)
    rc = subprocess.call(cmd, cwd=atlas_root)
    if rc != 0:
        raise eng.UpgradeError(f"`{' '.join(args[:2])}` failed (exit {rc})")


def _snapshot_digests(atlas_root: str) -> Dict[str, str]:
    """Current image digests per service (best-effort; empty on failure —
    the restore point still records the tag + .env backup)."""
    try:
        cmd = compose_config.command(
            atlas_root, ["images", "--format", "json"])
        out = subprocess.check_output(cmd, cwd=atlas_root, text=True,
                                      timeout=60)
    except (subprocess.SubprocessError, OSError):
        return {}
    import json
    digests: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        svc = rec.get("Service") or rec.get("Repository") or ""
        dig = rec.get("ID") or rec.get("Digest") or ""
        if svc and dig:
            digests[svc] = dig
    return digests


def _set_env_tag(atlas_root: str, tag: str) -> None:
    """Rewrite ATLAS_IMAGE_TAG in .env (append if missing)."""
    path = os.path.join(atlas_root, ".env")
    lines: List[str] = []
    found = False
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                if line.strip().startswith("ATLAS_IMAGE_TAG="):
                    lines.append(f"ATLAS_IMAGE_TAG={tag}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"ATLAS_IMAGE_TAG={tag}\n")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    os.replace(tmp, path)


def _readiness(atlas_root: str, timeout_s: int = 180) -> bool:
    """Poll the proxy /ready until 200 or timeout."""
    url = compose_config.service_url("proxy") + "/ready"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _smoke(atlas_root: str) -> bool:
    """Quick doctor as the post-upgrade smoke check."""
    from atlas.cli.commands import doctor
    try:
        rc = doctor.main(["--quick", "--json"])
    except SystemExit as e:
        rc = int(e.code or 0)
    return rc == 0


def _default_steps() -> eng.Steps:
    return eng.Steps(
        snapshot_digests=_snapshot_digests,
        set_env_tag=_set_env_tag,
        pull=lambda root: _compose(root, ["pull"]),
        up=lambda root: _compose(root, ["up", "-d"]),
        readiness=_readiness,
        smoke=_smoke,
        log=lambda m: print(f"  {m}"),
    )


def _stamp() -> str:
    # Filesystem-safe timestamp for the restore-point backup name.
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas upgrade",
        description="Staged upgrade with automatic restore on failure.")
    parser.add_argument("--to", default="latest",
                        help="target image tag (default: latest)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip the post-upgrade doctor smoke check")
    parser.add_argument("--yes", action="store_true",
                        help="don't prompt before starting")
    args = parser.parse_args(argv)

    atlas_root = compose_config.find_atlas_root()
    if not os.path.isfile(os.path.join(atlas_root, "docker-compose.yml")):
        print("atlas upgrade: run from an ATLAS checkout.", file=sys.stderr)
        return 1

    previous = eng.read_env_tag(atlas_root)
    if not args.yes and previous != args.to:
        print(f"Upgrade {previous} → {args.to}. A restore point is recorded "
              "first; a failed upgrade auto-restores the previous release.")
        try:
            if input("Continue? [y/N] ").strip().lower() != "y":
                print("aborted.")
                return 1
        except EOFError:
            print("non-interactive; pass --yes to proceed.", file=sys.stderr)
            return 1

    try:
        result = eng.run_upgrade(atlas_root, args.to, _default_steps(),
                                 _stamp(), run_smoke=not args.skip_smoke)
    except eng.UpgradeError as e:
        print(f"\natlas upgrade: {e}", file=sys.stderr)
        return 1

    if result["status"] == "noop":
        print(result["detail"])
        return 0
    print(f"\nUpgraded to {result['target_tag']}. "
          f"Roll back with: atlas rollback")
    return 0
