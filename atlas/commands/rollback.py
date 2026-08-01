"""atlas rollback — return the deployment to a working release.

With no argument, restores the last upgrade's previous release from the
recorded restore point (.atlas-upgrade/restore-point.json). With
`--to <tag>`, points the deployment at a specific immutable tag.
"""

import argparse
import os
import sys
from typing import List, Optional

from atlas import env as cli_env
from atlas import upgrade_engine as eng
from atlas.commands.upgrade import _default_steps


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas rollback",
        description="Roll back to the previous release or a specific tag.")
    parser.add_argument("--to", default=None,
                        help="roll back to this image tag (default: the "
                             "recorded restore point)")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    atlas_root = cli_env.atlas_root()
    if not os.path.isfile(os.path.join(atlas_root, "docker-compose.yml")):
        print("atlas rollback: run from an ATLAS checkout.", file=sys.stderr)
        return 1

    if not args.to:
        point = eng.read_restore_point(atlas_root)
        if not point:
            print("atlas rollback: no restore point found. Use "
                  "`atlas rollback --to <tag>` to target a release.",
                  file=sys.stderr)
            return 1
        target = point["previous_tag"]
    else:
        target = eng.normalize_image_tag(args.to)
        if target != args.to:
            print(f"(image tags carry no leading v — using {target})")

    if not args.yes:
        current = eng.read_env_tag(atlas_root)
        print(f"Roll back {current} → {target}.")
        try:
            if input("Continue? [y/N] ").strip().lower() != "y":
                print("aborted.")
                return 1
        except EOFError:
            print("non-interactive; pass --yes to proceed.", file=sys.stderr)
            return 1

    try:
        result = eng.run_rollback(atlas_root, _default_steps(),
                                  target_tag=target)
    except eng.UpgradeError as e:
        print(f"atlas rollback: {e}", file=sys.stderr)
        return 1
    print(f"Rolled back to {result['target_tag']}.")
    return 0
