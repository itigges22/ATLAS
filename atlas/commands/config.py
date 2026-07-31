"""atlas config — validate and migrate the ATLAS .env configuration.

    atlas config validate [.env]   type/range/enum + unknown/deprecated keys
    atlas config migrate  [.env]   forward-migrate to the current schema
                                   version (writes .env, backs up .env.bak)
"""

import argparse
import os
import sys
from typing import List, Optional

from atlas import compose as compose_config
from atlas import env as cli_env
from atlas import config_schema as cs


def _default_env() -> str:
    return os.path.join(cli_env.atlas_root(), ".env")


def _validate(path: str) -> int:
    env = compose_config.read_env_path(path)
    result = cs.validate(env)
    for w in result["warnings"]:
        print(f"  warning: {w}")
    for e in result["errors"]:
        print(f"  ERROR:   {e}")
    if result["errors"]:
        print(f"config validate: FAILED ({len(result['errors'])} errors)")
        return 1
    print(f"config validate: OK ({len(result['warnings'])} warnings)")
    return 0


def _migrate(path: str, dry_run: bool = False) -> int:
    env = compose_config.read_env_path(path)
    _migrated, notes = cs.migrate(env)
    for n in notes:
        print(f"  {n}")

    # Determine which keys are dropped (deprecated) and whether the
    # schema-version stamp needs adding — but rewrite LINE-BY-LINE so
    # comments, blank lines, and formatting survive the migration.
    dropped = {k for k in env if k not in _migrated}
    have_version = "ATLAS_CONFIG_SCHEMA_VERSION" in env

    if dry_run:
        added = [] if have_version else ["ATLAS_CONFIG_SCHEMA_VERSION"]
        print(f"config migrate (preview): +{len(added)} -{len(dropped)} keys, "
              f"target schema v{cs.CONFIG_SCHEMA_VERSION}")
        if dropped:
            print("  would remove: " + ", ".join(sorted(dropped)))
        if added:
            print("  would add:    " + ", ".join(added))
        print("  (no changes written — drop --dry-run to apply)")
        return 0

    if os.path.isfile(path):
        import shutil
        shutil.copy2(path, path + ".bak")
        print(f"  backed up {path} → {path}.bak")

    out_lines = []
    with open(path) as fh:
        for raw in fh:
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in dropped:
                    continue  # drop deprecated key line (comments preserved)
                if key == "ATLAS_CONFIG_SCHEMA_VERSION":
                    # Rewrite a stale stamp to the migration target —
                    # cs.migrate() force-sets it; the file must match.
                    out_lines.append(
                        f"ATLAS_CONFIG_SCHEMA_VERSION="
                        f"{cs.CONFIG_SCHEMA_VERSION}\n")
                    continue
            out_lines.append(raw if raw.endswith("\n") else raw + "\n")
    if not have_version:
        out_lines.append(
            f"ATLAS_CONFIG_SCHEMA_VERSION={cs.CONFIG_SCHEMA_VERSION}\n")

    tmp = path + ".migrating"
    with open(tmp, "w") as fh:
        fh.writelines(out_lines)
    os.replace(tmp, path)
    print(f"config migrate: wrote schema v{cs.CONFIG_SCHEMA_VERSION} "
          "(comments preserved)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas config")
    sub = parser.add_subparsers(dest="cmd")
    for name in ("validate", "migrate"):
        p = sub.add_parser(name)
        p.add_argument("path", nargs="?", default=None)
        if name == "migrate":
            p.add_argument("--dry-run", action="store_true",
                           help="preview changes without writing")
    args = parser.parse_args(argv)
    if args.cmd not in ("validate", "migrate"):
        parser.print_help()
        return 1
    path = args.path or _default_env()
    if not os.path.isfile(path):
        print(f"atlas config: no .env at {path}", file=sys.stderr)
        return 1
    if args.cmd == "migrate":
        return _migrate(path, dry_run=getattr(args, "dry_run", False))
    return _validate(path)
