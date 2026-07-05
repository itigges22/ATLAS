"""atlas config — validate and migrate the ATLAS .env configuration.

    atlas config validate [.env]   type/range/enum + unknown/deprecated keys
    atlas config migrate  [.env]   forward-migrate to the current schema
                                   version (writes .env, backs up .env.bak)
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

from atlas.cli import compose as compose_config
from atlas.cli import config_schema as cs


def _read_env(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _default_env() -> str:
    return os.path.join(compose_config.find_atlas_root(), ".env")


def _validate(path: str) -> int:
    env = _read_env(path)
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


def _migrate(path: str) -> int:
    env = _read_env(path)
    migrated, notes = cs.migrate(env)
    for n in notes:
        print(f"  {n}")
    # back up then rewrite as KEY=VALUE lines
    if os.path.isfile(path):
        import shutil
        shutil.copy2(path, path + ".bak")
        print(f"  backed up {path} → {path}.bak")
    tmp = path + ".migrating"
    with open(tmp, "w") as fh:
        for k, v in migrated.items():
            fh.write(f"{k}={v}\n")
    os.replace(tmp, path)
    print(f"config migrate: wrote schema v{cs.CONFIG_SCHEMA_VERSION}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas config")
    sub = parser.add_subparsers(dest="cmd")
    for name in ("validate", "migrate"):
        p = sub.add_parser(name)
        p.add_argument("path", nargs="?", default=None)
    args = parser.parse_args(argv)
    if args.cmd not in ("validate", "migrate"):
        parser.print_help()
        return 1
    path = args.path or _default_env()
    if not os.path.isfile(path):
        print(f"atlas config: no .env at {path}", file=sys.stderr)
        return 1
    return {"validate": _validate, "migrate": _migrate}[args.cmd](path)
