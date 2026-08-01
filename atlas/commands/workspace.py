"""atlas workspace — show or move the directory ATLAS operates on.

    atlas workspace              show the proxy + sandbox binds vs this directory
    atlas workspace align        point both at this directory (recreates them)

The alignment itself is runtime._align_workspace, the same code `atlas tui`
runs on launch — which is why the TUI already works in whatever directory you
started it from. This exposes it as a command so a client that is not the TUI
(the VS Code extension, a script) can ask for the same thing instead of
reimplementing the recreate and getting the two binds out of step.
"""

import argparse
import os
import sys
from typing import List, Optional

from atlas import env as cli_env
from atlas import runtime


def _binds() -> tuple:
    """(proxy, sandbox) host paths bound at /workspace, None when not Docker."""
    return runtime._docker_proxy_workspace(), runtime._docker_sandbox_workspace()


def _covers(bound: Optional[str], target: str) -> bool:
    """True when `target` is inside `bound` (or is it). Mirrors the rule
    _align_workspace uses, so `atlas workspace` reports what `align` would do."""
    if not bound:
        return False
    try:
        rel = os.path.relpath(target, os.path.realpath(bound))
    except ValueError:  # different drives (Windows); never covered
        return False
    return rel == "." or not rel.startswith("..")


def _show(target: str) -> int:
    proxy, sandbox = _binds()
    if proxy is None and sandbox is None:
        print("  proxy is not running under Docker — it works in its own "
              "process cwd, so there is no bind to align.")
        return 0

    print(f"  this directory : {target}")
    print(f"  proxy bind     : {proxy or '(not running)'}")
    print(f"  sandbox bind   : {sandbox or '(not running)'}")

    if proxy and sandbox and os.path.realpath(proxy) != os.path.realpath(sandbox):
        # The split-brain case: file tools read one tree, run_command uses the
        # other, and every health check stays green. atlas doctor's
        # workspace_mounts check exists for this.
        print("\n  SPLIT: the two binds differ. File tools and run_command are "
              "operating on different trees.\n  Fix: atlas workspace align")
        return 1

    if _covers(proxy, target) and _covers(sandbox, target):
        print("\n  aligned — ATLAS can see this directory.")
        return 0
    print("\n  NOT aligned — ATLAS is working somewhere else, so edits land "
          "outside this directory.\n  Fix: atlas workspace align")
    return 1


def _align(target: str) -> int:
    proxy, sandbox = _binds()
    if proxy is None and sandbox is None:
        print("  proxy is not running under Docker — nothing to align.")
        return 0
    if _covers(proxy, target) and _covers(sandbox, target):
        print(f"  already aligned — {target} is inside the current bind.")
        return 0

    # runtime does the work: it recreates proxy AND sandbox together with the
    # new ATLAS_PROJECT_DIR and waits for both healthy. Recreating one alone is
    # what produces the split above.
    if not runtime._recreate_docker_proxy(cli_env.atlas_root(), target):
        print("  alignment failed — the containers were left as they were.",
              file=sys.stderr)
        return 1
    return _show(target)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas workspace",
        description="Show or move the directory ATLAS operates on.")
    parser.add_argument("action", nargs="?", default="show",
                        choices=["show", "align"],
                        help="show (default) reports the binds; align moves "
                             "them to this directory")
    parser.add_argument("--dir", default=None,
                        help="align to this directory instead of the current one")
    args = parser.parse_args(argv)

    target = os.path.realpath(args.dir or os.getcwd())
    if not os.path.isdir(target):
        print(f"  not a directory: {target}", file=sys.stderr)
        return 2
    return _align(target) if args.action == "align" else _show(target)
