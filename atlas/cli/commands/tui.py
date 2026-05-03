"""atlas tui — Bubbletea TUI client (PC-062).

Replaces the Aider-based chat UI with a native Bubbletea TUI built into
the tui/ directory. Three panes (pipeline, events, chat) feed off
two SSE streams from atlas-proxy: /events (typed envelope visibility)
and /v1/agent (per-turn chat protocol). See docs/CLI.md for the full
keymap and slash-command reference.

Launch strategy:
  1. Locate the `atlas-tui` binary on PATH or in ~/.local/bin
  2. If missing and Go 1.24+ is available → build from tui/
  3. If still missing → print install instructions and exit
  4. Otherwise → ensure atlas-proxy is running, then exec the TUI

Pass-through args after `atlas tui` go straight to the binary, so e.g.
`atlas tui --proxy http://other:8090` works as expected.
"""

import os
import shutil
import subprocess
import sys
from typing import List, Optional


def _find_atlas_dir() -> str:
    """Walk up from this file looking for the ATLAS repo root."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "tui", "go.mod")):
            return d
        d = os.path.dirname(d)
    if os.path.exists(os.path.join(os.getcwd(), "tui", "go.mod")):
        return os.getcwd()
    return ""


def _find_tui_binary(atlas_dir: str) -> Optional[str]:
    """Locate the atlas-tui binary. Returns absolute path or None."""
    on_path = shutil.which("atlas-tui")
    if on_path:
        return on_path
    for cand in (
        os.path.expanduser("~/.local/bin/atlas-tui"),
        os.path.join(atlas_dir, "tui", "atlas-tui") if atlas_dir else None,
    ):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _build_tui(atlas_dir: str) -> Optional[str]:
    """Build atlas-tui from source. Returns the built path or None."""
    go_bin = shutil.which("go")
    if not go_bin or not atlas_dir:
        return None
    src = os.path.join(atlas_dir, "tui")
    if not os.path.isfile(os.path.join(src, "go.mod")):
        return None
    output = os.path.expanduser("~/.local/bin/atlas-tui")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print("  Building atlas-tui from source...")
    try:
        result = subprocess.run(
            [go_bin, "build", "-o", output, "."],
            cwd=src, capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            print(f"  Built: {output}")
            return output
        print(f"  Build failed: {result.stderr[:240]}")
    except Exception as e:
        print(f"  Build failed: {e}")
    return None


def main(argv: List[str]) -> int:
    """Entry point for `atlas tui [...args]`."""
    atlas_dir = _find_atlas_dir()

    binary = _find_tui_binary(atlas_dir)
    if not binary:
        binary = _build_tui(atlas_dir)
    if not binary:
        sys.stderr.write(
            "atlas tui: atlas-tui binary not found and Go is not "
            "available to build it.\n"
            "Install Go 1.24+ (https://go.dev/dl/) or build manually:\n"
            "  cd tui && go build -o ~/.local/bin/atlas-tui .\n"
        )
        return 1

    # Ensure proxy is up AND its /workspace bind covers the user's cwd.
    # _ensure_proxy() handles both: health check + auto-realign via
    # force-recreate when cwd is outside the bind. The recreate is ~5s
    # — fast enough to do unconditionally, and necessary for tool calls
    # to work (the proxy can only read/write paths under its mount).
    from atlas.cli.repl import _ensure_proxy, PROXY_URL
    if not _ensure_proxy():
        sys.stderr.write(
            "atlas tui: atlas-proxy not running and could not be "
            "started locally. Start it manually (docker compose up "
            "atlas-proxy) and rerun.\n"
        )
        return 1

    # Default --proxy from env/repl.py if the user didn't override it.
    args = list(argv)
    if "--proxy" not in args:
        args = ["--proxy", PROXY_URL] + args

    # Default --log to a stable path under ~/.cache so debugging the
    # TUI doesn't require the user to remember a flag. Alt-screen mode
    # makes it impractical to copy text out of the live view; the log
    # is the operator's read-only record of what the TUI received.
    # Override with --log <path> or ATLAS_TUI_LOG; "off" disables.
    if "--log" not in args and not os.environ.get("ATLAS_TUI_LOG"):
        log_dir = os.path.expanduser("~/.cache/atlas-tui")
        os.makedirs(log_dir, exist_ok=True)
        args = ["--log", os.path.join(log_dir, "debug.log")] + args
        print(f"  TUI debug log: {os.path.join(log_dir, 'debug.log')}")
    elif os.environ.get("ATLAS_TUI_LOG", "").lower() == "off":
        # Explicit opt-out — strip any default we'd set.
        os.environ.pop("ATLAS_TUI_LOG", None)

    # exec, not run — the TUI takes over the terminal and we want
    # signals (Ctrl+C, window resize) routed to it directly.
    try:
        os.execv(binary, [binary, *args])
    except OSError as e:
        sys.stderr.write(f"atlas tui: exec failed: {e}\n")
        return 1
    return 0  # unreachable
