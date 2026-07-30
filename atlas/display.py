"""Terminal display — ANSI formatting for command output.

Zero dependencies. Pure ANSI escape codes.
"""

import sys
import shutil

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GRAY = "\033[90m"

# Bright colors
BRIGHT_GREEN = "\033[92m"
BRIGHT_CYAN = "\033[96m"

# Glyphs
BOX_H = "─"
BULLET = "●"
CHECK = "✓"
CROSS = "✗"
DIAMOND = "◆"


# ---------------------------------------------------------------------------
# Unicode-safe output
#
# CLI commands emit a small fixed set of unicode glyphs (em-dash, checkmark,
# box drawing). On an ASCII-only stdout (LANG=C over SSH, stdout piped
# through a logger that defaulted to ASCII) a bare print() crashes with
# UnicodeEncodeError on the first em-dash, so command modules print through
# safe_print(), which degrades that glyph set instead.
# ---------------------------------------------------------------------------

def supports_unicode() -> bool:
    """True when stdout can strictly encode the unicode glyphs we emit."""
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if not enc:
        return False
    try:
        # Round-trip the chars we actually emit: em-dash + checkmark.
        "—✓".encode(enc, errors="strict")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Resolved at import time, like the color constants above.
UNICODE_OK = supports_unicode()

# Glyphs that degrade to ASCII when stdout can't encode them.
DASH = "—" if UNICODE_OK else "--"
OK_MARK = "✓" if UNICODE_OK else "[ok]"
NO_MARK = "✗" if UNICODE_OK else "[x]"
WARN_MARK = "⚠" if UNICODE_OK else "[!]"


def safe_print(s: str = "") -> None:
    """print() that survives an ASCII-only stdout.

    On a unicode-capable terminal this is a plain print(). Otherwise the
    known glyphs are rewritten to ASCII equivalents, then anything left is
    encoded with replacement so the command never dies mid-report.
    """
    if UNICODE_OK:
        try:
            print(s)
            return
        except UnicodeEncodeError:
            pass  # encoding probe was wrong for this string — degrade below
    s = (s.replace("—", "--")
          .replace("✓", "OK")
          .replace("✗", "X")
          .replace("⚠", "!")
          .replace("→", "->")
          .replace("│", "|")
          .replace("╭", "+").replace("╮", "+")
          .replace("╰", "+").replace("╯", "+")
          .replace("─", "-"))
    print(s.encode("ascii", errors="replace").decode("ascii"))


def w() -> int:
    """Terminal width."""
    return shutil.get_terminal_size().columns


def _write(s: str):
    sys.stdout.write(s)
    sys.stdout.flush()


# ── Chrome ───────────────────────────────────────────────

def separator():
    tw = min(w(), 64)
    print(f"  {GRAY}{BOX_H * (tw - 4)}{RESET}")


def phase_label(name: str):
    """Phase indicator."""
    print(f"\n  {GRAY}{BULLET}{RESET} {CYAN}{name}{RESET}")


# ── Info/Warning/Error ───────────────────────────────────

def info(msg: str):
    print(f"  {BLUE}{DIAMOND}{RESET} {msg}")


def success(msg: str):
    print(f"  {BRIGHT_GREEN}{CHECK}{RESET} {msg}")


def error(msg: str):
    print(f"  {RED}{CROSS}{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}{DIAMOND}{RESET} {msg}")


# ── Progress ─────────────────────────────────────────────

def progress_bar(current: int, total: int, pass_count: int, label: str = ""):
    """Inline progress bar for benchmarks."""
    bar_w = max(1, min(w() - 50, 25))
    filled = int(bar_w * current / max(total, 1))
    bar = f"{BRIGHT_CYAN}{'█' * filled}{GRAY}{'░' * (bar_w - filled)}{RESET}"
    pass_rate = pass_count / max(current, 1) * 100
    line = f"\r  {bar} {BOLD}{current}{RESET}/{total}  {BRIGHT_GREEN}{pass_rate:.1f}%{RESET}  {DIM}{label}{RESET}"
    _write(line)


def progress_done():
    print()
