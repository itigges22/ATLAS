"""Terminal display — ANSI formatting, streaming, panels.

Zero dependencies. Pure ANSI escape codes + cursor control.
Inspired by OpenCode/Crush terminal UI design.
"""

import sys
import shutil
import os

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
STRIKETHROUGH = "\033[9m"

# Colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
GRAY = "\033[90m"

# Bright colors
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Background
BG_GRAY = "\033[48;5;236m"
BG_DARK = "\033[48;5;233m"
BG_BLUE = "\033[44m"

# Cursor control
CLEAR_LINE = "\033[2K"
CURSOR_UP = "\033[A"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
SAVE_CURSOR = "\033[s"
RESTORE_CURSOR = "\033[u"

# Box drawing
BOX_H = "─"
BOX_V = "│"
BOX_TL = "╭"
BOX_TR = "╮"
BOX_BL = "╰"
BOX_BR = "╯"
BOX_T = "┬"
BOX_B = "┴"
BULLET = "●"
ARROW = "→"
CHECK = "✓"
CROSS = "✗"
DIAMOND = "◆"


def w() -> int:
    """Terminal width."""
    return shutil.get_terminal_size().columns


def h() -> int:
    """Terminal height."""
    return shutil.get_terminal_size().lines


def clear():
    """Clear screen."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _write(s: str):
    sys.stdout.write(s)
    sys.stdout.flush()


# ── Banner & Chrome ──────────────────────────────────────

def banner():
    """Print startup banner with box drawing."""
    tw = min(w(), 64)
    print()
    print(f"  {BOX_TL}{BOX_H * (tw - 4)}{BOX_TR}")
    title = f" {BOLD}{BRIGHT_CYAN}ATLAS{RESET} {DIM}v3.1{RESET} {GRAY}{DIAMOND} Adaptive Test-time Learning{RESET} "
    # Center the title
    padding = tw - 4 - 42  # rough visible length
    print(f"  {BOX_V}{title}{' ' * max(padding, 0)}{BOX_V}")
    print(f"  {BOX_BL}{BOX_H * (tw - 4)}{BOX_BR}")
    print()


def status_block(model: str, speed: str, lens: str, sandbox: str):
    """Print service status block."""
    _status_item("Model", model, BRIGHT_CYAN)
    _status_item("Speed", speed, BRIGHT_GREEN if "tok" in speed else YELLOW)
    _status_item("Lens", lens, BRIGHT_GREEN if lens == "connected" else RED)
    _status_item("Sandbox", sandbox, BRIGHT_GREEN if sandbox in ("ready", "healthy") else RED)
    print()


def _status_item(label: str, value: str, color: str):
    print(f"  {GRAY}{label:>8}{RESET}  {color}{value}{RESET}")


def separator():
    tw = min(w(), 64)
    print(f"  {GRAY}{BOX_H * (tw - 4)}{RESET}")


def thin_separator():
    tw = min(w(), 64)
    print(f"  {DIM}{BOX_H * (tw - 4)}{RESET}")


# ── Prompt ───────────────────────────────────────────────

def prompt() -> str:
    """Prompt for user input."""
    try:
        return input(f"\n  {BRIGHT_CYAN}{BOLD}{DIAMOND}{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "/quit"


# ── Messages ─────────────────────────────────────────────

def user_message(text: str):
    """Display user's message."""
    print(f"\n  {BOLD}{WHITE}{text}{RESET}")
    thin_separator()


def assistant_label():
    """Print the assistant response header."""
    print(f"\n  {BRIGHT_MAGENTA}{DIAMOND}{RESET} {BOLD}ATLAS{RESET}")


# ── Streaming ────────────────────────────────────────────

def stream_thinking_start():
    """Start thinking stream section."""
    _write(f"  {DIM}")


def stream_thinking_token(text: str):
    """Stream a thinking token (dimmed)."""
    # Indent and dim
    text = text.replace("\n", f"\n  ")
    _write(f"{DIM}{text}{RESET}{DIM}")


def stream_thinking_end():
    """End thinking stream."""
    _write(f"{RESET}\n")


def stream_code_start():
    """Start code stream section."""
    print(f"\n  {GRAY}{BOX_TL}{BOX_H * 3} Code {BOX_H * (min(w(), 60) - 12)}{RESET}")
    _write(f"  {GRAY}{BOX_V}{RESET} ")


def stream_code_token(text: str):
    """Stream a code token."""
    text = text.replace("\n", f"\n  {GRAY}{BOX_V}{RESET} ")
    _write(text)


def stream_code_end():
    """End code stream."""
    print(f"\n  {GRAY}{BOX_BL}{BOX_H * (min(w(), 60) - 4)}{RESET}")


# ── Results ──────────────────────────────────────────────

def phase_label(name: str):
    """Phase indicator."""
    print(f"\n  {GRAY}{BULLET}{RESET} {CYAN}{name}{RESET}")


def energy_score(energy: float, normalized: float):
    """Display Lens verification score."""
    if normalized < 0.3:
        color, icon, label = BRIGHT_GREEN, CHECK, "high confidence"
    elif normalized < 0.6:
        color, icon, label = YELLOW, DIAMOND, "moderate"
    else:
        color, icon, label = RED, CROSS, "low confidence"
    print(f"    {color}{icon}{RESET} C(x) = {energy:.2f}  {color}{label}{RESET}")


def sandbox_result(passed: bool, detail: str = ""):
    """Display sandbox test result."""
    if passed:
        print(f"    {BRIGHT_GREEN}{CHECK}{RESET} sandbox passed {DIM}{detail}{RESET}")
    else:
        print(f"    {RED}{CROSS}{RESET} sandbox failed {DIM}{detail}{RESET}")


def solution_accepted(tokens: int, time_s: float):
    """Solution accepted summary."""
    print(f"\n  {BRIGHT_GREEN}{CHECK} Solution ready{RESET}  {DIM}{tokens} tok  {time_s:.1f}s{RESET}")
    print()


def solution_failed(tokens: int, time_s: float, reason: str = ""):
    """Solution failed summary."""
    msg = f"  {RED}{CROSS} Failed{RESET}  {DIM}{tokens} tok  {time_s:.1f}s{RESET}"
    if reason:
        msg += f"  {DIM}{reason}{RESET}"
    print(f"\n{msg}")
    print()


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
    bar_w = min(w() - 50, 25)
    filled = int(bar_w * current / max(total, 1))
    bar = f"{BRIGHT_CYAN}{'█' * filled}{GRAY}{'░' * (bar_w - filled)}{RESET}"
    pct = current / max(total, 1) * 100
    pass_rate = pass_count / max(current, 1) * 100
    line = f"\r  {bar} {BOLD}{current}{RESET}/{total}  {BRIGHT_GREEN}{pass_rate:.1f}%{RESET}  {DIM}{label}{RESET}"
    _write(line)


def progress_done():
    print()


# ── Help ─────────────────────────────────────────────────

def help_text():
    tw = min(w(), 64)
    print(f"""
  {BOLD}Commands{RESET}
  {GRAY}{BOX_H * (tw - 4)}{RESET}
  {CYAN}/solve{RESET} {DIM}<file>{RESET}       Solve a problem from file
  {CYAN}/bench{RESET} {DIM}[options]{RESET}    Run benchmark
  {CYAN}/ablation{RESET} {DIM}[opts]{RESET}    Run ablation study
  {CYAN}/status{RESET}              Service health
  {CYAN}/help{RESET}                This help
  {CYAN}/quit{RESET}                Exit

  {BOLD}Usage{RESET}
  {GRAY}{BOX_H * (tw - 4)}{RESET}
  Type or paste a coding problem directly.
  Pipe: {DIM}cat problem.txt | atlas{RESET}
""")


def goodbye():
    print(f"\n  {DIM}Bye!{RESET}\n")
