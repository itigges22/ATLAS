"""Interactive REPL — the main ATLAS interface.

Routes to Aider (agent loop) when the proxy is available,
falls back to the built-in REPL for direct LLM interaction.
"""

import sys
import os
import shutil
import subprocess

from atlas.cli import display, client
from atlas.cli.commands import solve, status, bench


PROXY_URL = os.environ.get("ATLAS_PROXY_URL", "http://localhost:8090")


def _proxy_available() -> bool:
    """Check if atlas-proxy is reachable."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"{PROXY_URL}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _find_aider() -> str | None:
    """Find aider binary on PATH."""
    return shutil.which("aider")


def _find_atlas_dir() -> str:
    """Find the ATLAS project root (where .aider.model.settings.yml lives)."""
    # Walk up from this file to find the repo root
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, ".aider.model.settings.yml")):
            return d
        d = os.path.dirname(d)
    # Fallback: check if CWD has it
    if os.path.exists(".aider.model.settings.yml"):
        return os.getcwd()
    return ""


def launch_aider(extra_args: list[str] | None = None):
    """Launch Aider connected to the ATLAS proxy."""
    aider_bin = _find_aider()
    if not aider_bin:
        display.error("Aider not found. Install with: pip install aider-chat")
        display.info("Falling back to built-in REPL...")
        return False

    atlas_dir = _find_atlas_dir()
    settings_file = os.path.join(atlas_dir, ".aider.model.settings.yml") if atlas_dir else ""
    metadata_file = os.path.join(atlas_dir, ".aider.model.metadata.json") if atlas_dir else ""

    env = os.environ.copy()
    env["OPENAI_API_BASE"] = PROXY_URL
    env["OPENAI_API_KEY"] = "atlas-local"

    cmd = [
        aider_bin,
        "--model", "openai/atlas",
        "--edit-format", "whole",
        "--no-show-model-warnings",
        "--no-check-update",
        "--no-auto-commits",
        "--no-pretty",
    ]

    if settings_file and os.path.exists(settings_file):
        cmd.extend(["--model-settings-file", settings_file])
    if metadata_file and os.path.exists(metadata_file):
        cmd.extend(["--model-metadata-file", metadata_file])

    # Pass through any extra args (--message, filenames, etc.)
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


def startup_checks() -> bool:
    """Run startup health checks."""
    fox_ok, fox_model = client.check_fox()
    rag_ok, _ = client.check_rag_api()
    sandbox_ok, _ = client.check_sandbox()

    if fox_ok:
        display.status_block(
            model=fox_model,
            speed="47 tok/s",
            lens="connected" if rag_ok else "unavailable",
            sandbox="ready" if sandbox_ok else "unavailable",
        )
    else:
        display.error(f"Fox not running — {fox_model}")
        display.info("Start Fox first: fox serve --model-path <model.gguf>")
        return False

    if not rag_ok:
        display.warn("Lens unavailable — verification disabled")
    if not sandbox_ok:
        display.warn("Sandbox unavailable — code testing disabled")

    return True


def handle_command(line: str):
    """Dispatch slash commands."""
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        display.goodbye()
        sys.exit(0)

    elif cmd == "/help":
        display.help_text()

    elif cmd == "/status":
        status.status()

    elif cmd == "/solve":
        if not args:
            display.error("Usage: /solve <filename>")
            return
        filepath = args.strip()
        if not os.path.exists(filepath):
            display.error(f"File not found: {filepath}")
            return
        solve.solve_file(filepath)

    elif cmd == "/bench":
        import shlex
        bench_args = shlex.split(args) if args else []
        tasks = 0
        dataset = "livecodebench"
        strategy = "random"
        i = 0
        while i < len(bench_args):
            if bench_args[i] == "--tasks" and i + 1 < len(bench_args):
                tasks = int(bench_args[i + 1])
                i += 2
            elif bench_args[i] == "--dataset" and i + 1 < len(bench_args):
                dataset = bench_args[i + 1]
                i += 2
            elif bench_args[i] == "--strategy" and i + 1 < len(bench_args):
                strategy = bench_args[i + 1]
                i += 2
            else:
                i += 1
        bench.bench(dataset=dataset, max_tasks=tasks, selection_strategy=strategy)

    elif cmd == "/ablation":
        display.warn("Ablation mode coming soon")

    else:
        display.error(f"Unknown command: {cmd}")
        display.info("Type /help for commands")


def run():
    """Main entry point.

    If the atlas-proxy is running and Aider is installed, launches Aider
    connected to the proxy (full agent loop with tool calls).

    Otherwise, falls back to the built-in REPL for direct LLM interaction.
    """
    # Collect any CLI args (--message, filenames, etc.)
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else None

    # If proxy is available and aider is installed, use the agent loop
    if _proxy_available() and _find_aider():
        launch_aider(extra_args)
        return

    # Fall back to built-in REPL
    display.banner()

    if not startup_checks():
        return

    display.separator()

    # Pipe mode
    if not sys.stdin.isatty():
        problem = sys.stdin.read().strip()
        if problem:
            if problem.startswith("/"):
                handle_command(problem)
            else:
                display.user_message(problem[:80] + ("..." if len(problem) > 80 else ""))
                solve.solve(problem, stream=sys.stderr.isatty())
        return

    # Interactive mode
    while True:
        try:
            line = display.prompt()

            if not line:
                continue

            if line.startswith("/"):
                handle_command(line)
            else:
                # User typed/pasted a problem — solve it directly
                display.user_message(line[:80] + ("..." if len(line) > 80 else ""))
                solve.solve(line)

        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:
            display.error(str(e))
