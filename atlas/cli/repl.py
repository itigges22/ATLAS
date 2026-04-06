"""Interactive REPL — the main ATLAS interface."""

import sys
import os

from atlas.cli import display, client
from atlas.cli.commands import solve, status, bench


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
    """Main entry point."""
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
