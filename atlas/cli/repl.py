"""Interactive REPL — the main ATLAS interface.

Proxy launch strategy:
1. If atlas-proxy is already running (any method) → use it
2. If Go is installed → build and launch proxy locally (full CWD file access)
3. If Docker Compose proxy is running → use it (file access limited to ATLAS_PROJECT_DIR mount)
4. Fall back to built-in REPL (no file operations, /solve and /bench only)
"""

import sys
import os
import shutil
import subprocess
import time
import signal
import atexit
from typing import Optional, List

from atlas.cli import display, client
from atlas.cli.commands import solve, status, bench


PROXY_PORT = os.environ.get("ATLAS_PROXY_PORT", "8090")
PROXY_URL = os.environ.get("ATLAS_PROXY_URL", f"http://localhost:{PROXY_PORT}")
INFERENCE_URL = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
LENS_URL = os.environ.get("ATLAS_LENS_URL", "http://localhost:8099")
SANDBOX_URL = os.environ.get("ATLAS_SANDBOX_URL", "http://localhost:30820")
V3_URL = os.environ.get("ATLAS_V3_URL", "http://localhost:8070")
MODEL_NAME = os.environ.get("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")

_proxy_process = None


def _check_url(url: str, timeout: int = 3) -> bool:
    """Check if a URL is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _find_aider() -> Optional[str]:
    """Find aider binary on PATH."""
    return shutil.which("aider")


def _find_go() -> Optional[str]:
    """Find go binary on PATH."""
    return shutil.which("go")


def _find_atlas_dir() -> str:
    """Find the ATLAS repo root (where atlas-proxy/ and .aider configs live)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, "atlas-proxy", "main.go")):
            return d
        d = os.path.dirname(d)
    # Check CWD
    if os.path.exists(os.path.join(os.getcwd(), "atlas-proxy", "main.go")):
        return os.getcwd()
    return ""


def _find_proxy_binary(atlas_dir: str) -> Optional[str]:
    """Find or build the atlas-proxy-v2 binary."""
    # Check PATH first
    on_path = shutil.which("atlas-proxy-v2")
    if on_path:
        return on_path

    # Check common locations
    for candidate in [
        os.path.expanduser("~/.local/bin/atlas-proxy-v2"),
        os.path.join(atlas_dir, "atlas-proxy", "atlas-proxy-v2") if atlas_dir else None,
    ]:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def _build_proxy(atlas_dir: str) -> Optional[str]:
    """Build atlas-proxy-v2 from source using Go."""
    go_bin = _find_go()
    if not go_bin or not atlas_dir:
        return None

    proxy_src = os.path.join(atlas_dir, "atlas-proxy")
    if not os.path.isfile(os.path.join(proxy_src, "main.go")):
        return None

    output = os.path.expanduser("~/.local/bin/atlas-proxy-v2")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"  Building atlas-proxy from source...")
    try:
        result = subprocess.run(
            [go_bin, "build", "-o", output, "."],
            cwd=proxy_src,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"  Built: {output}")
            return output
        else:
            print(f"  Build failed: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  Build failed: {e}")
        return None


def _launch_local_proxy(proxy_bin: str) -> bool:
    """Launch atlas-proxy-v2 as a local background process."""
    global _proxy_process

    env = os.environ.copy()
    env["ATLAS_PROXY_PORT"] = PROXY_PORT
    env["ATLAS_INFERENCE_URL"] = INFERENCE_URL
    env["ATLAS_LLAMA_URL"] = INFERENCE_URL
    env["ATLAS_LENS_URL"] = LENS_URL
    env["ATLAS_SANDBOX_URL"] = SANDBOX_URL
    env["ATLAS_V3_URL"] = V3_URL
    env["ATLAS_AGENT_LOOP"] = "1"
    env["ATLAS_MODEL_NAME"] = MODEL_NAME

    try:
        _proxy_process = subprocess.Popen(
            [proxy_bin],
            env=env,
            cwd=os.getcwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Register cleanup
        atexit.register(_stop_local_proxy)

        # Wait for health
        for _ in range(30):
            time.sleep(0.5)
            if _check_url(PROXY_URL, timeout=1):
                return True

        print("  Proxy started but not responding on health check")
        return False

    except Exception as e:
        print(f"  Failed to start proxy: {e}")
        return False


def _stop_local_proxy():
    """Stop the locally-launched proxy on exit."""
    global _proxy_process
    if _proxy_process and _proxy_process.poll() is None:
        _proxy_process.terminate()
        try:
            _proxy_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proxy_process.kill()


def _ensure_proxy() -> bool:
    """Ensure atlas-proxy is running, launching it locally if needed.

    Strategy:
    1. Already running on PROXY_PORT → use it (Docker Compose or bare metal)
    2. Go available → build (if needed) and launch locally from CWD
    3. Nothing available → return False
    """
    # Already running?
    if _check_url(PROXY_URL):
        return True

    # Try to find or build and launch locally
    atlas_dir = _find_atlas_dir()
    proxy_bin = _find_proxy_binary(atlas_dir)

    if not proxy_bin and _find_go():
        proxy_bin = _build_proxy(atlas_dir)

    if proxy_bin:
        print(f"  Starting local proxy ({os.path.basename(proxy_bin)})...")
        if _launch_local_proxy(proxy_bin):
            print(f"  Proxy ready on port {PROXY_PORT}")
            return True

    return False


def launch_aider(extra_args: Optional[List[str]] = None):
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

    Launch strategy:
    1. Check if proxy is running or can be started locally (Go) or via Docker
    2. If proxy + Aider available → launch Aider (full coding assistant)
    3. Otherwise → fall back to built-in REPL (/solve, /bench only)
    """
    extra_args = sys.argv[1:] if len(sys.argv) > 1 else None

    # Try to get the proxy running
    if _find_aider() and _ensure_proxy():
        launch_aider(extra_args)
        return

    # Fall back to built-in REPL
    if _find_aider() and not _check_url(PROXY_URL):
        display.warn("Proxy not available — Aider needs the proxy for file operations.")
        display.info("Start services first: docker compose up -d")
        display.info("Or install Go 1.24+ for automatic local proxy: https://go.dev/dl/")
        display.separator()

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
                display.user_message(line[:80] + ("..." if len(line) > 80 else ""))
                solve.solve(line)

        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:
            display.error(str(e))
