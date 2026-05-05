"""Interactive REPL — the main ATLAS interface.

Proxy launch strategy:
1. If atlas-proxy is already running (any method) → use it
2. If Go is installed → build and launch proxy locally (full CWD file access)
3. Fall back to built-in REPL (no file operations, /solve and /bench only)
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


def _find_go() -> Optional[str]:
    """Find go binary on PATH."""
    return shutil.which("go")


def _find_atlas_dir() -> str:
    """Find the ATLAS repo root (where proxy/ source lives)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, "proxy", "main.go")):
            return d
        d = os.path.dirname(d)
    # Check CWD
    if os.path.exists(os.path.join(os.getcwd(), "proxy", "main.go")):
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
        os.path.join(atlas_dir, "proxy", "atlas-proxy-v2") if atlas_dir else None,
    ]:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def _build_proxy(atlas_dir: str) -> Optional[str]:
    """Build atlas-proxy-v2 from source using Go."""
    go_bin = _find_go()
    if not go_bin or not atlas_dir:
        return None

    proxy_src = os.path.join(atlas_dir, "proxy")
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


def _kill_stale_proxy() -> None:
    """Reap any pre-existing atlas-proxy-v2 process before launching a new
    one. Without this, an orphaned proxy from a previous `atlas` session
    (whose parent died ungracefully — terminal closed, SIGKILL, etc.)
    keeps running with the OLD binary in memory even after a rebuild has
    replaced the binary on disk. Users then see "I just fixed that bug"
    confusion: the on-disk fix is real, but the running process never
    picked it up. We re-find candidates two ways for robustness:

      1. /proc walk for processes whose exe basename is atlas-proxy-v2
         (catches orphans whose ppid=1).
      2. ss/lsof on PROXY_PORT (catches the case where the binary was
         renamed or replaced — /proc/<pid>/exe shows "(deleted)").

    Anything found gets SIGTERM, then SIGKILL after 2s if still alive.
    """
    pids: List[int] = []
    # /proc walk — works on Linux, where this proxy actually runs.
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == os.getpid():
                continue
            try:
                exe = os.readlink(f"/proc/{pid}/exe")
            except (OSError, PermissionError):
                continue
            base = os.path.basename(exe.split(" ")[0])
            # Match exact name or "<name> (deleted)" form.
            if base == "atlas-proxy-v2" or base.startswith("atlas-proxy-v2"):
                pids.append(pid)
    except FileNotFoundError:
        pass

    # Fallback: anything listening on PROXY_PORT.
    try:
        result = subprocess.run(
            ["ss", "-tlnpH", f"sport = :{PROXY_PORT}"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            # Format: ... users:(("name",pid=N,fd=M))
            if "pid=" in line:
                for tok in line.split("pid="):
                    digits = ""
                    for c in tok:
                        if c.isdigit():
                            digits += c
                        else:
                            break
                    if digits:
                        pid = int(digits)
                        if pid != os.getpid() and pid not in pids:
                            pids.append(pid)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    for pid in pids:
        try:
            print(f"  Reaping stale atlas-proxy-v2 (pid {pid})")
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"  WARN: can't kill pid {pid} — not owned by us")
            continue
    if pids:
        # Give SIGTERM ~2s to take, then escalate to SIGKILL.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            alive = [p for p in pids if _pid_alive(p)]
            if not alive:
                break
            time.sleep(0.1)
        for pid in pids:
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _pid_alive(pid: int) -> bool:
    """True if the given pid currently exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _proxy_log_path() -> str:
    """Path the proxy's stdout+stderr is redirected to. We want this in
    one well-known place so panics aren't silently lost (the previous
    /dev/null redirect ate every "[agent] error: ..." line and made
    debugging impossible)."""
    cache = os.path.expanduser("~/.cache/atlas")
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, "proxy.log")


def _launch_local_proxy(proxy_bin: str) -> bool:
    """Launch atlas-proxy-v2 as a local background process."""
    global _proxy_process

    # Defensive: kill any leftover proxy from a previous session before
    # we try to bind PROXY_PORT. Also handles the "rebuilt binary on
    # disk but old binary still in memory" case — the running orphan
    # gets reaped so the next launch picks up the fixed code.
    _kill_stale_proxy()

    env = os.environ.copy()
    env["ATLAS_PROXY_PORT"] = PROXY_PORT
    env["ATLAS_INFERENCE_URL"] = INFERENCE_URL
    env["ATLAS_LLAMA_URL"] = INFERENCE_URL
    env["ATLAS_LENS_URL"] = LENS_URL
    env["ATLAS_SANDBOX_URL"] = SANDBOX_URL
    env["ATLAS_V3_URL"] = V3_URL
    env["ATLAS_AGENT_LOOP"] = "1"
    env["ATLAS_MODEL_NAME"] = MODEL_NAME
    # Pin the proxy's "where to write files" target to the directory the user
    # invoked `atlas` from, overriding the proxy's stale-history heuristics.
    env["ATLAS_WORKSPACE_DIR"] = os.getcwd()

    log_path = _proxy_log_path()
    try:
        log_fd = open(log_path, "ab", buffering=0)
    except OSError as e:
        print(f"  WARN: can't open {log_path}: {e}; proxy logs disabled")
        log_fd = subprocess.DEVNULL

    try:
        _proxy_process = subprocess.Popen(
            [proxy_bin],
            env=env,
            cwd=os.getcwd(),
            stdout=log_fd,
            stderr=subprocess.STDOUT,
        )

        # Register cleanup
        atexit.register(_stop_local_proxy)

        # Wait for health
        for _ in range(30):
            time.sleep(0.5)
            if _check_url(PROXY_URL, timeout=1):
                print(f"  Proxy logs → {log_path}")
                return True

        print("  Proxy started but not responding on health check")
        print(f"  Last 20 lines of {log_path}:")
        try:
            with open(log_path, "rb") as f:
                tail = f.read()[-4096:].decode("utf-8", errors="replace")
            for line in tail.splitlines()[-20:]:
                print(f"    {line}")
        except OSError:
            pass
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


def _docker_workspace_for(container: str) -> Optional[str]:
    """Return the host path bind-mounted to /workspace inside the named
    Docker container, or None if the container isn't running (or docker
    isn't on PATH). See ISSUES.md PC-038, PC-189.
    """
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(
            [
                "docker", "inspect", container,
                "--format",
                "{{range .Mounts}}{{if eq .Destination \"/workspace\"}}{{.Source}}{{end}}{{end}}",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        src = result.stdout.strip()
        return src if src else None
    except Exception:
        return None


def _docker_proxy_workspace() -> Optional[str]:
    return _docker_workspace_for("atlas-atlas-proxy-1")


def _docker_sandbox_workspace() -> Optional[str]:
    return _docker_workspace_for("atlas-sandbox-1")


def _recreate_docker_proxy(atlas_dir: str, project_dir: str) -> bool:
    """Recreate the Docker atlas-proxy AND sandbox containers with a new
    ATLAS_PROJECT_DIR bind. Both services mount ${ATLAS_PROJECT_DIR}:/workspace
    and MUST share the same host path — the agent loop reads files via the
    proxy and runs commands via the sandbox; if their /workspace binds drift,
    the model can read app.py through the proxy but `python app.py` in the
    sandbox 404s. Returns True when both are back up and healthy.
    See PC-038, PC-189.
    """
    if not atlas_dir:
        return False
    print(f"  Aligning proxy + sandbox workspace → {project_dir}")
    env = os.environ.copy()
    env["ATLAS_PROJECT_DIR"] = project_dir
    try:
        result = subprocess.run(
            [
                "docker", "compose", "up", "-d",
                "atlas-proxy", "sandbox",
                "--no-deps", "--no-build", "--force-recreate",
            ],
            cwd=atlas_dir, env=env,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout)[:240]
            print(f"  Workspace recreate failed: {tail}")
            return False
        for _ in range(40):
            time.sleep(0.5)
            if _check_url(PROXY_URL, timeout=1):
                return True
        print("  Recreated but proxy not responding on health check")
        return False
    except Exception as e:
        print(f"  Recreate failed: {e}")
        return False


def _align_workspace(atlas_dir: str) -> None:
    """If the proxy is running in Docker and its /workspace bind doesn't
    cover the current working directory, recreate it so it does. See PC-038.

    Disable with ATLAS_AUTO_WORKSPACE=0 — the user keeps whatever bind the
    proxy was started with.
    """
    if os.environ.get("ATLAS_AUTO_WORKSPACE", "1") == "0":
        return
    cwd = os.path.realpath(os.getcwd())
    proxy_bound = _docker_proxy_workspace()
    if proxy_bound is None:
        # Local proxy (not in Docker) — _launch_local_proxy already pinned
        # ATLAS_WORKSPACE_DIR to os.getcwd(), so nothing to do.
        return
    sandbox_bound = _docker_sandbox_workspace()

    # No-op if BOTH binds cover cwd. Either being out of range triggers a
    # recreate of both — they must share a path (PC-189).
    def _covers_cwd(bound: Optional[str]) -> bool:
        if not bound:
            return False
        try:
            rel = os.path.relpath(cwd, os.path.realpath(bound))
        except ValueError:
            return False
        return rel == "." or not rel.startswith("..")

    if _covers_cwd(proxy_bound) and _covers_cwd(sandbox_bound):
        return
    if not _recreate_docker_proxy(atlas_dir, cwd):
        print("  Continuing with the existing workspace — file operations may not see your project.")


def _ensure_proxy() -> bool:
    """Ensure atlas-proxy is running, launching it locally if needed.

    Strategy:
    1. Already running on PROXY_PORT → use it (and align its workspace
       to the user's CWD if it's a Docker proxy — PC-038)
    2. Go available → build (if needed) and launch locally from CWD
    3. Nothing available → return False
    """
    # Already running?
    if _check_url(PROXY_URL):
        _align_workspace(_find_atlas_dir())
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


def startup_checks() -> bool:
    """Run startup health checks."""
    llm_ok, llm_model = client.check_llama()
    rag_ok, _ = client.check_rag_api()
    sandbox_ok, _ = client.check_sandbox()

    if llm_ok:
        display.status_block(
            model=llm_model,
            speed="~51 tok/s",
            lens="connected" if rag_ok else "unavailable",
            sandbox="ready" if sandbox_ok else "unavailable",
        )
    else:
        display.error(f"llama-server not running — {llm_model}")
        display.info("Start llama-server first (see inference/ entrypoints)")
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
    1. `atlas doctor [...]` → run install diagnostic and exit (PC-053)
    2. `atlas tier`/`atlas tui` → subcommand dispatch
    3. Default (interactive tty) → launch the Bubbletea TUI
    4. Pipe mode (no tty) → built-in REPL with /solve, /bench
    """
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from atlas.cli.commands import doctor
        sys.exit(doctor.main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "tier":
        from atlas.cli.commands import tier
        sys.exit(tier.main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "tui":
        from atlas.cli.commands import tui
        sys.exit(tui.main(sys.argv[2:]))

    # Interactive default → TUI. Pipe mode (e.g. `echo "..." | atlas`) skips
    # the TUI and runs the built-in /solve flow so scripts and CI usage
    # don't get a fullscreen UI they can't drive.
    if sys.stdin.isatty() and sys.stdout.isatty():
        from atlas.cli.commands import tui
        sys.exit(tui.main(sys.argv[1:]))

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
