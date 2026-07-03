"""
Multi-language sandbox execution server.

Supports: Python, JavaScript/TypeScript, Go, Rust, C/C++, Bash/Shell
Provides isolated code execution with resource limits and structured error reporting.

Security / trust model (load-bearing — read before "fixing" CodeQL alerts):
    This service IS the trust boundary. Its entire purpose is to execute
    agent-supplied code and shell commands on behalf of ATLAS. Isolation
    comes from the container, and only from what the container actually
    enforces: read-only rootfs, no-new-privileges, pids_limit (fork-bomb
    stop), tmpfs workspace, /workspace as the only writable host mount,
    and the compose `mem_limit` (written by `atlas init`; unlimited until
    then on a raw `compose up`). Per-call wall-clock is capped in-process
    via MAX_EXECUTION_TIME. Outbound network is NOT restricted — the
    sandbox sits on the regular bridge network so toolchains can fetch
    dependencies; do not describe it as egress-locked. The Python code in
    this file does NOT need to sanitize inputs to subprocess.run, validate
    user-controlled paths inside the workspace, or treat agent-supplied
    code as untrusted — that's the container's job.

    CodeQL routinely flags `py/command-line-injection` and
    `py/path-injection` here. Those alerts are by-design false positives:
    accepting + executing user-controlled commands is the requirement,
    and the cmd-list form (no shell=True at the Python layer) prevents
    Python-level injection. Don't add input validation that would break
    the sandbox's purpose; dismiss the alerts with rationale instead.
"""

import json
import os
import shutil
import signal
import tempfile
import subprocess
import logging
import re
import threading
import time
import uuid
from collections import deque
from typing import Dict, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
# Private-value filter on the root handler: every executor log record
# is masked before serialization (see private_values.py — canonical
# copy notice applies).
from private_values import PrivateValueLogFilter  # noqa: E402
for _h in logging.getLogger().handlers:
    _h.addFilter(PrivateValueLogFilter())
logger = logging.getLogger(__name__)

app = FastAPI(title="ATLAS Code Execution Sandbox")


def _load_service_token() -> str:
    """Internal-auth token (Authorization: Bearer). Empty = auth
    disabled (pre-token behavior; `atlas doctor` warns). Never logged.
    This is the highest-value enforcement point in the stack — /shell
    and /execute run arbitrary commands against the bind-mounted
    workspace."""
    path = os.getenv("ATLAS_SERVICE_TOKEN_FILE",
                     "/run/atlas-secrets/service-token")
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


SERVICE_TOKEN = _load_service_token()


@app.middleware("http")
async def _require_service_token(request, call_next):
    if SERVICE_TOKEN and request.url.path not in ("/health", "/languages"):
        import hmac
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got, f"Bearer {SERVICE_TOKEN}"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={
                "error": "unauthorized",
                "detail": "internal service auth is enabled; send "
                          "Authorization: Bearer <service-token> "
                          "(secrets/service-token)"})
    return await call_next(request)


MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "60"))
WORKSPACE_BASE = Path(os.getenv("WORKSPACE_BASE", "/tmp/sandbox"))

SUPPORTED_LANGUAGES = {
    "python", "py", "python3",
    "javascript", "js", "node",
    "typescript", "ts",
    "go", "golang",
    "rust", "rs",
    "c", "cpp", "c++",
    "bash", "sh", "shell",
    "html", "htm",
    "xml",
    "json",
    "yaml", "yml",
}

def normalize_language(lang: str) -> str:
    lang = lang.lower().strip()
    if lang in ("python", "py", "python3"):
        return "python"
    if lang in ("javascript", "js", "node"):
        return "javascript"
    if lang in ("typescript", "ts"):
        return "typescript"
    if lang in ("go", "golang"):
        return "go"
    if lang in ("rust", "rs"):
        return "rust"
    if lang in ("c",):
        return "c"
    if lang in ("cpp", "c++"):
        return "cpp"
    if lang in ("bash", "sh", "shell"):
        return "bash"
    if lang in ("html", "htm"):
        return "html"
    if lang in ("xml",):
        return "xml"
    if lang in ("json",):
        return "json"
    if lang in ("yaml", "yml"):
        return "yaml"
    return lang


class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    test_code: Optional[str] = None
    requirements: Optional[List[str]] = None
    timeout: int = 30
    # PC-046: Project-context files dropped into the workspace alongside
    # `solution.py` (or the language equivalent) so multi-file imports
    # resolve. Filename keys are relative to the workspace root; each is
    # validated to reject path traversal (`..`) and absolute paths
    # before being written. Used by V3's verified_sandbox and
    # smoke_compile_check to ship the rest of the project so e.g.
    # `import game_logic` resolves to the user's actual game_logic.py.
    files: Optional[Dict[str, str]] = None
    # Optional standard input piped to the run step. None (the default)
    # keeps existing behavior — the process inherits the server's stdin.
    stdin: Optional[str] = None


class ExecuteResponse(BaseModel):
    success: bool
    compile_success: bool
    tests_run: int
    tests_passed: int
    lint_score: Optional[float] = None
    stdout: str
    stderr: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: int


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/languages")
def list_languages():
    """List supported languages and their runtime versions."""
    versions = {}
    checks = {
        "python": ["python3", "--version"],
        "javascript": ["node", "--version"],
        "typescript": ["tsc", "--version"],
        "go": ["go", "version"],
        "rust": ["rustc", "--version"],
        "c": ["gcc", "--version"],
        "cpp": ["g++", "--version"],
        "bash": ["bash", "--version"],
    }
    for lang, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            versions[lang] = result.stdout.strip().split("\n")[0]
        except Exception:
            versions[lang] = "not installed"
    return {"languages": versions}


# ---------------------------------------------------------------------------
# /shell — arbitrary command execution against the bind-mounted workspace.
#
# The agent loop's run_command tool used to fork bash inside the proxy
# container, but the proxy is a slim Go binary with no python/pip/node/etc
# — every "verify your fix" call hit "command not found". We now route
# shell commands through the sandbox, which has the full language matrix
# pre-installed AND has /workspace bind-mounted (rw) at the same path the
# proxy sees, so paths the agent learned from read_file / list_directory
# carry over verbatim.
#
# Safety: the proxy's validateShellCommand blocks only CATASTROPHIC commands
# (whole-project wipe like `rm -rf /`, fork bombs, device destruction) BEFORE
# the call reaches us — ordinary file ops (mv/cp/rm of a file/mkdir) are
# allowed. This endpoint is the executor, not the gate. The real boundary is
# the container: no-new-privileges, read-only rootfs, and /workspace as the
# ONLY writable host mount, so the blast radius is the project folder.
# ---------------------------------------------------------------------------


class ShellRequest(BaseModel):
    command: str
    cwd: Optional[str] = None  # absolute path inside container, defaults to /workspace
    timeout: int = 30          # seconds; capped at MAX_EXECUTION_TIME
    env: Optional[Dict[str, str]] = None
    # Optional ephemeral overlay used by V3 build verification. When
    # present, /shell copies a bounded workspace snapshot into /tmp,
    # overlays these relative file paths, runs the command there, then
    # deletes the snapshot. This lets V3 test a candidate without
    # writing it into the real bind-mounted project.
    files: Optional[Dict[str, str]] = None


class ShellResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: int


# The bind-mounted project root. /workspace in every container
# deployment (compose/K3s mount it there); overridable so the executor
# can run directly on a host — e.g. the E2E acceptance test boots it
# under a pytest tmp dir.
WORKSPACE_ROOT = Path(os.getenv("ATLAS_SANDBOX_WORKSPACE_ROOT", "/workspace"))
SHELL_SNAPSHOT_IGNORE = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "coverage",
    ".coverage",
    "htmlcov",
    ".next",
    "dist",
    "target",
    "secrets",
}
SHELL_SNAPSHOT_IGNORE_SUFFIXES = (
    ".arrow",
    ".bin",
    ".db",
    ".gguf",
    ".gz",
    ".onnx",
    ".parquet",
    ".pt",
    ".safetensors",
    ".sqlite",
    ".tar",
    ".zip",
)
SHELL_SNAPSHOT_MAX_FILES = int(os.getenv("ATLAS_SHELL_SNAPSHOT_MAX_FILES", "20000"))
SHELL_SNAPSHOT_MAX_BYTES = int(os.getenv("ATLAS_SHELL_SNAPSHOT_MAX_BYTES", str(256 * 1024 * 1024)))
SHELL_SNAPSHOT_MAX_FILE_BYTES = int(os.getenv("ATLAS_SHELL_SNAPSHOT_MAX_FILE_BYTES", str(16 * 1024 * 1024)))


def _skip_shell_snapshot_path(path: Path) -> bool:
    name = path.name
    if name in SHELL_SNAPSHOT_IGNORE or name.startswith(".env"):
        return True
    return name.endswith(SHELL_SNAPSHOT_IGNORE_SUFFIXES)


def _copy_symlink_if_safe(src: Path, dest: Path, source_root: Path, snapshot: Path) -> bool:
    try:
        target = src.resolve(strict=True)
        rel_target = target.relative_to(source_root)
        link_target = os.readlink(src)
    except (OSError, ValueError):
        return False

    if os.path.isabs(link_target):
        snapshot_target = snapshot / rel_target
        link_target = os.path.relpath(snapshot_target, dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(link_target, dest)
    return True


def _copy_workspace_snapshot(source: Path, snapshot: Path):
    files_copied = 0
    bytes_copied = 0
    source = source.resolve()

    for root, dirs, filenames in os.walk(source, topdown=True, followlinks=False):
        root_path = Path(root)
        try:
            rel_root = root_path.relative_to(source)
        except ValueError:
            continue
        target_dir = snapshot / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)

        kept_dirs = []
        for name in dirs:
            src_dir = root_path / name
            if _skip_shell_snapshot_path(src_dir):
                continue
            if src_dir.is_symlink():
                _copy_symlink_if_safe(src_dir, target_dir / name, source, snapshot)
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs

        for name in filenames:
            src = root_path / name
            if _skip_shell_snapshot_path(src):
                continue
            if src.is_symlink():
                _copy_symlink_if_safe(src, target_dir / name, source, snapshot)
                continue
            try:
                stat = src.stat()
            except OSError:
                continue
            if not src.is_file():
                continue
            if stat.st_size > SHELL_SNAPSHOT_MAX_FILE_BYTES:
                logger.info(
                    "shell overlay snapshot skipped large file: %s (%d bytes)",
                    src, stat.st_size,
                )
                continue

            files_copied += 1
            bytes_copied += stat.st_size
            if files_copied > SHELL_SNAPSHOT_MAX_FILES:
                raise HTTPException(
                    status_code=413,
                    detail=f"workspace snapshot file limit exceeded ({SHELL_SNAPSHOT_MAX_FILES})",
                )
            if bytes_copied > SHELL_SNAPSHOT_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"workspace snapshot byte limit exceeded ({SHELL_SNAPSHOT_MAX_BYTES})",
                )
            shutil.copy2(src, target_dir / name)


def _safe_overlay_path(name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="overlay file path is required")
    rel = Path(name)
    if rel.is_absolute() or name.startswith("\\") or ".." in rel.parts:
        raise HTTPException(status_code=400, detail=f"unsafe overlay file path: {name!r}")
    return rel


def _write_overlay_files(root: Path, files: Dict[str, str]):
    for name, content in files.items():
        rel = _safe_overlay_path(name)
        content = content if isinstance(content, str) else ""
        if len(content.encode("utf-8", "ignore")) > SHELL_SNAPSHOT_MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"overlay file is too large: {name!r}")

        # Walk directories relative to an already-open snapshot root. O_NOFOLLOW
        # rejects symlink components and closes the resolve-then-write race that
        # a plain Path.write_text() would leave between containment checking and
        # the actual open.
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        parent_fd = os.dup(root_fd)
        try:
            for component in rel.parts[:-1]:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                os.close(parent_fd)
                parent_fd = next_fd

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
            file_fd = os.open(rel.name, flags, 0o600, dir_fd=parent_fd)
            with os.fdopen(file_fd, "w", encoding="utf-8") as overlay_file:
                overlay_file.write(content)
        except OSError as e:
            raise HTTPException(
                status_code=400,
                detail=f"unsafe overlay file path: {name!r}",
            ) from e
        finally:
            os.close(parent_fd)
            os.close(root_fd)


def _snapshot_workspace_with_overlay(files: Dict[str, str]) -> Path:
    snapshot = Path(tempfile.mkdtemp(prefix="shell-", dir=WORKSPACE_BASE))
    try:
        if WORKSPACE_ROOT.exists():
            _copy_workspace_snapshot(WORKSPACE_ROOT, snapshot)
        _write_overlay_files(snapshot, files)
        return snapshot
    except Exception:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def _resolve_shell_cwd(raw_cwd: Optional[str], root: Path) -> Path:
    if not raw_cwd:
        return root
    try:
        requested = Path(raw_cwd)
        if requested.is_absolute():
            rel = requested.resolve(strict=False).relative_to(WORKSPACE_ROOT.resolve(strict=False))
        else:
            rel = requested
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"cwd must be under {WORKSPACE_ROOT}")
        cwd = (root / rel).resolve(strict=False)
        cwd.relative_to(root.resolve(strict=False))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid cwd: {e}")
    if not cwd.exists():
        raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")
    return cwd


def _translate_workspace_command(command: str, root: Path) -> str:
    """Point absolute workspace paths at the ephemeral overlay root.

    V3 build commands often come from project detection and may include
    absolute `/workspace/...` paths. When /shell is running with an overlay
    snapshot, those paths must resolve inside the snapshot, not the real
    bind-mounted checkout.
    """
    workspace = str(WORKSPACE_ROOT.resolve(strict=False))
    replacement = str(root.resolve(strict=False))
    if workspace == replacement:
        return command
    pattern = rf"(?<![\w./-]){re.escape(workspace)}(?=$|/|[^\w./-])"
    return re.sub(pattern, replacement, command)


# ---------------------------------------------------------------------------
# Background jobs (PC-196)
# ---------------------------------------------------------------------------
#
# The agent's verify reflex is "run python app.py / npm start / cargo run
# and curl the result." Foreground /shell can't do that — the server
# blocks until killed. Models work around it with `timeout 5 ... || true`
# hacks that capture the startup banner but tear the server down before
# anything can curl it.
#
# Background jobs solve this cleanly: start_background spawns the
# command and returns a job_id immediately, tail_background lets the
# model peek at stdout/stderr, stop_background kills it. The model can
# now run a server, hit it from another command, then clean up.
#
# Process-global registry. Keyed by job_id (uuid4). Each entry holds:
#   proc:    subprocess.Popen
#   stdout:  deque of recent lines (bounded — long-running servers
#            otherwise eat unbounded memory)
#   stderr:  deque of recent lines
#   command: original command string for diagnostics
#   started: time.time() of spawn
#
# Cleanup: a janitor thread sweeps finished jobs every 30s, dropping
# entries older than BG_RETENTION_SEC. Models can still query a job
# right after it exits to read final output.

BG_MAX_LINES = 500          # ring buffer per stream
BG_MAX_JOBS = 32            # hard cap so a misbehaving model can't OOM us
BG_RETENTION_SEC = 600      # keep finished jobs around for 10 min
BG_MAX_AGE_SEC = 7200       # kill still-running jobs abandoned this long

_bg_jobs: Dict[str, dict] = {}
_bg_lock = threading.Lock()


def _bg_drain_stream(job_id: str, stream_name: str, fh):
    """Tail a Popen pipe in a background thread, append each line to
    the job's deque. Runs until the pipe closes (process exit)."""
    try:
        for raw in iter(fh.readline, ""):
            if raw == "":
                break
            with _bg_lock:
                job = _bg_jobs.get(job_id)
                if job is None:
                    return
                job[stream_name].append(raw.rstrip("\n"))
    except (OSError, ValueError):
        # pipe closed / process gone — normal end of life
        return


def _bg_janitor():
    """Sweep the job table every 30s. Daemon thread. Two responsibilities:

    * Finished jobs: stamp ended_at when the janitor first observes the
      process done (a job the caller never polls would otherwise carry no
      ended_at and be reaped on the first sweep, losing its output before
      the retention window), then drop the entry BG_RETENTION_SEC later.
    * Abandoned running jobs: kill the process group once a job passes
      BG_MAX_AGE_SEC so leaked servers can't exhaust pids_limit; the entry
      is then reaped through the normal finished-job path.
    """
    while True:
        time.sleep(30)
        now = time.time()
        with _bg_lock:
            for jid in list(_bg_jobs.keys()):
                job = _bg_jobs[jid]
                proc = job["proc"]
                if proc.poll() is not None:
                    ended = job.get("ended_at")
                    if not ended:
                        job["ended_at"] = now
                    elif ended < now - BG_RETENTION_SEC:
                        del _bg_jobs[jid]
                elif job["started_at"] < now - BG_MAX_AGE_SEC:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        # best-effort: swallow on failure (caller continues)
                        pass


threading.Thread(target=_bg_janitor, daemon=True).start()


class BackgroundStartRequest(BaseModel):
    command: str
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None


class BackgroundStartResponse(BaseModel):
    job_id: str
    pid: int
    started_at: float


class BackgroundOutputResponse(BaseModel):
    job_id: str
    running: bool
    exit_code: Optional[int]
    stdout: List[str]
    stderr: List[str]
    elapsed_sec: float
    command: str


class BackgroundStopResponse(BaseModel):
    job_id: str
    killed: bool
    exit_code: Optional[int]
    stdout: List[str]
    stderr: List[str]


def _resolve_bg_cwd(raw_cwd: Optional[str]) -> Path:
    """Same workspace-boundary check as run_shell."""
    if raw_cwd:
        try:
            cwd = Path(raw_cwd).resolve()
        except (OSError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"invalid cwd: {e}")
        if not (cwd == WORKSPACE_ROOT or WORKSPACE_ROOT in cwd.parents):
            raise HTTPException(
                status_code=400,
                detail=f"cwd must be under {WORKSPACE_ROOT}, got {cwd}",
            )
        if not cwd.exists():
            raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")
        return cwd
    return WORKSPACE_ROOT


@app.post("/jobs/start", response_model=BackgroundStartResponse)
def background_start(request: BackgroundStartRequest):
    """Spawn a background process and return a job_id.
    Returns immediately — does NOT wait for the process to print
    anything. Caller polls /jobs/{id}/output for stdout/stderr."""
    if not request.command or not request.command.strip():
        raise HTTPException(status_code=400, detail="command is required")
    cwd = _resolve_bg_cwd(request.cwd)
    with _bg_lock:
        if len(_bg_jobs) >= BG_MAX_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"too many active jobs ({BG_MAX_JOBS}). Stop existing jobs first.",
            )
    env = os.environ.copy()
    if request.env:
        env.update(request.env)
    try:
        proc = subprocess.Popen(
            ["bash", "-c", request.command],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,  # so /jobs/stop can kill the whole group
        )
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"spawn failed: {e}")
    job_id = uuid.uuid4().hex[:12]
    job = {
        "proc": proc,
        "command": request.command,
        "started_at": time.time(),
        "stdout": deque(maxlen=BG_MAX_LINES),
        "stderr": deque(maxlen=BG_MAX_LINES),
    }
    with _bg_lock:
        _bg_jobs[job_id] = job
    threading.Thread(target=_bg_drain_stream, args=(job_id, "stdout", proc.stdout), daemon=True).start()
    threading.Thread(target=_bg_drain_stream, args=(job_id, "stderr", proc.stderr), daemon=True).start()
    return BackgroundStartResponse(job_id=job_id, pid=proc.pid, started_at=job["started_at"])


@app.get("/jobs/{job_id}/output", response_model=BackgroundOutputResponse)
def background_output(job_id: str, lines: int = 50):
    """Snapshot of the job's recent stdout/stderr + run state."""
    with _bg_lock:
        job = _bg_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
        proc = job["proc"]
        rc = proc.poll()
        running = rc is None
        if not running and "ended_at" not in job:
            job["ended_at"] = time.time()
        # Snapshot the deques (thread-safe copy under lock)
        stdout = list(job["stdout"])[-max(1, lines):]
        stderr = list(job["stderr"])[-max(1, lines):]
        elapsed = time.time() - job["started_at"]
        cmd = job["command"]
    return BackgroundOutputResponse(
        job_id=job_id, running=running, exit_code=rc,
        stdout=stdout, stderr=stderr, elapsed_sec=elapsed, command=cmd,
    )


@app.post("/jobs/{job_id}/stop", response_model=BackgroundStopResponse)
def background_stop(job_id: str):
    """SIGTERM the process group, wait briefly, SIGKILL if still alive.
    Returns the final stdout/stderr buffer."""
    with _bg_lock:
        job = _bg_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
        proc = job["proc"]
    killed = False
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            # best-effort: swallow on failure (caller continues)
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                # best-effort: swallow on failure (caller continues)
                pass
            proc.wait(timeout=2)
        killed = True
    with _bg_lock:
        job["ended_at"] = time.time()
        stdout = list(job["stdout"])
        stderr = list(job["stderr"])
    return BackgroundStopResponse(
        job_id=job_id, killed=killed, exit_code=proc.poll(),
        stdout=stdout[-50:], stderr=stderr[-50:],
    )


@app.post("/shell", response_model=ShellResponse)
def run_shell(request: ShellRequest):
    """Run a shell command against the bind-mounted workspace."""
    if not request.command or not request.command.strip():
        raise HTTPException(status_code=400, detail="command is required")

    timeout = min(max(1, request.timeout), MAX_EXECUTION_TIME)

    snapshot = None
    root = WORKSPACE_ROOT
    try:
        if request.files:
            snapshot = _snapshot_workspace_with_overlay(request.files)
            root = snapshot
        # Resolve cwd. Default to /workspace (or the temp overlay root);
        # if the caller provides one, require it to live under the
        # workspace boundary. The path must already exist (no auto-mkdir).
        cwd = _resolve_shell_cwd(request.cwd, root)
        command = _translate_workspace_command(request.command, root)

        start = time.time()
        result = _run_cmd(["bash", "-c", command],
                          timeout=timeout, cwd=cwd, env=request.env)
        elapsed_ms = int((time.time() - start) * 1000)

        return ShellResponse(
            success=result["success"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            exit_code=result["returncode"],
            elapsed_ms=elapsed_ms,
        )
    finally:
        if snapshot is not None:
            shutil.rmtree(snapshot, ignore_errors=True)


@app.post("/execute", response_model=ExecuteResponse)
def execute_code(request: ExecuteRequest):
    """Execute code in isolated environment."""
    lang = normalize_language(request.language)

    if lang not in ("python", "javascript", "typescript", "go", "rust", "c", "cpp", "bash"):
        raise HTTPException(
            status_code=400,
            detail=f"Language '{request.language}' not supported. Supported: python, javascript, typescript, go, rust, c, cpp, bash"
        )

    workspace = tempfile.mkdtemp(dir=WORKSPACE_BASE)
    timeout = min(request.timeout, MAX_EXECUTION_TIME)

    # PC-046: Drop project-context files into the workspace BEFORE the
    # language handler runs so any `import other_module` in the candidate
    # resolves against the rest of the user's project. Routed through the
    # same O_NOFOLLOW/dir_fd walking helper as the /shell overlay writer so
    # both write paths enforce identical containment (no absolute paths, no
    # `..` traversal, no symlink components). Bad entries are skipped per
    # file (we don't want a malformed name to block legitimate verification).
    if request.files:
        for name, content in request.files.items():
            try:
                _write_overlay_files(Path(workspace), {name: content})
            except HTTPException as e:
                logger.warning(f"PC-046: rejected sandbox file {name!r}: {e.detail}")

    try:
        handler = LANGUAGE_HANDLERS[lang]
        result = handler(
            code=request.code,
            test_code=request.test_code,
            workspace=Path(workspace),
            timeout=timeout,
            requirements=request.requirements,
            stdin=request.stdin,
        )
        return result
    except Exception as e:
        logger.exception(f"Execution error for {lang}")
        return ExecuteResponse(
            success=False,
            compile_success=False,
            tests_run=0,
            tests_passed=0,
            stdout="",
            stderr=str(e),
            error_type=type(e).__name__,
            error_message=str(e),
            execution_time_ms=0,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


class SyntaxCheckRequest(BaseModel):
    code: str
    language: str = "python"
    filename: Optional[str] = None


class SyntaxCheckResponse(BaseModel):
    valid: bool
    errors: List[str]
    language: str
    check_time_ms: int


@app.post("/syntax-check", response_model=SyntaxCheckResponse)
def syntax_check(request: SyntaxCheckRequest):
    """Check code syntax without executing. Returns parse/compile errors."""
    lang = normalize_language(request.language)
    workspace = tempfile.mkdtemp(dir=WORKSPACE_BASE)
    start = time.time()

    try:
        errors = _syntax_check_impl(lang, request.code, Path(workspace), request.filename)
        elapsed = int((time.time() - start) * 1000)
        return SyntaxCheckResponse(
            valid=len(errors) == 0,
            errors=errors,
            language=lang,
            check_time_ms=elapsed,
        )
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return SyntaxCheckResponse(
            valid=False,
            errors=[str(e)],
            language=lang,
            check_time_ms=elapsed,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _syntax_check_impl(lang: str, code: str, workspace: Path, filename: Optional[str] = None) -> List[str]:
    """Language-specific syntax checking. Returns list of error strings."""
    errors = []

    if lang == "python":
        # Use py_compile for fast AST parse
        fpath = workspace / (filename or "check.py")
        fpath.write_text(code)
        result = _run_cmd(["python3", "-m", "py_compile", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            # Extract just the error line from py_compile output
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                line = line.strip()
                if line and any(kind in line for kind in ("SyntaxError", "IndentationError", "TabError")):
                    errors.append(line)
            if not errors and stderr.strip():
                errors.append(stderr.strip().split("\n")[-1])

    elif lang == "javascript":
        fpath = workspace / (filename or "check.js")
        fpath.write_text(code)
        result = _run_cmd(["node", "--check", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            errors.append(result.get("stderr", "").strip())

    elif lang == "typescript":
        fpath = workspace / (filename or "check.ts")
        fpath.write_text(code)
        # tsc --noEmit for type checking; fall back to tsx parse
        result = _run_cmd(["tsc", "--noEmit", "--strict", str(fpath)], timeout=10, cwd=workspace)
        if result["returncode"] != 0:
            for line in result.get("stderr", "").splitlines() + result.get("stdout", "").splitlines():
                line = line.strip()
                if line and ("error TS" in line or "Error" in line):
                    errors.append(line)

    elif lang == "go":
        fpath = workspace / (filename or "main.go")
        fpath.write_text(code)
        # Use gofmt -e for fast syntax-only checking (no compilation, no go.mod needed)
        result = _run_cmd(["gofmt", "-e", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                line = line.strip()
                if line:
                    errors.append(line)

    elif lang == "rust":
        fpath = workspace / (filename or "check.rs")
        fpath.write_text(code)
        # rustc --edition 2021 with no codegen for syntax-only
        result = _run_cmd(
            ["rustc", "--edition", "2021", "--crate-type", "bin", str(fpath), "-o", "/dev/null"],
            timeout=10, cwd=workspace
        )
        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip())
            if not errors and stderr.strip():
                errors.append(stderr.strip().split("\n")[-1])

    elif lang in ("c", "cpp"):
        ext = ".c" if lang == "c" else ".cpp"
        fpath = workspace / (filename or f"check{ext}")
        fpath.write_text(code)
        compiler = "gcc" if lang == "c" else "g++"
        flags = ["-std=c17"] if lang == "c" else ["-std=c++17"]
        # -fsyntax-only: parse and type-check only, no codegen
        result = _run_cmd(
            [compiler] + flags + ["-fsyntax-only", str(fpath)],
            timeout=10, cwd=workspace
        )
        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                if "error:" in line:
                    errors.append(line.strip())
            if not errors and stderr.strip():
                errors.append(stderr.strip().split("\n")[-1])

    elif lang == "bash":
        fpath = workspace / (filename or "check.sh")
        fpath.write_text(code)
        result = _run_cmd(["bash", "-n", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            errors.append(result.get("stderr", "").strip())

    elif lang == "json":
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            errors.append(str(e))

    elif lang in ("yaml", "yml"):
        try:
            import yaml
            yaml.safe_load(code)
        except Exception as e:
            errors.append(str(e))

    elif lang in ("html", "htm"):
        from html.parser import HTMLParser
        try:
            parser = HTMLParser()
            parser.feed(code)
            parser.close()
        except Exception as e:
            errors.append(str(e))

    elif lang == "xml":
        from defusedxml import ElementTree as ET
        from defusedxml.common import DefusedXmlException
        try:
            ET.fromstring(code)
        except (ET.ParseError, DefusedXmlException) as e:
            errors.append(str(e))

    else:
        errors.append(f"syntax verification is unavailable for language: {lang}")

    return errors


# ---------------------------------------------------------------------------
# Language handlers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], timeout: int, cwd: Path = None, env: dict = None,
             stdin: Optional[str] = None) -> Dict:
    """Run a command with timeout and return structured result.

    start_new_session + killpg mirrors the /jobs path: on timeout the whole
    process group dies, so children the command spawned can't outlive it and
    leak against the container's pids_limit. `stdin` (when not None) is piped
    to the process as its standard input.
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=run_env,
            start_new_session=True,
        )
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }
    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
        return {
            "success": proc.returncode == 0,
            "stdout": stdout[-4000:],
            "stderr": stderr[-2000:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            # best-effort: swallow on failure (caller continues)
            pass
        try:
            proc.communicate(timeout=5)  # reap + close pipes
        except Exception:
            proc.kill()
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # best-effort: swallow on failure (caller continues)
            pass
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def _classify_error(stderr: str) -> Optional[str]:
    """Extract error type from stderr."""
    patterns = [
        (r"SyntaxError", "SyntaxError"),
        (r"NameError", "NameError"),
        (r"TypeError", "TypeError"),
        (r"ValueError", "ValueError"),
        (r"ImportError|ModuleNotFoundError", "ImportError"),
        (r"IndexError", "IndexError"),
        (r"KeyError", "KeyError"),
        (r"AttributeError", "AttributeError"),
        (r"ZeroDivisionError", "ZeroDivisionError"),
        (r"FileNotFoundError", "FileNotFoundError"),
        (r"ReferenceError", "ReferenceError"),
        (r"error\[E\d+\]", "CompileError"),
        (r"error:", "CompileError"),
        (r"undefined reference", "LinkError"),
        (r"cannot find", "NotFoundError"),
        (r"timed out", "Timeout"),
    ]
    for pattern, error_type in patterns:
        if re.search(pattern, stderr):
            return error_type
    return "RuntimeError" if stderr.strip() else None


# --- Python ---

def execute_python(code, test_code, workspace, timeout, requirements, stdin=None, **_):
    start = time.time()
    main_file = workspace / "solution.py"
    main_file.write_text(code)

    # Syntax check
    try:
        compile(code, "solution.py", "exec")
    except SyntaxError as e:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=f"Line {e.lineno}: {e.msg}",
            error_type="SyntaxError", error_message=f"Line {e.lineno}: {e.msg}",
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Install requirements
    if requirements:
        r = _run_cmd(["pip", "install", "--target", str(workspace), "--quiet"] + requirements, timeout)
        if not r["success"]:
            return ExecuteResponse(
                success=False, compile_success=True,
                tests_run=0, tests_passed=0,
                stdout="", stderr=r["stderr"],
                error_type="DependencyError", error_message=r["stderr"][:500],
                execution_time_ms=int((time.time() - start) * 1000),
            )

    # Lint
    lint_score = None
    try:
        lr = subprocess.run(
            ["python", "-m", "pylint", "--score=y", "--exit-zero", str(main_file)],
            capture_output=True, text=True, timeout=15
        )
        m = re.search(r"rated at ([\d.]+)/10", lr.stdout)
        if m:
            lint_score = float(m.group(1))
    except Exception:
        # best-effort: swallow on failure (caller continues)
        pass

    # Run
    if test_code:
        (workspace / "test_solution.py").write_text(test_code)
        r = _run_cmd(["python", "-m", "pytest", "-v", "--tb=short", str(workspace)], timeout, cwd=workspace,
                     stdin=stdin)
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", r["stdout"])) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", r["stdout"])) else 0
        total = passed + failed or 1
    else:
        r = _run_cmd(
            ["python", "-c", f"import sys; sys.path.insert(0,'{workspace}'); import solution"],
            timeout, stdin=stdin
        )
        passed = 1 if r["success"] else 0
        total = 1

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=total, tests_passed=passed,
        lint_score=lint_score,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- JavaScript ---

def execute_javascript(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    main_file = workspace / "solution.js"
    main_file.write_text(code)

    # Syntax check via node --check
    r = _run_cmd(["node", "--check", str(main_file)], 10)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="SyntaxError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd(["node", str(main_file)], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- TypeScript ---

def execute_typescript(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    main_file = workspace / "solution.ts"
    main_file.write_text(code)

    # Type check via tsc --noEmit
    r = _run_cmd(["tsc", "--noEmit", "--strict", "--esModuleInterop", str(main_file)], 15)
    compile_success = r["success"]
    if not compile_success:
        # Still try to run — TS errors are often non-fatal for execution
        logger.info(f"TypeScript type errors: {r['stderr'][:200]}")

    # Run via tsx (faster than ts-node, handles ESM)
    r = _run_cmd(["tsx", str(main_file)], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=compile_success,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- Go ---

def execute_go(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    main_file = workspace / "main.go"
    main_file.write_text(code)

    # Init module
    _run_cmd(["go", "mod", "init", "sandbox"], 5, cwd=workspace)

    # Build (compile check)
    r = _run_cmd(["go", "build", "-o", str(workspace / "program"), str(main_file)], 30, cwd=workspace)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd([str(workspace / "program")], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- Rust ---

def execute_rust(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    main_file = workspace / "main.rs"
    main_file.write_text(code)

    # Compile
    binary = workspace / "program"
    r = _run_cmd(["rustc", str(main_file), "-o", str(binary)], 30)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd([str(binary)], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- C ---

def execute_c(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    main_file = workspace / "solution.c"
    main_file.write_text(code)

    binary = workspace / "program"
    r = _run_cmd(["gcc", "-o", str(binary), str(main_file), "-lm", "-Wall"], 15)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    r = _run_cmd([str(binary)], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- C++ ---

def execute_cpp(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    main_file = workspace / "solution.cpp"
    main_file.write_text(code)

    binary = workspace / "program"
    r = _run_cmd(["g++", "-o", str(binary), str(main_file), "-std=c++17", "-Wall"], 15)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    r = _run_cmd([str(binary)], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- Bash ---

def execute_bash(code, test_code, workspace, timeout, stdin=None, **_):
    start = time.time()
    script = workspace / "solution.sh"
    script.write_text(code)
    script.chmod(0o755)

    # Syntax check
    r = _run_cmd(["bash", "-n", str(script)], 5)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="SyntaxError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd(["bash", str(script)], timeout, cwd=workspace, stdin=stdin)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# Handler dispatch
LANGUAGE_HANDLERS = {
    "python": execute_python,
    "javascript": execute_javascript,
    "typescript": execute_typescript,
    "go": execute_go,
    "rust": execute_rust,
    "c": execute_c,
    "cpp": execute_cpp,
    "bash": execute_bash,
}


if __name__ == "__main__":
    import uvicorn
    WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)
    logger.info(f"Supported languages: {list(LANGUAGE_HANDLERS.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=8020)
