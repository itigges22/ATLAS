"""Candidate verification and scoring for the V3 pipeline: geometric-lens
scoring, task-type classification, language-aware smoke checks, build-command
verification, and the interactive-task lint."""

import json
import urllib.request
from pathlib import PurePath
from typing import Any, Dict, List, Tuple

import adapters


# --- Lens Scorer (calls Geometric Lens) ---------------------------------------------

def score_candidate_per_step(code: str) -> dict:
    """PC-207 wiring: per-step C(x)+G(x) scoring of a candidate.

    Returns the aggregate dict from `/internal/lens/score-per-step`
    (`first_off_rails_idx`, `gx_score_min`, `gx_score_mean`, etc.)
    plus `n_tokens`. Fail-soft: returns an empty dict on error so a
    lens outage degrades to "no per-step signal" instead of a
    pipeline-stopping exception.

    Cost on this hardware tier: ~7-15ms per token (lens batches the
    MLP + XGBoost calls), so a 500-token candidate adds ~3-7 seconds
    of latency. Worth it for the off-rails detection signal — see
    PC-207 in ISSUES.md for the empirical case (the May 6 53-min
    repetition loop would have been visible at first_off_rails_idx<5).
    """
    try:
        body = json.dumps({"text": code}).encode()
        req = urllib.request.Request(
            f"{adapters.LENS_URL}/internal/lens/score-per-step",
            data=body,
            headers=adapters._service_headers(),
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        if not data.get("enabled"):
            return {}
        agg = data.get("aggregate", {}) or {}
        if not int(data.get("n_tokens", 0)):
            # The endpoint answers 200 with an empty aggregate when it
            # could not score, so the defaults below would hand back
            # gx=0.500 for every candidate — a tie that reads exactly
            # like a real verdict. Report it as no signal instead.
            print(
                f"  [lens] per-step scoring returned no tokens "
                f"({data.get('error') or 'no error reported'}) "
                f"— degrading to no per-step signal",
                flush=True,
            )
            return {}
        thresholds = data.get("thresholds")
        if not (
            isinstance(thresholds, dict)
            and all(isinstance(thresholds.get(k), (int, float))
                    for k in ("off_rails", "low", "severe"))
        ):
            thresholds = None
        result = {
            "n_tokens":            int(data.get("n_tokens", 0)),
            "gx_available":        bool(data.get("gx_available", False)),
            "first_off_rails_idx": int(agg.get("first_off_rails_idx", -1)),
            "gx_score_min":        float(agg.get("gx_score_min", 0.5)),
            "gx_score_mean":       float(agg.get("gx_score_mean", 0.5)),
            "cx_norm_max":         float(agg.get("cx_norm_max", 0.0)),
            "cx_norm_mean":        float(agg.get("cx_norm_mean", 0.0)),
            "latency_ms":          float(data.get("latency_ms", 0.0)),
            "thresholds":          thresholds,
        }
        print(
            f"  [lens] candidate scored: n_tok={result['n_tokens']} "
            f"gx_min={result['gx_score_min']:.3f} gx_mean={result['gx_score_mean']:.3f} "
            f"off_rails={result['first_off_rails_idx']} lat={result['latency_ms']:.0f}ms",
            flush=True,
        )
        return result
    except Exception as e:
        print(f"  [lens] score_candidate_per_step failed: {e} — degrading to no per-step signal", flush=True)
        return {}


NEUTRAL_COMBINED = {
    "cx_energy": 0.0, "cx_normalized": 0.5, "cx_calibrated": False,
    "gx_score": 0.5, "gx_available": False, "verdict": "unavailable",
}


def score_candidate_combined(code: str) -> Dict[str, Any]:
    """Score code with Geometric Lens C(x) AND G(x) in one call.

    `/internal/lens/gx-score` extracts the embedding once and runs both
    models on it, so the pair costs no more than C(x) alone. Returns
    ``cx_energy``, ``cx_normalized``, ``cx_calibrated``, ``gx_score``,
    ``gx_available`` and ``verdict``; the CxGx allocation gate reads all
    six, everything else reads the C(x) three through score_candidate.

    Fail-soft: any transport error, a disabled lens, or a malformed body
    yields the neutral dict — ``cx_calibrated``/``gx_available`` false, so
    callers can tell "the lens said neutral" from "the lens said nothing"
    and the gate degrades to its k=3 floor instead of routing on noise.

    Timeout note: 10s was tight under load — the lens shares the box with
    V3's streaming generator and llama-server, and a single hot probe
    could starve scoring long enough to trip the fallback. Bumped to 30s
    so transient contention doesn't masquerade as a broken lens
    (symptom: C(x)=0.00 / gx=0.50 sentinel pair).
    """
    try:
        body = json.dumps({"text": code}).encode()
        req = urllib.request.Request(
            f"{adapters.LENS_URL}/internal/lens/gx-score",
            data=body,
            headers=adapters._service_headers(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        if not data.get("enabled", False):
            return dict(NEUTRAL_COMBINED)
        return {
            "cx_energy": data.get("cx_energy", 0.0),
            "cx_normalized": data.get("cx_normalized", 0.5),
            "cx_calibrated": bool(data.get("cx_calibrated", False)),
            "gx_score": data.get("gx_score", 0.5),
            "gx_available": bool(data.get("gx_available", False)),
            "verdict": data.get("verdict", "unavailable"),
        }
    except Exception as e:
        print(f"  [lens] score_candidate failed: {e} — using neutral uncalibrated score", flush=True)
        return dict(NEUTRAL_COMBINED)


def score_candidate(code: str) -> Tuple[float, float, bool]:
    """Score code with Geometric Lens C(x).

    Returns ``(raw_energy, normalized_energy, calibrated)``. The normalized
    value is neutral when this model has no calibration and must not drive
    adaptive routing in that case.
    """
    d = score_candidate_combined(code)
    return d["cx_energy"], d["cx_normalized"], d["cx_calibrated"]


# --- Task-type classifier (PC-022) -------------------------------------------

_INTERACTIVE_MARKERS = (
    "game", "tui", "terminal interface", "menu", "interactive",
    "pygame", "curses", "tkinter", "flask", "fastapi", "django",
    "streamlit", "gradio", "dashboard", "gui", "web app", "webapp",
    "cli tool", "command-line tool", "chat bot", "chatbot",
    "discord bot", "telegram bot", "snake", "tetris", "pong",
    "rpg", "shell", "repl", "live server", "scraper", "crawler",
    "watcher", "daemon",
)
_ALGORITHMIC_MARKERS = (
    "input:", "output:", "examples:", "sample input", "sample output",
    "constraints:", "test case", "leetcode", "codeforces", "hackerrank",
    "competitive programming", "function signature", "given an array",
    "given a string", "return the", "return an integer", "modulo 10",
)


def classify_task_type(problem: str) -> str:
    """Classify whether a task expects (input -> output) self-tests.

    Returns 'algorithmic' for problems with clear I/O contracts (the
    LiveCodeBench shape — synthesized self-tests are meaningful), or
    'interactive' for games/UIs/scripts/library code where I/O self-tests
    don't apply and would produce false failures (PC-022).
    """
    p = problem.lower()
    interactive_hits = sum(1 for m in _INTERACTIVE_MARKERS if m in p)
    algorithmic_hits = sum(1 for m in _ALGORITHMIC_MARKERS if m in p)
    if interactive_hits > 0 and interactive_hits >= algorithmic_hits:
        return "interactive"
    return "algorithmic"


def smoke_compile_check(code: str, sandbox, language: str = "python") -> Tuple[bool, str, str]:
    """Lightweight verification for interactive tasks: code parses + compiles.

    Replaces synthetic-I/O self-tests for tasks where (input -> output)
    pairs are nonsensical (curses games, pygame apps, flask servers, …).

    PC-048: language-aware. Everything the sandbox /syntax-check endpoint
    covers goes through it; anything else fails explicitly instead of
    being accepted without evidence.
    """
    lang = (language or "python").lower()

    verified_languages = {
        "python", "py", "javascript", "typescript", "go", "java", "kotlin",
        "rust", "c", "cpp", "ruby", "php", "bash", "html", "htm", "xml", "json", "yaml", "yml",
    }
    if lang not in verified_languages or not hasattr(sandbox, "syntax_check"):
        return False, "", f"syntax verification unavailable for language: {lang}"
    normalized = {
        "py": "python", "htm": "html", "yml": "yaml",
    }.get(lang, lang)
    return sandbox.syntax_check(code, normalized)


BUILD_EVIDENCE_LIMIT = 4000
ALLOWED_BUILD_PREFIXES = (
    "npm run build",
    "npm run test",
    "npm test",
    "pnpm run build",
    "pnpm run test",
    "pnpm test",
    "yarn build",
    "yarn test",
    "yarn run build",
    "yarn run test",
    "bun run build",
    "bun run test",
    "bun test",
    "npx tsc --noEmit",
    "npx next build",
    "python -m py_compile",
    "python -m pytest",
    "pytest",
    "go build",
    "go test",
    "cargo build",
    "cargo check",
    "make",
    "cmake --build",
    "bash -n",
)
DISALLOWED_BUILD_TOKENS = (
    ";", "&&", "||", "|", "&", "<", ">", "`", "$(", "\n", "\r", "\x00",
)


def _bounded_evidence(text: str, limit: int = BUILD_EVIDENCE_LIMIT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _build_command_allowed(command: str) -> bool:
    command = (command or "").strip()
    if not command or any(token in command for token in DISALLOWED_BUILD_TOKENS):
        return False
    for prefix in ALLOWED_BUILD_PREFIXES:
        if command == prefix or command.startswith(prefix + " "):
            return True
    return False


def _project_relative_path(file_path: str, working_dir: str = "/workspace") -> str:
    if not file_path:
        raise ValueError("file_path is required for build verification")
    path = PurePath(file_path)
    root = PurePath(working_dir or "/workspace")
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError(f"working_dir must be an absolute project root: {working_dir}")
    if path.is_absolute():
        try:
            rel = path.relative_to(root)
        except ValueError as e:
            raise ValueError(f"file_path must be under working_dir: {file_path}") from e
    else:
        rel = path
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe file_path for build verification: {file_path}")
    return str(rel)


def verify_build_command(
    code: str,
    sandbox,
    build_command: str,
    file_path: str,
    project_files: Dict[str, str],
    working_dir: str,
    emit=None,
) -> Tuple[bool, str, str, Dict[str, Any]]:
    command = (build_command or "").strip()
    evidence: Dict[str, Any] = {
        "verifier": "build_command",
        "command": command,
        "status": "unavailable",
        "exit_code": None,
        "duration_ms": 0,
        "stdout": "",
        "stderr": "",
    }
    if not command:
        return True, "", "", {}
    if not hasattr(sandbox, "run_command"):
        evidence["stderr"] = "sandbox build command runner is unavailable"
        if emit:
            emit("build_verify_unavailable", evidence["stderr"], command=command)
        return False, "", evidence["stderr"], evidence
    if not _build_command_allowed(command):
        evidence["stderr"] = f"build command is not allowed by verification policy: {command}"
        if emit:
            emit("build_verify_unavailable", evidence["stderr"], command=command)
        return False, "", evidence["stderr"], evidence

    try:
        rel_path = _project_relative_path(file_path, working_dir)
    except ValueError as e:
        evidence["stderr"] = str(e)
        if emit:
            emit("build_verify_unavailable", evidence["stderr"], command=command)
        return False, "", evidence["stderr"], evidence

    # Build verification snapshots the real /workspace inside the sandbox.
    # Do not overlay `project_files` here: that map is prompt context from
    # the proxy and may be intentionally truncated for token budget. Only
    # overlay the candidate under test so full project files remain intact.
    overlay = {rel_path: code}
    ok, out, err, meta = sandbox.run_command(
        command,
        files=overlay,
        cwd=working_dir or "/workspace",
        timeout=60,
    )
    evidence.update({
        "status": "passed" if ok else "failed",
        "exit_code": meta.get("exit_code"),
        "duration_ms": int(meta.get("elapsed_ms") or 0),
        "stdout": _bounded_evidence(out),
        "stderr": _bounded_evidence(err),
    })
    if emit:
        emit(
            "build_verify",
            f"{command}: {'OK' if ok else 'FAIL'}",
            command=command,
            status=evidence["status"],
            exit_code=evidence["exit_code"],
            duration_ms=evidence["duration_ms"],
        )
    if not ok:
        return False, out, err or out or f"build command failed: {command}", evidence
    return True, out, err, evidence


def interactive_lint(code: str) -> Tuple[bool, str]:
    """Heuristic checks beyond compile-OK for interactive (terminal/UI) tasks.

    Compile-OK is necessary but not sufficient: a snake game using
    `sys.stdin.read(1)` without termios setup parses fine, runs without
    crashing, and silently fails on every keypress. We've seen this in real
    user runs (ISSUES.md PC-034). Detect the most common failure shapes
    statically before accepting the probe.

    Returns (passed, reason). reason is empty when passed.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        # Compile gate above already caught this; treat as passed here so
        # we don't double-report.
        return True, ""

    has_curses = False
    has_termios_setraw = False
    has_raw_stdin_read = False
    has_blocking_input_loop = False
    # PC-047: track curses-bottom-row anti-patterns (unwrapped addstr to
    # LINES-N or COLS-N — addstr to the very last cell always returns ERR
    # in curses, which is why so many "snake game" runs crash with
    # `_curses.error: addwstr() returned ERR`).
    bottom_row_addstr_nodes: List[Tuple[int, str]] = []  # (lineno, snippet)
    try_except_curses_lines: set = set()

    def _is_lines_or_cols_minus(node: _ast.AST, name: str) -> bool:
        """True if node is a `curses.{name} - <int>` BinOp expression
        (or just `LINES - <int>` if the model imported the names)."""
        if not isinstance(node, _ast.BinOp) or not isinstance(node.op, _ast.Sub):
            return False
        left = node.left
        # curses.LINES - N
        if (isinstance(left, _ast.Attribute) and left.attr == name
                and isinstance(left.value, _ast.Name) and left.value.id == "curses"):
            return True
        # bare LINES - N (after `from curses import LINES, COLS`)
        if isinstance(left, _ast.Name) and left.id == name:
            return True
        return False

    # First pass: find every `try: ... except curses.error / except _curses.error`
    # block and record the line ranges they protect, so we can skip
    # already-wrapped addstr calls below.
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Try):
            handles_curses = False
            for handler in node.handlers:
                exc = handler.type
                if isinstance(exc, _ast.Attribute) and exc.attr == "error":
                    if (isinstance(exc.value, _ast.Name)
                            and exc.value.id in ("curses", "_curses")):
                        handles_curses = True
                elif isinstance(exc, _ast.Name) and exc.id == "Exception":
                    handles_curses = True  # broad catch covers curses.error
            if handles_curses:
                start = node.lineno
                end = max((getattr(n, "end_lineno", node.lineno) or node.lineno)
                          for n in _ast.walk(node))
                for ln in range(start, end + 1):
                    try_except_curses_lines.add(ln)

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                if alias.name == "curses":
                    has_curses = True
        elif isinstance(node, _ast.ImportFrom):
            if node.module in ("curses", "termios", "tty"):
                has_curses = has_curses or node.module == "curses"
                has_termios_setraw = has_termios_setraw or node.module in ("termios", "tty")
        elif isinstance(node, _ast.Call):
            func = node.func
            if isinstance(func, _ast.Attribute):
                # sys.stdin.read(1) without termios setup
                if (
                    func.attr == "read"
                    and isinstance(func.value, _ast.Attribute)
                    and func.value.attr == "stdin"
                    and isinstance(func.value.value, _ast.Name)
                    and func.value.value.id == "sys"
                ):
                    has_raw_stdin_read = True
                # termios.tcsetattr / tty.setraw / tty.setcbreak
                if func.attr in ("tcsetattr", "setraw", "setcbreak"):
                    has_termios_setraw = True
                # PC-047: addstr / addnstr / addch with a LINES-N first arg
                # (writing to a row near the bottom — last row always errors)
                # or a COLS-N second arg pair that targets the last column.
                if func.attr in ("addstr", "addnstr", "addch"):
                    args = node.args
                    if args:
                        first = args[0]
                        if _is_lines_or_cols_minus(first, "LINES"):
                            if node.lineno not in try_except_curses_lines:
                                snippet = f"line {node.lineno}: {func.attr}(curses.LINES - N, ...) without try/except curses.error"
                                bottom_row_addstr_nodes.append((node.lineno, snippet))
        elif isinstance(node, _ast.While):
            # Look for `while True: ... input(...)` shape — blocking input
            # in an interactive loop is almost always wrong.
            for sub in _ast.walk(node):
                if isinstance(sub, _ast.Call) and isinstance(sub.func, _ast.Name) and sub.func.id == "input":
                    has_blocking_input_loop = True
                    break

    # Raw stdin read without termios is a near-certain bug for interactive
    # keystroke handling — single-char read is line-buffered and can't see
    # arrow-key escape sequences.
    if has_raw_stdin_read and not has_termios_setraw and not has_curses:
        return False, "raw sys.stdin.read without termios/tty setup or curses — keystrokes won't register"

    # input() inside a `while True` of a TUI flow blocks until Enter; usually
    # intended to be a non-blocking key read.
    if has_blocking_input_loop and not has_curses and not has_termios_setraw:
        return False, "input() in a loop with no curses/termios — blocks on Enter, can't read single keystrokes"

    # PC-047: unwrapped addstr to the bottom row will always raise
    # `_curses.error: addwstr() returned ERR` at runtime (writing the last
    # cell of any window is undefined and historically returns ERR). The
    # idiomatic fix is `try: stdscr.addstr(...) except curses.error: pass`.
    # Fail the lint so V3 prefers a candidate that has the wrap.
    if has_curses and bottom_row_addstr_nodes:
        first = bottom_row_addstr_nodes[0][1]
        return False, f"curses bottom-row write without try/except curses.error wrap — {first} (will raise ERR at runtime)"

    return True, ""
