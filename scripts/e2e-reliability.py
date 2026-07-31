#!/usr/bin/env python3
"""Measured reliability of the live stack: proxy + v3-service + lens + sandbox.

`tests/e2e/` drives fake llama/lens/v3 handlers, which is what makes it
deterministic enough for CI — and also what stops it from telling you whether
the real services work together. This runs real sessions against the running
stack and reports two numbers that must not be conflated:

  Harness Integrity Rate  sessions in which ATLAS's own plumbing did nothing
                          wrong. Independent of whether the model succeeded at
                          the task, and the number that should be 100%.

  Task Success Rate       sessions where the requested change actually landed.
                          Bounded by the model's coding ability, so a low value
                          is evidence about the model, not about ATLAS.

The split matters because "it built a snake game" moves with model skill and
sampling luck, so it cannot tell you whether a pipeline regression shipped. A
harness defect is instead something provable from the event stream and the
workspace: a rejection whose stated reason does not hold against the file on
disk, steering that names a remedy the target cannot accept, an exit that
escaped the gates, a corrupt write, a service fault, an orphaned tool call, an
event the TUI cannot render, or a background job left running after the session.

Usage:
    python scripts/e2e-reliability.py                 # default suite, 2 reps
    python scripts/e2e-reliability.py --reps 5
    python scripts/e2e-reliability.py --tasks flask_pause,add_function
    python scripts/e2e-reliability.py --json out.json

Requires the stack to be up (docker compose ps) and ATLAS_PROJECT_DIR to be the
workspace the proxy has mounted at /workspace.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from code_quality import analyze as analyze_quality  # noqa: E402

# --------------------------------------------------------------------------
# Task suite
# --------------------------------------------------------------------------
#
# Each task is a deterministic fixture plus a post-condition that is decidable
# without reading the model's prose. `check` sees the workspace and returns
# (passed, detail) — it must answer "did the requested change land", never
# "did the model claim it did".


@dataclass
class Task:
    name: str
    prompt: str
    files: dict[str, str]
    check: Callable[[Path], tuple[bool, str]]
    # Files the session is expected to leave parseable. Any workspace file is
    # checked for corruption regardless; this is the subset that must exist.
    must_exist: tuple[str, ...] = ()


SNAKE_APP = (REPO / "scripts" / "fixtures" / "snake_app.py")


def _read_fixture(name: str) -> str:
    p = REPO / "scripts" / "fixtures" / name
    if not p.exists():
        raise SystemExit(f"missing fixture {p} — see scripts/fixtures/README.md")
    return p.read_text()


def _check_flask_pause(ws: Path) -> tuple[bool, str]:
    """A pause toggle wired into the game loop, with the JS still parsing.

    Three separate things, all required. An earlier, looser version of this
    check accepted the bare token `32` or a stray `Space` anywhere in the
    file and reported a pass against a fixture that had not been touched —
    a reliability check that can pass without the work being done is worse
    than no check, so each clause below names a distinct piece of the
    feature and the state variable has to be one the keydown handler
    actually assigns.
    """
    src = (ws / "app.py").read_text()
    js = _extract_script(src)
    if js is None:
        return False, "no <script> block survived"

    ok, err = _js_parses(js)
    if not ok:
        return False, f"embedded JS broken: {err}"

    # 1. A boolean the code flips, not merely a word that appears somewhere.
    state = None
    for m in re.finditer(r"\b(?:let|var|const)\s+(\w*[Pp]aused?\w*)\s*=", js):
        state = m.group(1)
        break
    if not state:
        return False, "no pause state variable declared"
    if not re.search(rf"\b{re.escape(state)}\s*=\s*(?:!\s*{re.escape(state)}|true|false)", js):
        return False, f"{state} is declared but never toggled"

    # 2. A keyboard branch on the spacebar specifically.
    if not re.search(r"(?:key|code)\s*===?\s*['\"](?: |Space|Spacebar)['\"]"
                     r"|keyCode\s*===?\s*32", js):
        return False, f"{state} exists but nothing binds the spacebar"

    # 3. The loop has to actually honour it, or the key toggles a dead flag.
    if not re.search(rf"if\s*\([^)]*\b{re.escape(state)}\b", js):
        return False, f"{state} is toggled but the game loop never checks it"

    return True, f"{state} declared, toggled, spacebar-bound, honoured by the loop"


def _check_add_function(ws: Path) -> tuple[bool, str]:
    src = (ws / "stats.py").read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"stats.py does not parse: {e}"
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "median" not in names:
        return False, f"no median() defined (found {sorted(names)})"
    # Behaviour, not just presence.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,%r); import stats;"
         "print(stats.median([3,1,2]), stats.median([4,1,3,2]))" % str(ws)],
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return False, f"median() raised: {proc.stderr.strip()[:160]}"
    if proc.stdout.split() != ["2", "2.5"]:
        return False, f"median() wrong: got {proc.stdout.strip()!r}, want '2 2.5'"
    return True, "median() defined and correct on odd + even input"


def _check_offbyone(ws: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,%r); import chunk;"
         "print(chunk.chunks([1,2,3,4,5], 2))" % str(ws)],
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return False, f"chunks() raised: {proc.stderr.strip()[:160]}"
    got = proc.stdout.strip()
    want = "[[1, 2], [3, 4], [5]]"
    if got != want:
        return False, f"chunks() wrong: got {got}, want {want}"
    return True, "chunks() drops no tail element"


# --- AoC-style puzzles: exact-integer answers, holdout-verified ----------
#
# The answer is a specific number, so "did it work" needs no judgement. The
# model sees input.txt; the check re-runs its program against a holdout input
# it never saw, so hardcoding the number it was shown fails. `shoal` is the
# one that separates understanding from transcription: the naive
# per-individual simulation reaches ~1.6e12 elements and cannot finish, so
# only the counting solution completes.

AOC_DIR = REPO / "scripts" / "fixtures" / "aoc"


def _aoc_answers() -> dict:
    return json.loads((AOC_DIR / "answers.json").read_text())


def _run_solution(ws: Path, timeout: int = 60) -> tuple[bool, str]:
    prog = ws / "solve.py"
    if not prog.exists():
        return False, "solve.py was never created"
    try:
        p = subprocess.run([sys.executable, "solve.py"], cwd=str(ws),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"solve.py did not finish within {timeout}s"
    if p.returncode != 0:
        return False, f"solve.py failed: {p.stderr.strip()[:160]}"
    nums = re.findall(r"-?\d+", p.stdout)
    if not nums:
        return False, f"no number printed (stdout={p.stdout.strip()[:80]!r})"
    return True, nums[-1]


def _check_aoc(name: str):
    def check(ws: Path) -> tuple[bool, str]:
        want = _aoc_answers()[name]
        ok, got = _run_solution(ws)
        if not ok:
            return False, got
        if got != str(want["input"]):
            return False, f"wrong answer: got {got}, want {want['input']}"
        # Same program, an input it never saw. A hardcoded answer dies here.
        original = (ws / "input.txt").read_text()
        (ws / "input.txt").write_text((AOC_DIR / name / "holdout.txt").read_text())
        try:
            ok2, got2 = _run_solution(ws)
        finally:
            (ws / "input.txt").write_text(original)
        if not ok2:
            return False, f"correct on its own input but broke on the holdout: {got2}"
        if got2 != str(want["holdout"]):
            return False, (f"holdout mismatch: got {got2}, want {want['holdout']} "
                           f"— the answer looks hardcoded rather than computed")
        return True, f"{got} correct, and correct on the holdout input"
    return check


_AOC_PROMPTS = {
    "sonar": ("input.txt holds one integer per line: a sonar depth reading. "
              "Consider sums of three-measurement sliding windows. Write "
              "solve.py that reads input.txt and prints how many such window "
              "sums are larger than the immediately previous window sum."),
    "course": ("input.txt holds one command per line: 'forward N', 'down N' or "
               "'up N'. Track horizontal position, depth and aim, all starting "
               "at 0. 'down N' increases aim by N, 'up N' decreases aim by N, "
               "and 'forward N' increases horizontal position by N AND "
               "increases depth by aim multiplied by N. Write solve.py that "
               "reads input.txt and prints the final horizontal position "
               "multiplied by the final depth."),
    "slope": ("input.txt is a grid of '.' (open) and '#' (tree). The pattern "
              "repeats infinitely to the right. Starting at the top-left and "
              "moving by a fixed (right, down) step until past the bottom, "
              "count the trees encountered. Write solve.py that reads "
              "input.txt and prints the PRODUCT of the tree counts for these "
              "five slopes: right 1 down 1, right 3 down 1, right 5 down 1, "
              "right 7 down 1, and right 1 down 2."),
    "shoal": ("input.txt is a comma-separated list of integers, each an "
              "internal timer for one fish. Each day every timer decreases by "
              "1. A fish whose timer is 0 resets to 6 and spawns a new fish "
              "with timer 8 (the new fish does not decrease that same day). "
              "Write solve.py that reads input.txt and prints how many fish "
              "exist after 256 days. Note: the population reaches roughly "
              "1e12, so simulating each fish individually will not finish."),
}


def _aoc_task(name: str) -> Task:
    return Task(
        name=f"aoc_{name}",
        prompt=_AOC_PROMPTS[name] + " Then run it and confirm the answer.",
        files={"input.txt": (AOC_DIR / name / "input.txt").read_text()},
        check=_check_aoc(name),
        must_exist=("input.txt",),
    )


TASKS: dict[str, Task] = {
    # Tier-2: Python file whose real logic is JS inside a template string. The
    # case ATLAS exists for, and the one that exposed D9/D10.
    "flask_pause": Task(
        name="flask_pause",
        prompt=("In app.py, the snake game has no pause. Add a pause toggle: "
                "pressing the spacebar should pause and resume the game loop. "
                "Keep the change small and targeted. Then verify the app still "
                "starts."),
        files={"app.py": None},  # filled from fixture at load time
        check=_check_flask_pause,
        must_exist=("app.py",),
    ),
    # Mid-tier: add a function to an existing module. Exercises the edit path
    # without the embedded-script complication.
    "add_function": Task(
        name="add_function",
        prompt=("In stats.py, add a median(values) function next to the "
                "existing mean(). It must return the middle value for an "
                "odd-length list and the average of the two middle values for "
                "an even-length list. Then verify it works."),
        files={"stats.py": (
            '"""Small statistics helpers."""\n'
            "\n"
            "\n"
            "def mean(values):\n"
            "    if not values:\n"
            "        raise ValueError('mean() of empty sequence')\n"
            "    return sum(values) / len(values)\n"
        )},
        check=_check_add_function,
        must_exist=("stats.py",),
    ),
    # Low-tier repair: a one-character bug with an unambiguous correct answer.
    # Isolates harness behaviour from model creativity.
    "offbyone": Task(
        name="offbyone",
        prompt=("chunk.py has a bug: chunks([1,2,3,4,5], 2) drops the last "
                "element instead of returning it as a short final chunk. Fix "
                "it and verify."),
        files={"chunk.py": (
            '"""Split a list into fixed-size chunks."""\n'
            "\n"
            "\n"
            "def chunks(values, size):\n"
            "    out = []\n"
            "    for i in range(0, len(values) - size + 1, size):\n"
            "        out.append(values[i:i + size])\n"
            "    return out\n"
        )},
        check=_check_offbyone,
        must_exist=("chunk.py",),
    ),
}


# --------------------------------------------------------------------------
# Helpers shared by the checks and the corruption detector
# --------------------------------------------------------------------------

for _n in ("sonar", "course", "slope", "shoal"):
    TASKS[f"aoc_{_n}"] = _aoc_task(_n)


def _check_multifile(ws: Path) -> tuple[bool, str]:
    """A real multi-file program: separate modules, working CLI, passing tests.

    The interesting failure is not "it did not work" — it is one 300-line
    todo.py with a `store` class glued on and a test file that imports
    nothing. So this checks the seams: the modules exist separately, the
    tests actually run and pass, and the CLI works end to end through them.
    """
    missing = [f for f in ("todo.py", "store.py", "test_store.py")
               if not (ws / f).exists()]
    if missing:
        return False, f"missing {', '.join(missing)}"

    # The split has to be real: store.py must carry the persistence, and
    # todo.py must go through it rather than reimplementing it.
    store_src = (ws / "store.py").read_text()
    todo_src = (ws / "todo.py").read_text()
    if "import store" not in todo_src and "from store" not in todo_src:
        return False, "todo.py never imports store.py — the split is cosmetic"
    if len(store_src.splitlines()) < 5:
        return False, "store.py is a stub"

    tp = subprocess.run([sys.executable, "-m", "pytest", "test_store.py", "-q"],
                        cwd=str(ws), capture_output=True, text=True, timeout=120)
    if tp.returncode != 0:
        tail = (tp.stdout or tp.stderr).strip().splitlines()
        return False, f"tests fail: {tail[-1][:120] if tail else 'no output'}"

    add = subprocess.run([sys.executable, "todo.py", "add", "buy milk"],
                         cwd=str(ws), capture_output=True, text=True, timeout=60)
    if add.returncode != 0:
        return False, f"`todo.py add` failed: {add.stderr.strip()[:120]}"
    lst = subprocess.run([sys.executable, "todo.py", "list"],
                         cwd=str(ws), capture_output=True, text=True, timeout=60)
    if lst.returncode != 0:
        return False, f"`todo.py list` failed: {lst.stderr.strip()[:120]}"
    if "buy milk" not in lst.stdout:
        return False, f"added item not listed (stdout={lst.stdout.strip()[:80]!r})"
    return True, "modules separate, tests pass, CLI round-trips through store"


TASKS["multifile_cli"] = Task(
    name="multifile_cli",
    prompt=("Build a small command-line todo app in this directory, as three "
            "files. store.py holds the persistence layer: load and save a "
            "list of items as JSON in todos.json, plus functions to add an "
            "item and to mark one done. todo.py is the CLI entry point and "
            "must use store.py rather than reimplementing it; support "
            "`add <text>`, `list`, and `done <index>`. test_store.py holds "
            "pytest tests for store.py, using a temporary file so the tests "
            "do not touch real data. Keep each file focused and small. Then "
            "run the tests and confirm they pass."),
    files={},
    check=_check_multifile,
)


def _extract_script(src: str) -> str | None:
    m = re.search(r"<script>(.*?)</script>", src, re.S)
    return m.group(1) if m else None


def _js_parses(js: str) -> tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return True, "node unavailable — skipped"
    tmp = Path("/tmp/.atlas_reliability_check.js")
    tmp.write_text(js)
    try:
        p = subprocess.run([node, "--check", str(tmp)],
                           capture_output=True, text=True, timeout=30)
        return p.returncode == 0, p.stderr.strip().splitlines()[0] if p.stderr else ""
    finally:
        tmp.unlink(missing_ok=True)


def _file_parses(path: Path) -> tuple[bool, str]:
    """Whole-file parse plus the embedded-script layer, mirroring the gates."""
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return True, ""  # binary or unreadable: not our concern
    if path.suffix == ".py":
        try:
            ast.parse(text)
        except SyntaxError as e:
            return False, f"python: {e}"
    js = _extract_script(text) if path.suffix in (".py", ".html", ".htm") else None
    if js:
        ok, err = _js_parses(js)
        if not ok:
            return False, f"embedded js: {err}"
    return True, ""


# --------------------------------------------------------------------------
# Harness-defect detectors
# --------------------------------------------------------------------------
#
# Each returns a list of human-readable defects. A session with an empty union
# counts toward the Harness Integrity Rate. These are deliberately conservative:
# anything ambiguous is NOT counted, so the reported rate is an upper bound on
# integrity only in the sense that undetected classes exist — never inflated by
# a false positive.

WRITE_TOOLS = {"write_file", "edit_file", "structural_edit", "delete_file",
               "move_file"}


@dataclass
class Session:
    task: str
    rep: int
    events: list[dict]
    workspace: Path
    wall_s: float
    stream_ok: bool
    defects: list[str] = field(default_factory=list)
    task_passed: bool = False
    task_detail: str = ""
    quality: dict = field(default_factory=dict)

    def of_type(self, t: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == t]


def h1_protocol(s: Session, known_types: set[str]) -> list[str]:
    """Every tool_call answered, stream terminated, every event type known."""
    out = []
    calls = len(s.of_type("tool_call"))
    results = len(s.of_type("tool_result"))
    if calls != results:
        out.append(f"H1 protocol: {calls} tool_call vs {results} tool_result "
                   f"(orphaned call)")
    if not s.of_type("done"):
        out.append("H1 protocol: stream ended without a done event")
    if not s.stream_ok:
        out.append("H1 protocol: stream terminated abnormally")
    seen = {e.get("type") for e in s.events}
    unknown = sorted(t for t in seen if t and t not in known_types)
    if unknown:
        out.append(f"H1 protocol: event type(s) the TUI cannot render: {unknown}")
    return out


def h2_false_rejection(s: Session) -> list[str]:
    """A rejection blaming the file for a defect the file does not have.

    The D9 class: V3 authored the broken content, the gate blocked it, but the
    message named the file, so the model went hunting a bug that was not there.
    Decided against the workspace as it stands after the session.
    """
    out = []
    for ev in s.of_type("tool_result"):
        d = ev.get("data") or {}
        if d.get("success"):
            continue
        err = str(d.get("error") or "")
        m = re.search(r"([\w./-]+\.(?:py|html|htm|js)) has a .*syntax error", err)
        if not m:
            continue
        target = s.workspace / Path(m.group(1)).name
        if not target.exists():
            continue
        ok, _ = _file_parses(target)
        if ok:
            out.append(f"H2 false rejection: {m.group(1)} was blamed for a "
                       f"syntax error it does not have")
    return out


def h3_dead_end_steering(s: Session) -> list[str]:
    """Steering that names a remedy the very next call is rejected for using.

    The D10 class: a nudge offered an HTML <tag> selector for a .py file, the
    model complied, and the tool refused it as unsupported.
    """
    out = []
    results = s.of_type("tool_result")
    calls = s.of_type("tool_call")
    for i, ev in enumerate(results[:-1]):
        d = ev.get("data") or {}
        if d.get("success"):
            continue
        advice = str(d.get("error") or "")
        nxt = (results[i + 1].get("data") or {})
        if nxt.get("success"):
            continue
        nxt_err = str(nxt.get("error") or "")
        if not re.search(r"unknown selector|unsupported|not supported|HTML-only",
                         nxt_err, re.I):
            continue
        used = ""
        if i + 1 < len(calls):
            used = str(((calls[i + 1].get("data") or {}).get("args") or {})
                       .get("selector") or "")
        if not used:
            continue
        # Match on FORM, not on the literal string. The observed defect was
        # advice offering `<body>` on a .py file and the model reaching for
        # `<script>`: a different tag, the same unsupported shape, so an
        # exact-string check would miss the very case this exists to catch.
        advised_forms = _selector_forms(advice)
        if _selector_form(used) in advised_forms:
            out.append(f"H3 dead-end steering: a rejection recommended a "
                       f"{_selector_form(used)} selector, and the next call was "
                       f"refused for using {used!r}")
    return out


def _selector_form(sel: str) -> str:
    sel = sel.strip()
    if sel.startswith("<") and sel.endswith(">"):
        return "<tag>"
    if sel.startswith("function:"):
        return "function:NAME"
    if sel.startswith("class:"):
        return "class:NAME"
    return "other"


def _selector_forms(text: str) -> set[str]:
    forms = set()
    if re.search(r"<[a-zA-Z][\w-]*>", text):
        forms.add("<tag>")
    if "function:" in text:
        forms.add("function:NAME")
    if "class:" in text:
        forms.add("class:NAME")
    return forms


def h4_gate_escape(s: Session) -> list[str]:
    """An action-intent session that exited having changed nothing, ungated.

    The D11 class: one gate spent the shared bounce budget, so the
    done-without-action gate never ran.
    """
    productive = any((e.get("data") or {}).get("success")
                     and (c.get("data") or {}).get("name") in WRITE_TOOLS
                     for c, e in zip(s.of_type("tool_call"),
                                     s.of_type("tool_result")))
    if productive:
        return []
    if not s.of_type("done"):
        return []
    # The breaker ending honestly is not an escape — it says it stopped.
    summary = " ".join(str((e.get("data") or {}).get("summary") or "")
                       for e in s.of_type("done"))
    texts = " ".join(str((e.get("data") or {}).get("content") or "")
                     for e in s.of_type("text"))
    if re.search(r"stopped after|could not|unable to|failed to", summary + texts,
                 re.I):
        return []
    return ["H4 gate escape: exited with no successful write on an "
            "action-intent prompt, without saying it had stopped"]


def h5_corrupt_write(s: Session, task: Task) -> list[str]:
    out = []
    for p in sorted(s.workspace.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".html", ".htm", ".js"):
            continue
        ok, why = _file_parses(p)
        if not ok:
            out.append(f"H5 corrupt write: {p.name} left unparseable ({why})")
    for name in task.must_exist:
        if not (s.workspace / name).exists():
            out.append(f"H5 corrupt write: {name} was deleted")
    return out


def h6_service_fault(s: Session) -> list[str]:
    out = []
    for ev in s.of_type("error"):
        d = ev.get("data") or {}
        # The proxy's error events carry "error" (see the TUI's own case);
        # "message" is what this harness uses for a stream-level failure it
        # synthesises. Reading only one of them reported every real error as
        # the string "None".
        detail = d.get("error") or d.get("message") or json.dumps(d)[:120]
        out.append(f"H6 service fault: error event {str(detail)[:160]!r}")
    for ev in s.of_type("tool_result"):
        err = str((ev.get("data") or {}).get("error") or "")
        if re.search(r"\b5\d\d\b.*(?:proxy|v3|lens|sandbox)|connection refused|"
                     r"service unavailable|internal server error", err, re.I):
            out.append(f"H6 service fault: {err[:120]!r}")
    return out


def h8_anchored_on_injected_text(s: Session) -> list[str]:
    """The model anchored an edit on text ATLAS injected, not on file content.

    read_file appends a call-graph footer to the content it returns, and the
    loop injects "[system note]:" correctives. Neither is on disk, so an
    old_str copied from them can never match — the edit fails through no fault
    of the model's, and it burns a turn (a measured session spent all three of
    its failures this way and stopped). Attributing that to the model would
    make the harness understate exactly the defects it exists to find.
    """
    markers = ("## Call graph (within this file)", "[system note]:",
               "--- end of ", "The lines below are ATLAS analysis")
    out = []
    for ev in s.of_type("tool_call"):
        d = ev.get("data") or {}
        old = str(((d.get("args") or {}).get("old_str")) or "")
        if not old:
            continue
        for m in markers:
            if m in old:
                out.append(f"H8 injected-text anchor: {d.get('name')} old_str "
                           f"copied ATLAS's own {m.strip()!r}, which is not on disk")
                break
    return out


def h7_background_leak(sandbox: str, s: Session) -> list[str]:
    """A background job may outlive the session, but not silently.

    Persistence is deliberate: an agent loop is one user message, so killing
    jobs at its end would break "start the dev server" followed by "now curl
    it". The defect is a job that keeps its port with nobody told — the next
    turn then fails on a bound port with no explanation. So this fires only
    when jobs are running AND the session never said so.
    """
    if not sandbox:
        return []
    p = subprocess.run(["docker", "exec", sandbox, "sh", "-c",
                        "ps -eo args | grep -v grep | grep -c 'python app' || true"],
                       capture_output=True, text=True, timeout=30)
    n = (p.stdout or "0").strip()
    if not (n.isdigit() and int(n) > 0):
        return []
    announced = any(
        "still running" in str((e.get("data") or {}).get("summary") or "").lower()
        or "stop_background" in str((e.get("data") or {}).get("summary") or "")
        for e in s.of_type("done"))
    if announced:
        return []
    return [f"H7 silent background leak: {n} job(s) still holding ports and the "
            f"session never said so"]


# --------------------------------------------------------------------------
# TUI coverage
# --------------------------------------------------------------------------

def tui_handled_types() -> set[str]:
    """Event types the TUI's dispatcher has a case for.

    Located by content marker rather than filename so a file move does not
    silently empty the set (the repo's contract-test convention).
    """
    out: set[str] = set()
    for go in sorted((REPO / "tui").glob("*.go")):
        if go.name.endswith("_test.go"):
            continue
        src = go.read_text()
        if "appendChatEvent" not in src:
            continue
        for m in re.finditer(r'^\s*case ((?:"[a-z0-9_]+"(?:,\s*)?)+):',
                             src, re.M):
            out.update(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    return out


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_session(task: Task, rep: int, url: str, workspace: Path,
                subdir: str, timeout: int) -> Session:
    # Wipe the workspace, then lay down only this task's fixtures. Resetting
    # the fixtures alone is not isolation: solve.py from a previous AoC task
    # survived into the next one, and a session that wrote nothing would have
    # been scored on the earlier task's program.
    if workspace.exists():
        for leftover in sorted(workspace.rglob("*"), reverse=True):
            try:
                if leftover.is_file() or leftover.is_symlink():
                    leftover.unlink()
                elif leftover.is_dir():
                    leftover.rmdir()
            except OSError:
                pass
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in task.files.items():
        (workspace / name).write_text(content)

    # sandbox_subdir, NOT working_dir. The proxy deliberately overrides the
    # client's working_dir with ATLAS_WORKSPACE_DIR (agent.go): the TUI sends
    # its HOST cwd, which does not exist inside the container, and the bind
    # mount is aligned so /workspace already IS the user's directory.
    # Passing a subdir through working_dir is therefore silently ignored, and
    # an earlier version of this harness had every session operating on
    # /workspace while the checks read the subdirectory — so real successes
    # were scored as failures. sandbox_subdir is the field that scopes a run.
    body = json.dumps({
        "message": task.prompt,
        "mode": "yolo",
        "sandbox_subdir": subdir,
        "session_id": f"reliability-{task.name}-{rep}",
    }).encode()
    req = urllib.request.Request(f"{url}/v1/agent", data=body,
                                 headers={"Content-Type": "application/json"})
    events: list[dict] = []
    stream_ok = False
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                # urlopen's timeout is per-read, so a session that keeps
                # streaming never trips it. One observed session looped past
                # 20 minutes against a 900s cap, trying to satisfy a self-test
                # it had written with the wrong expectation.
                if time.time() - t0 > timeout:
                    events.append({"type": "error", "data": {
                        "error": f"harness cap: session exceeded {timeout}s"}})
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    stream_ok = True
                    break
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    events.append({"type": "__unparseable__", "raw": payload[:200]})
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        events.append({"type": "error", "data": {"error": f"stream failed: {e}"}})
    wall = time.time() - t0

    s = Session(task=task.name, rep=rep, events=events, workspace=workspace,
                wall_s=wall, stream_ok=stream_ok)
    try:
        s.task_passed, s.task_detail = task.check(workspace)
    except Exception as e:  # a check that explodes is a failed task, not a crash
        s.task_passed, s.task_detail = False, f"check raised: {e}"
    # Quality of what the agent wrote, excluding the fixtures it was handed.
    try:
        s.quality = analyze_quality(workspace, set(task.files)).as_dict()
    except Exception as e:
        s.quality = {"error": str(e)}
    return s


def preflight(sandbox: str, subdir: str) -> list[str]:
    """Refuse to measure a stack that is misconfigured.

    The proxy and the sandbox each bind a host directory at /workspace, and
    nothing in a session fails loudly when those differ: file tools write
    through the proxy's mount while every run_command executes against the
    sandbox's. A whole run then produces confident numbers about an
    environment where edits and verification never met. That happened here —
    proxy on ~/demo, sandbox on ~/demo2 — and cost a full run, so the harness
    now checks before it spends an hour. `atlas doctor` reports the same thing
    under workspace_mounts.
    """
    problems: list[str] = []
    if not sandbox:
        return problems

    def mount_of(container: str) -> str:
        p = subprocess.run(
            ["docker", "inspect", container, "--format",
             '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}'],
            capture_output=True, text=True, timeout=30)
        return (p.stdout or "").strip()

    proxy_mount = mount_of("atlas-atlas-proxy-1")
    sandbox_mount = mount_of(sandbox)
    if proxy_mount and sandbox_mount and proxy_mount != sandbox_mount:
        problems.append(
            f"proxy and sandbox bind DIFFERENT host dirs at /workspace "
            f"(proxy={proxy_mount} sandbox={sandbox_mount}). Edits and "
            f"verification would run on separate filesystems. Fix: set "
            f"ATLAS_PROJECT_DIR in .env, then "
            f"`docker compose up -d --force-recreate atlas-proxy sandbox`")

    # The subdir has to be visible to the sandbox too, or every verification
    # command fails with "cwd does not exist" and the task looks unsolvable.
    if subdir:
        p = subprocess.run(["docker", "exec", sandbox, "test", "-d",
                            f"/workspace/{subdir}"], capture_output=True, timeout=30)
        if p.returncode != 0:
            problems.append(
                f"/workspace/{subdir} does not exist inside {sandbox} — every "
                f"run_command would fail with 'cwd does not exist'")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("ATLAS_PROXY_URL",
                                                    "http://127.0.0.1:8090"))
    ap.add_argument("--workspace", default=os.environ.get("ATLAS_PROJECT_DIR", ""),
                    help="host path of the subdirectory named by --subdir")
    ap.add_argument("--subdir", default="_reliability",
                    help="workspace subdirectory to confine each run to "
                         "(sent as sandbox_subdir); --workspace must be the "
                         "host path of this same subdirectory")
    ap.add_argument("--sandbox-container", default="atlas-sandbox-1",
                    help="'' to skip the background-leak check")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", dest="json_out", default="")
    ap.add_argument("--save-events", default="",
                    help="directory to write each session's raw event stream "
                         "to; without it a failure can only be diagnosed from "
                         "container logs, which roll")
    args = ap.parse_args()

    if not args.workspace:
        print("error: --workspace (or ATLAS_PROJECT_DIR) must be set — it has "
              "to be the host path the proxy mounts.", file=sys.stderr)
        return 2
    ws = Path(args.workspace)
    if not ws.is_dir():
        print(f"error: workspace {ws} is not a directory", file=sys.stderr)
        return 2

    TASKS["flask_pause"].files["app.py"] = _read_fixture("snake_app.py")

    selected = [TASKS[n] for n in args.tasks.split(",") if n in TASKS]
    if not selected:
        print(f"error: no known tasks in {args.tasks!r}", file=sys.stderr)
        return 2

    if problems := preflight(args.sandbox_container, args.subdir):
        for line in problems:
            print(f"error: {line}", file=sys.stderr)
        return 2

    known = tui_handled_types()
    if not known:
        print("error: could not read the TUI event dispatcher", file=sys.stderr)
        return 2
    # Types the harness itself synthesises are not TUI concerns.
    known |= {"__unparseable__"}

    sessions: list[Session] = []
    total = len(selected) * args.reps
    n = 0
    for rep in range(1, args.reps + 1):
        for task in selected:
            n += 1
            print(f"[{n}/{total}] {task.name} rep {rep} ...", flush=True)
            if args.sandbox_container:
                subprocess.run(["docker", "exec", args.sandbox_container,
                                "pkill", "-f", "python app"],
                               capture_output=True, timeout=30)
            s = run_session(task, rep, args.url, ws,
                            args.subdir, args.timeout)
            s.defects += h1_protocol(s, known)
            s.defects += h2_false_rejection(s)
            s.defects += h3_dead_end_steering(s)
            s.defects += h4_gate_escape(s)
            s.defects += h5_corrupt_write(s, task)
            s.defects += h6_service_fault(s)
            s.defects += h8_anchored_on_injected_text(s)
            s.defects += h7_background_leak(args.sandbox_container, s)
            sessions.append(s)
            if args.save_events:
                d = Path(args.save_events)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{task.name}-rep{rep}.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in s.events))
            turns = len(s.of_type("turn_start"))
            print(f"      task={'PASS' if s.task_passed else 'fail'} "
                  f"harness={'clean' if not s.defects else str(len(s.defects)) + ' defect(s)'} "
                  f"turns={turns} {s.wall_s:.0f}s — {s.task_detail[:70]}",
                  flush=True)
            for d in s.defects:
                print(f"      ! {d}", flush=True)

    report(sessions, known)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([{
            "task": s.task, "rep": s.rep, "task_passed": s.task_passed,
            "task_detail": s.task_detail, "defects": s.defects,
            "turns": len(s.of_type("turn_start")),
            "tools": len(s.of_type("tool_call")), "wall_s": round(s.wall_s, 1),
            "quality": s.quality,
        } for s in sessions], indent=2))
        print(f"\nwrote {args.json_out}")
    return 0 if all(not s.defects for s in sessions) else 1


def report(sessions: list[Session], known: set[str]) -> None:
    total = len(sessions)
    clean = sum(1 for s in sessions if not s.defects)
    passed = sum(1 for s in sessions if s.task_passed)
    print("\n" + "=" * 72)
    print(f"Harness Integrity Rate   {clean}/{total} "
          f"({100.0 * clean / total:.0f}%)   <- ATLAS's own plumbing")
    print(f"Task Success Rate        {passed}/{total} "
          f"({100.0 * passed / total:.0f}%)   <- bounded by model ability")
    print("=" * 72)

    by_class: dict[str, int] = {}
    for s in sessions:
        for d in s.defects:
            by_class[d.split(":")[0]] = by_class.get(d.split(":")[0], 0) + 1
    if by_class:
        print("\nHarness defects by class:")
        for cls, cnt in sorted(by_class.items(), key=lambda kv: -kv[1]):
            print(f"  {cnt:3d}  {cls}")
    else:
        print("\nNo harness defects detected.")

    print("\nPer task:")
    for name in sorted({s.task for s in sessions}):
        rows = [s for s in sessions if s.task == name]
        cl = sum(1 for s in rows if not s.defects)
        pa = sum(1 for s in rows if s.task_passed)
        turns = [len(s.of_type("turn_start")) for s in rows]
        print(f"  {name:14s} harness {cl}/{len(rows)}  task {pa}/{len(rows)}  "
              f"turns min/med/max {min(turns)}/{sorted(turns)[len(turns)//2]}/{max(turns)}")

    q = [s.quality for s in sessions if s.quality and "error" not in s.quality]
    if q:
        print("\nCode quality of what the agent wrote:")
        worst_cx = max(q, key=lambda r: r.get("max_complexity", 0))
        worst_fn = max(q, key=lambda r: r.get("max_function_lines", 0))
        worst_file = max(q, key=lambda r: r.get("max_file_lines", 0))
        lint = sum(r.get("lint_violations", 0) for r in q)
        unused = sum(r.get("unused_imports", 0) for r in q)
        broken = sum(len(r.get("syntax_errors") or []) for r in q)
        clean = sum(1 for r in q if not r.get("findings"))
        print(f"  sessions with no quality finding   {clean}/{len(q)}")
        print(f"  worst function complexity          {worst_cx.get('max_complexity', 0)}"
              f" ({worst_cx.get('max_complexity_where') or 'n/a'})")
        print(f"  longest function                   {worst_fn.get('max_function_lines', 0)} lines"
              f" ({worst_fn.get('max_function_where') or 'n/a'})")
        print(f"  longest file                       {worst_file.get('max_file_lines', 0)} lines"
              f" ({worst_file.get('max_file_where') or 'n/a'})")
        print(f"  lint violations / unused imports   {lint} / {unused}")
        print(f"  files left unparseable             {broken}")

    observed: set[str] = set()
    for s in sessions:
        observed |= {e.get("type") for e in s.events if e.get("type")}
    unrendered = sorted(observed - known)
    print(f"\nTUI coverage: {len(observed)} event types emitted, "
          f"{len(unrendered)} the TUI cannot render"
          + (f": {unrendered}" if unrendered else ""))


if __name__ == "__main__":
    raise SystemExit(main())
