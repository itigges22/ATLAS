"""The V3 pipeline orchestrator: probe, candidate generation, sandbox
verification, the lens/structural/call-graph vetoes, candidate selection,
the repair phases, stage telemetry, and the /v3/generate problem builder."""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from stages.llm_client import extract_code
from stages.budget_forcing import BudgetForcing, BudgetForcingConfig
from stages import cxgx_gate
from stages.plan_search import PlanSearch, PlanSearchConfig
from stages.div_sampling import DivSampling, DivSamplingConfig
from stages.failure_analysis import FailingCandidate
from stages.pr_cot import PRCoT, PRCoTConfig
from stages.refinement_loop import (
    RefinementLoop, RefinementLoopConfig,
    can_afford_iteration, estimate_iteration_ms,
)
from stages.self_test_gen import SelfTestGen, SelfTestGenConfig
from stages.candidate_selection import CandidateInfo, select_candidate

import adapters
import scoring
import symbols

BASE_TEMPERATURE = 0.6
DIVERSITY_TEMPERATURE = 0.8
MAX_TOKENS = 8192


# --- Stage telemetry ---------------------------------------------------------

# Serializes pipeline-summary appends across the ThreadingHTTPServer's
# request threads. Stage JSONL appends need no lock: each event is one
# small O_APPEND write.
_SUMMARY_LOCK = threading.Lock()

_TELEMETRY_DISABLE_VALUES = {"0", "off", "none", "disabled", "false"}

# stage name -> summary phase. Stages not listed (token/llm_*/task_type/…)
# don't contribute a phase row.
_STAGE_PHASE = {}
for _phase, _stages in {
    "probe": ("probe", "probe_light", "probe_retry", "probe_failed",
              "probe_error", "probe_scored", "probe_sandbox", "probe_pass"),
    "self_test": ("self_test_gen", "self_test_done", "self_test_error",
                  "self_test_skip", "self_test_inconclusive"),
    "allocation": ("phase2", "phase2_allocated"),
    "generation": ("phase1", "plansearch", "plansearch_done",
                   "plansearch_error", "divsampling", "divsampling_done",
                   "divsampling_error", "divsampling_stop", "lens_per_step"),
    "sandbox": ("sandbox_test", "sandbox_pass", "sandbox_fail",
                "sandbox_done", "smoke_check", "interactive_lint",
                "self_test_verify", "build_verify",
                "build_verify_unavailable"),
    "veto": ("lens_veto", "structural_veto", "call_graph_veto"),
    "selection": ("selected", "consensus"),
    "repair_pr_cot": ("phase3", "call_chain_context", "pr_cot",
                      "pr_cot_pass", "pr_cot_failed", "pr_cot_error"),
    "repair_refinement": ("refinement", "refinement_pass",
                          "refinement_failed", "refinement_error",
                          "refinement_verify_failed", "refinement_skip"),
    "fallback": ("fallback", "fallback_all_vetoed", "fallback_unverified",
                 "budget_exhausted", "budget_no_verified_candidate"),
}.items():
    for _s in _stages:
        _STAGE_PHASE[_s] = _phase

_VETO_STAGES = frozenset(("lens_veto", "structural_veto", "call_graph_veto"))


def _remaining_budget_ms(start: float) -> Optional[float]:
    """Remaining wall-clock (ms) in this run's ATLAS_V3_TIMEOUT budget.

    The proxy's V3 bridge abandons a live pipeline call after
    ``ATLAS_V3_TIMEOUT`` seconds (default 300; 0 disables the cap).
    The service reads the same knob so late phases can skip work the
    bridge would abandon mid-flight anyway. Returns None when the cap
    is disabled.
    """
    raw = os.environ.get("ATLAS_V3_TIMEOUT", "").strip()
    try:
        seconds = int(raw) if raw else 300
    except ValueError:
        seconds = 180
    if seconds <= 0:
        return None
    return seconds * 1000.0 - (time.time() - start) * 1000.0


def _resolve_telemetry_dir() -> Optional[Path]:
    """Resolve the stage-telemetry directory for the live service.

    ``ATLAS_V3_TELEMETRY_DIR`` names the directory; a disable value
    (``0``/``off``/``none``/``disabled``/``false``) turns telemetry off;
    unset/empty falls back to ``/data/telemetry`` when writable (the
    compose volume), else telemetry is disabled. Resolution never
    raises — telemetry must not break generation.
    """
    configured = os.environ.get("ATLAS_V3_TELEMETRY_DIR", "").strip()
    if configured.lower() in _TELEMETRY_DISABLE_VALUES:
        return None
    candidate = Path(configured) if configured else Path("/data/telemetry")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError as e:
        if configured:
            print(f"  [telemetry] {candidate} not writable ({e}) — "
                  f"stage telemetry disabled", flush=True)
        return None


def _summarize_phases(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold the run's progress events into ordered per-phase rows.

    Each row carries the phase name, the stage that closed it (its
    outcome marker), that stage's detail, and the span between the
    phase's first and last event. Derived purely from the events the
    run already emits — no extra instrumentation in the hot path.
    """
    rows: List[Dict[str, Any]] = []
    by_phase: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        phase = _STAGE_PHASE.get(ev.get("stage", ""))
        if phase is None:
            continue
        row = by_phase.get(phase)
        if row is None:
            row = {"phase": phase, "first_ms": round(ev.get("t", 0.0) * 1000)}
            by_phase[phase] = row
            rows.append(row)
        row["outcome"] = ev.get("stage", "")
        row["detail"] = str(ev.get("detail", ""))[:120]
        row["duration_ms"] = round(ev.get("t", 0.0) * 1000) - row["first_ms"]
    return rows


# --- V3 Pipeline Orchestrator ------------------------------------------------

def _candidate_by_index(candidates: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """Return the candidate dict whose original ``index`` field matches.

    Selection reports the winner by the candidate's original index, but
    the ``passing`` list has been sorted and filtered — positional
    indexing would pick the wrong candidate (or IndexError). Returns None
    when no candidate carries that index.
    """
    return next((c for c in candidates if c.get("index") == index), None)


def _entry_function(code: str) -> Optional[str]:
    """Name the function a self-test should call.

    The generator used to take the first `def` in the file, which is the
    entry point only when the solution happens to define no helpers above
    it. `def parse(...)` followed by `def solve(...)` meant every test
    called parse with the case input and compared it to the final answer —
    a guaranteed failure that says nothing about the code.

    Measured across a 28-session run: 0 of 44 candidates passed, and the
    self-test results were 0/5, 0/4, 0/3 — never partial. Imperfect code
    fails some cases; only a harness fault fails all of them uniformly.

    The entry point is the top-level function nothing else in the file
    calls. Where several qualify a conventional name wins, then the last
    one, since helpers are conventionally defined above their caller.
    Unparseable code falls back to the first def, which is what the old
    behaviour was.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        m = re.search(r'^def (\w+)\(', code, re.MULTILINE)
        return m.group(1) if m else None
    top = [n for n in tree.body if isinstance(n, _ast.FunctionDef)]
    if not top:
        return None
    names = {n.name for n in top}
    called = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            if node.func.id in names:
                called.add(node.func.id)
    roots = [n.name for n in top if n.name not in called]
    for preferred in ("solve", "main", "run"):
        if preferred in roots:
            return preferred
    if roots:
        return roots[-1]
    return top[-1].name


def _entry_takes_case_input(code: str, name: str) -> bool:
    """Whether the case input can be passed to `name` and its answer read
    back as a return value.

    The two self-test shapes are not interchangeable. A function-shaped
    solution takes the case input as arguments and returns the answer. A
    script-shaped one reads stdin and prints — calling it with the case
    input raises TypeError, and there is no return value to compare.

    Measured on aoc_sonar: candidates are `def main():` reading sys.stdin
    and printing, so the function path called main(case) and every case
    failed. The stdin/stdout path below handles exactly this and was never
    reached, because the choice keyed on a function merely existing.

    Requires all three: the function takes at least one parameter, it
    returns a value somewhere, and the file does not read stdin itself.
    """
    import ast as _ast
    if "sys.stdin" in code or "input()" in code:
        return False
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return True  # old behaviour for code that does not parse
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            args = node.args
            takes = bool(args.args or args.posonlyargs or args.kwonlyargs
                         or args.vararg)
            returns = any(isinstance(n, _ast.Return) and n.value is not None
                          for n in _ast.walk(node))
            return takes and returns
    return False


def _reads_input_file(code: str):
    """The literal filename a candidate opens for reading, if it opens one.

    Returns None when the program takes no named file, which is the case the
    stdin path below already covers.

    Only string literals count. A computed path cannot be materialised without
    running the program, and guessing one wrong is worse than falling through
    to the existing behaviour.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in ("open", "read_text", "read_bytes", "Path"):
            continue
        if not node.args:
            continue
        target = node.args[0]
        if not (isinstance(target, _ast.Constant) and isinstance(target.value, str)):
            continue
        # A write/append target is an output file, not the program's input.
        mode = node.args[1] if len(node.args) > 1 else None
        if isinstance(mode, _ast.Constant) and isinstance(mode.value, str):
            if any(ch in mode.value for ch in ("w", "a", "x")):
                continue
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, _ast.Constant):
                if any(ch in str(kw.value.value) for ch in ("w", "a", "x")):
                    break
        else:
            if "/" not in target.value and "\\" not in target.value:
                return target.value
    return None


def _make_self_test(code: str, tc) -> str:
    """Build executable assertion code for a single test case.

    Uses ast.literal_eval (safe — only parses Python literals) to convert
    I/O string representations to actual values for comparison.
    All code runs inside the sandboxed container.
    """
    inp = tc.input_str.strip()
    exp = tc.expected_output.strip()
    name = _entry_function(code)
    if name and _entry_takes_case_input(code, name):
        return (code + "\nimport ast as _a\n"
            + f"_i={repr(inp)}\n_e={repr(exp)}\n"
            + "try:\n _p=_a.literal_eval(_i)\nexcept:\n _p=_i\n"  # noqa: E722  -- bare except inside generated user code, intentional
            + f"_r={name}(*_p) if isinstance(_p,tuple) else {name}(_p) if isinstance(_p,list) else {name}(_p)\n"
            + "try:\n _ev=_a.literal_eval(_e)\nexcept:\n _ev=_e\n"  # noqa: E722  -- bare except inside generated user code, intentional
            + "assert str(_r)==str(_ev) or _r==_ev,f'got {_r}'\nprint('SELF_TEST_PASS')\n")
    # A program that reads a named file has to be given that file. Feeding it
    # stdin instead tests a contract the task never stated, and the verdict
    # comes out backwards: a candidate that correctly reads input.txt finds no
    # such file in the sandbox and FAILS, while one that reads stdin passes.
    # Verification then selects for the shape that cannot work when the caller
    # runs `python solve.py` with no stdin.
    #
    # Measured on the two AoC tasks whose input is a file: 9 of 12 candidates
    # ATLAS wrote read stdin, against a prompt that says "reads input.txt",
    # and both tasks sat at 7/26. The same model asked directly, with no
    # pipeline, wrote input.txt readers and scored 12/12.
    infile = _reads_input_file(code)
    if infile:
        return (
            "import sys as _s,io as _o\n"
            f"open({repr(infile)},'w').write({repr(inp)})\n"
            "_c=_o.StringIO()\n_old=_s.stdout\n_s.stdout=_c\n"
            f"_src={repr(code)}\n"
            "try:\n    exec(compile(_src,'solution.py','exec'),globals())\n"
            "finally:\n _s.stdout=_old\n"
            f"assert _c.getvalue().strip()=={repr(exp)},f'got {{_c.getvalue().strip()}}'\n"
            "print('SELF_TEST_PASS')\n")
    # exec the candidate from a string literal instead of splicing its lines
    # under `try:` — per-line indenting corrupts multiline string literals
    # inside the candidate. exec(..., globals()) keeps the namespace (and
    # __name__) identical to the previous inline form.
    return (
        "import sys as _s,io as _o\n"
        f"_s.stdin=_o.StringIO({repr(inp)})\n"
        "_c=_o.StringIO()\n_old=_s.stdout\n_s.stdout=_c\n"
        f"_src={repr(code)}\n"
        "try:\n    exec(compile(_src,'solution.py','exec'),globals())\nfinally:\n _s.stdout=_old\n"
        f"assert _c.getvalue().strip()=={repr(exp)},f'got {{_c.getvalue().strip()}}'\n"
        "print('SELF_TEST_PASS')\n")


_CONSENSUS_MARK = "V3_OUT:"


def _make_output_probe(code: str, tc) -> str:
    """Run a candidate on a case's INPUT and report what it printed.

    The self-test compares that output to the model's predicted answer. For a
    problem the model cannot reliably solve, producing the answer IS the
    problem, so the prediction is wrong and correct code fails its own suite.
    Measured across 42 verifications in one run: every one scored 0/N, while a
    candidate taken from those same logs passed immediately when given a
    correct expected value. Uniform zero is a broken answer key, not broken
    code.

    CodeT (Chen et al., 2022), which this pipeline already cites, does not use
    generated tests as an oracle, for exactly this reason. It runs candidates
    on the generated INPUTS and clusters them by agreement, so the signal is
    the answer candidates converge on rather than the answer the model
    guessed. This probe produces the raw material for that.
    """
    inp = tc.input_str.strip()
    name = _entry_function(code)
    if name and _entry_takes_case_input(code, name):
        return (code + "\nimport ast as _a\n"
                + f"_i={repr(inp)}\n"
                + "try:\n _p=_a.literal_eval(_i)\nexcept:\n _p=_i\n"
                + f"_r={name}(*_p) if isinstance(_p,tuple) else {name}(_p)\n"
                + f"print({repr(_CONSENSUS_MARK)}+repr(str(_r).strip()))\n")
    infile = _reads_input_file(code)
    setup = (f"open({repr(infile)},'w').write({repr(inp)})\n" if infile
             else f"_s.stdin=_o.StringIO({repr(inp)})\n")
    return ("import sys as _s,io as _o\n" + setup
            + "_c=_o.StringIO()\n_old=_s.stdout\n_s.stdout=_c\n"
            + f"_src={repr(code)}\n"
            + "try:\n    exec(compile(_src,'solution.py','exec'),globals())\n"
            + "except Exception:\n    pass\n"
            + "finally:\n _s.stdout=_old\n"
            + f"print({repr(_CONSENSUS_MARK)}+repr(_c.getvalue().strip()))\n")


def _consensus_winners(candidates, test_cases, sandbox, emit):
    """CodeT agreement: candidates whose outputs match the largest cluster.

    Returns [] when there is nothing to agree on — fewer than two candidates,
    no case produced output, or every candidate disagreed with every other.
    Agreement between independently generated programs is evidence; one
    program agreeing with itself is not, so a lone cluster does not win.
    """
    if len(candidates) < 2 or not test_cases:
        return []
    sigs = {}
    for c in candidates:
        outs = []
        for tc in test_cases:
            try:
                ok, out, _ = sandbox(_make_output_probe(c["code"], tc))
            except Exception:
                ok, out = False, ""
            marker = ""
            if ok and _CONSENSUS_MARK in (out or ""):
                marker = out.split(_CONSENSUS_MARK)[-1].strip()
            outs.append(marker)
        if any(outs):
            sigs.setdefault(tuple(outs), []).append(c)
    if not sigs:
        return []
    best = max(sigs.values(), key=len)
    if len(best) < 2:
        return []
    emit("consensus", f"{len(best)}/{len(candidates)} candidates agree",
         cluster=len(best), clusters=len(sigs))
    return best


class V3PipelineService:
    """Full V3 pipeline for a single coding task, with streaming progress."""

    def __init__(self):
        # ALL V3 components enabled — same as benchmark runner with all phases
        # active. Stage telemetry mirrors the bench runner's telemetry/*.jsonl
        # into ATLAS_V3_TELEMETRY_DIR so live-orchestrator runs are measurable.
        self.telemetry_dir = _resolve_telemetry_dir()
        t = self.telemetry_dir
        self.budget_forcing = BudgetForcing(BudgetForcingConfig(enabled=True),
                                            telemetry_dir=t)
        self.plan_search = PlanSearch(PlanSearchConfig(enabled=True),
                                      telemetry_dir=t)
        self.div_sampling = DivSampling(DivSamplingConfig(enabled=True),
                                        telemetry_dir=t)
        self.pr_cot = PRCoT(PRCoTConfig(enabled=True), telemetry_dir=t)
        self.refinement_loop = RefinementLoop(RefinementLoopConfig(enabled=True),
                                              telemetry_dir=t)
        self.self_test_gen = SelfTestGen(SelfTestGenConfig(enabled=True),
                                         telemetry_dir=t)

    def run(self, problem: str, task_id: str = "cli",
            progress_callback=None, files: Dict[str, str] = None,
            file_path: str = "", build_command: str = "",
            working_dir: str = "/workspace") -> Dict[str, Any]:
        """Run the full V3 pipeline on a coding problem.

        Args:
            problem: Problem description
            task_id: Task identifier
            progress_callback: SSE progress emitter
            files: Dict of filename→content from Aider's existing file context
            file_path: Target file path (used by PC-048 to detect language
                for the smoke check — `.html` files use HTML parser, not
                Python compile, etc.)
            build_command: Optional project build command to run against an
                ephemeral candidate overlay after syntax/self-tests pass.
            working_dir: Container workspace root used by the sandbox overlay.

        Writes one pipeline-summary telemetry line per task (fail-soft;
        see _write_pipeline_summary) around the actual pipeline body.
        """
        start = time.time()
        result: Optional[Dict[str, Any]] = None
        error = ""
        try:
            result = self._run_impl(
                problem, task_id=task_id, progress_callback=progress_callback,
                files=files, file_path=file_path, build_command=build_command,
                working_dir=working_dir)
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._write_pipeline_summary(task_id, result, error, start)

    def _write_pipeline_summary(self, task_id: str,
                                result: Optional[Dict[str, Any]],
                                error: str, start: float) -> None:
        """Append one summary line to telemetry/pipeline_summary.jsonl.

        Carries the per-task shape the bench runner gets for free from its
        per-task JSON files: phases run (outcome + duration, folded from the
        run's progress events), veto events, and the final result fields.
        Fail-soft by construction — a telemetry error never reaches the
        caller, so it can never break generation.
        """
        if self.telemetry_dir is None:
            return
        try:
            events = (result or {}).get("events") or []
            line = {
                "schema": "v3_pipeline_summary_v1",
                "ts": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
                "passed": bool((result or {}).get("passed")),
                "phase_solved": (result or {}).get("phase_solved", "none"),
                "task_type": (result or {}).get("task_type", ""),
                "candidates_generated": (result or {}).get("candidates_generated", 0),
                "total_tokens": (result or {}).get("total_tokens", 0),
                "total_time_ms": round(
                    (result or {}).get("total_time_ms")
                    or (time.time() - start) * 1000),
                "phases": _summarize_phases(events),
                "veto_events": [
                    {"stage": ev.get("stage", ""),
                     "index": (ev.get("data") or {}).get("index", -1),
                     "detail": str(ev.get("detail", ""))[:120]}
                    for ev in events if ev.get("stage") in _VETO_STAGES
                ],
            }
            if error:
                line["error"] = error[:300]
            with _SUMMARY_LOCK:
                with open(self.telemetry_dir / "pipeline_summary.jsonl", "a") as f:
                    f.write(json.dumps(line) + "\n")
        except Exception as e:
            print(f"  [telemetry] pipeline summary write failed (non-fatal): {e}",
                  flush=True)

    def _run_impl(self, problem: str, task_id: str = "cli",
                  progress_callback=None, files: Dict[str, str] = None,
                  file_path: str = "", build_command: str = "",
                  working_dir: str = "/workspace") -> Dict[str, Any]:
        """The pipeline body — see run() for the argument contract."""
        start = time.time()
        events = []
        files = files or {}

        # PC-048: derive language from the target file's extension. Used
        # only by smoke_compile_check below to pick the right syntax
        # checker. Defaults to Python when no file_path is supplied.
        _ext = Path(file_path).suffix.lower() if file_path else ""
        # Only languages scoring.smoke_compile_check can actually verify —
        # an entry here that the checker rejects would fail every candidate
        # with "verification unavailable" instead of checking anything.
        _ext_to_lang = {
            ".py": "python", ".pyw": "python",
            ".html": "html", ".htm": "html",
            ".json": "json",
            ".yaml": "yaml", ".yml": "yaml",
            ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".xml": "xml",
            ".sh": "bash", ".bash": "bash",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
        }
        smoke_language = _ext_to_lang.get(_ext, "python")

        # If existing file context is provided, prepend it to the problem
        # so all V3 modules (PlanSearch, PR-CoT, etc.) can see the code
        if files:
            file_context_parts = []
            for fname, content in files.items():
                file_context_parts.append(f"### Existing file: {fname}\n```\n{content}\n```")
            problem = (
                "The following files already exist in the project:\n\n"
                + "\n\n".join(file_context_parts)
                + "\n\n---\n\nTask:\n" + problem
            )

        def emit(stage, detail="", **data):
            ev = {"stage": stage, "detail": detail, "t": time.time() - start}
            if data:
                ev["data"] = data
            # Token deltas stream live through the callback but are not
            # stored: one dict per token would make the final `event: result`
            # frame multi-MB on long generations.
            if stage != "token":
                events.append(ev)
            if progress_callback:
                try:
                    progress_callback(stage, detail, **data)
                except TypeError:
                    progress_callback(stage, detail)

        def check_client():
            """Abort at phase boundaries once the SSE client disconnects.
            The handler sets `disconnected` on the callback when a write
            hits BrokenPipeError; a dead client must not keep burning GPU."""
            if getattr(progress_callback, "disconnected", False):
                raise adapters.ClientDisconnected(f"client disconnected during task {task_id}")

        # The adapter refuses to start a generation that cannot finish
        # before the cap, so every phase and every loop inside one is
        # covered by a single check rather than a boundary guard each.
        _budget_ms = _remaining_budget_ms(start)
        llm = adapters.LLMAdapter(progress_callback=emit)
        # Assigned rather than passed to the constructor: tests substitute
        # their own LLM doubles here, and a new required kwarg would break
        # every one of them for a value only the real adapter reads.
        if _budget_ms is not None:
            llm.deadline = start + _budget_ms / 1000.0
        # PC-046: ship the user's other project files into the sandbox so
        # multi-file imports resolve. `files` is the same Dict that V3
        # already prepends to the LLM prompt above; passing it to the
        # sandbox closes the gap where the model writes
        # `from utils import helper` and the sandbox imports a workspace
        # that contains only solution.py.
        sandbox = adapters.SandboxAdapter(project_files=files)
        embed = adapters.EmbedAdapter()

        result = {
            "task_id": task_id,
            "passed": False,
            "code": "",
            "phase_solved": "none",
            "candidates_generated": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "events": [],
            "verification_evidence": [],
        }

        # ===== PHASE 0: PROBE =====
        emit("probe", "Generating probe candidate...")
        # Light probe first (1024 thinking tokens), retry with standard if fails
        try:
            chatml = self.budget_forcing.format_chatml(problem, "light")
            response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
            probe_code = extract_code(response)
            if probe_code:
                emit("probe_light", f"Light probe: {len(probe_code)} chars, {tokens} tokens, {t_ms:.0f}ms")
        except Exception as e:
            emit("probe_error", str(e))
            probe_code = ""

        if not probe_code:
            emit("probe_retry", "Light probe failed — retrying with standard budget")
            try:
                chatml = self.budget_forcing.format_chatml(problem, "standard")
                response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
                probe_code = extract_code(response)
            except Exception as e:
                emit("probe_error", str(e))

        if not probe_code:
            emit("probe_failed", "No code extracted from probe")
            # Generate with the minimal reasoning budget
            chatml = self.budget_forcing.format_chatml(problem, "nothink")
            response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
            probe_code = extract_code(response)

        # Classify task type. Interactive tasks (games, UIs, framework code)
        # skip synthetic I/O self-tests entirely — those tests would fail by
        # construction, falsely triggering PR-CoT/refinement on working code.
        # See ISSUES.md PC-022.
        task_type = scoring.classify_task_type(problem)
        emit("task_type", task_type)
        result["task_type"] = task_type

        # Generate self-tests (algorithmic tasks only) — used for sandbox verification
        self_tests = None
        if task_type == "algorithmic":
            emit("self_test_gen", "Generating verification tests...")
            try:
                self_tests = self.self_test_gen.generate(problem, llm, task_id)
                emit("self_test_done", f"{len(self_tests.test_cases)} test cases")
                result["total_tokens"] += self_tests.generation_tokens
            except Exception as e:
                emit("self_test_error", str(e)[:200])
        else:
            emit("self_test_skip", "Interactive task — using compile smoke-test")

        def verified_sandbox(code, extra_test=""):
            """Sandbox + verification. Algorithmic tasks: I/O self-tests; interactive: compile smoke."""
            verification_evidence: List[Dict[str, Any]] = []

            def verify_build_if_requested(out="", err=""):
                ok, build_out, build_err, evidence = scoring.verify_build_command(
                    code=code,
                    sandbox=sandbox,
                    build_command=build_command,
                    file_path=file_path,
                    project_files=files,
                    working_dir=working_dir or "/workspace",
                    emit=emit,
                )
                if evidence:
                    verification_evidence.append(evidence)
                if not ok:
                    return False, build_out, build_err, verification_evidence
                return True, out, err, verification_evidence

            # Non-Python candidates always use the language-aware syntax path.
            # Python self-tests cannot establish correctness for another language.
            if smoke_language not in ("python", "py"):
                ok, out, err = scoring.smoke_compile_check(code, sandbox, language=smoke_language)
                emit("smoke_check", f"compile={'OK' if ok else 'FAIL'} ({smoke_language})")
                if not ok:
                    return ok, out, err, verification_evidence
                return verify_build_if_requested(out, err)

            # Interactive tasks: skip the run-and-test; just verify the code
            # parses and compiles. Running curses/pygame/flask in the sandbox
            # would fail for environmental reasons (no TTY, no display) even
            # when the code is correct — see PC-022.
            if task_type == "interactive":
                # PC-048: pass the detected language so HTML/JSON/etc. files
                # don't get parsed as Python (which produces spurious
                # SYNTAX_ERROR cascades into PR-CoT repair + LLM timeouts).
                ok, out, err = scoring.smoke_compile_check(code, sandbox, language=smoke_language)
                emit("smoke_check", f"compile={'OK' if ok else 'FAIL'} ({smoke_language})")
                if not ok:
                    return ok, out, err, verification_evidence
                # Interactive lint is Python-AST based — only meaningful for
                # Python files. Skip for HTML/CSS/JSON/etc.
                if smoke_language not in ("python", "py"):
                    return True, out, err, verification_evidence
                # Interactive lint: catch raw stdin reads / blocking input loops
                # that compile fine but don't actually work for keystroke
                # handling (PC-034).
                lint_ok, lint_reason = scoring.interactive_lint(code)
                if lint_ok:
                    emit("interactive_lint", "OK")
                    return verify_build_if_requested(out, err)
                emit("interactive_lint", f"FAIL: {lint_reason}")
                return False, out, f"interactive_lint: {lint_reason}", verification_evidence

            ok, out, err = sandbox(code)
            if not ok:
                return False, out, err, verification_evidence
            if self_tests and self_tests.test_cases:
                p, fails = 0, []
                for i, tc in enumerate(self_tests.test_cases):
                    try:
                        tc_code = _make_self_test(code, tc)
                        tp, to, te = sandbox(tc_code)
                        if tp and "SELF_TEST_PASS" in to:
                            p += 1
                        else:
                            fails.append(f"TC{i+1}:{te[:60] if te else 'wrong'}")
                    except Exception as ex:
                        fails.append(f"TC{i+1}:{str(ex)[:40]}")
                total = len(self_tests.test_cases)
                emit("self_test_verify", f"{p}/{total} passed")
                # A suite the candidate passes NO case of says nothing about
                # the candidate. The cases come from the same model that
                # writes the code, and for these tasks producing an expected
                # output IS solving the problem — so a wrong answer key and
                # wrong code are indistinguishable at zero.
                #
                # Measured: 0 of 44 candidates across a 28-session run, with
                # scores of 0/5, 0/4, 0/3 and never a partial. A check that
                # rejects every candidate it sees is not evidence, and it
                # cost the pipeline every winner it might have had — every
                # passing session was won by the model's own direct write.
                #
                # A partial score is different: the suite has demonstrated
                # some case is passable, so falling below half is the
                # candidate underperforming a bar something else cleared.
                if total > 0 and p == 0:
                    # Inconclusive is not a pass. An earlier version emitted
                    # this and fell through to the build check, which made
                    # "it compiles" the whole of verification: a candidate
                    # that failed every one of its own cases was marked
                    # passed, joined the selection pool, and could be written
                    # over the model's work. Measured on the two tasks whose
                    # answer is computed from a file the program has to read:
                    # 0/4 sessions correct on the run that shipped it,
                    # against 1-2 of 4 on each of the four runs before.
                    #
                    # "Cannot condemn" and "therefore promote" are different
                    # claims. A suite that passes nothing still establishes
                    # nothing, so the candidate stays unverified and the
                    # caller's own write stands — the rule the fallback path
                    # below already states, that executing but wrong is worse
                    # than an honest failure.
                    emit("self_test_inconclusive",
                         f"0/{total} — no case passed, so the suite cannot "
                         f"separate a wrong answer key from wrong code. Not "
                         f"verified; leaving the caller's own write in place",
                         cases=total)
                    return (False, out,
                            f"Self-test:0/{total} inconclusive — nothing verified",
                            verification_evidence)
                elif total > 0 and p < total / 2:
                    return False, out, f"Self-test:{p}/{total}. "+";".join(fails[:3]), verification_evidence
            return verify_build_if_requested(out, err)

        # Score and test probe with self-generated tests. The probe is the
        # only candidate the CxGx gate below can see, so it is scored with
        # the combined C(x)+G(x) call — one embedding extraction, both
        # models — rather than C(x) alone.
        probe_scores = dict(scoring.NEUTRAL_COMBINED)
        probe_energy_raw, probe_energy_norm = 0.0, 0.5
        probe_cx_calibrated = False
        probe_passed = False
        if probe_code:
            probe_scores = scoring.score_candidate_combined(probe_code)
            probe_energy_raw = probe_scores["cx_energy"]
            probe_energy_norm = probe_scores["cx_normalized"]
            probe_cx_calibrated = probe_scores["cx_calibrated"]
            norm_label = f"{probe_energy_norm:.2f}" if probe_cx_calibrated else "uncalibrated"
            emit("probe_scored",
                 f"C(x)={probe_energy_raw:.2f} norm={norm_label} "
                 f"G(x)={probe_scores['gx_score']:.2f} "
                 f"({probe_scores['verdict']})",
                 gx_score=probe_scores["gx_score"],
                 gx_available=probe_scores["gx_available"],
                 verdict=probe_scores["verdict"])
            probe_passed, probe_stdout, probe_stderr, probe_evidence = verified_sandbox(probe_code)
            emit("probe_sandbox", f"passed={probe_passed} stderr={probe_stderr[:80] if probe_stderr else ''}")
            result["total_tokens"] += tokens

        if probe_passed:
            emit("probe_pass", "Probe passed — returning early")
            result["passed"] = True
            result["code"] = probe_code
            result["phase_solved"] = "probe"
            result["candidates_generated"] = 1
            result["total_time_ms"] = (time.time() - start) * 1000
            result["verification_evidence"] = probe_evidence
            result["winning_score"] = probe_energy_norm
            result["events"] = events
            return result

        # ===== PHASE 2: CxGx K ALLOCATION =====
        # The probe failed verification, so this task is not trivial: C(x)
        # picks a base tier, G(x) escalates it, and k never drops below the
        # gate's k=3 floor (what this phase allocated unconditionally
        # before the gate existed).
        #
        # Live-path difference from the bench the gate was measured on: the
        # proxy's V3 bridge abandons this call after ATLAS_V3_TIMEOUT
        # (default 180s), a cap the bench never had. An unbounded escalation
        # to k=8 here would spend the whole budget on generation and hand
        # the user a timeout fallback instead of the k=3 answer the clock
        # could have produced — the failure mode the phase-3 refinement gate
        # already fixes. So the remaining wall-clock and the per-call
        # latency observed on THIS task go into the allocation, and the gate
        # lowers the tier to what the budget can actually generate. The
        # floor is not budget-dependent: k=3 is what would have run anyway.
        check_client()
        emit("phase2", "Allocating compute budget...")
        alloc = cxgx_gate.allocate(
            cx_normalized=probe_energy_norm,
            cx_calibrated=probe_cx_calibrated,
            gx_score=probe_scores["gx_score"],
            gx_available=probe_scores["gx_available"],
            gx_verdict=probe_scores["verdict"],
            remaining_ms=_remaining_budget_ms(start),
            observed_llm_call_ms=getattr(llm, "avg_call_ms", 0.0),
        )
        k, budget_tier = alloc.k, alloc.tier
        bf_tier = budget_tier
        emit("phase2_allocated", f"k={k} tier={budget_tier}",
             k=k, tier=budget_tier, base_tier=alloc.base_tier,
             gx_escalation=alloc.gx_escalation,
             capped_from=alloc.capped_from, reason=alloc.reason)

        # ===== PHASE 1: CONSTRAINT-DIVERSE CANDIDATE GENERATION =====
        emit("phase1", f"Generating {k} diverse candidates...", k=k)
        candidates = []

        def out_of_budget(reserve_ms: Optional[float] = None) -> bool:
            """True when too little of ATLAS_V3_TIMEOUT is left to start more
            work and still hand back a result.

            The reserve defaults to the cost of one more LLM call as observed
            on this run, plus room to serialize the result. A flat reserve is
            useless here: measured 2026-08-03, a phase-3 entry check with 20s
            left passed, PR-CoT then spent 31s on one call and the cap landed
            mid-way through the next. The unit of work is a generation, so
            that is what has to fit.

            The refinement loop was the only phase that checked at all, and it
            checked once. Everything ahead of it — probe, self-tests,
            PlanSearch, sandbox, PR-CoT — ran unguarded.
            """
            left = _remaining_budget_ms(start)
            if left is None:
                return False
            if reserve_ms is None:
                observed = getattr(llm, "avg_call_ms", 0.0) or 0.0
                reserve_ms = max(20000.0, observed * 1.2 + 10000.0)
            return left < reserve_ms

        def finish_with_best(reason: str) -> Dict[str, Any]:
            """Hand back the best candidate found so far.

            V3 is an anytime algorithm — the refinement loop is budget-gated,
            so it expands to fill the budget rather than converging early.
            Measured 2026-08-03 on one task: a 180s budget produced 9 LLM
            calls and 5 sandbox verifications, a 420s budget 21 calls and 10.
            It is not interrupted mid-something-it-would-finish; the clock is
            its terminal condition.

            An anytime algorithm whose clock expires has to return its best
            answer. This one returned nothing: the caller's deadline cancelled
            the request, every verified candidate was discarded, and the write
            fell back to the model's own output — 16 of 41 write calls in one
            28-session run, ~48 of the 56 minutes spent in them.

            Vetoed candidates stay excluded for the same reason the end-of-run
            fallback excludes them: "executes but is wrong" is worse than an
            honest failure.
            """
            emit("budget_exhausted", reason,
                 candidates=len(candidates),
                 remaining_ms=round(_remaining_budget_ms(start) or 0))
            pool = [c for c in candidates if not c.get("vetoed_by")]
            passing = [c for c in pool if c.get("passed")]
            chosen = None
            if passing:
                passing.sort(key=lambda c: c.get("energy", 999))
                chosen = passing[0]
                result["passed"] = True
                result["phase_solved"] = "budget"
            else:
                # No code rather than an unverified candidate. The caller's
                # baseline is the model's own write, which is syntax- and
                # structure-gated; a candidate that failed the sandbox is
                # not better than that, and ranking the failures by energy
                # picks among them without evidence.
                #
                # Measured across a 28-session run: 7 of 8 returns took the
                # unverified path and task success fell 20/28 to 17/28 —
                # aoc_course and aoc_slope went 2/2 to 0/2 — while harness
                # integrity reached 28/28. The same reasoning the vetoed
                # path already uses: "executes but is wrong" is worse than
                # an honest failure.
                emit("budget_no_verified_candidate",
                     f"{len(pool)} candidate(s), none verified — "
                     f"leaving the caller's gated baseline in place")
            if chosen is not None:
                result["code"] = chosen["code"]
                result["verification_evidence"] = chosen.get("verification_evidence", [])
                result["winning_score"] = chosen.get("energy_norm", 0.0)
            result["total_time_ms"] = (time.time() - start) * 1000
            result["events"] = events
            return result
        try:

            # Start with probe if it produced code
            if probe_code:
                candidates.append({
                    "index": 0, "code": probe_code,
                    "energy": probe_energy_raw, "energy_norm": probe_energy_norm,
                    "energy_calibrated": probe_cx_calibrated,
                    "passed": probe_passed, "stdout": "", "stderr": "",
                })

            remaining_k = max(0, k - len(candidates))

            # Step 1A: PlanSearch
            if remaining_k > 0:
                emit("plansearch", f"Generating {remaining_k} plans...",
                     plans=remaining_k)
                try:
                    ps_result = self.plan_search.generate(
                        problem, task_id, llm, num_plans=remaining_k,
                    )
                    for i, code in enumerate(ps_result.candidates):
                        if code:
                            energy_raw, energy_norm, energy_calibrated = scoring.score_candidate(code)
                            per_step = scoring.score_candidate_per_step(code)  # PC-207
                            cand_index = len(candidates)
                            candidates.append({
                                "index": cand_index, "code": code,
                                "energy": energy_raw, "energy_norm": energy_norm,
                                "energy_calibrated": energy_calibrated,
                                "passed": False, "stdout": "", "stderr": "",
                                "per_step": per_step,
                            })
                            if per_step:
                                emit("lens_per_step",
                                     f"cand {cand_index}: gx_min={per_step['gx_score_min']:.2f} "
                                     f"first_off_rails={per_step['first_off_rails_idx']}",
                                     index=cand_index,
                                     source="plansearch",
                                     first_off_rails_idx=per_step["first_off_rails_idx"],
                                     gx_score_min=per_step["gx_score_min"],
                                     gx_score_mean=per_step["gx_score_mean"],
                                     cx_norm_max=per_step["cx_norm_max"],
                                     n_tokens=per_step["n_tokens"])
                    result["total_tokens"] += ps_result.total_tokens
                    emit("plansearch_done",
                         f"{len(ps_result.candidates)} candidates from PlanSearch",
                         candidates=len(ps_result.candidates),
                         tokens=ps_result.total_tokens)
                except Exception as e:
                    emit("plansearch_error", str(e)[:200])

            # Step 1B: DivSampling to fill remaining slots
            remaining_k = max(0, k - len(candidates))
            if remaining_k > 0:
                emit("divsampling", f"Filling {remaining_k} slots with diverse sampling...",
                     slots=remaining_k)
                for idx in range(remaining_k):
                    check_client()
                    if out_of_budget():
                        emit("divsampling_stop",
                             f"budget spent after {len(candidates)} candidate(s)")
                        break
                    try:
                        perturbed = self.div_sampling.apply(problem, len(candidates) + idx, task_id)
                        chatml = self.budget_forcing.format_chatml(perturbed, bf_tier)
                        response, tokens, t_ms = llm(
                            chatml, DIVERSITY_TEMPERATURE,
                            self.budget_forcing.get_max_tokens(bf_tier),
                            42 + len(candidates) + idx,
                        )
                        code = extract_code(response)
                        if code:
                            energy_raw, energy_norm, energy_calibrated = scoring.score_candidate(code)
                            per_step = scoring.score_candidate_per_step(code)  # PC-207
                            cand_index = len(candidates)
                            candidates.append({
                                "index": cand_index, "code": code,
                                "energy": energy_raw, "energy_norm": energy_norm,
                                "energy_calibrated": energy_calibrated,
                                "passed": False, "stdout": "", "stderr": "",
                                "per_step": per_step,
                            })
                            if per_step:
                                emit("lens_per_step",
                                     f"cand {cand_index}: gx_min={per_step['gx_score_min']:.2f} "
                                     f"first_off_rails={per_step['first_off_rails_idx']}",
                                     index=cand_index,
                                     source="divsampling",
                                     first_off_rails_idx=per_step["first_off_rails_idx"],
                                     gx_score_min=per_step["gx_score_min"],
                                     gx_score_mean=per_step["gx_score_mean"],
                                     cx_norm_max=per_step["cx_norm_max"],
                                     n_tokens=per_step["n_tokens"])
                        result["total_tokens"] += tokens
                    except Exception as e:
                        emit("divsampling_error", str(e)[:200])
                emit("divsampling_done", f"{len(candidates)} total candidates",
                     total=len(candidates))

            result["candidates_generated"] = len(candidates)

            # ===== SANDBOX TESTING =====
            emit("sandbox_test", f"Testing {len(candidates)} candidates...",
                 candidates=len(candidates))
            # Sort by energy (easy first) for early-exit potential
            candidates.sort(key=lambda c: c.get("energy", 0))

            passing = []
            for c in candidates:
                check_client()
                if c.get("passed"):
                    passing.append(c)
                    continue
                sb_start = time.time()
                passed, stdout, stderr, verification_evidence = verified_sandbox(c["code"])
                sb_ms = int((time.time() - sb_start) * 1000)
                c["passed"] = passed
                c["stdout"] = stdout
                c["stderr"] = stderr
                c["verification_evidence"] = verification_evidence
                if passed:
                    passing.append(c)
                    emit("sandbox_pass", f"Candidate {c['index']} passed",
                         index=c["index"], elapsed_ms=sb_ms,
                         energy=c.get("energy_norm", 0.0))
                else:
                    emit("sandbox_fail", f"Candidate {c['index']} failed",
                         index=c["index"], elapsed_ms=sb_ms,
                         stderr=(stderr or "")[:120])

            emit("sandbox_done", f"{len(passing)}/{len(candidates)} passed",
                 passed=len(passing), total=len(candidates))

            # Nothing passed, which for these tasks usually means the answer
            # key was wrong rather than every candidate. Measured across 42
            # verifications in one run: all 42 scored 0/N, and a candidate
            # pulled from those logs passed immediately against a correct
            # expected value. The pipeline has never selected a candidate in
            # any measured run — every session shipped the model's own draft.
            #
            # Fall back to the agreement signal CodeT actually uses: run the
            # candidates on the generated INPUTS and take the largest cluster
            # that produced the same answers. Narrow on purpose — this only
            # runs where the oracle has already condemned everything, so a
            # working suite keeps deciding.
            if not passing and self_tests and self_tests.test_cases:
                agreed = _consensus_winners(
                    candidates, self_tests.test_cases, sandbox, emit)
                for c in agreed:
                    c["passed"] = True
                    passing.append(c)
                if agreed:
                    emit("sandbox_done",
                         f"{len(passing)}/{len(candidates)} by consensus",
                         passed=len(passing), total=len(candidates))

            # ===== LENS VETO =====
            # PC-207 alignment fix: hard-reject sandbox-passing candidates whose
            # geometric-lens gx_min sits below THIS model's calibrated severe band.
            # Sandbox is an ORM (does it execute?), lens is a PRM (is the
            # generation pattern collapsing into a stub?) — they answer
            # different questions. The May 7 dashboard.html session shipped
            # a 10-line `<h1>Dashboard</h1>` stub because sandbox said pass
            # while lens said gx_min=0.069. Without this filter, V3 returns
            # passed=True and the proxy's PC-044 nudges the agent to done.
            #
            # Language-agnostic by construction: the lens runs on the model's
            # residual stream; gx values don't depend on whether the file
            # being scored is HTML, Python, Rust, or Java.
            if passing:
                kept, vetoed = [], []
                for c in passing:
                    per_step = c.get("per_step") or {}
                    # Vetoes read the MEAN, not the MIN.
                    #
                    # gx_score_min is a minimum over every token, so it falls
                    # with length whatever the content: measured 2026-08-04 on
                    # one function repeated, 0.468 at 20 tokens down to 0.305
                    # at 305. It cannot separate — real code 0.325, a
                    # repetition loop 0.320, stub spam 0.286 all sit together,
                    # which is why severe=0.28 never fired once in 56 sessions.
                    #
                    # gx_score_mean holds across the same 15x length change
                    # (0.577 to 0.517) and does separate: real code 0.594,
                    # repetition 0.485, stub spam 0.467. Across 188 scores
                    # from live runs it ranged 0.547-0.651, so severe_mean at
                    # 0.52 sits below every clean sample observed and above
                    # both pathologies.
                    gx_mean = per_step.get("gx_score_mean")
                    gx_min = per_step.get("gx_score_min")
                    thresholds = per_step.get("thresholds") or {}
                    severe_mean = thresholds.get("severe_mean")
                    severe = thresholds.get("severe")
                    if (gx_mean is not None
                            and isinstance(severe_mean, (int, float))):
                        vetoed_now = gx_mean < severe_mean
                        gx_min, severe = gx_mean, severe_mean
                    else:
                        # Artifacts predating severe_mean keep the old check.
                        vetoed_now = (gx_min is not None
                                      and isinstance(severe, (int, float))
                                      and gx_min < severe)
                    if vetoed_now:
                        # A vetoed candidate is a failing candidate: mark it so
                        # the phase-3 pool (`not c.get("passed")`) picks it up
                        # and the energy fallback can never return it. The veto
                        # reason replaces the (empty) passing-run stderr so
                        # repair sees WHY it was rejected.
                        c["passed"] = False
                        c["vetoed_by"] = "lens"
                        c["stderr"] = (
                            f"lens veto: gx_min={gx_min:.3f} below the severe "
                            f"threshold {severe:.3f} — generation pattern "
                            f"collapsed toward a stub; the code executes but "
                            f"likely does not implement the task")
                        vetoed.append(c)
                        emit("lens_veto",
                             f"Candidate {c['index']} sandbox-passed but lens-vetoed "
                             f"(gx_min={gx_min:.3f} < {severe:.3f}) — likely a stub",
                             index=c["index"], gx_score_min=gx_min,
                             first_off_rails_idx=per_step.get("first_off_rails_idx", -1))
                    else:
                        kept.append(c)
                if vetoed:
                    print(
                        f"  [lens] vetoed {len(vetoed)}/{len(passing)} sandbox-passing "
                        f"candidates using per-model severe thresholds — falling "
                        f"{'through to phase-3 repair' if not kept else 'back to remaining %d' % len(kept)}",
                        flush=True,
                    )
                passing = kept

            # ===== STRUCTURAL VETO =====
            # GH #39 point 1: hard-reject candidates whose direct-identifier
            # calls don't resolve against (local defs, imports, builtins,
            # project symbols). Sandbox can pass for code where the unresolved
            # call is in a try/except ImportError fallback or a dead branch
            # that doesn't execute under the tests; tree-sitter sees the
            # surface bug regardless. Same architecture as lens veto.
            #
            # Language-agnostic fit: v1 supports Python only (matches the
            # rest of the GH #39 stack), but the resolution-order pattern
            # generalizes to any language with explicit imports + named
            # functions (Go, Rust, JS/TS modules). Adding a language adds
            # implementation surface, not model-facing API surface.
            if passing:
                # #147: gate on `passing` alone, not `passing and files`. The
                # edit path (improveContentWithV3) frequently sends no
                # project_context, so `files` was empty and the whole veto was
                # skipped — a NameError edit (render_template called with only
                # render_template_string imported) sailed through and landed as
                # verified. structural_score resolves against the candidate's
                # OWN imports/defs/builtins, so it catches an unresolved direct
                # call with empty project_symbols; project symbols only add
                # lenient cross-file crediting.
                project_symbols = symbols.build_project_symbols(files or {})
                kept = []
                for c in passing:
                    struct = symbols.structural_score(project_symbols, c.get("code", ""))
                    if struct.get("ok") and struct.get("n_unresolved", 0) >= 1:
                        # Same contract as the lens veto: vetoed = failing.
                        c["passed"] = False
                        c["vetoed_by"] = "structural"
                        c["stderr"] = (
                            "structural veto: unresolved direct call(s) that "
                            "would raise NameError at runtime: "
                            + ", ".join(struct["unresolved_calls"][:5]))
                        emit("structural_veto",
                             f"Candidate {c['index']} sandbox-passed but "
                             f"{struct['n_unresolved']} unresolved call(s): "
                             f"{', '.join(struct['unresolved_calls'][:3])}",
                             index=c["index"],
                             n_unresolved=struct["n_unresolved"],
                             unresolved_calls=struct["unresolved_calls"][:5],
                             n_calls_total=struct["n_calls_total"])
                        print(
                            f"  [structural] vetoed cand {c['index']} — "
                            f"{struct['n_unresolved']} unresolved: {struct['unresolved_calls'][:5]}",
                            flush=True,
                        )
                        continue
                    if struct.get("ok"):
                        c["structural"] = struct  # stash for phase 3 / repair
                    kept.append(c)
                if len(kept) < len(passing):
                    print(
                        f"  [structural] kept {len(kept)}/{len(passing)} candidates after structural veto"
                        f"{' — falling through to phase-3 repair' if not kept else ''}",
                        flush=True,
                    )
                passing = kept

            # ===== CALL-GRAPH VETO (issue #39, Phase 1) =====
            # Deepens the structural veto using the import graph: reject a candidate
            # whose direct calls don't resolve to a real, in-scope definition (local,
            # builtin, imported, or supplied by a resolved wildcard) — not merely
            # "some project file defines that name." Catches broken cross-file
            # references the shipped veto accepts. Flag-gated by ATLAS_CALL_GRAPH;
            # conservative — stays lenient on opaque wildcards and never empties the
            # candidate set (a fully-failing set falls through intact to repair).
            if passing and files and file_path:
                try:
                    from graph import call_graph_enabled, unresolved_calls
                    _cg_on = call_graph_enabled()
                except Exception:
                    _cg_on = False
                if _cg_on:
                    cg_kept, cg_vetoed = [], []
                    for c in passing:
                        try:
                            res = unresolved_calls(
                                file_path, c.get("code", ""), files, strict=True)
                        except Exception as cge:
                            print(f"  [call_graph] veto skipped for cand {c.get('index')}: {cge}",
                                  flush=True)
                            cg_kept.append(c)
                            continue
                        if res.get("ok") and res.get("unresolved"):
                            cg_vetoed.append((c, res["unresolved"]))
                            continue
                        cg_kept.append(c)
                    if cg_kept:  # only prune when at least one candidate survives
                        # Marking happens only when the prune actually applies —
                        # the conservative all-vetoed case keeps the full set
                        # (and its passed flags) intact.
                        for c, unresolved in cg_vetoed:
                            c["passed"] = False
                            c["vetoed_by"] = "call_graph"
                            c["stderr"] = (
                                "call-graph veto: cross-file call(s) that resolve "
                                "to no in-scope definition: "
                                + ", ".join(unresolved[:5]))
                            emit("call_graph_veto",
                                 f"Candidate {c.get('index')} has unresolved call(s): "
                                 f"{', '.join(unresolved[:3])}",
                                 index=c.get("index"), unresolved=unresolved[:5])
                            print(f"  [call_graph] vetoed cand {c.get('index')} — "
                                  f"unresolved: {unresolved[:5]}", flush=True)
                        passing = cg_kept

            # ===== CANDIDATE SELECTION =====
            # Lens selection: minimum C(x) energy among the passing candidates.
            # (S* tiebreaking used to run first for 2+ passers; across 118 H200
            # tiebreaks every pair scored 0-0 and 110/110 winners equaled the
            # lens min-energy pick, so it carried zero discriminating signal.)
            if passing:
                ci_list = [
                    CandidateInfo(c["index"], c["code"], c["energy"], c["passed"])
                    for c in passing
                ]
                selected = select_candidate(ci_list, strategy="lens")
                if selected:
                    emit("selected", f"Lens selected candidate {selected.index}",
                         index=selected.index, energy=getattr(selected, "energy", 0.0))
                    result["passed"] = True
                    result["code"] = selected.code
                    result["phase_solved"] = "phase1"
                    result["total_time_ms"] = (time.time() - start) * 1000
                    winner = _candidate_by_index(passing, selected.index)
                    result["verification_evidence"] = (winner or {}).get("verification_evidence", [])
                    result["winning_score"] = (winner or {}).get("energy_norm", 0.0)
                    result["events"] = events
                    return result

            # ===== PHASE 3: VERIFIED ITERATIVE REFINEMENT =====
            check_client()
            if out_of_budget():
                return finish_with_best("budget spent before the repair phase")
            emit("phase3", "All candidates failed — entering repair phase...",
                 failing=len([c for c in candidates if not c.get("passed")]))

            failing = [
                FailingCandidate(
                    index=c["index"], code=c["code"],
                    error_output=c.get("stderr", ""),
                )
                for c in candidates if not c.get("passed")
            ]

            # Repair verifies against the SAME self-tests phase 0 generated —
            # verified_sandbox closes over them. Regenerate only when phase 0
            # produced none (e.g. a transient LLM failure); a failed retry here
            # must not downgrade an existing good set to None. Interactive
            # tasks repair against compile-smoke (PC-022).
            if task_type == "algorithmic" and not (self_tests and self_tests.test_cases):
                emit("self_test_gen", "Generating self-tests...")
                try:
                    self_tests = self.self_test_gen.generate(problem, llm, task_id)
                    emit("self_test_done", f"{len(self_tests.test_cases)} test cases generated")
                    result["total_tokens"] += self_tests.generation_tokens
                except Exception as e:
                    emit("self_test_error", str(e)[:200])

            # GH #39 point 3: build call-graph context for the failing
            # function once, reuse across PR-CoT + refinement. Skips
            # cleanly when stderr isn't a Python traceback or the failing
            # function isn't defined in the project — both arms get plain
            # error_output in that case. When ATLAS_CALL_GRAPH is on the
            # block is a multi-hop reachability slice (entry-point path,
            # transitive impact, callees); flag-off it stays at direct
            # callers/callees (1 hop). Fail-soft on any graph failure.
            chain_context_block = ""
            if failing:
                failing_func = symbols._failing_function_from_stderr(failing[0].error_output)
                if failing_func and files:
                    try:
                        from graph import call_graph_enabled as _cg_on, repair_context as _cg_repair
                        chain_context_block = _cg_repair(files, failing_func, transitive=_cg_on())
                    except Exception as cge:
                        print(f"  [phase3] graph repair-context skipped: {cge}", flush=True)
                    if chain_context_block:
                        emit("call_chain_context",
                             f"Built call-chain for failing `{failing_func}`",
                             function=failing_func)
                        print(
                            f"  [phase3] call-chain context built for `{failing_func}`",
                            flush=True,
                        )

            def _enriched_error(stderr: str) -> str:
                """Append call-chain context to a candidate's stderr if available."""
                if not chain_context_block:
                    return stderr
                return (stderr or "") + "\n\n" + chain_context_block

            # Strategy 1: PR-CoT Quick Repair
            if failing and out_of_budget():
                return finish_with_best("budget spent before PR-CoT repair")
            if failing:
                emit("pr_cot", "Attempting PR-CoT repair...",
                     strategy="pr_cot", failing=len(failing))
                best_failing = failing[0]
                try:
                    pr_result = self.pr_cot.repair(
                        problem=problem,
                        code=best_failing.code,
                        error=_enriched_error(best_failing.error_output),
                        llm_call=llm,
                        task_id=task_id,
                    )
                    result["total_tokens"] += pr_result.total_tokens
                    for repair_code in pr_result.repairs:
                        passed, stdout, stderr, repair_evidence = verified_sandbox(repair_code)
                        if passed:
                            emit("pr_cot_pass", "PR-CoT repair succeeded!",
                                 strategy="pr_cot", tokens=pr_result.total_tokens)
                            result["passed"] = True
                            result["code"] = repair_code
                            result["phase_solved"] = "pr_cot"
                            result["total_time_ms"] = (time.time() - start) * 1000
                            result["verification_evidence"] = repair_evidence
                            result["events"] = events
                            return result
                    emit("pr_cot_failed", "PR-CoT repair did not produce passing code")
                except Exception as e:
                    emit("pr_cot_error", str(e)[:200])

            # Strategy 2: Refinement Loop — entered only when the remaining
            # wall-clock can afford one iteration. H200 join: 453/487
            # refinement entries timed out with ZERO completed iterations
            # while burning ~6 minutes each; one iteration is ~3 sequential
            # LLM calls, estimated at the per-call latency observed on THIS
            # run. The budget is the ATLAS_V3_TIMEOUT cap the proxy's V3
            # bridge enforces — starting work the bridge will abandon only
            # delays the fallback the user ends up with.
            run_refinement = bool(failing)
            if run_refinement:
                est_ms = estimate_iteration_ms(getattr(llm, "avg_call_ms", 0.0))
                remaining_ms = _remaining_budget_ms(start)
                if (remaining_ms is not None
                        and not can_afford_iteration(remaining_ms, est_ms)):
                    run_refinement = False
                    emit("refinement_skip",
                         f"remaining budget {remaining_ms / 1000:.0f}s cannot "
                         f"afford one iteration (~{est_ms / 1000:.0f}s) — "
                         f"skipping to fallback",
                         strategy="refinement",
                         remaining_ms=round(remaining_ms),
                         estimated_iteration_ms=round(est_ms))
            if run_refinement and out_of_budget():
                run_refinement = False
            if run_refinement:
                check_client()
                emit("refinement", "Starting refinement loop...",
                     strategy="refinement", failing=len(failing))
                # GH #39 point 3: enrich each failing candidate's error_output
                # with call-chain context so the refinement loop sees it on
                # every iteration. Cheap (chain_context_block is built once
                # above and reused).
                failing_for_refinement = failing
                if chain_context_block:
                    failing_for_refinement = [
                        FailingCandidate(
                            index=c.index,
                            code=c.code,
                            error_output=_enriched_error(c.error_output),
                        )
                        for c in failing
                    ]
                try:
                    ref_result = self.refinement_loop.run(
                        problem=problem,
                        failing_candidates=failing_for_refinement,
                        original_constraints=[],
                        llm_call=llm,
                        sandbox_run=sandbox,
                        embed_call=embed,
                        task_id=task_id,
                    )
                    result["total_tokens"] += ref_result.total_tokens
                    if ref_result.solved:
                        passed, stdout, stderr, refinement_evidence = verified_sandbox(ref_result.winning_code)
                        if passed:
                            emit("refinement_pass",
                                 f"Refinement solved in {ref_result.total_iterations} iterations!",
                                 strategy="refinement",
                                 iterations=ref_result.total_iterations,
                                 tokens=ref_result.total_tokens)
                            result["passed"] = True
                            result["code"] = ref_result.winning_code
                            result["phase_solved"] = "refinement"
                            result["total_time_ms"] = (time.time() - start) * 1000
                            result["verification_evidence"] = refinement_evidence
                            result["events"] = events
                            return result
                        emit("refinement_verify_failed", (stderr or "")[:200])
                    emit("refinement_failed", f"Exhausted {ref_result.total_iterations} iterations")
                except Exception as e:
                    emit("refinement_error", str(e)[:200])

            # ===== FALLBACK: Return best candidate even if none passed =====
            # Vetoed candidates are excluded outright: a veto means "executes
            # but is wrong" (stub, NameError-in-waiting), which is worse than
            # an honest sandbox failure — and returning one is exactly the
            # May 7 dashboard-stub failure mode. If every candidate was
            # vetoed, return no code; the caller falls back to its baseline.
            # Nothing verified, so nothing is returned. The caller's
            # baseline is the model's own write, which is syntax- and
            # structure-gated; a candidate that failed the sandbox is not
            # better than that, and ranking failures by energy picks among
            # them without evidence.
            #
            # Measured: across one 28-session run, 0 of 44 candidates passed
            # the sandbox, and this path still handed back a failing one 11
            # times — the proxy logged each as a V3 write and put it on disk
            # over the model's own. The run that shipped that behaviour
            # scored 20/28 to 17/28 against the run that did not.
            #
            # Same reasoning the vetoed branch below already used, and the
            # same as the budget boundary: "executes but is wrong" is worse
            # than an honest failure.
            unverified = [c for c in candidates if not c.get("vetoed_by")]
            if unverified:
                emit("fallback_unverified",
                     f"{len(unverified)} candidate(s), none passed verification — "
                     f"leaving the caller's gated baseline in place")
            elif candidates:
                emit("fallback_all_vetoed",
                     "Every candidate was vetoed — returning no code")
            result["total_time_ms"] = (time.time() - start) * 1000
            result["events"] = events
            return result
        except adapters.BudgetExhausted as exc:
            # An anytime algorithm whose clock expires owes its caller the
            # best answer it has. Raised from the adapter rather than
            # checked at phase boundaries: every phase runs its own loop —
            # PR-CoT alone issues two calls — so a boundary check that
            # reserves one call is already wrong by the second.
            return finish_with_best(f"budget exhausted mid-pipeline ({exc})")


# --- Problem Builder for /v3/generate ----------------------------------------

def _build_problem_from_request(
    file_path: str, baseline_code: str, project_context: Dict[str, str],
    framework: str, build_command: str, constraints: List[str],
    user_message: str = "",
) -> str:
    """Build a problem description for the V3 pipeline from a generate request.

    The user's own request leads, when the caller sends one. Without it the
    pipeline saw only "Create the file X", the project context and the
    baseline, under an instruction to improve on the baseline "preserving all
    functionality" — so every candidate could only mimic a draft whose
    requirement it had never been shown, and a baseline that misread the task
    was reproduced rather than corrected.

    Measured on the AoC tasks, whose prompt states "reads input.txt": 9 of the
    12 solutions ATLAS produced read stdin instead, and the caller runs
    `python solve.py` with no stdin. The same model given the task directly
    wrote file readers 12 times out of 12.
    """
    parts = []

    if user_message.strip():
        parts.append("## The request\n\n")
        parts.append(user_message.strip() + "\n\n")

    parts.append(f"Create the file `{file_path}`")
    if framework:
        parts.append(f" for a {framework} project")
    parts.append(".\n\n")

    # Project context
    if project_context:
        parts.append("## Existing project files:\n\n")
        for path, content in project_context.items():
            if len(content) < 500:
                parts.append(f"### {path}\n```\n{content}\n```\n\n")
            else:
                parts.append(f"### {path} (truncated)\n```\n{content[:300]}\n...\n```\n\n")

    # Constraints
    if constraints:
        parts.append("## Requirements:\n")
        for c in constraints:
            parts.append(f"- {c}\n")
        parts.append("\n")

    # Build command
    if build_command:
        parts.append(f"## Build verification:\nThe file must pass: `{build_command}`\n\n")

    # Baseline as reference
    if baseline_code:
        parts.append("## Reference implementation:\n")
        parts.append("Improve upon this baseline if possible, preserving all functionality.\n\n")
        parts.append(f"```\n{baseline_code}\n```\n")

    return "".join(parts)
