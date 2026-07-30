"""The V3 pipeline orchestrator: probe, candidate generation, sandbox
verification, the lens/structural/call-graph vetoes, candidate selection,
the repair phases, and the /v3/generate problem builder."""

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runner import extract_code
from benchmark.v3.budget_forcing import BudgetForcing, BudgetForcingConfig
from benchmark.v3.plan_search import PlanSearch, PlanSearchConfig
from benchmark.v3.div_sampling import DivSampling, DivSamplingConfig
from benchmark.v3.blend_asc import BlendASC, BlendASCConfig
from benchmark.v3.s_star import SStar, SStarConfig, CandidateScore
from benchmark.v3.failure_analysis import FailingCandidate
from benchmark.v3.pr_cot import PRCoT, PRCoTConfig
from benchmark.v3.refinement_loop import RefinementLoop, RefinementLoopConfig
from benchmark.v3.derivation_chains import DerivationChains, DerivationChainsConfig
from benchmark.v3.metacognitive import MetacognitiveProfile, MetacognitiveConfig
from benchmark.v3.self_test_gen import SelfTestGen, SelfTestGenConfig
from benchmark.v3.candidate_selection import CandidateInfo, select_candidate

import adapters
import scoring
import symbols

BASE_TEMPERATURE = 0.6
DIVERSITY_TEMPERATURE = 0.8
MAX_TOKENS = 8192


# --- V3 Pipeline Orchestrator ------------------------------------------------

def _candidate_by_index(candidates: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """Return the candidate dict whose original ``index`` field matches.

    Selection modules (S*, lens) report winners by the candidate's original
    index, but the ``passing`` list has been sorted and filtered — positional
    indexing would pick the wrong candidate (or IndexError). Returns None
    when no candidate carries that index.
    """
    return next((c for c in candidates if c.get("index") == index), None)


def _make_self_test(code: str, tc) -> str:
    """Build executable assertion code for a single test case.

    Uses ast.literal_eval (safe — only parses Python literals) to convert
    I/O string representations to actual values for comparison.
    All code runs inside the sandboxed container.
    """
    inp = tc.input_str.strip()
    exp = tc.expected_output.strip()
    fn = re.search(r'^def (\w+)\(', code, re.MULTILINE)
    if fn and 'input()' not in code:
        name = fn.group(1)
        return (code + "\nimport ast as _a\n"
            + f"_i={repr(inp)}\n_e={repr(exp)}\n"
            + "try:\n _p=_a.literal_eval(_i)\nexcept:\n _p=_i\n"  # noqa: E722  -- bare except inside generated user code, intentional
            + f"_r={name}(*_p) if isinstance(_p,tuple) else {name}(_p) if isinstance(_p,list) else {name}(_p)\n"
            + "try:\n _ev=_a.literal_eval(_e)\nexcept:\n _ev=_e\n"  # noqa: E722  -- bare except inside generated user code, intentional
            + "assert str(_r)==str(_ev) or _r==_ev,f'got {_r}'\nprint('SELF_TEST_PASS')\n")
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


class V3PipelineService:
    """Full V3 pipeline for a single coding task, with streaming progress."""

    def __init__(self):
        # ALL V3 components enabled — same as benchmark runner with all phases active
        self.budget_forcing = BudgetForcing(BudgetForcingConfig(enabled=True))
        self.plan_search = PlanSearch(PlanSearchConfig(enabled=True))
        self.div_sampling = DivSampling(DivSamplingConfig(enabled=True))
        self.blend_asc = BlendASC(BlendASCConfig(enabled=True))
        self.s_star = SStar(SStarConfig(enabled=True))
        self.pr_cot = PRCoT(PRCoTConfig(enabled=True))
        self.refinement_loop = RefinementLoop(RefinementLoopConfig(enabled=True))
        self.derivation_chains = DerivationChains(DerivationChainsConfig(enabled=True))
        self.metacognitive = MetacognitiveProfile(MetacognitiveConfig(enabled=True))
        self.self_test_gen = SelfTestGen(SelfTestGenConfig(enabled=True))

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
        """
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

        llm = adapters.LLMAdapter(progress_callback=emit)
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
                if total > 0 and p < total / 2:
                    return False, out, f"Self-test:{p}/{total}. "+";".join(fails[:3]), verification_evidence
            return verify_build_if_requested(out, err)

        # Score and test probe with self-generated tests
        probe_energy_raw, probe_energy_norm = 0.0, 0.5
        probe_cx_calibrated = False
        probe_passed = False
        if probe_code:
            probe_energy_raw, probe_energy_norm, probe_cx_calibrated = scoring.score_candidate(probe_code)
            norm_label = f"{probe_energy_norm:.2f}" if probe_cx_calibrated else "uncalibrated"
            emit("probe_scored", f"C(x)={probe_energy_raw:.2f} norm={norm_label}")
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

        # ===== PHASE 2: ADAPTIVE K ALLOCATION =====
        check_client()
        emit("phase2", "Allocating compute budget...")
        if probe_cx_calibrated:
            k, budget_tier = self.blend_asc.allocate(
                probe_energy_raw, task_id,
                normalized_energy=probe_energy_norm,
            )
        else:
            k, budget_tier = self.blend_asc.config.default_k, "standard"
        bf_tier = budget_tier
        emit("phase2_allocated", f"k={k} tier={budget_tier}", k=k, tier=budget_tier)

        # ===== PHASE 1: CONSTRAINT-DIVERSE CANDIDATE GENERATION =====
        emit("phase1", f"Generating {k} diverse candidates...", k=k)
        candidates = []

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
                gx_min = per_step.get("gx_score_min")
                severe = (per_step.get("thresholds") or {}).get("severe")
                if (gx_min is not None and isinstance(severe, (int, float))
                        and gx_min < severe):
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
                cg_kept = []
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
                        emit("call_graph_veto",
                             f"Candidate {c.get('index')} has unresolved call(s): "
                             f"{', '.join(res['unresolved'][:3])}",
                             index=c.get("index"), unresolved=res["unresolved"][:5])
                        print(f"  [call_graph] vetoed cand {c.get('index')} — "
                              f"unresolved: {res['unresolved'][:5]}", flush=True)
                        continue
                    cg_kept.append(c)
                if cg_kept:  # only prune when at least one candidate survives
                    passing = cg_kept

        # ===== CANDIDATE SELECTION =====
        if passing:
            # S* tiebreaking if multiple passing candidates
            if len(passing) >= 2:
                emit("s_star", "Tiebreaking with S*...")
                try:
                    s_star_candidates = [
                        CandidateScore(code=c["code"], raw_energy=c["energy"], index=c["index"])
                        for c in passing[:2]
                    ]
                    tb_result = self.s_star.tiebreak(
                        candidates=s_star_candidates,
                        problem=problem,
                        llm_call=llm,
                        sandbox_run=sandbox,
                        task_id=task_id,
                    )
                    if tb_result.triggered and tb_result.winner_index >= 0:
                        # winner_index is the candidate's ORIGINAL index, not a
                        # position in the sorted/filtered `passing` list — match
                        # by field like the lens path below. No match falls
                        # through to lens selection.
                        winner = _candidate_by_index(passing, tb_result.winner_index)
                        if winner is not None:
                            emit("s_star_winner", f"Winner: candidate {winner['index']}",
                                 index=winner["index"], energy=winner.get("energy_norm", 0.0))
                            result["passed"] = True
                            result["code"] = winner["code"]
                            result["phase_solved"] = "phase1_sstar"
                            result["total_time_ms"] = (time.time() - start) * 1000
                            result["verification_evidence"] = winner.get("verification_evidence", [])
                            result["winning_score"] = winner.get("energy_norm", 0.0)
                            result["events"] = events
                            return result
                except Exception as e:
                    emit("s_star_error", str(e)[:200])

            # Lens selection from passing candidates
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
        emit("phase3", "All candidates failed — entering repair phase...",
             failing=len([c for c in candidates if not c.get("passed")]))

        failing = [
            FailingCandidate(
                index=c["index"], code=c["code"],
                error_output=c.get("stderr", ""),
            )
            for c in candidates if not c.get("passed")
        ]

        # Self-test generation for repair verification — algorithmic only.
        # Interactive tasks repair against compile-smoke (PC-022).
        if task_type == "algorithmic":
            emit("self_test_gen", "Generating self-tests...")
            try:
                self_tests = self.self_test_gen.generate(problem, llm, task_id)
                emit("self_test_done", f"{len(self_tests.test_cases)} test cases generated")
            except Exception as e:
                self_tests = None
                emit("self_test_error", str(e)[:200])
        else:
            self_tests = None

        # Metacognitive warnings
        metacog_warnings = self.metacognitive.get_warnings([], task_id)

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

        # Strategy 2: Refinement Loop
        if failing:
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
                    metacognitive_warnings=metacog_warnings,
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

        # Strategy 3: Derivation Chains
        if failing:
            check_client()
            emit("derivation", "Attempting derivation chains...",
                 strategy="derivation", failing=len(failing))
            failure_context = "; ".join(
                f"Candidate {c.index}: {c.error_output[:200]}"
                for c in failing[:3]
            )
            # GH #39 point 3: append call-chain context to the failure
            # context so derivation chains gets the structural hints
            # alongside the truncated stderrs from each failing candidate.
            if chain_context_block:
                failure_context = failure_context + "\n\n" + chain_context_block
            try:
                dc_result = self.derivation_chains.solve(
                    problem=problem,
                    failure_context=failure_context,
                    llm_call=llm,
                    sandbox_run=sandbox,
                    task_id=task_id,
                )
                result["total_tokens"] += dc_result.total_tokens
                if dc_result.solved:
                    # Verify with real sandbox
                    passed, _, _, derivation_evidence = verified_sandbox(dc_result.final_code)
                    if passed:
                        emit("derivation_pass", "Derivation chains solved!",
                             strategy="derivation")
                        result["passed"] = True
                        result["code"] = dc_result.final_code
                        result["phase_solved"] = "derivation"
                        result["total_time_ms"] = (time.time() - start) * 1000
                        result["verification_evidence"] = derivation_evidence
                        result["events"] = events
                        return result
                emit("derivation_failed", dc_result.reason)
            except Exception as e:
                emit("derivation_error", str(e)[:200])

        # ===== FALLBACK: Return best candidate even if none passed =====
        emit("fallback", "No passing solution found — returning best candidate by energy")
        if candidates:
            candidates.sort(key=lambda c: c.get("energy", 999))
            result["code"] = candidates[0]["code"]
        result["total_time_ms"] = (time.time() - start) * 1000
        result["events"] = events
        return result


# --- Problem Builder for /v3/generate ----------------------------------------

def _build_problem_from_request(
    file_path: str, baseline_code: str, project_context: Dict[str, str],
    framework: str, build_command: str, constraints: List[str],
) -> str:
    """Build a problem description for the V3 pipeline from a generate request."""
    parts = []

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
