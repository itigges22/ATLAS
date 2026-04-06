#!/usr/bin/env python3
"""
ATLAS V3 Pipeline Service — HTTP wrapper around the V3 benchmark pipeline.

Exposes the full V3 pipeline (PlanSearch, DivSampling, BudgetForcing, BlendASC,
S*, PR-CoT, RefinementLoop, DerivationChains, etc.) as an HTTP service that
the Go proxy can call for T2/T3 tasks.

For CLI use, test cases are generated via SelfTestGen since we don't have
benchmark ground truth. The sandbox runs syntax/runtime checks on all candidates.

Streams progress events back as SSE for real-time CLI feedback.
"""

import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import io

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runner import extract_code
from benchmark.v3.budget_forcing import BudgetForcing, BudgetForcingConfig
from benchmark.v3.plan_search import PlanSearch, PlanSearchConfig
from benchmark.v3.div_sampling import DivSampling, DivSamplingConfig
from benchmark.v3.blend_asc import BlendASC, BlendASCConfig
from benchmark.v3.s_star import SStar, SStarConfig, CandidateScore
from benchmark.v3.failure_analysis import FailureAnalyzer, FailureAnalysisConfig, FailingCandidate
from benchmark.v3.constraint_refinement import ConstraintRefiner, ConstraintRefinementConfig
from benchmark.v3.pr_cot import PRCoT, PRCoTConfig
from benchmark.v3.refinement_loop import RefinementLoop, RefinementLoopConfig
from benchmark.v3.derivation_chains import DerivationChains, DerivationChainsConfig
from benchmark.v3.metacognitive import MetacognitiveProfile, MetacognitiveConfig
from benchmark.v3.self_test_gen import SelfTestGen, SelfTestGenConfig
from benchmark.v3.candidate_selection import CandidateInfo, select_candidate


# --- Configuration -----------------------------------------------------------

INFERENCE_URL = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
LENS_URL = os.environ.get("ATLAS_LENS_URL", "http://localhost:8099")
SANDBOX_URL = os.environ.get("ATLAS_SANDBOX_URL", "http://localhost:30820")
PORT = int(os.environ.get("ATLAS_V3_PORT", "8070"))

BASE_TEMPERATURE = 0.6
DIVERSITY_TEMPERATURE = 0.8
MAX_TOKENS = 8192


# --- LLM Adapter (calls llama-server /v1/completions) ---------------------------------

class LLMAdapter:
    """Calls Fox's /v1/completions with ChatML prompt."""

    _lock = threading.Lock()

    def __init__(self, progress_callback=None):
        self.call_count = 0
        self.total_tokens = 0
        self.last_logprobs: List[float] = []
        self._progress = progress_callback

    def _emit(self, stage: str, detail: str = ""):
        if self._progress:
            self._progress(stage, detail)

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.call_count += 1

        body = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "stop": ["\n\n\n\n"],
            "top_k": 20,
            "top_p": 0.95,
        }
        if seed is not None:
            body["seed"] = seed

        start = time.time()
        data = self._send(body)

        # Parse response
        content = ""
        tokens = 0
        if "choices" in data:
            content = data["choices"][0].get("text", "")
            tokens = data.get("usage", {}).get("completion_tokens", 0)

        # Strip thinking blocks
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        if '</think>' in content and '<think>' not in content:
            content = content[content.index('</think>') + len('</think>'):].strip()

        t_ms = (time.time() - start) * 1000
        self.total_tokens += tokens
        return content, tokens, t_ms

    def _send(self, body: dict) -> dict:
        """Send to Fox via /v1/chat/completions.

        V3 modules generate ChatML prompts. We parse them into messages format
        for Fox's chat endpoint. ChatML format:
            <|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n
        """
        prompt = body.pop("prompt", "")
        model_name = os.environ.get("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")

        # Parse ChatML into messages
        messages = []
        parts = re.split(r'<\|im_start\|>(\w+)\n', prompt)
        # parts = ['', 'system', 'content...<|im_end|>\n', 'user', 'content...<|im_end|>\n', ...]
        i = 1
        while i < len(parts) - 1:
            role = parts[i]
            content = parts[i + 1].replace('<|im_end|>', '').strip()
            # Remove think pre-fill from assistant messages
            content = content.replace('<think>\n\n</think>', '').strip()
            if content:
                messages.append({"role": role, "content": content})
            i += 2

        # If parsing failed, just send as user message
        if not messages:
            print(f"  [LLM] ChatML parse failed, using raw prompt ({len(prompt)} chars)", flush=True)
            messages = [{"role": "user", "content": "/nothink\n" + prompt}]
        else:
            print(f"  [LLM] Parsed {len(messages)} messages from ChatML", flush=True)
            # Ensure /nothink in last user message
            for msg in messages:
                if msg["role"] == "user" and not msg["content"].startswith("/nothink"):
                    msg["content"] = "/nothink\n" + msg["content"]

        chat_body = {
            "model": model_name,
            "messages": messages,
            "max_tokens": body.get("max_tokens", body.pop("n_predict", 4096)),
            "temperature": body.get("temperature", 0.6),
            "stream": False,
        }
        if "seed" in body:
            chat_body["seed"] = body["seed"]

        req = urllib.request.Request(
            f"{INFERENCE_URL}/v1/chat/completions",
            data=json.dumps(chat_body).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(5):
            try:
                with LLMAdapter._lock:
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        data = json.loads(resp.read())
                        # Convert chat response to completions format
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "message" in choice:
                                choice["text"] = choice["message"].get("content", "")
                        return data
            except (urllib.error.HTTPError, OSError) as e:
                print(f"  [LLM] Attempt {attempt+1} failed: {e}", flush=True)
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise


# --- Sandbox Adapter (calls sandbox /execute) ---------------------------------

class SandboxAdapter:
    """Calls the K3s sandbox for code execution."""

    def __call__(self, code: str, test_input: str = "") -> Tuple[bool, str, str]:
        body = {
            "code": code,
            "language": "python",
            "timeout": 15,
        }
        try:
            req = urllib.request.Request(
                f"{SANDBOX_URL}/execute",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
                return data.get("success", False), data.get("stdout", ""), data.get("stderr", "")
        except Exception as e:
            return False, "", str(e)


# --- Embedding Adapter --------------------------------------------------------

class EmbedAdapter:
    """Calls llama-server /v1/embeddings for code embeddings."""

    def __call__(self, text: str) -> List[float]:
        body = {"model": "default", "input": text}
        try:
            req = urllib.request.Request(
                f"{INFERENCE_URL}/v1/embeddings",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("data", [{}])[0].get("embedding", [])
        except Exception:
            return []


# --- Lens Scorer (calls Geometric Lens) ---------------------------------------------

def score_candidate(code: str) -> Tuple[float, float]:
    """Score code with Geometric Lens C(x). Returns (raw_energy, normalized)."""
    try:
        body = json.dumps({"text": code}).encode()
        req = urllib.request.Request(
            f"{LENS_URL}/internal/lens/gx-score",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("cx_energy", 0.0), data.get("gx_score", 0.5)
    except Exception:
        return 0.0, 0.5


# --- V3 Pipeline Orchestrator ------------------------------------------------

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
        self.failure_analyzer = FailureAnalyzer(FailureAnalysisConfig(enabled=True))
        self.constraint_refiner = ConstraintRefiner(ConstraintRefinementConfig(enabled=True))
        self.metacognitive = MetacognitiveProfile(MetacognitiveConfig(enabled=True))
        self.self_test_gen = SelfTestGen(SelfTestGenConfig(enabled=True))

    def run(self, problem: str, task_id: str = "cli",
            progress_callback=None, files: Dict[str, str] = None) -> Dict[str, Any]:
        """Run the full V3 pipeline on a coding problem.

        Args:
            problem: Problem description
            task_id: Task identifier
            progress_callback: SSE progress emitter
            files: Dict of filename→content from Aider's existing file context
        """
        start = time.time()
        events = []
        files = files or {}

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

        def emit(stage, detail=""):
            events.append({"stage": stage, "detail": detail, "t": time.time() - start})
            if progress_callback:
                progress_callback(stage, detail)

        llm = LLMAdapter(progress_callback=emit)
        sandbox = SandboxAdapter()
        embed = EmbedAdapter()

        result = {
            "task_id": task_id,
            "passed": False,
            "code": "",
            "phase_solved": "none",
            "candidates_generated": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "events": [],
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
            # Generate with /nothink
            chatml = self.budget_forcing.format_chatml(problem, "nothink")
            response, tokens, t_ms = llm(chatml, BASE_TEMPERATURE, MAX_TOKENS, 42)
            probe_code = extract_code(response)

        # Generate self-tests first — used for all sandbox verification
        emit("self_test_gen", "Generating verification tests...")
        self_tests = None
        try:
            self_tests = self.self_test_gen.generate(problem, llm, task_id)
            emit("self_test_done", f"{len(self_tests.test_cases)} test cases")
            result["total_tokens"] += self_tests.generation_tokens
        except Exception as e:
            emit("self_test_error", str(e)[:200])

        def _make_test(code, tc):
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
                    + "try:\n _p=_a.literal_eval(_i)\nexcept:\n _p=_i\n"  # noqa: safe literal parse
                    + f"_r={name}(*_p) if isinstance(_p,tuple) else {name}(_p) if isinstance(_p,list) else {name}(_p)\n"
                    + "try:\n _ev=_a.literal_eval(_e)\nexcept:\n _ev=_e\n"  # noqa: safe literal parse
                    + "assert str(_r)==str(_ev) or _r==_ev,f'got {_r}'\nprint('SELF_TEST_PASS')\n")
            return (
                "import sys as _s,io as _o\n"
                f"_s.stdin=_o.StringIO({repr(inp)})\n"
                "_c=_o.StringIO()\n_old=_s.stdout\n_s.stdout=_c\n"
                "try:\n" + "\n".join("    "+l for l in code.split("\n"))
                + "\nfinally:\n _s.stdout=_old\n"
                f"assert _c.getvalue().strip()=={repr(exp)},f'got {{_c.getvalue().strip()}}'\n"
                "print('SELF_TEST_PASS')\n")

        def verified_sandbox(code, extra_test=""):
            """Sandbox + self-test correctness. Fails if code crashes OR >50% assertions fail."""
            ok, out, err = sandbox(code)
            if not ok:
                return False, out, err
            if self_tests and self_tests.test_cases:
                p, fails = 0, []
                for i, tc in enumerate(self_tests.test_cases):
                    try:
                        tc_code = _make_test(code, tc)
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
                    return False, out, f"Self-test:{p}/{total}. "+";".join(fails[:3])
            return True, out, err

        # Score and test probe with self-generated tests
        probe_energy_raw, probe_energy_norm = 0.0, 0.5
        probe_passed = False
        if probe_code:
            probe_energy_raw, probe_energy_norm = score_candidate(probe_code)
            emit("probe_scored", f"C(x)={probe_energy_raw:.2f} norm={probe_energy_norm:.2f}")
            probe_passed, probe_stdout, probe_stderr = verified_sandbox(probe_code)
            emit("probe_sandbox", f"passed={probe_passed} stderr={probe_stderr[:80] if probe_stderr else ''}")
            result["total_tokens"] += tokens

        if probe_passed:
            emit("probe_pass", "Probe passed — returning early")
            result["passed"] = True
            result["code"] = probe_code
            result["phase_solved"] = "probe"
            result["candidates_generated"] = 1
            result["total_time_ms"] = (time.time() - start) * 1000
            result["events"] = events
            return result

        # ===== PHASE 2: ADAPTIVE K ALLOCATION =====
        emit("phase2", "Allocating compute budget...")
        k, budget_tier = self.blend_asc.allocate(probe_energy_raw, task_id)
        bf_tier = budget_tier
        emit("phase2_allocated", f"k={k} tier={budget_tier}")

        # ===== PHASE 1: CONSTRAINT-DIVERSE CANDIDATE GENERATION =====
        emit("phase1", f"Generating {k} diverse candidates...")
        candidates = []

        # Start with probe if it produced code
        if probe_code:
            candidates.append({
                "index": 0, "code": probe_code,
                "energy": probe_energy_raw, "energy_norm": probe_energy_norm,
                "passed": probe_passed, "stdout": "", "stderr": "",
            })

        remaining_k = max(0, k - len(candidates))

        # Step 1A: PlanSearch
        if remaining_k > 0:
            emit("plansearch", f"Generating {remaining_k} plans...")
            try:
                ps_result = self.plan_search.generate(
                    problem, task_id, llm, num_plans=remaining_k,
                )
                for i, code in enumerate(ps_result.candidates):
                    if code:
                        energy_raw, energy_norm = score_candidate(code)
                        candidates.append({
                            "index": len(candidates), "code": code,
                            "energy": energy_raw, "energy_norm": energy_norm,
                            "passed": False, "stdout": "", "stderr": "",
                        })
                result["total_tokens"] += ps_result.total_tokens
                emit("plansearch_done", f"{len(ps_result.candidates)} candidates from PlanSearch")
            except Exception as e:
                emit("plansearch_error", str(e)[:200])

        # Step 1B: DivSampling to fill remaining slots
        remaining_k = max(0, k - len(candidates))
        if remaining_k > 0:
            emit("divsampling", f"Filling {remaining_k} slots with diverse sampling...")
            for idx in range(remaining_k):
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
                        energy_raw, energy_norm = score_candidate(code)
                        candidates.append({
                            "index": len(candidates), "code": code,
                            "energy": energy_raw, "energy_norm": energy_norm,
                            "passed": False, "stdout": "", "stderr": "",
                        })
                    result["total_tokens"] += tokens
                except Exception as e:
                    emit("divsampling_error", str(e)[:200])
            emit("divsampling_done", f"{len(candidates)} total candidates")

        result["candidates_generated"] = len(candidates)

        # ===== SANDBOX TESTING =====
        emit("sandbox_test", f"Testing {len(candidates)} candidates...")
        # Sort by energy (easy first) for early-exit potential
        candidates.sort(key=lambda c: c.get("energy", 0))

        passing = []
        for c in candidates:
            if c.get("passed"):
                passing.append(c)
                continue
            passed, stdout, stderr = verified_sandbox(c["code"])
            c["passed"] = passed
            c["stdout"] = stdout
            c["stderr"] = stderr
            if passed:
                passing.append(c)
                emit("sandbox_pass", f"Candidate {c['index']} passed")

        emit("sandbox_done", f"{len(passing)}/{len(candidates)} passed")

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
                        winner = passing[tb_result.winner_index]
                        emit("s_star_winner", f"Winner: candidate {winner['index']}")
                        result["passed"] = True
                        result["code"] = winner["code"]
                        result["phase_solved"] = "phase1_sstar"
                        result["total_time_ms"] = (time.time() - start) * 1000
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
                emit("selected", f"Lens selected candidate {selected.index}")
                result["passed"] = True
                result["code"] = selected.code
                result["phase_solved"] = "phase1"
                result["total_time_ms"] = (time.time() - start) * 1000
                result["events"] = events
                return result

        # ===== PHASE 3: VERIFIED ITERATIVE REFINEMENT =====
        emit("phase3", "All candidates failed — entering repair phase...")

        failing = [
            FailingCandidate(
                index=c["index"], code=c["code"],
                error_output=c.get("stderr", ""),
            )
            for c in candidates if not c.get("passed")
        ]

        # Self-test generation for repair verification
        emit("self_test_gen", "Generating self-tests...")
        try:
            self_tests = self.self_test_gen.generate(problem, llm, task_id)
            emit("self_test_done", f"{len(self_tests.test_cases)} test cases generated")
        except Exception as e:
            self_tests = None
            emit("self_test_error", str(e)[:200])

        # Metacognitive warnings
        metacog_warnings = self.metacognitive.get_warnings([], task_id)

        # Strategy 1: PR-CoT Quick Repair
        if failing:
            emit("pr_cot", "Attempting PR-CoT repair...")
            best_failing = failing[0]
            try:
                pr_result = self.pr_cot.repair(
                    problem=problem,
                    code=best_failing.code,
                    error=best_failing.error_output,
                    llm_call=llm,
                    task_id=task_id,
                )
                result["total_tokens"] += pr_result.total_tokens
                for repair_code in pr_result.repairs:
                    passed, stdout, stderr = verified_sandbox(repair_code)
                    if passed:
                        emit("pr_cot_pass", "PR-CoT repair succeeded!")
                        result["passed"] = True
                        result["code"] = repair_code
                        result["phase_solved"] = "pr_cot"
                        result["total_time_ms"] = (time.time() - start) * 1000
                        result["events"] = events
                        return result
                emit("pr_cot_failed", "PR-CoT repair did not produce passing code")
            except Exception as e:
                emit("pr_cot_error", str(e)[:200])

        # Strategy 2: Refinement Loop
        if failing:
            emit("refinement", "Starting refinement loop...")
            constraints = []  # from PlanSearch
            try:
                ref_result = self.refinement_loop.run(
                    problem=problem,
                    failing_candidates=failing,
                    original_constraints=constraints,
                    llm_call=llm,
                    sandbox_run=sandbox,
                    embed_call=embed,
                    metacognitive_warnings=metacog_warnings,
                    task_id=task_id,
                )
                result["total_tokens"] += ref_result.total_tokens
                if ref_result.solved:
                    emit("refinement_pass", f"Refinement solved in {ref_result.total_iterations} iterations!")
                    result["passed"] = True
                    result["code"] = ref_result.winning_code
                    result["phase_solved"] = "refinement"
                    result["total_time_ms"] = (time.time() - start) * 1000
                    result["events"] = events
                    return result
                emit("refinement_failed", f"Exhausted {ref_result.total_iterations} iterations")
            except Exception as e:
                emit("refinement_error", str(e)[:200])

        # Strategy 3: Derivation Chains
        if failing:
            emit("derivation", "Attempting derivation chains...")
            failure_context = "; ".join(
                f"Candidate {c.index}: {c.error_output[:200]}"
                for c in failing[:3]
            )
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
                    passed, _, _ = verified_sandbox(dc_result.final_code)
                    if passed:
                        emit("derivation_pass", "Derivation chains solved!")
                        result["passed"] = True
                        result["code"] = dc_result.final_code
                        result["phase_solved"] = "derivation"
                        result["total_time_ms"] = (time.time() - start) * 1000
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


# --- Build Verification (per-file-type) --------------------------------------

class BuildVerifier:
    """Generates file-type-appropriate verification commands.

    Instead of stdin/stdout test pairs (for algorithm problems), this generates
    build/compile/import commands appropriate for arbitrary code files.
    """

    # Extension → (verification commands, description)
    VERIFY_MAP = {
        ".py": (["python -m py_compile {file}"], "Python compile check"),
        ".ts": (["npx tsc --noEmit"], "TypeScript type check"),
        ".tsx": (["npx tsc --noEmit"], "TypeScript/React type check"),
        ".js": (["node --check {file}"], "JavaScript syntax check"),
        ".jsx": (["node --check {file}"], "JavaScript/React syntax check"),
        ".go": (["go build ."], "Go build"),
        ".rs": (["cargo check"], "Rust cargo check"),
        ".c": (["gcc -fsyntax-only {file}"], "C syntax check"),
        ".h": (["gcc -fsyntax-only {file}"], "C header syntax check"),
        ".cpp": (["g++ -fsyntax-only {file}"], "C++ syntax check"),
        ".sh": (["bash -n {file}"], "Shell syntax check"),
        ".bash": (["bash -n {file}"], "Shell syntax check"),
        ".json": (['python -c "import json; json.load(open(\'{file}\'))"'], "JSON validation"),
    }

    # Framework → build command override
    FRAMEWORK_BUILD = {
        "nextjs": "npx next build",
        "react": "npx react-scripts build",
        "flask": "python -m py_compile {file}",
        "django": "python manage.py check",
        "express": "node --check {file}",
    }

    def __init__(self, file_path: str, framework: str = "",
                 build_command: str = "", working_dir: str = ""):
        self.file_path = file_path
        self.framework = framework
        self.build_command = build_command
        self.working_dir = working_dir
        self._ext = Path(file_path).suffix.lower()

    def describe(self) -> str:
        cmds = self.get_commands()
        return " && ".join(cmds) if cmds else "no verification available"

    def get_commands(self) -> List[str]:
        """Return verification commands for this file type."""
        # Framework-specific override
        if self.framework and self.framework in self.FRAMEWORK_BUILD:
            cmd = self.FRAMEWORK_BUILD[self.framework].format(file=self.file_path)
            return [cmd]

        # Explicit build command from project detection
        if self.build_command:
            return [self.build_command]

        # Extension-based
        if self._ext in self.VERIFY_MAP:
            cmds, _ = self.VERIFY_MAP[self._ext]
            return [c.format(file=self.file_path) for c in cmds]

        return []

    def verify_code_in_sandbox(self, code: str, sandbox: SandboxAdapter) -> Tuple[bool, str, str]:
        """Run the code through sandbox with appropriate verification.

        For Python files, we can execute directly.
        For other languages, we check syntax/compilation.
        """
        if self._ext == ".py":
            return sandbox(code)

        # For non-Python, the sandbox only supports Python execution.
        # Wrap verification in a Python script that writes the file
        # and runs the verification command.
        if self.get_commands():
            verify_script = self._build_verify_script(code)
            return sandbox(verify_script)

        # Fallback: basic syntax check
        return sandbox(code)

    def _build_verify_script(self, code: str) -> str:
        """Build a Python script that writes the file and runs verification."""
        import shlex
        cmds = self.get_commands()
        safe_code = code.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

        lines = [
            "import subprocess, tempfile, os, sys",
            "tmpdir = tempfile.mkdtemp()",
            f"filepath = os.path.join(tmpdir, '{Path(self.file_path).name}')",
            f"with open(filepath, 'w') as f:",
            f"    f.write('''{code}''')",
            "os.chdir(tmpdir)",
        ]
        for cmd in cmds:
            lines.append(f"r = subprocess.run({shlex.quote(cmd)}, shell=True, capture_output=True, text=True, timeout=30)")
            lines.append("if r.returncode != 0:")
            lines.append("    print(r.stderr, file=sys.stderr)")
            lines.append("    sys.exit(1)")

        lines.append("print('BUILD_VERIFY_PASS')")
        return "\n".join(lines)


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


# --- HTTP Handler (SSE streaming) --------------------------------------------

pipeline = V3PipelineService()


class V3Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v3/run":
            self._handle_run()
        elif self.path == "/v3/generate":
            self._handle_generate()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "v3-pipeline"})
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_run(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))

        problem = body.get("problem", "")
        task_id = body.get("task_id", "cli")
        stream = body.get("stream", True)
        files = body.get("files", {})

        if not problem:
            self._json_response(400, {"error": "missing 'problem' field"})
            return

        if stream:
            # SSE streaming
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit_sse(stage, detail=""):
                event = json.dumps({"stage": stage, "detail": detail})
                try:
                    self.wfile.write(f"data: {event}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    pass

            result = pipeline.run(problem, task_id, progress_callback=emit_sse, files=files)

            # Final result event
            final = json.dumps(result, default=str)
            self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            result = pipeline.run(problem, task_id, files=files)
            self._json_response(200, result)

    def _handle_generate(self):
        """Handle /v3/generate — accepts arbitrary file generation requests from Go proxy.

        Request format (V3GenerateRequest):
            file_path: str          — target file path
            baseline_code: str      — model's initial content (candidate #0)
            project_context: dict   — other files in project {path: content}
            framework: str          — detected framework
            build_command: str      — build verification command
            constraints: list[str]  — extracted requirements
            tier: int               — 2 or 3
            working_dir: str        — project root

        Response format (V3GenerateResponse):
            code: str               — winning candidate
            passed: bool            — whether it passed verification
            phase_solved: str       — which phase solved it
            candidates_tested: int
            winning_score: float
            total_tokens: int
            total_time_ms: float
        """
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))

        file_path = body.get("file_path", "")
        baseline_code = body.get("baseline_code", "")
        project_context = body.get("project_context", {})
        framework = body.get("framework", "")
        build_command = body.get("build_command", "")
        constraints = body.get("constraints", [])
        tier = body.get("tier", 2)
        working_dir = body.get("working_dir", "")

        if not file_path and not baseline_code:
            self._json_response(400, {"error": "file_path or baseline_code required"})
            return

        # Build problem description from the adapter request
        problem = _build_problem_from_request(
            file_path, baseline_code, project_context,
            framework, build_command, constraints,
        )

        # Build file context for the pipeline
        files = dict(project_context) if project_context else {}

        # Determine build verification for this file type
        build_verifier = BuildVerifier(file_path, framework, build_command, working_dir)

        print(f"[generate] file={file_path} framework={framework} tier=T{tier}", flush=True)
        print(f"[generate] build_verify: {build_verifier.describe()}", flush=True)
        print(f"[generate] constraints: {constraints}", flush=True)

        # Stream V3 pipeline progress as SSE events, then final result as JSON
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit_progress(stage, detail=""):
            """Stream progress events to the Go proxy."""
            event = json.dumps({"stage": stage, "detail": detail})
            try:
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
                # Also log for debugging
                print(f"  [SSE] {stage}: {detail[:80]}", flush=True)
            except BrokenPipeError:
                pass
            except Exception as e:
                print(f"  [SSE ERROR] {e}", flush=True)

        # Run V3 pipeline with streaming progress
        result = pipeline.run(
            problem=problem,
            task_id=f"gen-{Path(file_path).stem}",
            progress_callback=emit_progress,
            files=files,
        )

        # If baseline code was provided and pipeline didn't produce anything better,
        # use the baseline
        if not result.get("code") and baseline_code:
            result["code"] = baseline_code
            result["phase_solved"] = "baseline"

        # Send final result
        response = {
            "code": result.get("code", ""),
            "passed": result.get("passed", False),
            "phase_solved": result.get("phase_solved", "none"),
            "candidates_tested": result.get("candidates_generated", 0),
            "winning_score": 0.0,
            "total_tokens": result.get("total_tokens", 0),
            "total_time_ms": result.get("total_time_ms", 0.0),
        }
        final = json.dumps(response)
        self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    print(f"ATLAS V3 Pipeline Service starting on :{PORT}")
    print(f"  Inference:     {INFERENCE_URL}")
    print(f"  Geometric Lens: {LENS_URL}")
    print(f"  Sandbox: {SANDBOX_URL}")

    server = HTTPServer(("0.0.0.0", PORT), V3Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
