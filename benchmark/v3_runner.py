#!/usr/bin/env python3
"""
ATLAS V3 Benchmark Runner.

Orchestrates the full V3 pipeline on LiveCodeBench:

  For each task:
    Phase 1: Generate k constraint-diverse candidates
      - PlanSearch → constraints
      - DivSampling → diverse prompts
      - Budget Forcing → token control
      - Sandbox test all k
      - If any pass → Lens selects best → DONE

    Phase 2: Adaptive compute allocation
      - Blend-ASC → adaptive K per difficulty
      - ReASC → early stopping on low-confidence
      - S* → tiebreaking for borderline candidates

    Phase 3: Verified iterative refinement (if 0/k pass)
      - PR-CoT repair (quick fix, 1-2 attempts)
      - Full refinement loop:
        - 3A: Failure analysis
        - 3F: Metacognitive compensations
        - 3B: Constraint refinement
        - 3D: Derivation chains (if complex)
        - 3E: Loop orchestration (max 5 iterations)
      - 3G: ACE learning from successes

Telemetry: results/<run_id>/telemetry/v3_events.jsonl
"""

import json
import math
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.config import config
from benchmark.models import BenchmarkTask, AttemptResult, TaskResult
from benchmark.runner import BenchmarkRunner, LLMConnectionError, extract_code
from benchmark.runner import execute_code, execute_code_stdio
from benchmark.geo_learning import extract_embedding_urllib
from benchmark.best_of_k import score_candidate

# V3 components
from benchmark.v3.budget_forcing import BudgetForcing, BudgetForcingConfig, extract_thinking
from benchmark.v3.plan_search import PlanSearch, PlanSearchConfig
from benchmark.v3.div_sampling import DivSampling, DivSamplingConfig
from benchmark.v3.blend_asc import BlendASC, BlendASCConfig
from benchmark.v3.reasc import ReASC, ReASCConfig
from benchmark.v3.s_star import SStar, SStarConfig, CandidateScore
from benchmark.v3.failure_analysis import (
    FailureAnalyzer, FailureAnalysisConfig, FailingCandidate,
)
from benchmark.v3.constraint_refinement import (
    ConstraintRefiner, ConstraintRefinementConfig,
)
from benchmark.v3.pr_cot import PRCoT, PRCoTConfig
from benchmark.v3.refinement_loop import (
    RefinementLoop, RefinementLoopConfig,
)
from benchmark.v3.derivation_chains import (
    DerivationChains, DerivationChainsConfig,
)
from benchmark.v3.metacognitive import (
    MetacognitiveProfile, MetacognitiveConfig, BenchmarkResult,
)
from benchmark.v3.ace_pipeline import ACEPipeline, ACEConfig
from benchmark.v3.self_test_gen import SelfTestGen, SelfTestGenConfig
from benchmark.v3.lens_feedback import LensFeedbackCollector, LensFeedbackConfig


# --- Constants ----------------------------------------------------------------

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:31144")
LLAMA_URL = os.environ.get("LLAMA_URL", f"http://localhost:{config._conf.get('ATLAS_LLAMA_NODEPORT', '32735')}")
MAX_TOKENS = 16384
BASE_TEMPERATURE = 0.0
DIVERSITY_TEMPERATURE = 0.6


# --- Atomic I/O (reused from v2_runner) ----------------------------------------

def atomic_write_json(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        shutil.move(str(tmp), str(filepath))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def append_jsonl(filepath, record):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + '\n')


def wrap_class_solution(code: str, task: BenchmarkTask) -> str:
    """Wrap 'class Solution' code with a stdin/stdout harness for stdio eval.

    Many LCB tasks provide a 'class Solution' method signature in the prompt.
    The model completes the class but doesn't add stdin/stdout handling.
    This wrapper parses the method signature from the task prompt and appends
    a harness that reads stdin, calls the method, and prints the result.

    Returns the original code unchanged if it's not a class Solution pattern,
    already has input() calls, or the task is not stdio eval.
    """
    if task.eval_mode != "stdio":
        return code
    if "class Solution" not in code:
        return code
    if "input()" in code:
        return code  # already handles stdin

    # Extract method signature from task prompt
    sig_match = re.search(
        r'class Solution:.*?def (\w+)\(self,?\s*(.*?)\)\s*(?:->.*?)?:',
        task.prompt, re.DOTALL,
    )
    if not sig_match:
        return code

    method_name = sig_match.group(1)
    params_str = sig_match.group(2).strip()

    # Parse parameter names (ignore type annotations)
    param_names = []
    if params_str:
        for p in params_str.split(','):
            name = p.split(':')[0].strip()
            if name:
                param_names.append(name)

    # Prepend typing imports (class may use List, Dict, etc. without importing)
    # then append stdin/stdout harness after the class definition.
    preamble = "from typing import List, Optional, Tuple, Dict, Set\nimport ast"

    reader_lines = []
    for name in param_names:
        reader_lines.append(f"{name} = ast.literal_eval(input())")
    call_args = ", ".join(param_names)
    reader_lines.append(f"result = Solution().{method_name}({call_args})")
    reader_lines.append("print(result)")

    harness = "\n".join(reader_lines)
    return preamble + "\n" + code + "\n\n" + harness


def find_completed_tasks(phase_dir):
    completed = set()
    per_task_dir = Path(phase_dir) / "per_task"
    if per_task_dir.exists():
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    if 'task_id' in data:
                        completed.add(data['task_id'])
            except (json.JSONDecodeError, IOError):
                pass
    return completed


# --- Callable adapters for V3 components --------------------------------------

def self_verify_execute(results: List[Tuple[bool, str, str]],
                        threshold: float = 0.6) -> Tuple[bool, str, str]:
    """Majority-vote self-verification from multiple test case results.

    Args:
        results: List of (passed, stdout, stderr) per self-test case.
        threshold: Fraction of tests that must pass (0.0-1.0).

    Returns:
        (majority_passed, combined_stdout, combined_stderr)
    """
    if not results:
        return False, "", "no self-test results"

    passes = sum(1 for p, _, _ in results if p)
    ratio = passes / len(results)

    all_stderr = [s for _, _, s in results if s]
    all_stdout = [s for _, s, _ in results if s]

    return (
        ratio >= threshold,
        "\n".join(all_stdout),
        "\n".join(all_stderr),
    )


class LLMAdapter:
    """Adapts BenchmarkRunner._call_llm to the V3 LLMCallable signature.

    V3 components expect: (prompt, temperature, max_tokens, seed) -> (response, tokens, time_ms)
    The prompt is already ChatML-formatted by the V3 components.

    Budget Forcing enforcement: if the model's <think> block consumes >80%
    of the token budget and no useful output remains, the call is retried
    with /nothink injected into the prompt. This prevents infinite reasoning
    from starving code generation.
    """

    # Thinking consumes too much if it's >80% of tokens and output is tiny
    THINK_BUDGET_RATIO = 0.80
    MIN_OUTPUT_CHARS = 50

    def __init__(self, runner: BenchmarkRunner, max_retries: int = 4):
        self.runner = runner
        self.max_retries = max_retries
        self.call_count = 0
        self.total_tokens = 0
        self.last_logprobs: List[float] = []

    @staticmethod
    def _parse_logprobs(data: dict) -> List[float]:
        """Extract per-token log-probabilities from llama-server response."""
        logprobs = []
        for tok in data.get("completion_probabilities", []):
            probs = tok.get("probs", [])
            if probs:
                p = probs[0].get("prob", 0.0)
                if p > 0:
                    logprobs.append(math.log(p))
        return logprobs

    def _send_request(self, request_body: dict) -> dict:
        """Send a single request to llama-server with retry on connection errors."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.runner.llm_url}/completion",
                    data=json.dumps(request_body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 10 * (2 ** attempt)
                    time.sleep(wait)
        raise LLMConnectionError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
        )

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.call_count += 1
        request_body = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
            "cache_prompt": False,
            "stop": ["<|im_end|>", "<|im_start|>"],
            "n_probs": 1,
        }
        if seed is not None:
            request_body["seed"] = seed

        start_time = time.time()
        data = self._send_request(request_body)
        content = data.get("content", "")
        tokens = data.get("tokens_predicted", 0)
        self.last_logprobs = self._parse_logprobs(data)

        # --- Budget Forcing enforcement ---
        # If the model spent most of its budget thinking and produced little
        # useful output, retry with nothink to get an actual answer.
        thinking, output = extract_thinking(content)
        thinking_heavy = (
            len(thinking) > 0
            and tokens > 0
            and len(output) < self.MIN_OUTPUT_CHARS
        )

        if thinking_heavy and "/nothink" not in prompt:
            # Retry: inject /nothink into the system prompt
            nothink_prompt = prompt.replace(
                "<|im_end|>\n<|im_start|>user",
                " /nothink<|im_end|>\n<|im_start|>user",
                1,
            )
            request_body["prompt"] = nothink_prompt
            data = self._send_request(request_body)
            content = data.get("content", "")
            tokens += data.get("tokens_predicted", 0)
            self.last_logprobs = self._parse_logprobs(data)

        # Strip think blocks from final output
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        if '<think>' in content:
            content = content[:content.index('<think>')].strip()

        t_ms = (time.time() - start_time) * 1000
        self.total_tokens += tokens
        return content, tokens, t_ms


class SandboxAdapter:
    """Adapts execute_code/execute_code_stdio to V3 SandboxCallable.

    V3 components expect: (code, test_case) -> (passed, stdout, stderr)

    In self_verify_mode, runs code against model-generated test cases
    instead of real benchmark tests. Uses majority vote for pass/fail.
    """

    def __init__(self, task: BenchmarkTask, timeout_sec: int = 30,
                 memory_mb: int = 512,
                 self_verify_mode: bool = False,
                 custom_test_cases: Optional[List] = None,
                 majority_threshold: float = 0.6):
        self.task = task
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb
        self.call_count = 0
        self.self_verify_mode = self_verify_mode
        self.custom_test_cases = custom_test_cases or []
        self.majority_threshold = majority_threshold

    def __call__(self, code: str, test_case: str) -> Tuple[bool, str, str]:
        self.call_count += 1
        code = wrap_class_solution(code, self.task)

        if self.self_verify_mode and self.custom_test_cases:
            return self._run_self_tests(code)

        if self.task.eval_mode == "stdio":
            passed, stdout, stderr, _ = execute_code_stdio(
                code, self.task.test_inputs, self.task.test_outputs,
                timeout_sec=self.timeout_sec, memory_mb=self.memory_mb,
            )
        else:
            test_code = test_case or self.task.test_code
            passed, stdout, stderr, _ = execute_code(
                code, test_code,
                timeout_sec=self.timeout_sec, memory_mb=self.memory_mb,
            )
        return passed, stdout, stderr

    def _run_self_tests(self, code: str) -> Tuple[bool, str, str]:
        """Run code against self-generated test cases with majority vote."""
        results = []
        for tc in self.custom_test_cases:
            try:
                passed, stdout, stderr, _ = execute_code_stdio(
                    code, [tc.input_str], [tc.expected_output],
                    timeout_sec=self.timeout_sec, memory_mb=self.memory_mb,
                )
                results.append((passed, stdout, stderr))
            except Exception as e:
                results.append((False, "", str(e)))
        return self_verify_execute(results, self.majority_threshold)


class EmbedAdapter:
    """Adapts extract_embedding_urllib to V3 EmbedCallable."""

    def __init__(self, llama_url: str):
        self.llama_url = llama_url
        self.call_count = 0

    def __call__(self, text: str) -> List[float]:
        self.call_count += 1
        emb = extract_embedding_urllib(text, self.llama_url)
        if emb is None:
            raise RuntimeError("Embedding extraction failed")
        return emb


# --- V3 Pipeline Orchestrator -------------------------------------------------

class V3Pipeline:
    """Orchestrates all V3 features for a single task.

    This is the core: given a task, run it through the full V3
    cascade and return the result.
    """

    def __init__(self, runner: BenchmarkRunner, telemetry_dir: Path,
                 llama_url: str = LLAMA_URL,
                 enable_phase1: bool = True,
                 enable_phase2: bool = True,
                 enable_phase3: bool = True,
                 enable_feedback: bool = False):
        self.runner = runner
        self.telemetry_dir = telemetry_dir
        self.llama_url = llama_url
        self.enable_phase1 = enable_phase1
        self.enable_phase2 = enable_phase2
        self.enable_phase3 = enable_phase3

        # Read V3 config from atlas.conf (with defaults)
        self._v3_conf = self._load_v3_config()

        # Initialize V3 components
        self._init_phase1(telemetry_dir)
        self._init_phase2(telemetry_dir)
        self._init_phase3(telemetry_dir)
        self._init_feedback(telemetry_dir, enable_feedback)

    @staticmethod
    def _load_v3_config() -> Dict[str, str]:
        """Load V3-specific config values from atlas.conf."""
        v3 = {}
        try:
            conf = config._conf
            v3["bf_default_tier"] = conf.get(
                "ATLAS_V3_BUDGET_FORCING_DEFAULT_TIER", "standard",
            ).strip('"')
            v3["bf_max_wait"] = int(conf.get(
                "ATLAS_V3_BUDGET_FORCING_MAX_WAIT_INJECTIONS", "3",
            ))
            v3["ps_num_plans"] = int(conf.get(
                "ATLAS_V3_PLAN_SEARCH_NUM_PLANS", "3",
            ))
            v3["ba_default_k"] = int(conf.get(
                "ATLAS_V3_BLEND_ASC_DEFAULT_K", "3",
            ))
            v3["reasc_confidence"] = float(conf.get(
                "ATLAS_V3_REASC_CONFIDENCE_THRESHOLD", "-0.5",
            ))
            v3["reasc_energy"] = float(conf.get(
                "ATLAS_V3_REASC_ENERGY_THRESHOLD", "0.10",
            ))
            v3["s_star_delta"] = float(conf.get(
                "ATLAS_V3_S_STAR_ENERGY_DELTA", "1.0",
            ))
            v3["ewc_lambda"] = float(conf.get(
                "ATLAS_V3_EWC_LAMBDA", "1000.0",
            ))
            v3["replay_max_size"] = int(conf.get(
                "ATLAS_V3_REPLAY_BUFFER_MAX_SIZE", "5000",
            ))
            v3["replay_ratio"] = float(conf.get(
                "ATLAS_V3_REPLAY_BUFFER_REPLAY_RATIO", "0.30",
            ))
            v3["feedback_enabled"] = conf.get(
                "ATLAS_V3_LENS_FEEDBACK_ENABLED", "false",
            ).lower() in ("true", "1")
            v3["feedback_interval"] = int(conf.get(
                "ATLAS_V3_LENS_FEEDBACK_RETRAIN_INTERVAL", "50",
            ))
        except Exception:
            pass
        return v3

    def _init_phase1(self, telemetry_dir):
        self.budget_forcing = BudgetForcing(
            BudgetForcingConfig(
                enabled=self.enable_phase1,
                default_tier=self._v3_conf.get("bf_default_tier", "standard"),
                max_wait_injections=self._v3_conf.get("bf_max_wait", 3),
            ),
            telemetry_dir=telemetry_dir,
        )
        self.plan_search = PlanSearch(
            PlanSearchConfig(
                enabled=self.enable_phase1,
                num_plans=self._v3_conf.get("ps_num_plans", 3),
            ),
            telemetry_dir=telemetry_dir,
        )
        self.div_sampling = DivSampling(
            DivSamplingConfig(enabled=self.enable_phase1),
            telemetry_dir=telemetry_dir,
        )

    def _init_phase2(self, telemetry_dir):
        self.blend_asc = BlendASC(
            BlendASCConfig(
                enabled=self.enable_phase2,
                default_k=self._v3_conf.get("ba_default_k", 3),
            ),
            telemetry_dir=telemetry_dir,
        )
        self.reasc = ReASC(
            ReASCConfig(
                enabled=self.enable_phase2,
                confidence_threshold=self._v3_conf.get("reasc_confidence", -0.5),
                energy_threshold=self._v3_conf.get("reasc_energy", 0.10),
            ),
            telemetry_dir=telemetry_dir,
        )
        self.s_star = SStar(
            SStarConfig(
                enabled=self.enable_phase2,
                energy_delta=self._v3_conf.get("s_star_delta", 1.0),
            ),
            telemetry_dir=telemetry_dir,
        )

    def _init_phase3(self, telemetry_dir):
        fa_config = FailureAnalysisConfig(enabled=self.enable_phase3)
        cr_config = ConstraintRefinementConfig(enabled=self.enable_phase3)
        self.failure_analyzer = FailureAnalyzer(fa_config, telemetry_dir=telemetry_dir)
        self.constraint_refiner = ConstraintRefiner(cr_config, telemetry_dir=telemetry_dir)
        self.pr_cot = PRCoT(
            PRCoTConfig(enabled=self.enable_phase3),
            telemetry_dir=telemetry_dir,
        )
        self.refinement_loop = RefinementLoop(
            RefinementLoopConfig(enabled=self.enable_phase3),
            failure_analyzer=self.failure_analyzer,
            constraint_refiner=self.constraint_refiner,
            telemetry_dir=telemetry_dir,
        )
        self.derivation_chains = DerivationChains(
            DerivationChainsConfig(enabled=self.enable_phase3),
            telemetry_dir=telemetry_dir,
        )
        self.metacognitive = MetacognitiveProfile(
            MetacognitiveConfig(enabled=self.enable_phase3),
            telemetry_dir=telemetry_dir,
        )
        self.ace = ACEPipeline(
            ACEConfig(enabled=self.enable_phase3),
            telemetry_dir=telemetry_dir,
        )
        self.self_test_gen = SelfTestGen(
            SelfTestGenConfig(enabled=self.enable_phase3),
            telemetry_dir=telemetry_dir,
        )

    def _init_feedback(self, telemetry_dir, enable_feedback):
        self.lens_feedback = LensFeedbackCollector(
            LensFeedbackConfig(
                enabled=enable_feedback,
                retrain_interval=self._v3_conf.get("feedback_interval", 50),
                rag_api_url=RAG_API_URL,
            ),
            telemetry_dir=telemetry_dir,
        ) if enable_feedback else None

    def run_task(self, task: BenchmarkTask, task_id: str = "") -> Dict[str, Any]:
        """Run a single task through the full V3 pipeline.

        Returns a dict with:
          - passed: bool
          - code: str (winning code)
          - phase_solved: str ("phase1", "pr_cot", "refinement", "derivation", "none")
          - candidates_generated: int
          - total_tokens: int
          - total_time_ms: float
          - telemetry: dict (per-phase details)
        """
        start_time = time.time()
        task_id = task_id or task.task_id
        llm = LLMAdapter(self.runner)
        sandbox = SandboxAdapter(task)
        embed = EmbedAdapter(self.llama_url)

        result = {
            "task_id": task_id,
            "passed": False,
            "code": "",
            "phase_solved": "none",
            "candidates_generated": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "telemetry": {},
        }

        # ===== PROBE: Quick candidate for Lens energy estimation =====
        # Generate a single nothink candidate to get energy signal for Phase 2
        # adaptive K allocation and Budget Forcing tier selection.
        probe_candidate = None
        probe_energy_raw = None

        if self.enable_phase1:
            try:
                chatml = self.budget_forcing.format_chatml(task.prompt, "nothink")
                response, tokens, t_ms = llm(
                    chatml, BASE_TEMPERATURE, MAX_TOKENS, 42,
                )
                probe_code = extract_code(response)
                if probe_code:
                    try:
                        energy_raw, energy_norm = score_candidate(
                            probe_code, RAG_API_URL,
                        )
                        # Sentinel check: (0.0, 0.5) means Lens models
                        # not loaded. Leave probe_energy_raw as None so
                        # Phase 2 falls back to default k=3.
                        if not (energy_raw == 0.0 and energy_norm == 0.5):
                            probe_energy_raw = energy_raw
                    except Exception:
                        energy_raw, energy_norm = 0.0, 0.5
                    probe_candidate = {
                        "index": 0,
                        "code": probe_code,
                        "response": response,
                        "tokens": tokens,
                        "time_ms": t_ms,
                        "energy": energy_raw,
                        "energy_norm": energy_norm,
                        "passed": None,
                    }
                    result["total_tokens"] += tokens
            except LLMConnectionError:
                raise
            except Exception as e:
                result["telemetry"]["probe_error"] = str(e)

        # ===== Phase 2: Adaptive K + Budget Tier + Early Stopping =====
        if self.enable_phase2 and probe_energy_raw is not None:
            # Blend-ASC: determine how many candidates to generate
            k, budget_tier = self.blend_asc.allocate(
                raw_energy=probe_energy_raw,
                task_id=task_id,
                probe_tokens=(
                    probe_candidate.get("tokens", 0)
                    if probe_candidate else 0
                ),
                probe_time_ms=(
                    probe_candidate.get("time_ms", 0.0)
                    if probe_candidate else 0.0
                ),
            )
            # Budget Forcing: select thinking tier for remaining generations
            bf_tier = self.budget_forcing.select_tier(
                raw_energy=probe_energy_raw,
            )

            # ReASC: early stopping — if task is easy and model is confident,
            # skip generating more candidates (just use the probe)
            should_stop, reasc_reason = self.reasc.evaluate(
                probe_energy_raw, llm.last_logprobs, task_id=task_id,
            )
            if should_stop:
                k = 1
                result["telemetry"]["reasc_stopped"] = True
                result["telemetry"]["reasc_reason"] = reasc_reason
        else:
            k = 3
            budget_tier = "standard"
            bf_tier = self.budget_forcing.select_tier()

        result["telemetry"]["adaptive_k"] = k
        result["telemetry"]["budget_tier"] = budget_tier

        # ===== Phase 1: Build candidate pool =====
        candidates = []
        constraints = []

        # Include probe as first candidate
        if probe_candidate:
            candidates.append(probe_candidate)

        # Get ACE playbook context for this task
        ace_context = ""
        if self.enable_phase3:
            try:
                categories = self._infer_categories(task)
                ace_context = self.ace.get_context(categories, task_id=task_id)
            except Exception:
                pass

        # Generate constraint-diverse candidates via PlanSearch
        remaining_k = max(0, k - len(candidates))
        if self.enable_phase1 and remaining_k > 0:
            try:
                problem_with_context = task.prompt
                if ace_context:
                    problem_with_context = f"{task.prompt}\n\n{ace_context}"

                ps_result = self.plan_search.generate(
                    problem=problem_with_context, task_id=task_id,
                    llm_call=llm, num_plans=remaining_k,
                )
                for cs in ps_result.constraint_sets:
                    constraints.extend(cs.constraints)
                for i, code in enumerate(ps_result.candidates):
                    if not code:
                        continue
                    try:
                        energy_raw, energy_norm = score_candidate(
                            code, RAG_API_URL,
                        )
                    except Exception:
                        energy_raw, energy_norm = 0.0, 0.5
                    candidates.append({
                        "index": len(candidates),
                        "code": code,
                        "response": "",
                        "tokens": 0,
                        "time_ms": 0.0,
                        "energy": energy_raw,
                        "energy_norm": energy_norm,
                        "passed": None,
                    })
                result["total_tokens"] += ps_result.total_tokens
                result["telemetry"]["plansearch_tokens"] = ps_result.total_tokens
            except Exception as e:
                result["telemetry"]["plansearch_error"] = str(e)

        # Fill remaining slots with DivSampling + Budget Forcing
        if self.enable_phase1 and len(candidates) < k:
            for extra_idx in range(len(candidates), k):
                try:
                    perturbed = self.div_sampling.apply(
                        task.prompt, candidate_index=extra_idx,
                        task_id=task_id,
                    )
                    chatml = self.budget_forcing.format_chatml(
                        perturbed, bf_tier,
                    )
                    max_tok = self.budget_forcing.get_max_tokens(bf_tier)
                    response, tokens, t_ms = llm(
                        chatml, DIVERSITY_TEMPERATURE, max_tok,
                        42 + extra_idx,
                    )
                    code = extract_code(response)
                    if not code:
                        continue
                    try:
                        energy_raw, energy_norm = score_candidate(
                            code, RAG_API_URL,
                        )
                    except Exception:
                        energy_raw, energy_norm = 0.0, 0.5
                    candidates.append({
                        "index": len(candidates),
                        "code": code,
                        "response": response,
                        "tokens": tokens,
                        "time_ms": t_ms,
                        "energy": energy_raw,
                        "energy_norm": energy_norm,
                        "passed": None,
                    })
                    result["total_tokens"] += tokens
                except Exception:
                    continue

        # Fallback: if no candidates at all, direct generation
        if not candidates:
            try:
                response, tokens, t_ms = self.runner._call_llm(
                    task.prompt, temperature=BASE_TEMPERATURE,
                    max_tokens=MAX_TOKENS, seed=42,
                )
                code = extract_code(response)
                try:
                    energy_raw, energy_norm = score_candidate(
                        code, RAG_API_URL,
                    )
                except Exception:
                    energy_raw, energy_norm = 0.0, 0.5
                candidates.append({
                    "index": 0,
                    "code": code,
                    "response": response,
                    "tokens": tokens,
                    "time_ms": t_ms,
                    "energy": energy_raw,
                    "energy_norm": energy_norm,
                    "passed": None,
                })
                result["total_tokens"] += tokens
            except LLMConnectionError as e:
                result["telemetry"]["fallback_error"] = str(e)

        # Get metacognitive warnings (for Phase 3 reuse)
        metacog_warnings = []
        if self.enable_phase3:
            try:
                categories = self._infer_categories(task)
                metacog_warnings = self.metacognitive.get_warnings(
                    categories, task_id=task_id,
                )
            except Exception:
                pass

        result["candidates_generated"] = len(candidates)

        # ===== Test ALL candidates in sandbox =====
        candidates.sort(key=lambda c: c["energy"])
        passing_candidates = []

        for cand in candidates:
            if not cand["code"]:
                cand["passed"] = False
                continue
            try:
                passed, stdout, stderr = sandbox(cand["code"], "")
                cand["passed"] = passed
                cand["stdout"] = stdout or ""
                cand["stderr"] = stderr or ""
                if passed:
                    passing_candidates.append(cand)
            except Exception as e:
                cand["passed"] = False
                cand["stdout"] = ""
                cand["stderr"] = str(e)

        # Store best candidate code even on failure (for feedback + analysis)
        if candidates and not passing_candidates:
            result["code"] = candidates[0]["code"]  # Best by energy (sorted)

        # ===== Select best passing candidate (with S* tiebreaking) =====
        if passing_candidates:
            result["passed"] = True
            result["phase_solved"] = "phase1"

            if len(passing_candidates) >= 2 and self.enable_phase2:
                # S* tiebreaking: generate edge-case inputs to distinguish
                # the top-2 passing candidates by energy
                try:
                    s_candidates = [
                        CandidateScore(
                            code=c["code"], raw_energy=c["energy"],
                            index=c["index"],
                        )
                        for c in passing_candidates[:2]
                    ]
                    tb_result = self.s_star.tiebreak(
                        candidates=s_candidates, problem=task.prompt,
                        llm_call=llm, sandbox_run=sandbox,
                        task_id=task_id,
                    )
                    if tb_result.triggered and tb_result.winner_index >= 0:
                        winner = next(
                            (c for c in passing_candidates
                             if c["index"] == tb_result.winner_index),
                            passing_candidates[0],
                        )
                        result["code"] = winner["code"]
                        result["telemetry"]["s_star_triggered"] = True
                    else:
                        result["code"] = passing_candidates[0]["code"]
                except Exception:
                    result["code"] = passing_candidates[0]["code"]
            else:
                result["code"] = passing_candidates[0]["code"]

            result["total_tokens"] = llm.total_tokens
            result["total_time_ms"] = (time.time() - start_time) * 1000
            self._record_feedback(task_id, result)
            self._log_v3_event(task_id, result)
            return result

        # ===== Phase 3: Refinement cascade =====
        if not self.enable_phase3:
            result["total_time_ms"] = (time.time() - start_time) * 1000
            self._record_feedback(task_id, result)
            self._log_v3_event(task_id, result)
            return result

        # Build failing candidates list for Phase 3 (with actual error output)
        failing = [
            FailingCandidate(
                code=c["code"],
                error_output=c.get("stderr", "") or c.get("stdout", ""),
                index=c["index"],
            )
            for c in candidates if c.get("passed") is False and c["code"]
        ]

        # --- Self-Test Generation (generate ONCE, cache for all iterations) ---
        self_tests = self.self_test_gen.generate(
            problem=task.prompt, llm_call=llm, task_id=task_id,
        )
        result["telemetry"]["self_tests_generated"] = len(self_tests.test_cases)

        # Create self-verify sandbox if we have self-tests
        if self_tests.test_cases:
            self_verify_sandbox = SandboxAdapter(
                task, self_verify_mode=True,
                custom_test_cases=self_tests.test_cases,
                majority_threshold=self.self_test_gen.config.majority_threshold,
            )
        else:
            # Fallback: no self-tests generated, use real sandbox
            # (this is a degraded mode, logged for analysis)
            self_verify_sandbox = sandbox
            result["telemetry"]["self_test_fallback"] = True

        # Step 3a: PR-CoT Quick Repair (uses self-verify sandbox)
        if failing:
            try:
                best_failing = failing[0]
                error_msg = best_failing.error_output or "All test cases failed"
                repair_result = self.pr_cot.repair(
                    problem=task.prompt,
                    code=best_failing.code,
                    error=error_msg,
                    llm_call=llm,
                    task_id=task_id,
                )
                for repair_code in repair_result.repairs:
                    if not repair_code:
                        continue
                    try:
                        passed, stdout, stderr = self_verify_sandbox(repair_code, "")
                        if passed:
                            # Self-tests pass — verify with real tests for final score
                            real_passed, _, _ = sandbox(repair_code, "")
                            if real_passed:
                                result["passed"] = True
                                result["code"] = repair_code
                                result["phase_solved"] = "pr_cot"
                                break
                    except Exception:
                        continue
            except Exception as e:
                result["telemetry"]["pr_cot_error"] = str(e)

        if result["passed"]:
            self._learn_from_success(task, task_id, "pr_cot")
            result["total_tokens"] = llm.total_tokens
            result["total_time_ms"] = (time.time() - start_time) * 1000
            self._record_feedback(task_id, result)
            self._log_v3_event(task_id, result)
            return result

        # Step 3b: Full Refinement Loop (uses self-verify sandbox)
        try:
            ref_result = self.refinement_loop.run(
                problem=task.prompt,
                failing_candidates=failing,
                original_constraints=constraints,
                llm_call=llm,
                sandbox_run=self_verify_sandbox,  # Self-verify, not real tests
                embed_call=embed,
                metacognitive_warnings=metacog_warnings,
                task_id=task_id,
            )
            if ref_result.solved:
                # Self-tests pass — verify with real tests for final score
                real_passed, _, _ = sandbox(ref_result.winning_code, "")
                if real_passed:
                    result["passed"] = True
                    result["code"] = ref_result.winning_code
                    result["phase_solved"] = "refinement"
                    result["telemetry"]["refinement_iterations"] = ref_result.total_iterations
        except Exception as e:
            result["telemetry"]["refinement_error"] = str(e)

        if result["passed"]:
            self._learn_from_success(task, task_id, "refinement")
            result["total_tokens"] = llm.total_tokens
            result["total_time_ms"] = (time.time() - start_time) * 1000
            self._record_feedback(task_id, result)
            self._log_v3_event(task_id, result)
            return result

        # Step 3c: Derivation Chains (real sandbox — sub-problems have own test cases)
        try:
            failure_context = "; ".join(
                f"Candidate {c['index']}: {c.get('stderr', 'failed')[:200]}"
                for c in candidates if c.get("passed") is False
            )
            dc_result = self.derivation_chains.solve(
                problem=task.prompt,
                failure_context=failure_context,
                llm_call=llm,
                sandbox_run=sandbox,
                task_id=task_id,
            )
            if dc_result.solved and dc_result.final_code:
                try:
                    # Verify with real tests for final score
                    passed, stdout, stderr = sandbox(dc_result.final_code, "")
                    if passed:
                        result["passed"] = True
                        result["code"] = dc_result.final_code
                        result["phase_solved"] = "derivation"
                except Exception:
                    pass
        except Exception as e:
            result["telemetry"]["derivation_error"] = str(e)

        if result["passed"]:
            self._learn_from_success(task, task_id, "derivation")

        result["total_tokens"] = llm.total_tokens
        result["total_time_ms"] = (time.time() - start_time) * 1000
        self._record_feedback(task_id, result)
        self._log_v3_event(task_id, result)
        return result

    def _record_feedback(self, task_id: str, result: Dict) -> None:
        """Record pass/fail embedding for Lens feedback loop."""
        if not self.lens_feedback or not self.lens_feedback.config.enabled:
            return
        code = result.get("code", "")
        if not code:
            return
        try:
            embed = EmbedAdapter(self.llama_url)
            embedding = embed(code)
            label = "PASS" if result.get("passed") else "FAIL"
            self.lens_feedback.record(embedding, label, task_id)
            if self.lens_feedback.needs_propagation:
                self.lens_feedback.apply_to_components(
                    self.blend_asc, self.budget_forcing,
                )
        except Exception:
            pass  # Never crash benchmark for feedback

    def _learn_from_success(self, task: BenchmarkTask,
                            task_id: str, method: str) -> None:
        """Extract and store a principle from a successfully solved task."""
        try:
            categories = self._infer_categories(task)
            category = categories[0] if categories else ""

            # Check if this relates to existing principles
            related = self.ace.find_related(
                f"Solved via {method}", categories,
            )

            if len(related) >= 2:
                # Derive a composed principle from related ones
                self.ace.derive(
                    parent_ids=[r.entry_id for r in related[:3]],
                    new_principle=f"Solved via {method}: {task_id} (builds on {category} principles)",
                    category=category,
                    task_id=task_id,
                )
            else:
                self.ace.learn(
                    principle=f"Solved via {method}: {task_id}",
                    category=category,
                    task_id=task_id,
                )
        except Exception:
            pass

    def _build_generation_prompt(self, task: BenchmarkTask,
                                  constraints: List[str],
                                  metacog_warnings: List[str],
                                  ace_context: str,
                                  candidate_index: int) -> str:
        """Build a generation prompt with V3 enhancements."""
        parts = [task.prompt]

        if constraints:
            parts.append("\n\nIMPORTANT constraints to satisfy:")
            for c in constraints:
                parts.append(f"- {c}")

        if metacog_warnings:
            parts.append("\n\nKnown pitfalls for this problem type:")
            for w in metacog_warnings:
                parts.append(f"- {w}")

        if ace_context:
            parts.append(f"\n\n{ace_context}")

        return '\n'.join(parts)

    def _infer_categories(self, task: BenchmarkTask) -> List[str]:
        """Infer problem categories from task metadata."""
        categories = []
        prompt_lower = task.prompt.lower()

        if any(w in prompt_lower for w in ["sort", "binary search", "heap"]):
            categories.append("sorting_searching")
        if any(w in prompt_lower for w in ["graph", "tree", "bfs", "dfs", "node"]):
            categories.append("graph_theory")
        if any(w in prompt_lower for w in ["dynamic programming", "dp", "memoiz"]):
            categories.append("dynamic_programming")
        if any(w in prompt_lower for w in ["string", "substring", "palindrome"]):
            categories.append("string_processing")
        if any(w in prompt_lower for w in ["bit", "xor", "bitwise", "shift"]):
            categories.append("bitwise")
        if any(w in prompt_lower for w in ["math", "prime", "gcd", "modulo"]):
            categories.append("mathematics")

        if not categories:
            categories.append("general")

        return categories

    def _log_v3_event(self, task_id: str, result: Dict) -> None:
        """Log a V3 pipeline event to JSONL."""
        event = {
            "task_id": task_id,
            "passed": result["passed"],
            "phase_solved": result["phase_solved"],
            "candidates_generated": result["candidates_generated"],
            "total_tokens": result["total_tokens"],
            "total_time_ms": result["total_time_ms"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            append_jsonl(self.telemetry_dir / "v3_events.jsonl", event)
        except Exception:
            pass

    def collect_benchmark_results(self, results: Dict[str, Dict]) -> None:
        """Post-benchmark: feed results to metacognitive + ACE."""
        benchmark_results = []
        for task_id, r in results.items():
            categories = r.get("telemetry", {}).get("categories", ["general"])
            benchmark_results.append(BenchmarkResult(
                task_id=task_id,
                category=categories[0] if categories else "general",
                passed=r["passed"],
                code=r.get("code", ""),
            ))

        # Metacognitive analysis
        try:
            llm = LLMAdapter(self.runner)
            self.metacognitive.analyze_benchmark(benchmark_results, llm_call=llm)
        except Exception:
            pass


# --- V3 Benchmark Runner -------------------------------------------------------

class V3BenchmarkRunner:
    """Runs V3 benchmark with full pipeline."""

    def __init__(self, run_dir: Path, enable_phase1=True,
                 enable_phase2=True, enable_phase3=True,
                 enable_feedback=False):
        self.run_dir = Path(run_dir)
        self.telemetry_dir = self.run_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.runner = BenchmarkRunner(max_retries=10)
        self.pipeline = V3Pipeline(
            self.runner, self.telemetry_dir,
            enable_phase1=enable_phase1,
            enable_phase2=enable_phase2,
            enable_phase3=enable_phase3,
            enable_feedback=enable_feedback,
        )
        self._start_time = time.time()

    def close(self):
        self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def run_lcb(self, tasks: List[BenchmarkTask],
                phase_name: str = "v3_lcb") -> Dict[str, Dict]:
        """Run LiveCodeBench tasks through V3 pipeline."""
        phase_dir = self.run_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        per_task_dir = phase_dir / "per_task"
        per_task_dir.mkdir(parents=True, exist_ok=True)

        completed = find_completed_tasks(phase_dir)
        remaining = [t for t in tasks if t.task_id not in completed]
        total = len(tasks)
        done = len(completed)

        if completed:
            print(f"  Resuming: {done}/{total} complete, {len(remaining)} remaining")

        # Load already-completed results
        results: Dict[str, Dict] = {}
        for f in per_task_dir.glob("*.json"):
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    results[data['task_id']] = data
            except Exception:
                pass

        for idx, task in enumerate(remaining):
            task_start = time.time()

            try:
                task_result = self.pipeline.run_task(task, task_id=task.task_id)
            except Exception as e:
                task_result = {
                    "task_id": task.task_id,
                    "passed": False,
                    "code": "",
                    "phase_solved": "error",
                    "candidates_generated": 0,
                    "total_tokens": 0,
                    "total_time_ms": (time.time() - task_start) * 1000,
                    "error": str(e),
                    "telemetry": {},
                }

            results[task.task_id] = task_result

            # Save per-task result atomically
            safe_name = task.task_id.replace('/', '_')
            atomic_write_json(per_task_dir / f"{safe_name}.json", task_result)

            done += 1
            status = "PASS" if task_result["passed"] else "FAIL"
            phase = task_result.get("phase_solved", "?")
            elapsed = time.time() - self._start_time
            rate = done / (elapsed / 3600) if elapsed > 0 else 0
            tokens = task_result.get("total_tokens", 0)
            print(
                f"  [{done}/{total}] {task.task_id}: {status} "
                f"(via {phase}, {tokens} tok) "
                f"[{rate:.0f} tasks/hr]",
                flush=True,
            )

        # Save phase summary
        passed = sum(1 for r in results.values() if r.get("passed"))
        summary = {
            "phase": phase_name,
            "total_tasks": len(results),
            "passed_tasks": passed,
            "pass_rate": passed / max(len(results), 1),
            "phase_breakdown": self._phase_breakdown(results),
        }
        atomic_write_json(phase_dir / "results.json", summary)

        return results

    def _phase_breakdown(self, results: Dict[str, Dict]) -> Dict:
        """Compute breakdown of which phase solved each task."""
        breakdown = {
            "phase1": 0, "pr_cot": 0, "refinement": 0,
            "derivation": 0, "none": 0, "error": 0,
        }
        for r in results.values():
            phase = r.get("phase_solved", "none")
            if phase in breakdown:
                breakdown[phase] += 1
            else:
                breakdown["none"] += 1
        return breakdown


# --- Main Entry Point ----------------------------------------------------------

def load_lcb_tasks():
    """Load LiveCodeBench dataset."""
    from benchmark.datasets import LiveCodeBenchDataset
    ds = LiveCodeBenchDataset()
    ds.load()
    return ds.tasks


def run_v3_benchmark(run_id=None, smoke_only=False, max_tasks=None,
                     enable_phase1=True, enable_phase2=True,
                     enable_phase3=True):
    """Run V3 benchmark on LiveCodeBench."""
    if run_id is None:
        run_id = f"v3_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = config.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    meta = {
        "run_id": run_id,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "version": "v3",
        "enable_phase1": enable_phase1,
        "enable_phase2": enable_phase2,
        "enable_phase3": enable_phase3,
        "smoke_only": smoke_only,
        "max_tasks": max_tasks,
    }
    atomic_write_json(run_dir / "run_meta.json", meta)

    print("=" * 60)
    print(f"  ATLAS V3 Benchmark")
    print(f"  Run ID: {run_id}")
    print(f"  Results: {run_dir}")
    print(f"  Phase 1: {'ON' if enable_phase1 else 'OFF'}")
    print(f"  Phase 2: {'ON' if enable_phase2 else 'OFF'}")
    print(f"  Phase 3: {'ON' if enable_phase3 else 'OFF'}")
    print("=" * 60)

    # Pre-flight checks
    print("\nPre-flight checks...")
    try:
        req = urllib.request.Request(f"{LLAMA_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        print(f"  llama-server: OK ({data.get('status', '?')})")
    except Exception as e:
        print(f"  llama-server: FAILED ({e})")
        print("  Aborting benchmark — llama-server not reachable")
        return None

    try:
        req = urllib.request.Request(f"{RAG_API_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        print(f"  RAG API: OK ({data.get('status', '?')})")
    except Exception:
        print("  RAG API: WARNING — lens scoring unavailable")

    # Check Lens model availability
    try:
        test_body = json.dumps({"text": "test"}).encode("utf-8")
        req = urllib.request.Request(
            f"{RAG_API_URL}/internal/lens/score-text",
            data=test_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            lens_data = json.loads(resp.read().decode('utf-8'))
        if lens_data.get("error"):
            print(f"  Lens model: NOT LOADED ({lens_data['error']})")
            print("    Phase 2 (adaptive K) will use default k=3")
        else:
            print(f"  Lens model: OK (energy={lens_data.get('energy', '?')})")
    except Exception:
        print("  Lens model: UNAVAILABLE — Phase 2 will use default k=3")

    # Load dataset
    print("\nLoading LiveCodeBench...", end=" ", flush=True)
    tasks = load_lcb_tasks()
    print(f"{len(tasks)} tasks")

    if smoke_only:
        tasks = tasks[:10]
        print(f"  SMOKE MODE: running {len(tasks)} tasks only")
    elif max_tasks:
        tasks = tasks[:max_tasks]
        print(f"  LIMITED MODE: running {len(tasks)} tasks")

    # Run benchmark
    print(f"\nRunning V3 pipeline on {len(tasks)} tasks...")
    print("-" * 60)

    with V3BenchmarkRunner(
        run_dir,
        enable_phase1=enable_phase1,
        enable_phase2=enable_phase2,
        enable_phase3=enable_phase3,
    ) as runner:
        results = runner.run_lcb(tasks)

        # Post-benchmark analysis
        print("\n" + "-" * 60)
        print("Post-benchmark analysis...")
        runner.pipeline.collect_benchmark_results(results)

    # Summary
    passed = sum(1 for r in results.values() if r.get("passed"))
    total = len(results)
    rate = passed / max(total, 1)

    # Phase breakdown
    breakdown = {}
    for r in results.values():
        phase = r.get("phase_solved", "none")
        breakdown[phase] = breakdown.get(phase, 0) + 1

    print("\n" + "=" * 60)
    print(f"  V3 BENCHMARK COMPLETE")
    print(f"  pass@1: {passed}/{total} ({rate*100:.1f}%)")
    print(f"  Solved by:")
    for phase, count in sorted(breakdown.items()):
        print(f"    {phase}: {count}")
    print(f"  Results: {run_dir}")
    print("=" * 60)

    # Update metadata
    meta["end_time"] = datetime.now(timezone.utc).isoformat()
    meta["total_tasks"] = total
    meta["passed_tasks"] = passed
    meta["pass_rate"] = rate
    meta["phase_breakdown"] = breakdown
    atomic_write_json(run_dir / "run_meta.json", meta)

    return run_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS V3 Benchmark Runner")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test (10 tasks only)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks")
    parser.add_argument("--no-phase1", action="store_true",
                        help="Disable Phase 1 features")
    parser.add_argument("--no-phase2", action="store_true",
                        help="Disable Phase 2 features")
    parser.add_argument("--no-phase3", action="store_true",
                        help="Disable Phase 3 features")
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline mode: all V3 features OFF (equivalent to V2)")
    args = parser.parse_args()

    if args.baseline:
        args.no_phase1 = True
        args.no_phase2 = True
        args.no_phase3 = True

    run_dir = run_v3_benchmark(
        run_id=args.run_id,
        smoke_only=args.smoke,
        max_tasks=args.max_tasks,
        enable_phase1=not args.no_phase1,
        enable_phase2=not args.no_phase2,
        enable_phase3=not args.no_phase3,
    )

    if run_dir:
        print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
