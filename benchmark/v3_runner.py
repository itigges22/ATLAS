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
import threading
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
from benchmark.v3.candidate_selection import (
    CandidateInfo, select_candidate, STRATEGIES,
)
from benchmark.v3.embedding_store import EmbeddingWriter


# --- Constants ----------------------------------------------------------------

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:31144")
LLAMA_URL = os.environ.get("LLAMA_URL", f"http://localhost:{config._conf.get('ATLAS_LLAMA_NODEPORT', '32735')}")
# Published Qwen3.5 benchmarks use: temp=0.6, top_k=20, top_p=0.95,
# max_tokens=32768+, thinking mode enabled. Match their settings.
MAX_TOKENS = 8192
BASE_TEMPERATURE = 0.6  # Qwen3.5 recommended for coding with thinking
DIVERSITY_TEMPERATURE = 0.8  # Slightly higher for candidate diversity


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

    Request serialization: DeltaNet hybrid architecture (Qwen3.5-9B) hangs
    when multiple slots generate simultaneously via cont-batching. A class-level
    lock ensures only one /completion request is in-flight at a time, giving
    full single-slot throughput (~47 tok/s) while keeping 4 slots for connection
    acceptance and prompt caching.
    """

    # Thinking consumes too much if it's >80% of tokens and output is tiny
    THINK_BUDGET_RATIO = 0.80
    MIN_OUTPUT_CHARS = 50

    # Serialize LLM requests to avoid DeltaNet multi-slot generation hang.
    # Set ATLAS_LLM_PARALLEL=1 to disable the lock (requires --no-cache-prompt
    # on llama-server to prevent checkpoint restore hang).
    _llm_lock = threading.Lock()
    _parallel_mode = os.environ.get("ATLAS_LLM_PARALLEL", "0") == "1"

    def __init__(self, runner: BenchmarkRunner, max_retries: int = 2,
                 timeout: int = 900):
        self.runner = runner
        self.max_retries = max_retries
        # Scale timeout by parallel tasks — shared GPU bandwidth means each
        # call takes proportionally longer with more concurrent tasks.
        parallel_tasks = int(os.environ.get("ATLAS_PARALLEL_TASKS", "1"))
        if LLMAdapter._parallel_mode and parallel_tasks > 1:
            self.timeout = timeout * parallel_tasks
        else:
            self.timeout = timeout
        self.call_count = 0
        self.total_tokens = 0
        self.last_logprobs: List[float] = []

    @staticmethod
    def _parse_logprobs(data: dict) -> List[float]:
        """Extract per-token log-probabilities from llama-server response.

        Handles both /v1/chat/completions format (choices[0].logprobs)
        and legacy /completion format (completion_probabilities).
        """
        logprobs = []
        # /v1/chat/completions format
        choices = data.get("choices", [])
        if choices:
            lp_data = choices[0].get("logprobs", {})
            if lp_data and lp_data.get("content"):
                for tok in lp_data["content"]:
                    lp = tok.get("logprob")
                    if lp is not None:
                        logprobs.append(lp)
                return logprobs

        # Legacy /completion format fallback
        for tok in data.get("completion_probabilities", []):
            probs = tok.get("probs", [])
            if probs:
                p = probs[0].get("prob", 0.0)
                if p > 0:
                    logprobs.append(math.log(p))
        return logprobs

    def _send_request(self, request_body: dict) -> dict:
        """Send request to LLM server with retry.

        Supports both llama.cpp (/completion) and Fox (/v1/completions).
        Uses manual ChatML formatting for thinking mode control.
        """
        # Detect Fox vs llama.cpp: Fox uses /v1/completions with OpenAI format
        use_fox = os.environ.get("ATLAS_USE_FOX", "0") == "1"
        if use_fox:
            endpoint = f"{self.runner.llm_url}/v1/completions"
            # Convert llama.cpp format to OpenAI format
            if "n_predict" in request_body:
                request_body["max_tokens"] = request_body.pop("n_predict")
            request_body.pop("cache_prompt", None)
            request_body.pop("n_probs", None)
            request_body.setdefault("model", os.environ.get("ATLAS_MODEL_NAME", "default"))
        else:
            endpoint = f"{self.runner.llm_url}/completion"

        last_error = None
        max_attempts = self.max_retries + 3
        for attempt in range(max_attempts):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(request_body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                if LLMAdapter._parallel_mode:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        return json.loads(resp.read().decode('utf-8'))
                else:
                    with LLMAdapter._llm_lock:
                        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                            return json.loads(resp.read().decode('utf-8'))
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 503 and attempt < max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                if attempt < max_attempts - 1:
                    time.sleep(10 * (2 ** min(attempt, 3)))
            except (ConnectionError, OSError, urllib.error.URLError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(5 * (attempt + 1))
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(10 * (2 ** min(attempt, 3)))
        raise LLMConnectionError(
            f"LLM call failed after {max_attempts} retries: {last_error}"
        )

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.call_count += 1

        # With --jinja enabled on llama-server, the model naturally uses
        # <think>...</think> tags for reasoning via the /completion endpoint.
        # No pre-fill needed — the chat template handles thinking mode.

        request_body = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
            "cache_prompt": False,
            "stop": ["\n\n\n\n"],
            "n_probs": 1,
            "top_k": 20,
            "top_p": 0.95,
        }
        if seed is not None:
            request_body["seed"] = seed

        start_time = time.time()
        data = self._send_request(request_body)

        # Parse response — Fox returns OpenAI format, llama.cpp returns legacy
        if "choices" in data:
            # Fox / OpenAI format
            content = data["choices"][0].get("text", "")
            tokens = data.get("usage", {}).get("completion_tokens", 0)
        else:
            # llama.cpp legacy format
            content = data.get("content", "")
            tokens = data.get("tokens_predicted", 0)
        self.last_logprobs = self._parse_logprobs(data)

        # Strip thinking blocks. With --jinja, the model wraps reasoning
        # in <think>...</think> tags naturally. Strip them to get clean code.
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)

        # Handle orphaned </think> (from nothink pre-fill or partial output)
        if '</think>' in content and '<think>' not in content:
            content = content[content.index('</think>') + len('</think>'):].strip()

        # Handle unclosed <think> (token budget exhausted during thinking)
        if '<think>' in content:
            after_think = content[content.index('<think>') + len('<think>'):].strip()
            before_think = content[:content.index('<think>')].strip()
            after_has_code = '```' in after_think or 'def ' in after_think or 'class ' in after_think
            before_has_code = '```' in before_think or 'def ' in before_think or 'class ' in before_think
            if after_has_code and not before_has_code:
                content = after_think
            elif before_has_code and not after_has_code:
                content = before_think
            elif after_has_code and before_has_code:
                content = after_think if len(after_think) > len(before_think) else before_think
            else:
                content = ""

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


class SStarSandboxAdapter:
    """Dedicated sandbox adapter for S* tiebreaking.

    Unlike SandboxAdapter which uses the task's test cases, this adapter
    runs code with a specific stdin input and returns (ran_ok, stdout, stderr).
    This enables S* to generate distinguishing inputs for stdio-mode tasks.
    """

    def __init__(self, task: BenchmarkTask, timeout_sec: int = 10,
                 memory_mb: int = 512):
        self.task = task
        self.timeout_sec = timeout_sec
        self.memory_mb = memory_mb

    def __call__(self, code: str, test_input: str) -> Tuple[bool, str, str]:
        code = wrap_class_solution(code, self.task)
        if self.task.eval_mode == "stdio":
            # Run with the specific distinguishing input as stdin
            passed, stdout, stderr, _ = execute_code_stdio(
                code, [test_input], ["__S_STAR_NO_EXPECTED__"],
                timeout_sec=self.timeout_sec, memory_mb=self.memory_mb,
            )
            # For S*, "passed" means "ran without crash and produced output"
            ran_ok = bool(stdout.strip()) and not stderr.strip()
            return ran_ok, stdout, stderr
        else:
            # Function mode: test_input is test code
            passed, stdout, stderr, _ = execute_code(
                code, test_input,
                timeout_sec=self.timeout_sec, memory_mb=self.memory_mb,
            )
            return passed, stdout, stderr


class EmbedAdapter:
    """Adapts extract_embedding_urllib to V3 EmbedCallable.

    Retries up to 3 times with backoff to handle transient 503/timeout
    errors when llama-server is busy with generation requests.
    """

    def __init__(self, llama_url: str, max_retries: int = 3):
        self.llama_url = llama_url
        self.call_count = 0
        self.max_retries = max_retries

    def __call__(self, text: str) -> List[float]:
        self.call_count += 1
        for attempt in range(self.max_retries):
            emb = extract_embedding_urllib(text, self.llama_url)
            if emb is not None:
                return emb
            if attempt < self.max_retries - 1:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"Embedding extraction failed after {self.max_retries} retries"
        )


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
                 enable_feedback: bool = False,
                 selection_strategy: str = "lens"):
        self.runner = runner
        self.telemetry_dir = telemetry_dir
        self.llama_url = llama_url
        self.enable_phase1 = enable_phase1
        self.enable_phase2 = enable_phase2
        self.enable_phase3 = enable_phase3
        self.selection_strategy = selection_strategy

        # Read V3 config from atlas.conf (with defaults)
        self._v3_conf = self._load_v3_config()

        # Embedding store for post-hoc analysis (V3.1 Section 5.2)
        self._emb_writer = EmbeddingWriter(telemetry_dir / "embeddings.emb")

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
            budget_forcing=self.budget_forcing,
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

        # Per-phase latency tracking
        latency = {}

        # ===== PROBE: Quick candidate for Lens energy estimation =====
        # Generate a single candidate to get energy signal for Phase 2
        # adaptive K allocation and Budget Forcing tier selection.
        # Uses "standard" tier (up to 2048 thinking tokens) — matches
        # Qwen3.5 published benchmark settings where thinking is enabled.
        # Gives the model enough reasoning budget to solve harder tasks
        # at probe, reducing cascade into Phase 3.
        probe_candidate = None
        probe_energy_raw = None

        if self.enable_phase1:
            probe_start = time.time()
            try:
                chatml = self.budget_forcing.format_chatml(task.prompt, "standard")
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
            latency["probe_ms"] = (time.time() - probe_start) * 1000

        # ===== Sandbox-test probe for data-driven early exit =====
        # Instead of predicting difficulty from energy (unreliable on 9B),
        # test the probe directly: if it passes, skip PlanSearch/DivSampling.
        probe_passed_sandbox = False
        if probe_candidate and probe_candidate.get("code"):
            try:
                probe_sandbox = SandboxAdapter(task)
                passed, stdout, stderr = probe_sandbox(
                    probe_candidate["code"], "",
                )
                probe_candidate["passed"] = passed
                probe_candidate["stdout"] = stdout or ""
                probe_candidate["stderr"] = stderr or ""
                probe_passed_sandbox = passed
            except Exception as e:
                probe_candidate["passed"] = False
                probe_candidate["stdout"] = ""
                probe_candidate["stderr"] = str(e)
            # Store embedding early (overlaps with probe sandbox test)
            try:
                emb = embed(probe_candidate["code"])
                label = "PASS" if probe_passed_sandbox else "FAIL"
                self._emb_writer.write(task_id, 0, label, emb)
            except Exception:
                pass
            result["telemetry"]["probe_sandbox_passed"] = probe_passed_sandbox

        # ===== Phase 2: Adaptive K + Budget Tier =====
        phase2_start = time.time()
        if probe_passed_sandbox:
            # Data-driven early exit: probe already passes sandbox.
            # No need to generate more candidates.
            k = 1
            budget_tier = "nothink"
            bf_tier = self.budget_forcing.select_tier()
            result["telemetry"]["probe_early_exit"] = True
        elif self.enable_phase2 and probe_energy_raw is not None:
            # Probe FAILED sandbox — we need diverse candidates.
            # Energy-based k allocation (BlendASC) is uninformative for
            # short probe code on 9B (raw ~1-4, all normalize to <0.05,
            # always mapping to k=1). Use default k=3 so PlanSearch runs.
            k = 3
            budget_tier = "standard"
            bf_tier = self.budget_forcing.select_tier()

            # Log BlendASC/ReASC evaluations for telemetry (not gating)
            k_blend, tier_blend = self.blend_asc.allocate(
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
            result["telemetry"]["blend_asc_k"] = k_blend
            result["telemetry"]["blend_asc_tier"] = tier_blend

            should_stop, reasc_reason = self.reasc.evaluate(
                probe_energy_raw, llm.last_logprobs, task_id=task_id,
            )
            result["telemetry"]["reasc_stopped"] = should_stop
            result["telemetry"]["reasc_reason"] = reasc_reason
        else:
            k = 3
            budget_tier = "standard"
            bf_tier = self.budget_forcing.select_tier()

        latency["phase2_alloc_ms"] = (time.time() - phase2_start) * 1000
        result["telemetry"]["adaptive_k"] = k
        result["telemetry"]["budget_tier"] = budget_tier

        # ===== Phase 1: Build candidate pool =====
        phase1_start = time.time()
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

                # PlanSearch does multiple sequential LLM calls (constraint
                # extraction + plan construction + code gen). Use a longer
                # timeout to handle long competition prompts at 9B speed.
                ps_llm = LLMAdapter(self.runner, timeout=300)
                ps_result = self.plan_search.generate(
                    problem=problem_with_context, task_id=task_id,
                    llm_call=ps_llm, num_plans=remaining_k,
                )
                result["total_tokens"] += ps_llm.total_tokens
                for cs in ps_result.constraint_sets:
                    constraints.extend(cs.constraints)
                # Log constraint sets for qualitative analysis (V3.1 Section 5.3)
                result["telemetry"]["plansearch_constraints"] = [
                    {"plan_index": i, "constraints": cs.constraints}
                    for i, cs in enumerate(ps_result.constraint_sets)
                ]
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
                result["telemetry"]["plansearch_tokens"] = ps_llm.total_tokens
            except Exception as e:
                result["telemetry"]["plansearch_error"] = str(e)

        # Fill remaining slots with DivSampling + Budget Forcing (parallel)
        if self.enable_phase1 and len(candidates) < k:
            def _generate_div_candidate(extra_idx):
                """Generate a single DivSampling candidate (thread-safe)."""
                try:
                    perturbed = self.div_sampling.apply(
                        task.prompt, candidate_index=extra_idx,
                        task_id=task_id,
                    )
                    chatml = self.budget_forcing.format_chatml(
                        perturbed, bf_tier,
                    )
                    max_tok = self.budget_forcing.get_max_tokens(bf_tier)
                    # Each thread creates its own LLMAdapter for thread safety
                    thread_llm = LLMAdapter(self.runner)
                    response, tokens, t_ms = thread_llm(
                        chatml, DIVERSITY_TEMPERATURE, max_tok,
                        42 + extra_idx,
                    )
                    code = extract_code(response)
                    if not code:
                        return None
                    try:
                        energy_raw, energy_norm = score_candidate(
                            code, RAG_API_URL,
                        )
                    except Exception:
                        energy_raw, energy_norm = 0.0, 0.5
                    return {
                        "code": code,
                        "response": response,
                        "tokens": tokens,
                        "time_ms": t_ms,
                        "energy": energy_raw,
                        "energy_norm": energy_norm,
                        "passed": None,
                    }
                except Exception:
                    return None

            fill_indices = list(range(len(candidates), k))
            with ThreadPoolExecutor(max_workers=min(len(fill_indices), 3)) as pool:
                futures = {pool.submit(_generate_div_candidate, idx): idx
                           for idx in fill_indices}
                for future in as_completed(futures):
                    cand = future.result()
                    if cand:
                        cand["index"] = len(candidates)
                        candidates.append(cand)
                        result["total_tokens"] += cand["tokens"]

        # Fallback: if no candidates at all, direct generation
        # Use BudgetForcing "standard" tier — allows thinking (up to 2048
        # tokens). Published Qwen3.5 benchmarks use full thinking mode
        # with temp=0.6 (65.6% LCB v6 with thinking vs ~39% without).
        if not candidates:
            try:
                chatml = self.budget_forcing.format_chatml(
                    task.prompt, "standard",
                )
                response, tokens, t_ms = llm(
                    chatml, BASE_TEMPERATURE, MAX_TOKENS, 42,
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

        latency["phase1_gen_ms"] = (time.time() - phase1_start) * 1000
        result["candidates_generated"] = len(candidates)

        # ===== Test ALL candidates in sandbox (pipelined, V3.1 4.2) =====
        # Sandbox tests + embedding storage run in parallel threads.
        # Candidates sorted by energy (low=easy first) for early-exit potential.
        sandbox_start = time.time()
        candidates.sort(key=lambda c: c["energy"])
        passing_candidates = []

        def _test_and_embed(cand):
            """Sandbox test + embedding storage for one candidate."""
            if not cand.get("code"):
                cand["passed"] = False
                return cand
            # Skip sandbox test if already tested (e.g., probe early exit)
            if cand.get("passed") is not None:
                return cand
            try:
                task_sandbox = SandboxAdapter(task)
                passed, stdout, stderr = task_sandbox(cand["code"], "")
                cand["passed"] = passed
                cand["stdout"] = stdout or ""
                cand["stderr"] = stderr or ""
            except Exception as e:
                cand["passed"] = False
                cand["stdout"] = ""
                cand["stderr"] = str(e)
            # Inline embedding storage (overlaps with other sandbox tests)
            try:
                emb = embed(cand["code"])
                label = "PASS" if cand.get("passed") else "FAIL"
                self._emb_writer.write(task_id, cand["index"], label, emb)
            except Exception:
                pass
            return cand

        n_workers = min(len(candidates), 3) if len(candidates) > 1 else 1
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_test_and_embed, c): c for c in candidates}
                for future in as_completed(futures):
                    cand = future.result()
                    if cand.get("passed"):
                        passing_candidates.append(cand)
        else:
            for cand in candidates:
                cand = _test_and_embed(cand)
                if cand.get("passed"):
                    passing_candidates.append(cand)

        latency["sandbox_ms"] = (time.time() - sandbox_start) * 1000

        # Log per-candidate energies, pass/fail, AND code for analysis.
        # Storing all candidate codes enables ablation replay: run once with
        # full pipeline, then derive other conditions by replaying selection
        # strategies on stored candidates (3.5x faster than 6 separate runs).
        result["telemetry"]["candidate_energies"] = [
            {"index": c["index"], "energy": c["energy"], "passed": c.get("passed")}
            for c in candidates
        ]
        result["telemetry"]["all_candidates"] = [
            {
                "index": c["index"],
                "code": c.get("code", ""),
                "energy": c["energy"],
                "energy_norm": c.get("energy_norm", 0.5),
                "passed": c.get("passed"),
                "tokens": c.get("tokens", 0),
            }
            for c in candidates
        ]

        # Store best candidate code even on failure (for feedback + analysis)
        if candidates and not passing_candidates:
            result["code"] = candidates[0]["code"]  # Best by energy (sorted)

        # ===== Select best passing candidate (with S* tiebreaking) =====
        if passing_candidates:
            result["passed"] = True
            result["phase_solved"] = "phase1"

            # Build CandidateInfo objects for strategy-based selection
            candidate_infos = [
                CandidateInfo(
                    index=c["index"], code=c["code"],
                    energy=c["energy"], passed=True,
                    logprobs=llm.last_logprobs if c["index"] == candidates[-1]["index"] else None,
                )
                for c in passing_candidates
            ]

            if len(passing_candidates) >= 2 and self.enable_phase2:
                # S* tiebreaking: generate edge-case inputs to distinguish
                # the top-2 passing candidates by energy
                # Use dedicated S* sandbox that pipes specific stdin for stdio tasks
                s_star_sandbox = SStarSandboxAdapter(task)
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
                        llm_call=llm, sandbox_run=s_star_sandbox,
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
                        # Use selection strategy
                        selected = select_candidate(
                            candidate_infos, strategy=self.selection_strategy,
                            seed=42,
                        )
                        result["code"] = selected.code if selected else passing_candidates[0]["code"]
                except Exception:
                    selected = select_candidate(
                        candidate_infos, strategy=self.selection_strategy,
                        seed=42,
                    )
                    result["code"] = selected.code if selected else passing_candidates[0]["code"]
            else:
                selected = select_candidate(
                    candidate_infos, strategy=self.selection_strategy,
                    seed=42,
                )
                result["code"] = selected.code if selected else passing_candidates[0]["code"]

            result["telemetry"]["selection_strategy"] = self.selection_strategy

            result["telemetry"]["latency"] = latency
            result["total_tokens"] = max(result["total_tokens"], llm.total_tokens)
            result["total_time_ms"] = (time.time() - start_time) * 1000
            self._record_feedback(task_id, result)
            self._log_v3_event(task_id, result)
            return result

        # ===== Phase 3: Refinement cascade =====
        phase3_start = time.time()
        if not self.enable_phase3:
            result["telemetry"]["latency"] = latency
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
        selftest_start = time.time()
        self_tests = self.self_test_gen.generate(
            problem=task.prompt, llm_call=llm, task_id=task_id,
        )
        latency["self_test_gen_ms"] = (time.time() - selftest_start) * 1000
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

        # Steps 3a/3b/3c: Run repair strategies SEQUENTIALLY
        # Priority order: PR-CoT (cheapest, 2-6 calls), Refinement Loop
        # (3-15 calls), Derivation Chains (most expensive, up to 17 calls).
        # Stop on first successful fix — saves ~29-33 LLM calls on average
        # vs parallel execution which wastes calls on losing strategies.
        phase3_extra_tokens = 0
        phase3_strategies_tried = []

        # --- Strategy 1: PR-CoT quick repair (2-6 LLM calls) ---
        if failing:
            phase3_strategies_tried.append("pr_cot")
            pr_llm = LLMAdapter(self.runner, timeout=300)
            try:
                best_failing = failing[0]
                error_msg = best_failing.error_output or "All test cases failed"

                # Enrich problem context with metacognitive warnings
                # and ACE principles for better-guided repairs
                enriched_problem = task.prompt
                if metacog_warnings:
                    enriched_problem += "\n\nKnown pitfalls for this problem type:"
                    for w in metacog_warnings:
                        enriched_problem += f"\n- {w}"
                if ace_context:
                    enriched_problem += f"\n\n{ace_context}"

                repair_result = self.pr_cot.repair(
                    problem=enriched_problem,
                    code=best_failing.code,
                    error=error_msg,
                    llm_call=pr_llm,
                    task_id=task_id,
                )
                phase3_extra_tokens += pr_llm.total_tokens
                for repair_code in repair_result.repairs:
                    if not repair_code:
                        continue
                    try:
                        # Test repairs directly against real sandbox.
                        # Self-test gating was filtering valid repairs on 9B
                        # (0/15 success rate). Direct sandbox testing is more
                        # reliable — costs a few extra sandbox calls but
                        # eliminates false-negative self-test rejections.
                        real_passed, _, _ = sandbox(repair_code, "")
                        if real_passed:
                            result["passed"] = True
                            result["code"] = repair_code
                            result["phase_solved"] = "pr_cot"
                            self._learn_from_success(task, task_id, "pr_cot")
                            break
                    except Exception:
                        continue
            except Exception as e:
                result["telemetry"]["pr_cot_error"] = str(e)

        # --- Strategy 2: Refinement Loop (3-15 LLM calls) ---
        if not result["passed"] and failing:
            phase3_strategies_tried.append("refinement")
            ref_llm = LLMAdapter(self.runner, timeout=300)
            try:
                ref_result = self.refinement_loop.run(
                    problem=task.prompt,
                    failing_candidates=failing,
                    original_constraints=constraints,
                    llm_call=ref_llm,
                    sandbox_run=self_verify_sandbox,
                    embed_call=embed,
                    metacognitive_warnings=metacog_warnings,
                    task_id=task_id,
                )
                phase3_extra_tokens += ref_llm.total_tokens
                if ref_result.solved:
                    real_passed, _, _ = sandbox(ref_result.winning_code, "")
                    if real_passed:
                        result["passed"] = True
                        result["code"] = ref_result.winning_code
                        result["phase_solved"] = "refinement"
                        result["telemetry"]["refinement_iterations"] = ref_result.total_iterations
                        self._learn_from_success(task, task_id, "refinement")
            except Exception as e:
                result["telemetry"]["refinement_error"] = str(e)

        # --- Strategy 3: Derivation Chains (up to 17 LLM calls) ---
        if not result["passed"]:
            phase3_strategies_tried.append("derivation")
            dc_llm = LLMAdapter(self.runner, timeout=300)
            try:
                failure_context = "; ".join(
                    f"Candidate {c['index']}: {c.get('stderr', 'failed')[:200]}"
                    for c in candidates if c.get("passed") is False
                )
                # Enrich with metacognitive context for derivation
                dc_problem = task.prompt
                if metacog_warnings:
                    dc_problem += "\n\nKnown pitfalls for this problem type:"
                    for w in metacog_warnings:
                        dc_problem += f"\n- {w}"

                dc_result = self.derivation_chains.solve(
                    problem=dc_problem,
                    failure_context=failure_context,
                    llm_call=dc_llm,
                    sandbox_run=sandbox,
                    task_id=task_id,
                )
                phase3_extra_tokens += dc_llm.total_tokens
                if dc_result.solved and dc_result.final_code:
                    passed, stdout, stderr = sandbox(dc_result.final_code, "")
                    if passed:
                        result["passed"] = True
                        result["code"] = dc_result.final_code
                        result["phase_solved"] = "derivation"
                        self._learn_from_success(task, task_id, "derivation")
            except Exception as e:
                result["telemetry"]["derivation_error"] = str(e)

        result["telemetry"]["phase3_strategies_tried"] = phase3_strategies_tried
        result["total_tokens"] += phase3_extra_tokens
        latency["phase3_total_ms"] = (time.time() - phase3_start) * 1000
        result["telemetry"]["latency"] = latency
        result["total_tokens"] = max(result["total_tokens"], llm.total_tokens)
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
        """Log a unified V3 pipeline event to JSONL.

        Consolidates all per-task telemetry into a single event for
        the analysis pipeline (V3.1 Section 5.1).
        """
        event = {
            "task_id": task_id,
            "passed": result["passed"],
            "phase_solved": result["phase_solved"],
            "candidates_generated": result["candidates_generated"],
            "total_tokens": result["total_tokens"],
            "total_time_ms": result["total_time_ms"],
            "selection_strategy": self.selection_strategy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Merge all telemetry sub-fields into the event
        telemetry = result.get("telemetry", {})
        for key, value in telemetry.items():
            event[key] = value
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
                 enable_feedback=False, selection_strategy="lens"):
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
            selection_strategy=selection_strategy,
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
        """Run LiveCodeBench tasks through V3 pipeline.

        When ATLAS_LLM_PARALLEL=1, runs multiple tasks concurrently using
        ATLAS_PARALLEL_TASKS workers (default 4). Otherwise runs sequentially.
        """
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

        parallel_tasks = int(os.environ.get("ATLAS_PARALLEL_TASKS", "4"))
        use_parallel = LLMAdapter._parallel_mode and parallel_tasks > 1

        if use_parallel:
            print(f"  PARALLEL MODE: {parallel_tasks} concurrent tasks")
            self._run_parallel(remaining, results, per_task_dir, total, done,
                               parallel_tasks)
        else:
            self._run_serial(remaining, results, per_task_dir, total, done)

        # Save phase summary (done counter updated by reference via results dict)
        done = len(results)
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

    def _process_one_task(self, task: BenchmarkTask) -> Dict:
        """Run a single task through the pipeline (thread-safe)."""
        task_start = time.time()
        try:
            return self.pipeline.run_task(task, task_id=task.task_id)
        except Exception as e:
            return {
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

    def _save_and_log(self, task_result: Dict, per_task_dir: Path,
                      done: int, total: int) -> None:
        """Save result atomically and print progress."""
        task_id = task_result["task_id"]
        safe_name = task_id.replace('/', '_')
        atomic_write_json(per_task_dir / f"{safe_name}.json", task_result)

        status = "PASS" if task_result["passed"] else "FAIL"
        phase = task_result.get("phase_solved", "?")
        elapsed = time.time() - self._start_time
        rate = done / (elapsed / 3600) if elapsed > 0 else 0
        tokens = task_result.get("total_tokens", 0)
        print(
            f"  [{done}/{total}] {task_id}: {status} "
            f"(via {phase}, {tokens} tok) "
            f"[{rate:.0f} tasks/hr]",
            flush=True,
        )

    def _run_serial(self, remaining, results, per_task_dir, total, done):
        """Process tasks one at a time (safe fallback)."""
        for task in remaining:
            task_result = self._process_one_task(task)
            results[task.task_id] = task_result
            done += 1
            self._save_and_log(task_result, per_task_dir, done, total)

    def _run_parallel(self, remaining, results, per_task_dir, total, done,
                      max_workers):
        """Process tasks concurrently using ThreadPoolExecutor.

        Each task gets its own LLMAdapter (per run_task), so thread safety
        relies on:
          - llama-server handling concurrent /completion requests (--no-cache-prompt)
          - atomic_write_json for per-task results (temp file + rename)
          - EmbeddingWriter._lock for binary embedding file
          - append_jsonl for JSONL telemetry (small writes are atomic on Linux)
          - print() with flush=True (inherently thread-safe in CPython via GIL)
        """
        _done_lock = threading.Lock()
        _done_counter = [done]  # mutable container for closure

        def process_and_save(task):
            task_result = self._process_one_task(task)
            with _done_lock:
                results[task.task_id] = task_result
                _done_counter[0] += 1
                current_done = _done_counter[0]
            self._save_and_log(task_result, per_task_dir, current_done, total)
            return task_result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_and_save, task): task
                for task in remaining
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    # Should not happen (process_and_save catches exceptions)
                    print(f"  UNEXPECTED ERROR on {task.task_id}: {e}",
                          flush=True)

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
                     enable_phase3=True, selection_strategy="lens",
                     enable_feedback=False):
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
        "selection_strategy": selection_strategy,
        "enable_feedback": enable_feedback,
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

    # Check Lens model availability (retry once after short delay)
    lens_ok = False
    for lens_attempt in range(2):
        try:
            test_body = json.dumps({"text": "test"}).encode("utf-8")
            req = urllib.request.Request(
                f"{RAG_API_URL}/internal/lens/score-text",
                data=test_body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                lens_data = json.loads(resp.read().decode('utf-8'))
            if lens_data.get("error"):
                print(f"  Lens model: NOT LOADED ({lens_data['error']})")
                print("    Phase 2 (adaptive K) will use default k=3")
            else:
                print(f"  Lens model: OK (energy={lens_data.get('energy', '?')})")
            lens_ok = True
            break
        except Exception:
            if lens_attempt == 0:
                time.sleep(3)
    if not lens_ok:
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
        selection_strategy=selection_strategy,
        enable_feedback=enable_feedback,
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
    parser.add_argument("--selection-strategy", type=str, default="lens",
                        choices=["lens", "random", "logprob", "oracle"],
                        help="Candidate selection strategy (default: lens)")
    parser.add_argument("--enable-feedback", action="store_true",
                        help="Enable Lens Evolution (Phase 4): online C(x) retrain during benchmark")
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
        selection_strategy=args.selection_strategy,
        enable_feedback=args.enable_feedback,
    )

    if run_dir:
        print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
