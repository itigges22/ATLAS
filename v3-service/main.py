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
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runner import extract_code
from benchmark.llm_client import chatml_to_messages
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


def _load_service_token() -> str:
    """Internal-auth token (Authorization: Bearer). Empty = auth
    disabled — an install without `atlas init` keeps the open-localhost
    behavior and `atlas doctor` warns. The value is never logged."""
    path = os.environ.get("ATLAS_SERVICE_TOKEN_FILE",
                          "/run/atlas-secrets/service-token")
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


SERVICE_TOKEN = _load_service_token()

if SERVICE_TOKEN:
    # Outbound injection: one opener covers every urllib call site
    # (llama, lens, sandbox). urllib merges addheaders under explicit
    # per-request headers, so requests that already set Authorization
    # keep their own value.
    _opener = urllib.request.build_opener()
    _opener.addheaders = [("Authorization", f"Bearer {SERVICE_TOKEN}")]
    urllib.request.install_opener(_opener)
PORT = int(os.environ.get("ATLAS_V3_PORT", "8070"))


def _service_headers(rid: str = "") -> dict:
    """Headers for outbound service-to-service calls: forwards the
    current request's correlation ID so lens/sandbox/llama log records
    join the same trace. Pass rid explicitly from background threads —
    a new thread doesn't inherit the request's ContextVar."""
    headers = {"Content-Type": "application/json"}
    if not rid:
        try:
            from structured_log import get_request_id
            rid = get_request_id()
        except ImportError:
            rid = ""
    if rid:
        headers["X-ATLAS-Request-ID"] = rid
    return headers

BASE_TEMPERATURE = 0.6
DIVERSITY_TEMPERATURE = 0.8
MAX_TOKENS = 8192


# --- Pattern Cache write hook -------------------------------------------------
# Maps the V3 phase that produced the winning solution to a retry_count value.
# The pattern cache uses retry_count / max_retries as a "surprise" proxy — higher
# retries mean the pattern was harder to find and worth caching with more weight.
_PHASE_RETRY_COUNT = {
    "probe": 1,             # solved on first probe (phase_solved="probe")
    "phase1": 2,            # plan-search candidates passed
    "phase1_sstar": 2,      # S* tiebreak among passing candidates
    "pr_cot": 3,            # required PR-CoT repair
    "refinement": 4,        # required refinement loop
    "derivation": 5,        # required derivation chains
    "fallback": 5,          # nothing passed; best-by-energy returned
    "none": 5,
}


def _post_pattern_outcome(problem: str, result: dict):
    """Fire-and-forget: post the pipeline outcome to geometric-lens for caching.

    Runs in a background thread so it never delays the response. Errors are
    logged but never raised — the pattern cache is best-effort, not load-bearing.
    """
    import threading

    # Capture the correlation ID on the request thread — the ContextVar
    # doesn't propagate into a newly created thread.
    try:
        from structured_log import get_request_id
        rid = get_request_id()
    except ImportError:
        rid = ""

    def _do_post():
        payload = {
            "query": problem,
            "solution": result.get("code", ""),
            "retry_count": _PHASE_RETRY_COUNT.get(result.get("phase_solved", "none"), 5),
            "max_retries": 5,
            "error_context": None,
            "source_files": [],
            "active_pattern_ids": [],
            "success": bool(result.get("passed")),
        }
        try:
            req = urllib.request.Request(
                f"{LENS_URL}/internal/patterns/write",
                data=json.dumps(payload).encode(),
                headers=_service_headers(rid),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception as e:
            print(f"  [pattern-write] POST failed (non-fatal): {e}", flush=True)

    threading.Thread(target=_do_post, daemon=True).start()


# --- PC-061 step B: typed event emission ------------------------------------
# --- LLM Adapter (calls llama-server /v1/chat/completions) ----------------------------

class LLMAdapter:
    """Calls llama-server's /v1/chat/completions, parsing ChatML prompts into messages.

    PC-206: `thinking` controls template-level reasoning when supported.
    - False (default) — `enable_thinking=False`.
      Required for grammar-constrained JSON output (the agent's tool-call
      shape) and for the tight V3 sampling loop where reasoning would 5-20×
      output token cost. This matches the previously hardcoded behavior.
    - True — `enable_thinking=True`. Use for
      high-reasoning-value calls (planner, verification, claim-check) where
      the output can absorb a preamble and the strip pattern in __call__
      cleans up `<think>...</think>` blocks before downstream JSON parse.

    The default is set per-instance; individual __call__ invocations can
    override via the `thinking` keyword for ad-hoc switches.
    """

    _lock = threading.Lock()

    def __init__(self, progress_callback=None, thinking: bool = False):
        self.call_count = 0
        self.total_tokens = 0
        self.last_logprobs: List[float] = []
        self._progress = progress_callback
        self.thinking = thinking

    def _emit(self, stage: str, detail: str = "", **data):
        if self._progress:
            try:
                self._progress(stage, detail, **data)
            except TypeError:
                # Older two-arg callbacks don't accept **data — call back
                # to the legacy signature so we stay compatible.
                self._progress(stage, detail)

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int],
                 thinking: Optional[bool] = None) -> Tuple[str, int, float]:
        self.call_count += 1

        # Resolve per-call override against the instance default (PC-206).
        thinking_resolved = self.thinking if thinking is None else thinking

        body = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,  # streaming: per-token visibility + no
                              # 300s urllib read-timeout on long gens.
            "stop": ["\n\n\n\n"],
            "top_k": 20,
            "top_p": 0.95,
            "_thinking": thinking_resolved,  # consumed by _send, popped before send
        }
        if seed is not None:
            body["seed"] = seed

        start = time.time()
        # Marker so the TUI can frame this LLM call. Mirrors what
        # atlas-proxy emits around its own llama.cpp calls.
        self._emit("llm_start", f"call #{self.call_count}",
                   call=self.call_count, max_tokens=max_tokens,
                   temperature=temperature)
        data = self._send(body)
        # The streaming send already emitted token events; emit a
        # closing marker with totals so the TUI can replace the live
        # row with a compact summary.
        elapsed_ms = (time.time() - start) * 1000
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0) \
            or data.get("usage", {}).get("total_tokens", 0)
        self._emit("llm_end", f"{completion_tokens} tok · {elapsed_ms:.0f}ms",
                   call=self.call_count, tokens=completion_tokens,
                   elapsed_ms=int(elapsed_ms))

        # Parse response
        content = ""
        tokens = completion_tokens
        if "choices" in data:
            content = data["choices"][0].get("text", "")

        # Strip thinking blocks
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        if '</think>' in content and '<think>' not in content:
            content = content[content.index('</think>') + len('</think>'):].strip()

        self.total_tokens += tokens
        return content, tokens, elapsed_ms

    def _send(self, body: dict) -> dict:
        """Send to llama-server via /v1/chat/completions.

        V3 modules generate ChatML prompts. We parse them into messages format
        for the chat endpoint. ChatML format:
            <|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n
        """
        prompt = body.pop("prompt", "")
        model_name = os.environ.get("ATLAS_MODEL_NAME", "local-model")

        # PC-206: thinking flag drops down from __call__. Default False so
        # callers get bounded generation unless they opt into reasoning.
        thinking = bool(body.pop("_thinking", False))

        # Convert the internal prompt carrier into structured messages before
        # llama-server applies the selected model's own template.
        messages = chatml_to_messages(prompt)
        if "<|im_start|>" not in prompt:
            print(f"  [LLM] ChatML parse failed, using raw prompt ({len(prompt)} chars)", flush=True)
        else:
            print(f"  [LLM] Parsed {len(messages)} messages from ChatML"
                  f" (thinking={'on' if thinking else 'off'})", flush=True)
            if thinking:
                # Strip the legacy directive from old prompt templates when a
                # caller explicitly enables reasoning.
                for msg in messages:
                    if msg["role"] == "user" and msg["content"].startswith("/nothink"):
                        msg["content"] = msg["content"][len("/nothink"):].lstrip("\n")

        chat_body = {
            "model": model_name,
            "messages": messages,
            "max_tokens": body.get("max_tokens", body.pop("n_predict", 4096)),
            "temperature": body.get("temperature", 0.6),
            "stream": bool(body.get("stream", False)),
            # The chat template may honor enable_thinking; templates that do
            # not support it ignore the kwarg. Reasoning blocks are stripped
            # in __call__ before downstream JSON parsing.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if chat_body["stream"]:
            # Need usage in the final chunk so we can report token counts.
            chat_body["stream_options"] = {"include_usage": True}
        if "seed" in body:
            chat_body["seed"] = body["seed"]

        req = urllib.request.Request(
            f"{INFERENCE_URL}/v1/chat/completions",
            data=json.dumps(chat_body).encode(),
            headers=_service_headers(),
        )
        for attempt in range(5):
            try:
                with LLMAdapter._lock:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        if not chat_body["stream"]:
                            data = json.loads(resp.read())
                            # Convert chat response to completions format
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "message" in choice:
                                    choice["text"] = choice["message"].get("content", "")
                            return data
                        # Streaming path: parse SSE chunks, accumulate
                        # delta content, and forward each delta to the
                        # progress callback as ("token", text). The 600s
                        # urllib timeout is per-read; with continuous
                        # token flow each read is sub-second, so long
                        # generations no longer hit the old 300s ceiling.
                        full = []
                        reasoning = []
                        usage = {}
                        first_chunk_logged = False
                        for raw in resp:
                            line = raw.decode("utf-8", "replace").rstrip("\r\n")
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].lstrip()
                            if payload == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices") or []
                            if choices:
                                delta_obj = choices[0].get("delta", {}) or {}
                                if not first_chunk_logged and delta_obj:
                                    print(f"  [LLM] first delta keys={list(delta_obj.keys())} sample={json.dumps(delta_obj)[:200]}",
                                          flush=True)
                                    first_chunk_logged = True
                                delta = delta_obj.get("content", "") or ""
                                # Some llama.cpp builds split <think>…</think>
                                # into delta.reasoning_content. Capture it as
                                # a fallback so we don't end up with 2048 tok
                                # of reasoning and zero parseable text.
                                rdelta = delta_obj.get("reasoning_content", "") or ""
                                if delta:
                                    full.append(delta)
                                    self._emit("token", delta)
                                if rdelta:
                                    reasoning.append(rdelta)
                            u = chunk.get("usage")
                            if u:
                                usage = u
                        text = "".join(full)
                        if not text and reasoning:
                            # Reasoning-only response: surface it so the
                            # parser at least sees the JSON the model
                            # buried inside its think block.
                            print(f"  [LLM] reasoning-only response ({len(reasoning)} chunks, "
                                  f"{sum(len(r) for r in reasoning)} chars) — using as content",
                                  flush=True)
                            text = "".join(reasoning)
                        return {
                            "choices": [{"text": text}],
                            "usage": usage,
                        }
            except (urllib.error.HTTPError, OSError) as e:
                print(f"  [LLM] Attempt {attempt+1} failed: {e}", flush=True)
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise
        # Unreachable: the for loop above always either returns inside
        # the success branch or raises on the 5th failure. Explicit
        # for py/mixed-returns (the implicit fall-through returns None,
        # which violates the -> dict signature).
        raise RuntimeError("unreachable: _send loop must return or raise")


class ClientDisconnected(Exception):
    """SSE client went away mid-pipeline. Raised at phase boundaries in
    V3PipelineService.run so a dead client doesn't keep burning GPU minutes;
    the HTTP handlers catch it and stop without writing a response."""


# --- Sandbox Adapter (calls sandbox /execute) ---------------------------------

class SandboxAdapter:
    """Calls the sandbox service for code execution.

    PC-046: optional `project_files` dict ships supporting files (other
    modules from the user's project) into the sandbox workspace so
    multi-file imports resolve. Without this, a candidate that does
    `from utils import helper` fails ImportError in the sandbox even
    though it would work on the user's machine.
    """

    def __init__(self, project_files: Optional[Dict[str, str]] = None):
        self.project_files = project_files or {}

    def __call__(self, code: str, test_input: str = "") -> Tuple[bool, str, str]:
        body = {
            "code": code,
            "language": "python",
            "timeout": 15,
        }
        if self.project_files:
            body["files"] = self.project_files
        try:
            req = urllib.request.Request(
                f"{SANDBOX_URL}/execute",
                data=json.dumps(body).encode(),
                headers=_service_headers(),
            )
            # 45s client timeout: the sandbox's server-side budgets (syntax
            # check + optional pip install + lint + the 15s run cap) can sum
            # past 30s, and the old 20s read timeout gave up on executions
            # the sandbox would still have completed.
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read())
                return data.get("success", False), data.get("stdout", ""), data.get("stderr", "")
        except Exception as e:
            return False, "", str(e)

    def syntax_check(self, code: str, language: str, filename: str = "") -> Tuple[bool, str, str]:
        """Ask the sandbox to parse or compile source without executing it."""
        body = {
            "code": code,
            "language": language,
            "filename": filename or None,
        }
        try:
            req = urllib.request.Request(
                f"{SANDBOX_URL}/syntax-check",
                data=json.dumps(body).encode(),
                headers=_service_headers(),
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            errors = data.get("errors", [])
            error_text = "\n".join(str(error) for error in errors)
            return bool(data.get("valid", False)), "", error_text
        except Exception as e:
            return False, "", f"syntax verification unavailable: {e}"

    def run_command(
        self,
        command: str,
        files: Optional[Dict[str, str]] = None,
        cwd: str = "/workspace",
        timeout: int = 60,
    ) -> Tuple[bool, str, str, Dict[str, Any]]:
        """Run a project command through the sandbox /shell endpoint.

        `files` is an ephemeral overlay: the sandbox snapshots /workspace,
        applies these relative paths in the temp copy, runs the command there,
        then deletes the temp copy. It lets V3 verify a candidate without
        writing it to the user's real workspace.
        """
        body = {
            "command": command,
            "cwd": cwd or "/workspace",
            "timeout": timeout,
        }
        if files:
            body["files"] = files
        try:
            req = urllib.request.Request(
                f"{SANDBOX_URL}/shell",
                data=json.dumps(body).encode(),
                headers=_service_headers(),
            )
            with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
                data = json.loads(resp.read())
            return (
                bool(data.get("success", False)),
                data.get("stdout", ""),
                data.get("stderr", ""),
                data,
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            return False, "", detail, {"exit_code": None, "elapsed_ms": 0}
        except Exception as e:
            return False, "", f"build verification unavailable: {e}", {"exit_code": None, "elapsed_ms": 0}


# --- Embedding Adapter --------------------------------------------------------

class EmbedAdapter:
    """Calls llama-server /v1/embeddings for code embeddings."""

    def __call__(self, text: str) -> List[float]:
        body = {"model": "default", "input": text}
        try:
            req = urllib.request.Request(
                f"{INFERENCE_URL}/v1/embeddings",
                data=json.dumps(body).encode(),
                headers=_service_headers(),
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("data", [{}])[0].get("embedding", [])
        except Exception:
            return []


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
            f"{LENS_URL}/internal/lens/score-per-step",
            data=body,
            headers=_service_headers(),
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        if not data.get("enabled"):
            return {}
        agg = data.get("aggregate", {}) or {}
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


def score_candidate(code: str) -> Tuple[float, float, bool]:
    """Score code with Geometric Lens C(x).

    Returns ``(raw_energy, normalized_energy, calibrated)``. The normalized
    value is neutral when this model has no calibration and must not drive
    adaptive routing in that case.

    Timeout note: 10s was tight under load — the lens shares the box with
    V3's streaming generator and llama-server, and a single hot probe
    could starve scoring long enough to trip the fallback. Bumped to 30s
    so transient contention doesn't masquerade as a broken lens
    (symptom: C(x)=0.00 / gx=0.50 sentinel pair).
    """
    try:
        body = json.dumps({"text": code}).encode()
        req = urllib.request.Request(
            f"{LENS_URL}/internal/lens/gx-score",
            data=body,
            headers=_service_headers(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return (
                data.get("cx_energy", 0.0),
                data.get("cx_normalized", 0.5),
                bool(data.get("cx_calibrated", False)),
            )
    except Exception as e:
        print(f"  [lens] score_candidate failed: {e} — using neutral uncalibrated score", flush=True)
        return 0.0, 0.5, False


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
    Runs inside the sandbox so any import-time crashes show up as stderr.

    PC-048: language-aware. Python, JavaScript, TypeScript, Go, Rust,
    C/C++, Bash, HTML, XML, JSON, and YAML files use the sandbox syntax
    checker. Unknown formats fail explicitly instead of being accepted
    without evidence.
    """
    lang = (language or "python").lower()

    verified_languages = {
        "python", "py", "javascript", "typescript", "go", "java", "kotlin",
        "rust", "c", "cpp", "ruby", "php", "bash", "html", "htm", "xml", "json", "yaml", "yml",
    }
    if hasattr(sandbox, "syntax_check") and lang in verified_languages:
        normalized = {
            "py": "python", "htm": "html", "yml": "yaml",
        }.get(lang, lang)
        return sandbox.syntax_check(code, normalized)

    if lang in ("html", "htm"):
        smoke = (
            "import sys\n"
            "from html.parser import HTMLParser\n"
            f"_src = {code!r}\n"
            "class _Strict(HTMLParser):\n"
            "    def error(self, msg):\n"
            "        raise ValueError(msg)\n"
            "try:\n"
            "    _p = _Strict()\n"
            "    _p.feed(_src)\n"
            "    _p.close()\n"
            "    print('SMOKE_OK')\n"
            "except Exception as e:\n"
            "    print(f'HTML_PARSE_ERROR: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        ok, out, err = sandbox(smoke)
        return (ok and "SMOKE_OK" in out), out, err

    if lang == "json":
        smoke = (
            "import json, sys\n"
            f"_src = {code!r}\n"
            "try:\n"
            "    json.loads(_src)\n"
            "    print('SMOKE_OK')\n"
            "except json.JSONDecodeError as e:\n"
            "    print(f'JSON_PARSE_ERROR: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        ok, out, err = sandbox(smoke)
        return (ok and "SMOKE_OK" in out), out, err

    if lang in ("yaml", "yml"):
        smoke = (
            "import sys\n"
            f"_src = {code!r}\n"
            "try:\n"
            "    import yaml\n"
            "    yaml.safe_load(_src)\n"
            "    print('SMOKE_OK')\n"
            "except ImportError:\n"
            # PyYAML not installed in sandbox — pass-through so we don't
            # block legitimate edits on a missing optional dep.
            "    print('SMOKE_OK')\n"
            "except Exception as e:\n"
            "    print(f'YAML_PARSE_ERROR: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        ok, out, err = sandbox(smoke)
        return (ok and "SMOKE_OK" in out), out, err

    if lang not in ("python", "py"):
        return False, "", f"syntax verification unavailable for language: {lang}"

    # Default: Python compile smoke
    smoke = (
        "import ast, sys\n"
        f"_src = {code!r}\n"
        "try:\n"
        "    ast.parse(_src)\n"
        "    compile(_src, '<smoke>', 'exec')\n"
        "    print('SMOKE_OK')\n"
        "except SyntaxError as e:\n"
        "    print(f'SYNTAX_ERROR: {e}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    ok, out, err = sandbox(smoke)
    return (ok and "SMOKE_OK" in out), out, err


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
        self.failure_analyzer = FailureAnalyzer(FailureAnalysisConfig(enabled=True))
        self.constraint_refiner = ConstraintRefiner(ConstraintRefinementConfig(enabled=True))
        self.metacognitive = MetacognitiveProfile(MetacognitiveConfig(enabled=True))
        self.self_test_gen = SelfTestGen(SelfTestGenConfig(enabled=True))

    def run(self, problem: str, task_id: str = "cli",
            progress_callback=None, files: Dict[str, str] = None,
            file_path: str = "", build_command: str = "",
            working_dir: str = "/workspace",
            constraints: List[str] = None) -> Dict[str, Any]:
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
            constraints: raw constraint strings from the request. When the V3.2
                RPG planner is active (issue #120) these carry the node's planned
                signatures, used by the RPG signature veto below. Backward
                compatible — defaults to none, leaving existing callers unchanged.
        """
        start = time.time()
        events = []
        files = files or {}
        constraints = constraints or []

        # PC-048: derive language from the target file's extension. Used
        # only by smoke_compile_check below to pick the right syntax
        # checker. Defaults to Python when no file_path is
        # supplied, preserving previous behavior for /v3/run callers.
        _ext = Path(file_path).suffix.lower() if file_path else ""
        _ext_to_lang = {
            ".py": "python", ".pyw": "python",
            ".html": "html", ".htm": "html",
            ".json": "json",
            ".yaml": "yaml", ".yml": "yaml",
            ".css": "css",
            ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".md": "markdown", ".markdown": "markdown",
            ".txt": "text", ".rst": "text",
            ".toml": "toml",
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
                raise ClientDisconnected(f"client disconnected during task {task_id}")

        llm = LLMAdapter(progress_callback=emit)
        # PC-046: ship the user's other project files into the sandbox so
        # multi-file imports resolve. `files` is the same Dict that V3
        # already prepends to the LLM prompt above; passing it to the
        # sandbox closes the gap where the model writes
        # `from utils import helper` and the sandbox imports a workspace
        # that contains only solution.py.
        sandbox = SandboxAdapter(project_files=files)
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
        task_type = classify_task_type(problem)
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
                ok, build_out, build_err, evidence = verify_build_command(
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
                ok, out, err = smoke_compile_check(code, sandbox, language=smoke_language)
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
                ok, out, err = smoke_compile_check(code, sandbox, language=smoke_language)
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
                lint_ok, lint_reason = interactive_lint(code)
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
            probe_energy_raw, probe_energy_norm, probe_cx_calibrated = score_candidate(probe_code)
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
                        energy_raw, energy_norm, energy_calibrated = score_candidate(code)
                        per_step = score_candidate_per_step(code)  # PC-207
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
                        energy_raw, energy_norm, energy_calibrated = score_candidate(code)
                        per_step = score_candidate_per_step(code)  # PC-207
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
            project_symbols = build_project_symbols(files or {})
            kept = []
            for c in passing:
                struct = structural_score(project_symbols, c.get("code", ""))
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
                            file_path, c.get("code", ""), files,
                            builtins=set(PY_BUILTINS), strict=True)
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

        # ===== RPG SIGNATURE VETO (V3.2, issue #120) =====
        # When the RPG planner is active, the request constraints carry the
        # node's planned signatures. Extend the #39-pt-1 veto from "imports
        # survive" to "the planned interface exists": reject sandbox-passing
        # candidates that don't define the planned functions. Additive and
        # conservative — only fires when the flag is on AND constraints carry
        # planned signatures, and never empties the candidate set (so it can't
        # do worse than the flat path; a fully-failing set falls through intact
        # to phase-3 repair).
        if passing and constraints:
            try:
                from wavelet import rpg_planning_enabled as _rpg_on
                _rpg_active = _rpg_on()
            except Exception:
                _rpg_active = False
            if _rpg_active:
                try:
                    import rpg as _rpgmod
                    planned_sigs = _rpgmod.planned_signatures_from_constraints(constraints)
                except Exception:
                    planned_sigs = []
                if planned_sigs:
                    sig_kept = []
                    vetoed = []  # (candidate, missing) — emitted only if the prune applies
                    for c in passing:
                        # Guarded per-candidate like the call-graph veto: an
                        # unexpected parser error must skip the veto for that
                        # candidate, not fail the whole /v3/generate request.
                        try:
                            missing = _rpgmod.missing_planned_signatures(
                                c.get("code", ""), planned_sigs, file_path)
                        except Exception as ve:
                            print(f"  [rpg] signature check failed on cand "
                                  f"{c.get('index')}: {ve} — keeping candidate",
                                  flush=True)
                            missing = []
                        if missing:
                            vetoed.append((c, missing))
                            continue
                        sig_kept.append(c)
                    # Only prune when at least one candidate realizes the plan
                    # — and only announce vetoes that actually took effect
                    # (when every candidate drifts, all of them survive).
                    if sig_kept:
                        passing = sig_kept
                        for c, missing in vetoed:
                            emit("rpg_signature_veto",
                                 f"Candidate {c['index']} sandbox-passed but missing "
                                 f"planned signature(s): {', '.join(missing[:3])}",
                                 index=c["index"], missing=missing[:5])
                            print(f"  [rpg] vetoed cand {c['index']} — missing: {missing[:5]}",
                                  flush=True)

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

        # GH #39 point 3: build call-chain context for the failing
        # function once, reuse across PR-CoT + refinement. Skips
        # cleanly when stderr isn't a Python traceback or the failing
        # function isn't defined in the project — both arms get plain
        # error_output in that case.
        chain_context_block = ""
        if failing:
            failing_func = _failing_function_from_stderr(failing[0].error_output)
            if failing_func and files:
                # Phase 2 (#39 point 3): when the call graph is enabled, build a
                # multi-hop reachability slice (entry-point path, transitive
                # impact, callees) instead of the 1-hop callers/callees block.
                # Falls back to the shipped builder on flag-off or any failure.
                chain_context_block = ""
                try:
                    from graph import call_graph_enabled as _cg_on, repair_context as _cg_repair
                    if _cg_on():
                        chain_context_block = _cg_repair(files, failing_func)
                except Exception as cge:
                    print(f"  [phase3] graph repair-context skipped: {cge}", flush=True)
                if not chain_context_block:
                    chain_context_block = call_chain_context(files, failing_func)
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
            constraints = []  # from PlanSearch
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
                    original_constraints=constraints,
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


# --- Plan generation (/v3/plan) ----------------------------------------------
#
# Generates a structured plan for an agent task. Reuses the same LLMAdapter
# the code-generation pipeline uses, but with a planning prompt template and
# a heuristic scorer (V3's lens-based scorer is for code embeddings, not prose
# plans, so it doesn't apply here).
#
# Why bother: when a compact coding model gets a multi-step task without a plan, it
# wanders through 12+ turns of recon before any real work. Forcing the
# model to commit to an ordered set of steps up front cuts the wander to
# zero — even a wrong plan beats no plan, because at least the wrongness
# is visible in one screen instead of buried in a trace.

PLAN_PROMPT_TEMPLATE = """You are an architect. Output ONLY a JSON plan, no other text. No markdown fences. No prose preamble.

User goal: {user_message}
Working directory: {working_dir}

{project_context}

Produce a plan as a SINGLE JSON object:
{{
  "steps": [
    {{"id": "s1", "action": "<concrete action>", "target": "<file path or url>", "why": "<one short sentence>"}},
    ...
  ],
  "verify_step": "<id of the step that verifies the fix works>",
  "rationale": "<one sentence on why this plan shape is right>"
}}

Rules:
- Each step is a single tool call: read_file, write_file, edit_file, ast_edit, delete_file, run_command, list_directory.
- Tool selection guidance:
    * read_file        — inspect a file before editing it
    * write_file       — create a NEW file (rejected for files >5 lines that already exist)
    * edit_file        — small, targeted string change (one function, one block) inside an existing file
    * ast_edit         — replace a WHOLE function, class, or HTML element by selector. Use for any
                         "replace the dashboard function" / "rewrite <body>" / "swap the validate method"
                         step. Selectors: `function:NAME`, `class:NAME`, `<tag>` (.py and .html only).
                         Strongly preferred over edit_file when the change is a whole-unit swap —
                         edit_file truncates on long old_str/new_str pairs (>1.5 KB hits max_tokens
                         mid-string and the JSON parse fails). ast_edit takes no old_str so it
                         doesn't truncate.
    * run_command      — build, test, run, curl. Verifies behavior.
    * delete_file      — remove a file
    * list_directory   — list a directory's contents
- The verify_step MUST run a verification command — curl, pytest, python <script>, go test, npm test, cargo test, make test. ls / cat / grep do NOT verify; they only inspect.
- Minimum 2 steps, maximum 6. Tighter is better.
- Address the user's STATED problem only. Don't add unrelated work, don't re-architect.
- For "fix" intents, the plan shape should be: investigate (1 step) → change (1-3 steps) → verify (1 step).

JSON plan:"""


def _build_plan_prompt(user_message: str, working_dir: str,
                       project_context: Dict[str, str]) -> str:
    """Render the planning prompt with project files inlined (truncated)."""
    if project_context:
        ctx_lines = ["Files in project:"]
        for path, content in project_context.items():
            preview = content[:200]
            if len(content) > 200:
                preview += "\n..."
            ctx_lines.append(f"### {path}\n```\n{preview}\n```")
        ctx_str = "\n".join(ctx_lines)
    else:
        ctx_str = "(no project files inspected yet)"
    return PLAN_PROMPT_TEMPLATE.format(
        user_message=user_message,
        working_dir=working_dir,
        project_context=ctx_str,
    )


def _parse_plan_json(raw: str) -> Optional[dict]:
    """Extract a plan dict from raw LLM output. Tolerates leading/trailing
    prose and markdown fences — the agent's output sanitizer normally
    strips these for tool args, but plans cross the wire as raw text so
    we strip here too."""
    if not raw:
        return None
    # Strip ```json ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    # Find the first {...} block — model sometimes prefixes "Here's the plan:".
    brace_start = raw.find("{")
    if brace_start < 0:
        return None
    # Scan to matching closing brace (depth-aware, ignores braces in strings
    # to be safe — though plan JSON shouldn't have embedded strings with `{`).
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(brace_start, len(raw)):
        c = raw[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(raw[brace_start:end])
    except (json.JSONDecodeError, ValueError):
        return None


# Verification-command pattern. Mirrors proxy/guardrails.go:verificationCommandRe
# so the plan scorer agrees with the agent loop on what counts as "verifies".
_VERIFY_CMD_RE = re.compile(
    r"\b(pytest|python\b|python3\b|node\b|deno\b|bun\b|"
    r"cargo\s+(run|test|check|build)|go\s+(run|test|build|vet)|"
    r"npm\s+(test|run|start)|yarn\s+(test|run|start)|pnpm\s+(test|run|start)|"
    r"make\b|just\b|curl\b|wget\b|http\b|httpie\b|"
    r"mypy\b|ruff\b|pylint\b|tsc\b|eslint\b|"
    r"markdownlint\b|stylelint\b|shellcheck\b|hadolint\b|flake8\b|"
    r"rubocop\b|golangci-lint\b)"
)


def _score_plan(plan: dict, user_message: str) -> Tuple[float, List[str]]:
    """Heuristic plan scorer. Returns (score in [0,1], reasons[]).

    Plans aren't sandbox-buildable so the lens doesn't help us pick a
    winner. Instead we check structural properties that correlate with
    "this plan will actually solve the user's problem":
      - has a verify_step
      - step count is in [2, 6]
      - verify_step's action runs an actual verification command
      - target paths reference files the user named
      - rationale is present

    Reasons are returned alongside the score so the picker can stream
    "plan #2 won because: has verify, target matches" to the TUI.
    """
    reasons: List[str] = []
    score = 0.0

    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        return 0.0, ["steps is not a list"]

    n = len(steps)
    if 2 <= n <= 6:
        score += 0.2
        reasons.append(f"step count {n} in range")
    elif n > 0:
        # A 1-step or 7+ step plan is a yellow flag, not a fail.
        score += 0.05
        reasons.append(f"step count {n} outside [2,6]")
    else:
        return 0.0, ["empty plan"]

    verify_step_id = plan.get("verify_step")
    verify_step = None
    for s in steps:
        if isinstance(s, dict) and s.get("id") == verify_step_id:
            verify_step = s
            break
    if verify_step is not None:
        score += 0.3
        reasons.append(f"verify_step={verify_step_id}")
        action = (verify_step.get("action") or "") + " " + (verify_step.get("target") or "")
        if _VERIFY_CMD_RE.search(action.lower()):
            score += 0.2
            reasons.append("verify_step references a real verification command")
        else:
            reasons.append("verify_step doesn't reference a verification command")
    else:
        reasons.append("missing or invalid verify_step")

    # Target-vs-user-message overlap. If the user said "fix index.html",
    # plans that touch index.html beat plans that don't.
    # Length bounds (max 128 for the stem, max 16 for the extension) cap
    # regex backtracking on pathological input like "a." repeated many
    # times — both groups need `.` so without bounds this is polynomial.
    mentioned_files = set(re.findall(
        r"[\w./-]{1,128}\.[a-zA-Z0-9]{1,16}", user_message.lower()))
    target_hits = 0
    for s in steps:
        if not isinstance(s, dict):
            continue
        target = (s.get("target") or "").lower()
        for f in mentioned_files:
            if f in target:
                target_hits += 1
                break
    if mentioned_files:
        target_score = min(0.2, target_hits * 0.1)
        score += target_score
        if target_hits:
            reasons.append(f"{target_hits} step(s) target user-mentioned files")

    if plan.get("rationale"):
        score += 0.1
        reasons.append("rationale present")

    return min(score, 1.0), reasons


def generate_plan(
    user_message: str,
    working_dir: str,
    project_context: Dict[str, str],
    n_candidates: int = 3,
    progress_callback=None,
) -> dict:
    """Generate a plan via diverse LLM sampling + heuristic scoring.

    Returns a dict matching the proxy's expected schema:
      {
        "steps": [...],
        "verify_step": "sN",
        "candidates_tested": int,
        "winning_score": float,
        "winning_index": int,
        "rationale": str,
        "reasons": [str],  # why the winner won
      }

    On total failure (no candidate parses), returns a single-step
    fallback plan that asks the model to plan inline. Better than
    blocking the agent loop on planner-pipeline errors.
    """

    def emit(stage: str, detail: str = "", **data):
        if progress_callback:
            try:
                progress_callback(stage, detail, **data)
            except TypeError:
                progress_callback(stage, detail)

    emit("plan_start", f"generating {n_candidates} candidate plans")

    # V3.2 RPG-style architecture-first planning (issue #120), flag-gated by
    # ATLAS_RPG_PLANNING. Strictly additive: on any failure (flag off, modules
    # missing, model output unusable) we fall through to the flat planner below.
    # Pre-bound so a partial import failure can never leave a name unbound
    # on the guarded paths below (py/uninitialized-local-variable).
    decompose_project = decompose_file_map = _rpg_mod = None
    try:
        from wavelet import rpg_planning_enabled, decompose_project, decompose_file_map
        import rpg as _rpg_mod
        _rpg_on = rpg_planning_enabled()
    except Exception:
        _rpg_on = False
    if _rpg_on:
        try:
            # Build the coarse structural band that seeds the proposal stage.
            # Prefer the in-memory project_context the proxy already sent — the
            # v3-service container has no project volume mount, so reading
            # working_dir off disk only works in non-container / dev setups.
            # Fall back to a disk scan when no context was passed.
            coarse_map = None
            try:
                if project_context:
                    coarse_map = [
                        {"label": p.label, "filename": p.filename}
                        for p in decompose_file_map(project_context, limit=30)
                    ]
                elif working_dir and os.path.isdir(working_dir):
                    coarse_map = [
                        {"label": p.label, "filename": p.filename}
                        for p in decompose_project(working_dir, limit=30)
                    ]
            except Exception as ce:
                emit("rpg_coarse_error", f"coarse decomposition failed: {ce}")
            rpg_thinking = os.environ.get("ATLAS_PLAN_THINKING", "0").lower() in ("1", "true", "yes")
            rpg_llm = LLMAdapter(progress_callback=progress_callback, thinking=rpg_thinking)

            def _complete(prompt, temperature, mt, seed):
                raw, _, _ = rpg_llm(prompt, temperature, mt, seed)
                return raw

            result = _rpg_mod.construct_rpg(
                user_message=user_message,
                complete_fn=_complete,
                project_context=project_context,
                coarse_map=coarse_map,
                max_tokens=(8192 if rpg_thinking else 2048),
                emit=emit,
            )
            if result.ok and result.plan and result.plan.get("steps"):
                plan = dict(result.plan)
                plan["candidates_tested"] = 1
                plan["winning_score"] = result.score
                plan["winning_index"] = 0
                plan["reasons"] = ["RPG two-stage plan"] + result.issues
                plan["rpg"] = result.rpg.to_dict()
                emit("plan_selected",
                     f"RPG plan ({len(plan['steps'])} steps, score={result.score:.2f})",
                     index=0, score=result.score, steps=len(plan["steps"]))
                return plan
            emit("rpg_fallback",
                 f"RPG not usable (stage={result.stage_reached}); using flat planner")
        except Exception as re_:
            emit("rpg_error", f"RPG planning failed: {re_}; using flat planner")

    # PC-206: thinking-aware infrastructure shipped — planner CAN run with
    # Template-level reasoning ON via ATLAS_PLAN_THINKING=1. Default is OFF
    # because empirically on the reference local model + this codebase's
    # hardware tier, thinking pushes planner latency from ~5-30s to >4min
    # per candidate (model spends the full token budget reasoning before
    # emitting JSON). On faster GPU tiers the design's aspirational
    # "reasoning > latency cost" trade may be worth it — flip the env var
    # there. When ON, max_tokens jumps to 8192 to fit reasoning + answer.
    plan_thinking = os.environ.get("ATLAS_PLAN_THINKING", "0").lower() in ("1", "true", "yes")
    plan_max_tokens = 8192 if plan_thinking else 2048
    llm = LLMAdapter(progress_callback=progress_callback, thinking=plan_thinking)
    prompt = _build_plan_prompt(user_message, working_dir, project_context)

    candidates: List[Tuple[Optional[dict], float, List[str]]] = []
    # Diverse sampling via temperature spread. Cheap version of V3's
    # PlanSearch — three samples at 0.3 / 0.5 / 0.7 give us breadth
    # without the full plansearch infrastructure.
    temperatures = [0.3, 0.5, 0.7][:n_candidates]
    for i, temp in enumerate(temperatures):
        emit("plan_candidate", f"candidate {i+1}/{n_candidates} (temp={temp})",
             index=i, temperature=temp)
        try:
            # plan_max_tokens varies with thinking mode (PC-206): 2048 covers
            # a 6-step plan + rationale when thinking is off, 8192 leaves
            # room for ~6KB of reasoning preamble plus the JSON answer.
            raw, tokens, t_ms = llm(prompt, temp, plan_max_tokens, 42 + i)
        except Exception as e:
            emit("plan_candidate_error", f"candidate {i+1} failed: {e}", index=i)
            candidates.append((None, 0.0, [f"llm error: {e}"]))
            continue
        plan = _parse_plan_json(raw)
        if plan is None:
            preview = raw[:500] if raw else "(empty)"
            print(f"  [plan] candidate {i+1} unparseable. raw preview:\n{preview}\n",
                  flush=True)
            emit("plan_candidate_unparseable", f"candidate {i+1} didn't parse",
                 index=i)
            candidates.append((None, 0.0, ["unparseable"]))
            continue
        score, reasons = _score_plan(plan, user_message)
        emit("plan_candidate_scored", f"candidate {i+1} score={score:.2f}",
             index=i, score=score, reasons=reasons)
        candidates.append((plan, score, reasons))

    # Pick winner. Tie-break: shorter plan wins (less waffle).
    best_idx = -1
    best_score = -1.0
    best_steps = 999
    for i, (plan, score, _) in enumerate(candidates):
        if plan is None:
            continue
        n_steps = len(plan.get("steps") or [])
        if score > best_score or (score == best_score and n_steps < best_steps):
            best_score = score
            best_steps = n_steps
            best_idx = i

    if best_idx < 0:
        # All candidates failed. Return a minimal fallback so the agent
        # loop doesn't block — the plan-adherence gate will be lenient
        # if it sees this shape.
        emit("plan_failed", "no candidate parsed — returning fallback")
        return {
            "steps": [
                {"id": "s1", "action": "investigate the request and act",
                 "target": working_dir, "why": "planner failed; deferring to agent"},
            ],
            "verify_step": None,
            "candidates_tested": len(candidates),
            "winning_score": 0.0,
            "winning_index": -1,
            "rationale": "planner-pipeline fallback (no parseable candidate)",
            "reasons": ["all candidates failed to parse"],
        }

    plan, score, reasons = candidates[best_idx]
    plan["candidates_tested"] = len([c for c in candidates if c[0] is not None])
    plan["winning_score"] = score
    plan["winning_index"] = best_idx
    plan["reasons"] = reasons
    emit("plan_selected", f"plan {best_idx+1} won (score={score:.2f})",
         index=best_idx, score=score, steps=len(plan.get("steps") or []))
    return plan


# --- AST-edit (GH #39 v1) ----------------------------------------------------
#
# Friendly-selector-driven structural edits. Replaces the model's edit_file
# old_str/new_str pair (which truncates on long blocks: 2716-char Flask
# template hit max_tokens mid-JSON in the May 7 session) with a tree-sitter
# AST node selector.
#
# v1 supports:
#   - Python: function:NAME, class:NAME
#   - HTML:   <tag>
# Single-match enforcement: ambiguous selectors fail with a clear error so
# the model knows to be more specific. Returns new content for the proxy to
# write, preserving the lens-score-before-write pattern that write_file uses.

try:
    import tree_sitter as _ts
    import tree_sitter_python as _tsp
    import tree_sitter_html as _tsh
    _PY_LANG = _ts.Language(_tsp.language())
    _HTML_LANG = _ts.Language(_tsh.language())
    _AST_EDIT_AVAILABLE = True
except ImportError as _e:
    print(f"[ast_edit] tree-sitter not available: {_e} — endpoint will return 501", flush=True)
    _AST_EDIT_AVAILABLE = False
    _PY_LANG = None
    _HTML_LANG = None


def _ast_language_for_path(path: str):
    p = path.lower()
    if p.endswith(".py"):
        return "python", _PY_LANG
    if p.endswith((".html", ".htm")):
        return "html", _HTML_LANG
    return None, None


def _ast_selector_to_query(selector: str, language: str):
    """Translate friendly selector → (tree-sitter query string, target capture).
    Returns (None, None, error_message) for unknown selectors.
    """
    s = selector.strip()
    if language == "python":
        if s.startswith("function:"):
            name = s[len("function:"):].strip()
            if not name:
                return None, None, "selector 'function:' missing name (e.g. 'function:dashboard')"
            return (
                f'(function_definition name: (identifier) @_name (#eq? @_name "{name}")) @target',
                "target", None,
            )
        if s.startswith("class:"):
            name = s[len("class:"):].strip()
            if not name:
                return None, None, "selector 'class:' missing name (e.g. 'class:UserModel')"
            return (
                f'(class_definition name: (identifier) @_name (#eq? @_name "{name}")) @target',
                "target", None,
            )
        return None, None, (
            f"unknown selector '{selector}' for python. Supported: function:NAME, class:NAME"
        )
    if language == "html":
        if s.startswith("<") and s.endswith(">") and len(s) > 2:
            tag = s[1:-1].strip().lower()
            if not tag.replace("-", "").replace("_", "").isalnum():
                return None, None, (
                    f"selector '{selector}' has invalid tag name — use a bare "
                    f"tag like <script> or <body>, not attributes"
                )
            # tree-sitter-html parses <script> and <style> as dedicated
            # script_element / style_element nodes (their bodies are raw
            # JS/CSS, not HTML), NOT generic `element` nodes — so the generic
            # element query matches them 0 times. Target their real node type.
            if tag == "script":
                return "(script_element) @target", "target", None
            if tag == "style":
                return "(style_element) @target", "target", None
            return (
                f'(element (start_tag (tag_name) @_tag (#eq? @_tag "{tag}"))) @target',
                "target", None,
            )
        return None, None, (
            f"unknown selector '{selector}' for html. Supported: <tag> (e.g. <body>, <head>, <h1>, <script>, <style>)"
        )
    return None, None, f"unsupported language: {language}"


# GH #39 point 4: project-aware symbol resolution. Caller (proxy) extracts
# candidate symbols from the user message and ships a file_map of relevant
# project files; we tree-sitter-walk each, build a symbol index, return
# snippets for the symbols that are actually defined in the project.
# Stateless — no caching, fresh index per call. v1 supports Python only.

def _symbol_index_for_python_source(source: bytes):
    """Return list of (name, kind, start_byte, end_byte) for each top-level
    function/class definition in source. Decorator-aware: function with
    @app.route(...) returns the byte range that includes the decorator,
    so callers paste the whole decorated unit."""
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return []
    out = []
    # Walk root children only — top-level definitions. Skip nested functions
    # and methods inside classes for v1 (they'd noise up the index without
    # adding much value for the kinds of references users actually make).
    for node in tree.root_node.children:
        target = node
        kind = None
        if node.type == "function_definition":
            kind = "function"
        elif node.type == "class_definition":
            kind = "class"
        elif node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    target = child
                    kind = "function"
                    break
                if child.type == "class_definition":
                    target = child
                    kind = "class"
                    break
            # Use the wrapper's byte range so the decorator is included
        if not kind:
            continue
        # Find name child of the function/class itself
        name = None
        for child in target.children:
            if child.type == "identifier":
                name = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        if not name:
            continue
        # Use outer node's byte range (decorator wrapper if present)
        out.append((name, kind, node.start_byte, node.end_byte))
    return out


def symbol_index(file_map: dict, candidate_symbols: list, max_snippets: int = 3, max_lines_per_snippet: int = 200) -> dict:
    """Resolve candidate_symbols against a project's Python files.

    file_map: {path: source_text} of project .py files
    candidate_symbols: ['dashboard', 'UserModel', ...] extracted from user msg
    Returns:
        matched: [{name, kind, file, snippet, n_lines}] for symbols defined in the project
        skipped: [{name, reason}] for symbols mentioned but not found
    """
    if not _AST_EDIT_AVAILABLE:
        return {"matched": [], "skipped": [{"name": s, "reason": "tree-sitter not installed"} for s in candidate_symbols]}

    # Build {symbol_name: [(file, kind, start_byte, end_byte)]} index
    index: dict = {}
    for path, source_text in (file_map or {}).items():
        if not path.lower().endswith(".py"):
            continue
        try:
            source_bytes = source_text.encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            continue
        for name, kind, sb, eb in _symbol_index_for_python_source(source_bytes):
            index.setdefault(name, []).append((path, kind, sb, eb, source_bytes))

    matched, skipped, seen = [], [], set()
    for sym in candidate_symbols:
        if sym in seen:
            continue
        seen.add(sym)
        if len(matched) >= max_snippets:
            skipped.append({"name": sym, "reason": "snippet cap reached"})
            continue
        hits = index.get(sym)
        if not hits:
            skipped.append({"name": sym, "reason": "not defined in scanned project files"})
            continue
        if len(hits) > 1:
            # Ambiguous — multiple files define the same symbol. Skip
            # rather than guess; the model can read_file directly if the
            # context matters.
            skipped.append({"name": sym, "reason": f"ambiguous ({len(hits)} definitions)"})
            continue
        path, kind, sb, eb, source_bytes = hits[0]
        snippet_bytes = source_bytes[sb:eb]
        snippet = snippet_bytes.decode("utf-8", errors="replace")
        # Trim very long snippets — keep the head only. The model can
        # read_file for the full content if it actually needs it.
        snippet_lines = snippet.split("\n")
        truncated = False
        if len(snippet_lines) > max_lines_per_snippet:
            snippet = "\n".join(snippet_lines[:max_lines_per_snippet]) + f"\n# ... ({len(snippet_lines) - max_lines_per_snippet} more lines truncated)"
            truncated = True
        matched.append({
            "name": sym,
            "kind": kind,
            "file": path,
            "snippet": snippet,
            "n_lines": len(snippet_lines),
            "truncated": truncated,
        })
    return {"matched": matched, "skipped": skipped}


# GH #39 point 1: structural verification of V3 candidates.
#
# Sandbox tests whether code RUNS; structural verification tests whether
# the candidate's calls actually resolve. The two answer different
# questions — sandbox can pass for code with try/except ImportError
# fallbacks, lazy imports, or dead branches that never execute the
# unresolved call. Tree-sitter sees what sandbox can't.
#
# v1 supports Python only. Direct-identifier calls only (skips method
# calls like `obj.foo()` and chained calls — they'd need import-graph
# resolution that's a v2 problem). Resolution order:
#   1. Local function/class definition in the same file
#   2. Imported name (top-of-file imports only, no conditional imports)
#   3. Python builtin
#   4. Project-wide symbol (any function/class in any scanned file)
# Anything that doesn't match → unresolved. Strict: 1+ unresolved → veto.

PY_BUILTINS = frozenset({
    # Subset of common builtins — anything heavily used in idiomatic
    # Python that we don't want to false-positive on. Hand-curated to
    # keep the set small; obscure builtins (intern, breakpoint,
    # __import__) caught here are vanishingly rare in generated code.
    "print", "len", "range", "str", "int", "float", "bool", "bytes", "bytearray",
    "list", "dict", "tuple", "set", "frozenset", "complex",
    "open", "input", "sum", "min", "max", "abs", "round", "pow", "divmod",
    "type", "isinstance", "issubclass", "callable",
    "getattr", "setattr", "hasattr", "delattr", "super",
    "enumerate", "zip", "sorted", "reversed", "map", "filter", "any", "all",
    "id", "vars", "dir", "iter", "next", "slice",
    "ord", "chr", "hex", "oct", "bin", "repr", "format", "hash",
    "eval", "exec", "compile", "globals", "locals",
    "object", "classmethod", "staticmethod", "property",
    # Common exception classes — frequently raised, treated as calls
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "IndexError", "RuntimeError", "AttributeError", "ImportError",
    "FileNotFoundError", "IOError", "OSError", "StopIteration",
    "GeneratorExit", "NotImplementedError", "ZeroDivisionError",
    "ArithmeticError", "OverflowError", "AssertionError", "LookupError",
    "MemoryError", "NameError", "ReferenceError", "SyntaxError",
    "SystemExit", "UnicodeError", "Warning", "DeprecationWarning",
})


def _extract_python_imports(source: bytes) -> set:
    """Names introduced into the file's namespace by import statements.

    Handles `import foo`, `import foo.bar`, `import foo as bar`,
    `from foo import bar`, `from foo import bar as baz`. Doesn't track
    star imports — `from foo import *` returns nothing because we don't
    know what's in `foo` without resolving the import. Star imports are
    a known v1 gap; conservative behavior is "treat the file's calls
    as more likely unresolved" rather than silently passing them.
    """
    if not _AST_EDIT_AVAILABLE:
        return set()
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return set()

    imported = set()

    def text_of(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def walk(node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    # `import foo.bar` introduces `foo` into namespace
                    imported.add(text_of(child).split(".")[0])
                elif child.type == "aliased_import":
                    # `import foo as bar` — alias is the trailing identifier
                    last_ident = None
                    for c in child.children:
                        if c.type == "identifier":
                            last_ident = c
                    if last_ident is not None:
                        imported.add(text_of(last_ident))
        elif node.type == "import_from_statement":
            past_import_kw = False
            for child in node.children:
                if not past_import_kw:
                    if child.type == "import" or text_of(child) == "import":
                        past_import_kw = True
                    continue
                # After `import` keyword: dotted_name, identifier,
                # aliased_import, or wildcard_import
                if child.type == "dotted_name":
                    imported.add(text_of(child).split(".")[0])
                elif child.type == "identifier":
                    imported.add(text_of(child))
                elif child.type == "aliased_import":
                    last_ident = None
                    for c in child.children:
                        if c.type == "identifier":
                            last_ident = c
                    if last_ident is not None:
                        imported.add(text_of(last_ident))
                elif child.type == "wildcard_import":
                    # `from foo import *` — can't enumerate without
                    # resolving the import. Best we can do: bail out
                    # of strict mode for this file by adding a sentinel.
                    imported.add("*")
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return imported


def _extract_python_call_targets(source: bytes) -> list:
    """All direct-identifier call targets. Skips attribute / subscript /
    chained calls — those need full import-graph resolution and are out
    of scope for v1. Returns a list (not set) because duplicate calls
    matter when reporting — caller may dedup later."""
    if not _AST_EDIT_AVAILABLE:
        return []
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return []

    out = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            # `function:` field is the first non-paren child
            for child in node.children:
                if child.type == "identifier":
                    out.append(source[child.start_byte:child.end_byte].decode("utf-8", errors="replace"))
                    break
                # attribute / subscript / lambda → skip silently
                # so we don't false-positive on `obj.method()`
                if child.type not in ("(",):
                    break
        stack.extend(node.children)
    return out


def _extract_python_top_level_defs(source: bytes) -> set:
    """Top-level function and class names defined in the file. Used as
    one input to call resolution. Skips nested functions and class
    methods — those don't introduce names into the file's top-level
    namespace."""
    if not _AST_EDIT_AVAILABLE:
        return set()
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return set()

    names = set()
    for node in tree.root_node.children:
        target = node
        if node.type == "decorated_definition":
            for c in node.children:
                if c.type in ("function_definition", "class_definition"):
                    target = c
                    break
        if target.type in ("function_definition", "class_definition"):
            for c in target.children:
                if c.type == "identifier":
                    names.add(source[c.start_byte:c.end_byte].decode("utf-8", errors="replace"))
                    break
    return names


def build_project_symbols(file_map: dict) -> set:
    """Aggregate top-level function/class names across every .py file
    in file_map. Built once per V3 run, reused across all candidates."""
    out = set()
    for path, source_text in (file_map or {}).items():
        if not path.lower().endswith(".py"):
            continue
        try:
            out |= _extract_python_top_level_defs(source_text.encode("utf-8"))
        except Exception:
            continue
    return out


def structural_score(project_symbols, candidate_code: str) -> dict:
    """Check a candidate for unresolved direct-identifier calls.

    project_symbols: set built by build_project_symbols(file_map). Pass
    {} or set() if the project is empty / unavailable — every call
    will fall through to imports/builtins/unresolved.

    Returns:
        ok: True if parse succeeded
        n_calls_total / n_unresolved: aggregate counts
        unresolved_calls: list of unique unresolved names (capped at 10)
        wildcard_imports: True if the candidate has `from x import *`,
                          which makes the unresolved set a lower bound
    """
    if not _AST_EDIT_AVAILABLE:
        return {"ok": False, "error": "tree-sitter not installed"}
    try:
        candidate_bytes = candidate_code.encode("utf-8")
    except (UnicodeEncodeError, AttributeError) as e:
        return {"ok": False, "error": f"candidate not utf-8: {e}"}

    try:
        local_defs = _extract_python_top_level_defs(candidate_bytes)
        imports = _extract_python_imports(candidate_bytes)
        calls = _extract_python_call_targets(candidate_bytes)
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {type(e).__name__}: {e}"}

    has_wildcard = "*" in imports
    if has_wildcard:
        # Star import in scope → can't reliably mark anything unresolved.
        # Be lenient and only flag calls that aren't obviously local /
        # builtin — wildcard might supply the rest.
        pass

    unresolved = []
    seen_unresolved = set()
    for name in calls:
        if name in seen_unresolved:
            continue
        if name in local_defs:
            continue
        if name in imports:
            continue
        if name in PY_BUILTINS:
            continue
        if name in (project_symbols or set()):
            continue
        if has_wildcard:
            # Wildcard import might supply this — treat as resolved-by-
            # wildcard rather than unresolved. False negatives possible
            # but better than blocking valid code.
            continue
        seen_unresolved.add(name)
        unresolved.append(name)

    return {
        "ok": True,
        "n_calls_total": len(calls),
        "n_unresolved": len(unresolved),
        "unresolved_calls": unresolved[:10],
        "wildcard_imports": has_wildcard,
        "n_local_defs": len(local_defs),
        "n_imports": len(imports),
    }


# GH #39 point 3: Phase 3 repair with call-chain context.
#
# When all candidates fail sandbox and we drop to PR-CoT / refinement,
# the repair model gets `error` (raw stderr) + `code` (the failing
# candidate). It has to guess from the traceback alone what the
# failing function does inside the project, who calls it, what it
# depends on. With a call graph we can hand it that context directly.
#
# v1 approach: parse the deepest frame from a Python traceback to get
# the failing function name, then walk file_map to find:
#   - which file defines that function
#   - which other project functions call it (direct callers, 1 hop)
#   - which other project functions IT calls (direct callees, 1 hop)
# Format as a markdown block, append to the error field passed to
# PR-CoT / refinement so the repair LLM sees it as part of failure
# context.

# Python traceback frame: `File "path", line N, in funcname`
_TRACEBACK_FRAME_RE = re.compile(r'File "[^"]+", line \d+, in (\S+)')


def _failing_function_from_stderr(stderr: str):
    """Return the deepest function name in a Python traceback, or None
    if stderr doesn't look like a traceback. The deepest frame is the
    one nearest the actual error; earlier frames are callers."""
    if not stderr:
        return None
    matches = _TRACEBACK_FRAME_RE.findall(stderr)
    if not matches:
        return None
    # Filter sentinels — `<module>`, `<lambda>`, `<genexpr>` aren't
    # callable names we can look up. Walk back until we find one.
    for name in reversed(matches):
        if not name.startswith("<"):
            return name
    return None


def _python_call_targets_per_function(source: bytes):
    """Return {function_name: list[called_identifier_names]} for the
    file. Top-level functions only; class methods aggregate under their
    class name (we don't track method-level callers in v1)."""
    if not _AST_EDIT_AVAILABLE:
        return {}
    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source)
    except Exception:
        return {}

    out = {}

    def text_of(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    for node in tree.root_node.children:
        target = node
        if node.type == "decorated_definition":
            for c in node.children:
                if c.type in ("function_definition", "class_definition"):
                    target = c
                    break
        if target.type not in ("function_definition", "class_definition"):
            continue
        # Find function/class name
        name = None
        for c in target.children:
            if c.type == "identifier":
                name = text_of(c)
                break
        if not name:
            continue
        # Extract direct-identifier calls from the function body
        calls = []
        stack = list(target.children)
        while stack:
            n = stack.pop()
            if n.type == "call":
                for child in n.children:
                    if child.type == "identifier":
                        calls.append(text_of(child))
                        break
                    if child.type not in ("(",):
                        break
            stack.extend(n.children)
        out[name] = calls
    return out


def call_chain_context(file_map: dict, function_name: str, max_callers: int = 6, max_callees: int = 6) -> str:
    """Build a markdown block describing direct callers + callees of
    function_name across file_map's project. Returns empty string when
    the function isn't found anywhere — caller should skip injection
    in that case rather than dilute the error context with a useless
    'no matches' block."""
    if not function_name or not file_map or not _AST_EDIT_AVAILABLE:
        return ""

    # Pass 1: per-file map of {func: callees}. Also locate definition.
    per_file = {}  # path -> {func: [calls]}
    defined_in = None
    for path, source_text in file_map.items():
        if not path.lower().endswith(".py"):
            continue
        try:
            src_bytes = source_text.encode("utf-8")
        except (UnicodeEncodeError, AttributeError):
            continue
        funcs = _python_call_targets_per_function(src_bytes)
        per_file[path] = funcs
        if defined_in is None and function_name in funcs:
            defined_in = path

    if defined_in is None:
        return ""

    # Pass 2: callers — any (path, func) where func's body calls function_name
    callers = []
    for path, funcs in per_file.items():
        for fname, calls in funcs.items():
            if fname == function_name and path == defined_in:
                continue  # don't list the function as its own caller
            if function_name in calls:
                callers.append((path, fname))

    # Callees: the target function's own calls
    callees = per_file.get(defined_in, {}).get(function_name, [])
    # Dedup callees while preserving order
    seen = set()
    unique_callees = []
    for c in callees:
        if c in seen:
            continue
        seen.add(c)
        unique_callees.append(c)

    sb = [f"## Call-chain context for failing function `{function_name}`"]
    sb.append("")
    sb.append(f"Defined in: `{defined_in}`")
    sb.append("")

    if callers:
        capped = callers[:max_callers]
        sb.append(f"**Direct callers in project ({len(callers)} found):**")
        for path, fname in capped:
            sb.append(f"- `{fname}` in {path}")
        if len(callers) > max_callers:
            sb.append(f"- ... and {len(callers) - max_callers} more")
        sb.append("")
    else:
        sb.append("**Direct callers in project:** (none found — this function may be an entry point or only called by external code)")
        sb.append("")

    if unique_callees:
        capped = unique_callees[:max_callees]
        sb.append(f"**Functions called by `{function_name}` ({len(unique_callees)} unique):**")
        for c in capped:
            sb.append(f"- `{c}`")
        if len(unique_callees) > max_callees:
            sb.append(f"- ... and {len(unique_callees) - max_callees} more")
        sb.append("")
    else:
        sb.append(f"**Functions called by `{function_name}`:** (none — leaf function)")
        sb.append("")

    sb.append("Use this map to scope your fix: changing what `" + function_name + "` calls may require updating its callers; changing its callees may not.")

    return "\n".join(sb)


def cyclomatic_complexity(path: str, source_text: str) -> dict:
    """McCabe-style cyclomatic complexity from tree-sitter AST.

    Counts decision points across the whole file (sum of per-function CC,
    not strictly McCabe's per-function definition — we want one number for
    tier classification, not a per-symbol map). Decision-point set targets
    the things that actually predict V3-pipeline benefit: branches, loops,
    exception handlers, short-circuit booleans, comprehensions with filters,
    match/case clauses.

    v1 supports Python only. HTML CC isn't meaningful (markup, no real
    branching in tree-sitter's view of it — Jinja control blocks parse as
    text content). Other languages return {"ok": False} so the proxy's
    regex-based classifyFileTier stays the fallback floor.
    """
    if not _AST_EDIT_AVAILABLE:
        return {"ok": False, "error": "tree-sitter not installed in this build"}

    p = (path or "").lower()
    if not p.endswith(".py"):
        return {"ok": False, "error": f"cyclomatic_complexity v1 supports .py only (got {path})"}

    try:
        parser = _ts.Parser(_PY_LANG)
        tree = parser.parse(source_text.encode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {type(e).__name__}: {e}"}

    # Decision-point node types in Python's tree-sitter grammar.
    # Each adds 1 to CC. `if_clause` inside a comprehension is the
    # filter clause (e.g. `[x for x in xs if x > 0]`) and counts as a branch.
    DECISION = {
        "if_statement", "elif_clause",
        "for_statement", "while_statement",
        "except_clause",
        "conditional_expression",  # ternary x if cond else y
        "boolean_operator",        # and / or short-circuit
        "case_clause",             # match-case
        "if_clause",               # comprehension filter
    }

    cc = 1  # base path
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type in DECISION:
            cc += 1
        stack.extend(n.children)

    return {"ok": True, "language": "python", "cyclomatic_complexity": cc}


def ast_edit(path: str, source_text: str, selector: str, content: str) -> dict:
    """Apply a friendly-selector AST edit. Stateless transform — caller provides
    the source bytes (read from their own filesystem) and gets back new content.
    v3-service does no file IO; the proxy reads + writes via its existing
    workspace mount, which keeps lens-score-before-write intact."""
    if not _AST_EDIT_AVAILABLE:
        return {"success": False, "error": "ast_edit unavailable: tree-sitter not installed in this v3-service build"}

    # Empty-content guard (defense-in-depth; the proxy also checks). Splicing
    # empty content over a node deletes it — a model that omits `content`
    # would silently remove the function instead of fixing it.
    if not content.strip():
        return {"success": False, "error": (
            f"ast_edit: content is empty — that would DELETE '{selector}', not edit it. "
            f"Provide the full replacement body of the node."
        )}

    language, lang_obj = _ast_language_for_path(path)
    if not language:
        return {"success": False, "error": (
            f"unsupported file type for ast_edit: {path}. v1 supports .py, .html, .htm — "
            f"use edit_file for other languages."
        )}

    query_str, target_cap, err = _ast_selector_to_query(selector, language)
    if err:
        return {"success": False, "error": err}

    try:
        source = source_text.encode("utf-8")
    except (UnicodeEncodeError, AttributeError) as e:
        return {"success": False, "error": f"source not valid utf-8 string: {e}"}

    try:
        parser = _ts.Parser(lang_obj)
        tree = parser.parse(source)
        query = _ts.Query(lang_obj, query_str)
        # tree_sitter ≥0.23 moved captures off Query onto QueryCursor; older
        # versions exposed Query.captures directly. Support both so the
        # service works whichever wheel pip resolves.
        if hasattr(_ts, "QueryCursor"):
            captures = _ts.QueryCursor(query).captures(tree.root_node)
        else:
            captures = query.captures(tree.root_node)
    except Exception as e:
        return {"success": False, "error": f"tree-sitter parse/query error: {type(e).__name__}: {e}"}

    targets = captures.get(target_cap, [])
    if len(targets) == 0:
        # Ground the retry in the file's REAL symbols. A weak model
        # hallucinates selectors for functions that don't exist
        # (observed: function:get_inventory_count, function:calculate_inventory
        # against a file that defines item_subtotal / total_value). The lens
        # can't catch this — the replacement text is plausible code; the
        # TARGET is the problem. Listing what's actually defined turns a
        # dead-end "verify the symbol exists" into an actionable retry.
        available = ""
        if language == "python":
            try:
                names = []
                for name, kind, _sb, _eb in _symbol_index_for_python_source(source_text.encode("utf-8")):
                    names.append(f"{kind}:{name}")
                if names:
                    available = " This file defines: " + ", ".join(names[:30]) + ". Use one of these exact selectors, or read the file to confirm."
            except Exception:
                available = ""
        return {"success": False, "error": (
            f"selector '{selector}' matched 0 nodes in {path} — that symbol does not exist in this file."
            + (available or " Read the file first to see what's defined.")
        )}
    if len(targets) > 1:
        return {"success": False, "error": (
            f"selector '{selector}' matched {len(targets)} nodes in {path}. "
            f"ast_edit requires exactly one match — use a more specific selector."
        )}

    target = targets[0]
    # Python grammar wraps decorated functions/classes in decorated_definition.
    # function:dashboard matches the inner function_definition; if its parent
    # is decorated_definition we want THAT byte range so @app.route(...) lines
    # get replaced too. Otherwise the model writes new @decorator lines and
    # the old ones stay, double-decorating the function.
    if language == "python" and target.parent is not None and target.parent.type == "decorated_definition":
        target = target.parent
    try:
        new_bytes = source[:target.start_byte] + content.encode("utf-8") + source[target.end_byte:]
        new_content = new_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        return {"success": False, "error": f"replacement produced invalid utf-8: {e}"}

    # Post-splice syntax gate (Python). Tree-sitter is error-tolerant: it
    # happily locates the node and splices in replacement content that is
    # not valid Python — observed live: a model emitted `item["id""]` and
    # `&quot;`-escaped quotes, ast_edit reported success, and a previously
    # runnable Flask app shipped with a SyntaxError. Refuse to hand back a
    # broken file; return the parse error so the model can fix its quoting
    # on the retry. Keyed off file type, not the model.
    if language == "python":
        try:
            compile(new_content, path, "exec")
        except SyntaxError as e:
            snippet = (e.text or "").strip()
            return {"success": False, "error": (
                f"ast_edit: the replacement makes {path} invalid Python — "
                f"SyntaxError at line {e.lineno}: {e.msg}"
                + (f" (offending line: {snippet})" if snippet else "")
                + '. The file was NOT modified. Check your quoting (no doubled '
                  'quotes like ["id""], no escaped \\" inside the content, no '
                  'HTML entities like &quot;) and re-emit the full node.'
            )}

    return {
        "success": True,
        "language": language,
        "selector": selector,
        "new_content": new_content,
        "byte_range": [target.start_byte, target.end_byte],
        "old_size": len(source),
        "new_size": len(new_bytes),
    }


# --- HTTP Handler (SSE streaming) --------------------------------------------

pipeline = V3PipelineService()


class V3Handler(BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        """Token check for every route except /health. Constant-time
        compare; 401 bodies never echo token material."""
        if not SERVICE_TOKEN or self.path == "/health":
            return True
        import hmac
        got = self.headers.get("Authorization", "")
        want = f"Bearer {SERVICE_TOKEN}"
        if hmac.compare_digest(got, want):
            return True
        self._json_response(401, {
            "error": "unauthorized",
            "detail": "internal service auth is enabled; send "
                      "Authorization: Bearer <service-token> "
                      "(secrets/service-token)"})
        return False

    def do_POST(self):
        from structured_log import set_request_id as _set_rid
        _set_rid(self.headers.get("X-ATLAS-Request-ID", ""))
        if not self._authorized():
            return
        if self.path == "/v3/run":
            self._handle_run()
        elif self.path == "/v3/generate":
            self._handle_generate()
        elif self.path == "/v3/plan":
            self._handle_plan()
        elif self.path == "/internal/ast_edit":
            self._handle_ast_edit()
        elif self.path == "/internal/cyclomatic_complexity":
            self._handle_cyclomatic_complexity()
        elif self.path == "/internal/symbol_index":
            self._handle_symbol_index()
        elif self.path == "/internal/call_graph":
            self._handle_call_graph()
        elif self.path == "/internal/outline":
            self._handle_outline()
        elif self.path == "/internal/pycheck":
            self._handle_pycheck()
        elif self.path == "/internal/structural_check":
            self._handle_structural_check()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_GET(self):
        from structured_log import set_request_id as _set_rid
        _set_rid(self.headers.get("X-ATLAS-Request-ID", ""))
        if not self._authorized():
            return
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "v3-pipeline"})
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_run(self):
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"invalid JSON body: {e}"})
            return

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

            def emit_sse(stage, detail="", **data):
                payload = {"stage": stage, "detail": detail}
                if data:
                    payload["data"] = data
                event = json.dumps(payload)
                try:
                    self.wfile.write(f"data: {event}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Client gone — flag it so pipeline.run aborts at the
                    # next phase boundary instead of grinding on.
                    emit_sse.disconnected = True
                except Exception:
                    # best-effort: swallow on failure (caller continues)
                    pass

            try:
                result = pipeline.run(problem, task_id, progress_callback=emit_sse, files=files)
            except ClientDisconnected as e:
                print(f"[run] pipeline aborted: {e}", flush=True)
                return
            _post_pattern_outcome(problem, result)

            # Final result event
            final = json.dumps(result, default=str)
            self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            result = pipeline.run(problem, task_id, files=files)
            _post_pattern_outcome(problem, result)
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
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"invalid JSON body: {e}"})
            return

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

        print(f"[generate] file={file_path} framework={framework} tier=T{tier}", flush=True)
        if build_command:
            print(f"[generate] requested build command: {build_command}", flush=True)
        print(f"[generate] constraints: {constraints}", flush=True)

        # Stream V3 pipeline progress as SSE events, then final result as JSON
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit_progress(stage, detail="", **data):
            """Stream progress events to the Go proxy."""
            payload = {"stage": stage, "detail": detail}
            if data:
                payload["data"] = data
            event = json.dumps(payload)
            try:
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
                # Also log for debugging
                print(f"  [SSE] {stage}: {detail[:80]}", flush=True)
            except (BrokenPipeError, ConnectionResetError):
                # Client gone — flag it so pipeline.run aborts at the
                # next phase boundary instead of grinding on.
                emit_progress.disconnected = True
            except Exception as e:
                print(f"  [SSE ERROR] {e}", flush=True)

        # Run V3 pipeline with streaming progress
        try:
            result = pipeline.run(
                problem=problem,
                task_id=f"gen-{Path(file_path).stem}",
                progress_callback=emit_progress,
                files=files,
                file_path=file_path,  # PC-048: language-aware smoke check
                build_command=build_command,
                working_dir=working_dir or "/workspace",
                constraints=constraints,  # V3.2: RPG signature veto (issue #120)
            )
        except ClientDisconnected as e:
            print(f"[generate] pipeline aborted: {e}", flush=True)
            return
        _post_pattern_outcome(problem, result)

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
            "winning_score": result.get("winning_score", 0.0),
            "total_tokens": result.get("total_tokens", 0),
            "total_time_ms": result.get("total_time_ms", 0.0),
            "verification_evidence": result.get("verification_evidence", []),
        }

        # V3.2 RPG (issue #120): report which planned signatures the WINNING
        # code failed to realize, so the proxy's re-plan loop can react (the
        # signature veto rejects failing candidates mid-pipeline, but the
        # winner can still drift when every candidate fell short and one was
        # kept anyway). Flag-gated and conservative — empty unless RPG is on,
        # constraints carry signatures, and a planned function is genuinely
        # absent from parseable code.
        try:
            from wavelet import rpg_planning_enabled as _rpg_on
            _rpg_active = _rpg_on()
        except Exception:
            _rpg_active = False
        if _rpg_active and response["code"] and constraints:
            try:
                import rpg as _rpgmod
                planned_sigs = _rpgmod.planned_signatures_from_constraints(constraints)
                missing = _rpgmod.missing_planned_signatures(
                    response["code"], planned_sigs, file_path)
                if missing:
                    response["rpg_signature_missing"] = missing
            except Exception as _e:
                print(f"  [rpg] drift check skipped: {_e}", flush=True)

        final = json.dumps(response)
        try:
            self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client closed mid-stream (timed out, cancelled, etc).
            # See ISSUES.md PC-026.
            pass

    def _handle_plan(self):
        """Handle /v3/plan — generate a structured plan for an agent task.

        Same SSE shape as /v3/generate so the Go proxy's SSE parser can
        reuse its frame-reading logic; only the stage names and the
        final result envelope differ.

        Request:
            user_message: str       — the prompt the user typed
            working_dir: str        — proxy's container working dir
            project_context: dict   — files the agent has read so far
            tier: int               — 2 or 3
            n_candidates: int       — optional; default 3

        Response (event: result):
            steps: list[dict]
            verify_step: str | null
            candidates_tested: int
            winning_score: float
            winning_index: int
            rationale: str
            reasons: list[str]
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"invalid JSON body: {e}"})
            return

        user_message = body.get("user_message", "")
        working_dir = body.get("working_dir", "")
        project_context = body.get("project_context", {}) or {}
        n_candidates = int(body.get("n_candidates", 3))

        if not user_message:
            self._json_response(400, {"error": "user_message required"})
            return

        print(f"[plan] msg={user_message[:80]!r} cwd={working_dir} files={len(project_context)} n={n_candidates}",
              flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit_progress(stage, detail="", **data):
            payload = {"stage": stage, "detail": detail}
            if data:
                payload["data"] = data
            event = json.dumps(payload)
            try:
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.flush()
                print(f"  [SSE plan] {stage}: {detail[:80]}", flush=True)
            except BrokenPipeError:
                # best-effort: swallow on failure (caller continues)
                pass
            except Exception as e:
                print(f"  [SSE plan ERROR] {e}", flush=True)

        try:
            plan = generate_plan(
                user_message=user_message,
                working_dir=working_dir,
                project_context=project_context,
                n_candidates=n_candidates,
                progress_callback=emit_progress,
            )
        except Exception as e:
            print(f"  [plan ERROR] {e}", flush=True)
            plan = {
                "steps": [], "verify_step": None,
                "candidates_tested": 0, "winning_score": 0.0,
                "winning_index": -1,
                "rationale": f"planner failed: {e}",
                "reasons": [str(e)],
            }

        final = json.dumps(plan)
        try:
            self.wfile.write(f"event: result\ndata: {final}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # best-effort: swallow on failure (caller continues)
            pass

    def _handle_ast_edit(self):
        """POST /internal/ast_edit — friendly-selector-driven AST edit.

        Request:
            {"path": "...",  "source": "<full file content>",
             "selector": "function:foo" | "<body>" | ..., "content": "..."}
        Response (success):
            {"success": true, "new_content": "...", "byte_range": [start, end],
             "old_size": int, "new_size": int, "language": "python" | "html"}
        Response (failure):
            {"success": false, "error": "..."}

        Stateless transform — caller (the proxy) reads the file, sends content
        in, gets new content out. v3-service does no file IO; proxy writes
        after lens-scoring, matching write_file's flow so PC-207 lens-veto
        can still reject stub-shaped replacements.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"success": False, "error": f"invalid JSON body: {e}"})
            return

        path = body.get("path", "")
        source_text = body.get("source", "")
        selector = body.get("selector", "")
        content = body.get("content", "")
        if not path or not selector or not source_text:
            self._json_response(400, {"success": False, "error": "missing required field(s): path, source, selector"})
            return

        result = ast_edit(path, source_text, selector, content)
        # Log per-call signal — matches the verbose-logging pattern we added
        # to score_candidate_per_step. Lets `docker logs atlas-v3-service-1`
        # answer "what did the model ask ast_edit to do" without SSE capture.
        if result.get("success"):
            print(
                f"  [ast_edit] {result['language']} {path} selector={selector!r} "
                f"matched bytes [{result['byte_range'][0]}-{result['byte_range'][1]}] "
                f"old={result['old_size']}B new={result['new_size']}B",
                flush=True,
            )
        else:
            print(f"  [ast_edit] FAIL path={path} selector={selector!r}: {result['error']}", flush=True)
        self._json_response(200, result)

    def _handle_symbol_index(self):
        """POST /internal/symbol_index — resolve candidate symbols to project snippets.

        Request:
            {"file_map": {"app.py": "...", "utils.py": "..."},
             "symbols": ["dashboard", "UserModel", ...],
             "max_snippets": 3, "max_lines_per_snippet": 200}
        Response:
            {"matched": [{name, kind, file, snippet, n_lines, truncated}],
             "skipped": [{name, reason}]}

        Caps default to 3 snippets / 200 lines each. Caller is the proxy,
        which extracts symbols from the user message via regex and walks
        the working directory for .py files (with its own size cap).
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"matched": [], "skipped": [], "error": f"invalid JSON body: {e}"})
            return

        file_map = body.get("file_map") or {}
        symbols = body.get("symbols") or []
        max_snippets = int(body.get("max_snippets", 3))
        max_lines = int(body.get("max_lines_per_snippet", 200))

        result = symbol_index(file_map, symbols, max_snippets=max_snippets, max_lines_per_snippet=max_lines)
        n_matched = len(result.get("matched", []))
        n_skipped = len(result.get("skipped", []))
        n_files = len(file_map)

        # Phase 3 (#39 point 4): when the call graph is enabled, attach each
        # matched symbol's graph neighborhood (callers / callees / impact) so
        # the proxy can inject structurally related code instead of name-matched
        # snippets alone. Additive: the "matched"/"skipped" shape is unchanged,
        # so flag-off callers see exactly today's response.
        try:
            from graph import call_graph_enabled, symbol_neighborhood, build_graph
            if call_graph_enabled() and result.get("matched"):
                # Build the project graph ONCE, then neighborhood each matched
                # symbol against it (not once per symbol).
                g = build_graph(file_map)
                related = []
                for m in result["matched"]:
                    nb = symbol_neighborhood(file_map, m["name"], graph=g)
                    if nb["callers"] or nb["callees"] or nb["impact"]:
                        related.append(nb)
                if related:
                    result["graph"] = related
        except Exception as cge:
            print(f"  [symbol_index] graph neighborhood skipped: {cge}", flush=True)

        print(
            f"  [symbol_index] {n_files} files, {len(symbols)} candidates → "
            f"matched={n_matched} skipped={n_skipped}",
            flush=True,
        )
        self._json_response(200, result)

    def _handle_call_graph(self):
        """POST /internal/call_graph — structural call-graph query (issue #39).

        Builds (cached) a project call graph from the supplied files and runs a
        native analysis on it. No solver; O(V+E) traversal.

        Request:
            {"file_map": {"app.py": "...", "pkg/util.py": "..."},
             "analysis": "callers" | "callees" | "reachability" | "path" |
                         "impact" | "cycles" | "dead-code" | "entry-points" | "facts",
             "target": "<name>",        # callers/callees/impact
             "from": "<name>", "to": "<name>",   # reachability/path
             "entry_points": ["main"]}  # dead-code (optional)
        Response:
            {"ok": true, "analysis": "...", "result": <analysis result>}
            or {"ok": false, "error": "..."}

        Gated by ATLAS_CALL_GRAPH; returns ok=false when the flag is off so the
        feature stays inert until enabled.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return

        try:
            from graph import (
                build_graph, run_analysis, graph_to_prolog, call_graph_enabled,
                reachable_pairs,
            )
        except Exception as e:  # pragma: no cover - import guard
            self._json_response(200, {"ok": False, "error": f"graph package unavailable: {e}"})
            return

        if not call_graph_enabled():
            self._json_response(200, {"ok": False, "error": "ATLAS_CALL_GRAPH disabled"})
            return

        file_map = body.get("file_map") or {}
        analysis = body.get("analysis", "facts")
        try:
            g = build_graph(file_map)
            if analysis == "facts":
                # Prolog facts + rules for an external solver (chiasmus_verify / SWI).
                result = graph_to_prolog(g, entry_points=body.get("entry_points"))
            elif analysis == "closure":
                # Phase 5: the transitive reaches/2 relation via the in-process
                # Datalog engine — a relation the native single-pair API doesn't
                # expose directly. Returned as [from, to] pairs.
                result = [list(p) for p in reachable_pairs(g)]
            else:
                result = run_analysis(
                    g, analysis,
                    target=body.get("target"),
                    frm=body.get("from"),
                    to=body.get("to"),
                    entry_points=body.get("entry_points"),
                )
        except ValueError as e:
            self._json_response(400, {"ok": False, "error": str(e)})
            return
        except Exception as e:
            self._json_response(200, {"ok": False, "error": f"call_graph failed: {e}"})
            return

        print(f"  [call_graph] {len(file_map)} files, analysis={analysis} → ok", flush=True)
        self._json_response(200, {"ok": True, "analysis": analysis, "result": result})

    def _handle_pycheck(self):
        """POST /internal/pycheck — does this Python source parse?

        Request:  {"path": "app.py", "source": "<file text>"}
        Response: {"ok": bool, "error": "...", "line": N}

        Used by the proxy's edit_file path to refuse writing a .py file the
        edit would break — the same gate ast_edit applies post-splice. Pure
        compile() check, no execution.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return
        path = body.get("path", "") or "<edit>"
        source = body.get("source", "") or ""
        try:
            compile(source, path, "exec")
            self._json_response(200, {"ok": True})
        except SyntaxError as e:
            snippet = (e.text or "").strip()
            msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if snippet:
                msg += f" (offending line: {snippet})"
            self._json_response(200, {"ok": False, "error": msg, "line": e.lineno or 0})

    def _handle_structural_check(self):
        """POST /internal/structural_check — does every direct-identifier call
        in this Python source resolve (local def, import, builtin, or a
        supplied project symbol)?

        Request:  {"path": "app.py", "source": "<file text>",
                   "project_context": {"other.py": "..."}}   # optional
        Response: {"ok": bool, "unresolved": ["render_template", ...],
                   "wildcard_imports": bool}

        Used by the proxy's edit_file/ast_edit path to refuse writing a .py
        file whose edit introduces a NameError — a direct call to a name the
        file neither imports, defines, nor gets from builtins (#147:
        render_template called with only render_template_string imported
        landed as verified because the in-pipeline veto was skipped when no
        project_context was sent). Same resolver as the V3 structural veto;
        no execution. `ok:false` means the check couldn't run (tree-sitter
        missing / parse error) — the caller treats that as pass (fail-open).
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return
        source = body.get("source", "") or ""
        project_symbols = build_project_symbols(body.get("project_context") or {})
        struct = structural_score(project_symbols, source)
        if not struct.get("ok"):
            # couldn't parse / tree-sitter unavailable → fail-open
            self._json_response(200, {"ok": False, "error": struct.get("error", "unavailable")})
            return
        self._json_response(200, {
            "ok": True,
            "unresolved": struct.get("unresolved_calls", []),
            "n_unresolved": struct.get("n_unresolved", 0),
            "wildcard_imports": struct.get("wildcard_imports", False),
        })

    def _handle_outline(self):
        """POST /internal/outline — list a file's top-level functions/classes.

        Request:  {"path": "app.py", "source": "<file text>"}
        Response: {"symbols": [{name, kind, start_line, end_line}], "supported": bool}

        Reuses the same decorator-aware tree-sitter walk ast_edit uses, so a
        symbol the outline names is selectable by ast_edit with the same name.
        Bodies are NOT returned — this is the cheap "what's in here" probe so
        the model can then read just the one function's line range instead of
        the whole file. .py only here; the proxy regex-falls-back for the rest.

        When ATLAS_CALL_GRAPH is on, each symbol also carries its intra-file
        call-graph neighborhood (`calls` / `called_by`). The outline is the
        artifact the model inspects right before it decides WHICH symbol to
        edit, so this is where structural context earns its keep: it lets the
        model follow `total_value -> item_subtotal` to a callee-rooted bug
        instead of editing the function where the symptom merely surfaces
        (issue #39). Scoped to this one file — no project-wide scan — so it's
        cheap and never misses the file in a large repo. Additive: the
        symbols/supported shape is unchanged, so flag-off callers see exactly
        today's response.
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"symbols": [], "supported": False, "error": f"invalid JSON body: {e}"})
            return
        path = body.get("path", "") or ""
        source = body.get("source", "") or ""
        symbols = []
        supported = False
        if path.endswith(".py") and _AST_EDIT_AVAILABLE:
            supported = True
            src = source.encode("utf-8")
            for name, kind, sb_, eb in _symbol_index_for_python_source(src):
                symbols.append({
                    "name": name,
                    "kind": kind,
                    "start_line": src[:sb_].count(b"\n") + 1,
                    "end_line": src[:eb].count(b"\n") + 1,
                })

        # Call-graph neighborhood (issue #39, flag-gated). Build the single-file
        # graph once and attach callers/callees to each symbol the model can see.
        if symbols:
            try:
                from graph import call_graph_enabled, symbol_neighborhood, build_graph
                if call_graph_enabled():
                    file_map = {path: source}
                    g = build_graph(file_map)
                    for s in symbols:
                        nb = symbol_neighborhood(file_map, s["name"], graph=g)
                        if nb["callees"]:
                            s["calls"] = nb["callees"]
                        if nb["callers"]:
                            s["called_by"] = nb["callers"]
            except Exception as cge:  # pragma: no cover - import/extract guard
                print(f"  [outline] call-graph neighborhood skipped: {cge}", flush=True)

        self._json_response(200, {"symbols": symbols, "supported": supported})

    def _handle_cyclomatic_complexity(self):
        """POST /internal/cyclomatic_complexity — McCabe CC for tier classification.

        Request:  {"path": "...", "source": "<full file content>"}
        Response: {"ok": true, "language": "python", "cyclomatic_complexity": 12}
                  or {"ok": false, "error": "..."}
        """
        content_len = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(content_len) or b"{}")
        except json.JSONDecodeError as e:
            self._json_response(400, {"ok": False, "error": f"invalid JSON body: {e}"})
            return

        path = body.get("path", "")
        source_text = body.get("source", "")
        if not path or source_text == "":
            self._json_response(400, {"ok": False, "error": "missing required field(s): path, source"})
            return

        result = cyclomatic_complexity(path, source_text)
        # Per-call signal — same pattern as ast_edit. Lets us correlate
        # tier upgrades to the file that triggered them in docker logs.
        if result.get("ok"):
            print(
                f"  [cc] {result['language']} {path} cc={result['cyclomatic_complexity']}",
                flush=True,
            )
        # Don't log the not-supported case — it'd flood the log on every HTML/JSON write.
        self._json_response(200, result)

    def _json_response(self, code, data):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except (BrokenPipeError, ConnectionResetError):
            # Client closed before we finished writing — typically a
            # docker healthcheck that hit its timeout. Not actionable.
            # See ISSUES.md PC-026.
            pass

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


# --- Main --------------------------------------------------------------------

class _PrivateValueStream:
    """Line-filtering wrapper for stdout/stderr: the v3 service logs
    via print(), so the stream is the serialization choke point (the
    equivalent of the root-handler filter in the FastAPI services)."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        from private_values import filter_private_values
        filtered = filter_private_values(text)
        if os.environ.get("ATLAS_LOG_FORMAT", "").lower() == "json" \
                and filtered.strip():
            # Wrap each non-empty print line as a structured record so v3
            # matches the other services' JSON logs (it logs via print()).
            import json as _json
            from structured_log import get_request_id as _get_rid
            for line in filtered.splitlines():
                if not line.strip():
                    continue
                rec = {"service": "v3-service", "level": "info", "msg": line}
                rid = _get_rid()
                if rid:
                    rec["request_id"] = rid
                self._stream.write(_json.dumps(rec) + "\n")
            return
        self._stream.write(filtered)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


if __name__ == "__main__":
    sys.stdout = _PrivateValueStream(sys.stdout)
    sys.stderr = _PrivateValueStream(sys.stderr)
    from structured_log import install as _install_logging
    _install_logging("v3-service")
    print(f"ATLAS V3 Pipeline Service starting on :{PORT}")
    print(f"  Inference:     {INFERENCE_URL}")
    print(f"  Geometric Lens: {LENS_URL}")
    print(f"  Sandbox: {SANDBOX_URL}")

    # ThreadingHTTPServer: a long pipeline call must not starve /health and
    # the /internal/* endpoints. Shared state is thread-safe by construction:
    # LLMAdapter._lock serializes the llama.cpp backend, pipeline components
    # are read-only after __init__ (telemetry_dir=None so nothing appends),
    # per-run state (events, candidates, adapters) is local to each request,
    # and the graph package's FileGraphCache carries its own lock.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), V3Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
