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


# --- Pattern Cache write hook -------------------------------------------------
# Maps the V3 phase that produced the winning solution to a retry_count value.
# The pattern cache uses retry_count / max_retries as a "surprise" proxy — higher
# retries mean the pattern was harder to find and worth caching with more weight.
_PHASE_RETRY_COUNT = {
    "probe_pass": 1,        # solved on first probe
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
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except Exception as e:
            print(f"  [pattern-write] POST failed (non-fatal): {e}", flush=True)

    threading.Thread(target=_do_post, daemon=True).start()


# --- LLM Adapter (calls llama-server /v1/chat/completions) ----------------------------

class LLMAdapter:
    """Calls llama-server's /v1/chat/completions, parsing ChatML prompts into messages.

    PC-206: `thinking` controls Qwen3.5's hybrid reasoning mode.
    - False (default) — `/nothink` injected, `enable_thinking=False`.
      Required for grammar-constrained JSON output (the agent's tool-call
      shape) and for the tight V3 sampling loop where reasoning would 5-20×
      output token cost. This matches the previously hardcoded behavior.
    - True — `/nothink` NOT injected, `enable_thinking=True`. Use for
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
        model_name = os.environ.get("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")

        # PC-206: thinking flag drops down from __call__. Default False so
        # any caller that constructs a body dict directly preserves the
        # pre-PC-206 /nothink behavior.
        thinking = bool(body.pop("_thinking", False))

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
            user_content = prompt if thinking else "/nothink\n" + prompt
            messages = [{"role": "user", "content": user_content}]
        else:
            print(f"  [LLM] Parsed {len(messages)} messages from ChatML"
                  f" (thinking={'on' if thinking else 'off'})", flush=True)
            if not thinking:
                # Ensure /nothink in last user message — NOT enough on its
                # own (Qwen3 reasoning still streams into reasoning_content
                # without the chat_template_kwargs flip below), but cheap
                # belt-and-braces signal to the model.
                for msg in messages:
                    if msg["role"] == "user" and not msg["content"].startswith("/nothink"):
                        msg["content"] = "/nothink\n" + msg["content"]
            else:
                # PC-206: strip any /nothink the prompt template hardcoded.
                # Lets a caller flip thinking on without rewriting prompts.
                for msg in messages:
                    if msg["role"] == "user" and msg["content"].startswith("/nothink"):
                        msg["content"] = msg["content"][len("/nothink"):].lstrip("\n")

        chat_body = {
            "model": model_name,
            "messages": messages,
            "max_tokens": body.get("max_tokens", body.pop("n_predict", 4096)),
            "temperature": body.get("temperature", 0.6),
            "stream": bool(body.get("stream", False)),
            # PC-206: when thinking=True, Qwen3.5's hybrid reasoning mode is
            # allowed; the <think>...</think> blocks get stripped in __call__
            # before downstream JSON parse. When False (the agent-loop default),
            # enable_thinking=False is the load-bearing one — /nothink in the
            # prompt alone is NOT enough; confirmed via logs where 2048 tok of
            # reasoning streamed into delta.reasoning_content with empty content.
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
            headers={"Content-Type": "application/json"},
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
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        if not data.get("enabled"):
            return {}
        agg = data.get("aggregate", {}) or {}
        return {
            "n_tokens":            int(data.get("n_tokens", 0)),
            "gx_available":        bool(data.get("gx_available", False)),
            "first_off_rails_idx": int(agg.get("first_off_rails_idx", -1)),
            "gx_score_min":        float(agg.get("gx_score_min", 0.5)),
            "gx_score_mean":       float(agg.get("gx_score_mean", 0.5)),
            "cx_norm_max":         float(agg.get("cx_norm_max", 0.0)),
            "cx_norm_mean":        float(agg.get("cx_norm_mean", 0.0)),
            "latency_ms":          float(data.get("latency_ms", 0.0)),
        }
    except Exception as e:
        print(f"  [lens] score_candidate_per_step failed: {e} — degrading to no per-step signal", flush=True)
        return {}


def score_candidate(code: str) -> Tuple[float, float]:
    """Score code with Geometric Lens C(x). Returns (raw_energy, normalized).

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
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("cx_energy", 0.0), data.get("gx_score", 0.5)
    except Exception as e:
        print(f"  [lens] score_candidate failed: {e} — falling back to (0.0, 0.5)", flush=True)
        return 0.0, 0.5


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

    PC-048: language-aware. Python files run the AST parse / compile
    smoke. HTML/JSON/YAML files run a stdlib parse for well-formedness.
    Everything else (CSS, JS, MD, plain text, …) returns OK without a
    sandbox round-trip — we don't have a cheap, accurate validator and
    the LLM is more reliable on those formats than spurious-failure
    pressure from a half-built validator would be.
    """
    lang = (language or "python").lower()

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
        # CSS, JS, TS, MD, plain text, anything else — no cheap validator,
        # trust the LLM. Returning OK avoids false-positive failures that
        # cascade into PR-CoT repair attempts and LLM timeouts.
        return True, "SMOKE_SKIP (non-Python)", ""

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
            file_path: str = "") -> Dict[str, Any]:
        """Run the full V3 pipeline on a coding problem.

        Args:
            problem: Problem description
            task_id: Task identifier
            progress_callback: SSE progress emitter
            files: Dict of filename→content from Aider's existing file context
            file_path: Target file path (used by PC-048 to detect language
                for the smoke check — `.html` files use HTML parser, not
                Python compile, etc.)
        """
        start = time.time()
        events = []
        files = files or {}

        # PC-048: derive language from the target file's extension. Used
        # only by smoke_compile_check below to pick the right parser
        # (Python compile vs HTML parser vs JSON loads vs skip-and-pass
        # for unknown formats). Defaults to Python when no file_path is
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
            ".xml": "html",  # treat XML same as HTML for parsing
            ".sh": "bash", ".bash": "bash",
            ".go": "go",
            ".rs": "rust",
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
            events.append(ev)
            if progress_callback:
                try:
                    progress_callback(stage, detail, **data)
                except TypeError:
                    progress_callback(stage, detail)

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
            """Sandbox + verification. Algorithmic tasks: I/O self-tests; interactive: compile smoke."""
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
                    return ok, out, err
                # Interactive lint is Python-AST based — only meaningful for
                # Python files. Skip for HTML/CSS/JSON/etc.
                if smoke_language not in ("python", "py"):
                    return True, out, err
                # Interactive lint: catch raw stdin reads / blocking input loops
                # that compile fine but don't actually work for keystroke
                # handling (PC-034).
                lint_ok, lint_reason = interactive_lint(code)
                if lint_ok:
                    emit("interactive_lint", "OK")
                    return True, out, err
                emit("interactive_lint", f"FAIL: {lint_reason}")
                return False, out, f"interactive_lint: {lint_reason}"

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
        emit("phase2_allocated", f"k={k} tier={budget_tier}", k=k, tier=budget_tier)

        # ===== PHASE 1: CONSTRAINT-DIVERSE CANDIDATE GENERATION =====
        emit("phase1", f"Generating {k} diverse candidates...", k=k)
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
            emit("plansearch", f"Generating {remaining_k} plans...",
                 plans=remaining_k)
            try:
                ps_result = self.plan_search.generate(
                    problem, task_id, llm, num_plans=remaining_k,
                )
                for i, code in enumerate(ps_result.candidates):
                    if code:
                        energy_raw, energy_norm = score_candidate(code)
                        per_step = score_candidate_per_step(code)  # PC-207
                        cand_index = len(candidates)
                        candidates.append({
                            "index": cand_index, "code": code,
                            "energy": energy_raw, "energy_norm": energy_norm,
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
                        per_step = score_candidate_per_step(code)  # PC-207
                        cand_index = len(candidates)
                        candidates.append({
                            "index": cand_index, "code": code,
                            "energy": energy_raw, "energy_norm": energy_norm,
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
            if c.get("passed"):
                passing.append(c)
                continue
            sb_start = time.time()
            passed, stdout, stderr = verified_sandbox(c["code"])
            sb_ms = int((time.time() - sb_start) * 1000)
            c["passed"] = passed
            c["stdout"] = stdout
            c["stderr"] = stderr
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
                        emit("s_star_winner", f"Winner: candidate {winner['index']}",
                             index=winner["index"], energy=winner.get("energy_norm", 0.0))
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
                emit("selected", f"Lens selected candidate {selected.index}",
                     index=selected.index, energy=getattr(selected, "energy", 0.0))
                result["passed"] = True
                result["code"] = selected.code
                result["phase_solved"] = "phase1"
                result["total_time_ms"] = (time.time() - start) * 1000
                result["events"] = events
                return result

        # ===== PHASE 3: VERIFIED ITERATIVE REFINEMENT =====
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

        # Strategy 1: PR-CoT Quick Repair
        if failing:
            emit("pr_cot", "Attempting PR-CoT repair...",
                 strategy="pr_cot", failing=len(failing))
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
                        emit("pr_cot_pass", "PR-CoT repair succeeded!",
                             strategy="pr_cot", tokens=pr_result.total_tokens)
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
            emit("refinement", "Starting refinement loop...",
                 strategy="refinement", failing=len(failing))
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
                    emit("refinement_pass",
                         f"Refinement solved in {ref_result.total_iterations} iterations!",
                         strategy="refinement",
                         iterations=ref_result.total_iterations,
                         tokens=ref_result.total_tokens)
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
            emit("derivation", "Attempting derivation chains...",
                 strategy="derivation", failing=len(failing))
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
                        emit("derivation_pass", "Derivation chains solved!",
                             strategy="derivation")
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


# --- Plan generation (/v3/plan) ----------------------------------------------
#
# Generates a structured plan for an agent task. Reuses the same LLMAdapter
# the code-generation pipeline uses, but with a planning prompt template and
# a heuristic scorer (V3's lens-based scorer is for code embeddings, not prose
# plans, so it doesn't apply here).
#
# Why bother: when qwen-coder gets a multi-step task without a plan, it
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
- Each step is a single tool call: read_file, write_file, edit_file, delete_file, run_command, list_directory.
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
    r"mypy\b|ruff\b|pylint\b|tsc\b|eslint\b)"
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
    mentioned_files = set(re.findall(r"[\w./-]+\.[a-zA-Z0-9]+", user_message.lower()))
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

    # PC-206: thinking-aware infrastructure shipped — planner CAN run with
    # Qwen3.5 hybrid reasoning ON via ATLAS_PLAN_THINKING=1. Default is OFF
    # because empirically on the reference Qwen3.5-9B-Q6_K + this codebase's
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


# --- HTTP Handler (SSE streaming) --------------------------------------------

pipeline = V3PipelineService()


class V3Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v3/run":
            self._handle_run()
        elif self.path == "/v3/generate":
            self._handle_generate()
        elif self.path == "/v3/plan":
            self._handle_plan()
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

            def emit_sse(stage, detail="", **data):
                payload = {"stage": stage, "detail": detail}
                if data:
                    payload["data"] = data
                event = json.dumps(payload)
                try:
                    self.wfile.write(f"data: {event}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    pass

            result = pipeline.run(problem, task_id, progress_callback=emit_sse, files=files)
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
            file_path=file_path,  # PC-048: language-aware smoke check
        )
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
            "winning_score": 0.0,
            "total_tokens": result.get("total_tokens", 0),
            "total_time_ms": result.get("total_time_ms", 0.0),
        }
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
        body = json.loads(self.rfile.read(content_len))

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
            pass

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
