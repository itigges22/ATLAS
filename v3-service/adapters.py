"""Outbound service adapters for the V3 service: llama-server chat/embedding
clients, the sandbox client, internal-auth plumbing, and the pattern-cache
write hook."""

import json
import os
import re
import threading
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from stages.llm_client import chatml_to_messages

# --- Configuration -----------------------------------------------------------

INFERENCE_URL = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
LENS_URL = os.environ.get("ATLAS_LENS_URL", "http://localhost:8099")
SANDBOX_URL = os.environ.get("ATLAS_SANDBOX_URL", "http://localhost:30820")


def _max_inflight() -> int:
    """How many generations may be in flight against llama.cpp at once.

    Defaults to the server's slot count, since a request beyond that queues
    behind a slot rather than gaining anything — llama.cpp oversubscribed
    past its slots degrades latency sharply. ATLAS_V3_MAX_INFLIGHT overrides;
    1 restores the fully serialized behaviour.
    """
    for var in ("ATLAS_V3_MAX_INFLIGHT", "ATLAS_PARALLEL_SLOTS", "PARALLEL_SLOTS"):
        raw = os.environ.get(var)
        if raw and raw.strip().isdigit() and int(raw) > 0:
            return int(raw)
    return 4


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


# --- Pattern Cache write hook -------------------------------------------------
# Maps the V3 phase that produced the winning solution to a retry_count value.
# The pattern cache uses retry_count / max_retries as a "surprise" proxy — higher
# retries mean the pattern was harder to find and worth caching with more weight.
_PHASE_RETRY_COUNT = {
    "probe": 1,             # solved on first probe (phase_solved="probe")
    "phase1": 2,            # plan-search candidates passed
    "pr_cot": 3,            # required PR-CoT repair
    "refinement": 4,        # required refinement loop
    "none": 5,              # nothing passed; best-by-energy returned
}


def _post_pattern_outcome(problem: str, result: dict):
    """Fire-and-forget: post the pipeline outcome to geometric-lens for caching.

    Runs in a background thread so it never delays the response. Errors are
    logged but never raised — the pattern cache is best-effort, not load-bearing.
    """
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

    # Bounds how many generations this service has in flight at once.
    #
    # This was a Lock, added when the service moved to ThreadingHTTPServer so
    # a long pipeline call could not starve /health and /internal/* — its job
    # was to keep concurrent REQUESTS from oversubscribing llama.cpp. Being
    # class-level, it also serialized the calls inside a single pipeline run,
    # which is where the cost landed: PlanSearch generates its candidates from
    # independent plans and seeds, and 4 of them ran end to end at ~22s each
    # while three of llama-server's four slots sat idle. Measured 2026-08-03,
    # 8 sequential calls totalling 166s against the proxy's 180s cap.
    #
    # A semaphore keeps the original guarantee — never more in flight than the
    # backend has slots — while letting independent generations share them.
    # llama.cpp batches concurrent slots itself; serializing here did that job
    # a second time, worse.
    _slots = threading.BoundedSemaphore(_max_inflight())

    # Counter updates are read-modify-write and now run under concurrency.
    _counter_lock = threading.Lock()

    def __init__(self, progress_callback=None, thinking: bool = False,
                 deadline: Optional[float] = None):
        self.call_count = 0
        self.total_tokens = 0
        self.total_time_ms = 0.0
        self._progress = progress_callback
        self.thinking = thinking
        # Monotonic wall-clock (time.time()) after which no new generation
        # may start. None leaves the adapter unbounded, which is what the
        # bench and any caller without a cap want.
        self.deadline = deadline

    # Fallback decode rate (tokens/sec) for sizing the first call, before
    # this run has observed one. Measured on a 12B Q4 at 4 slots: ~25 tok/s
    # single-stream. Deliberately conservative — overestimating the rate
    # asks for more tokens than the clock can deliver, which is the failure
    # this exists to prevent.
    _ASSUMED_TOK_PER_SEC = 20.0

    # Below this many tokens a generation cannot produce anything useful,
    # so the budget is spent rather than nearly spent.
    _MIN_USEFUL_TOKENS = 128

    def _observed_tok_per_sec(self) -> float:
        if self.total_time_ms <= 0 or self.total_tokens <= 0:
            return self._ASSUMED_TOK_PER_SEC
        return max(1.0, self.total_tokens / (self.total_time_ms / 1000.0))

    def _budget_max_tokens(self, max_tokens: int) -> int:
        """Shrink max_tokens to what the remaining clock can actually decode.

        A generation runs until it stops or hits max_tokens, so an 8192-token
        ceiling at ~25 tok/s is a 327-second call — longer than the whole
        180s budget. Measured 2026-08-04: every V3 hang-up in a 28-session
        run was the pipeline cut mid-probe, the first generation, which had
        the full budget and still did not finish. Refusing the call is not
        the answer either — that produces nothing at all. Asking for a
        length the clock can deliver is.

        Raises BudgetExhausted when even a minimal generation will not fit.
        """
        if self.deadline is None:
            return max_tokens
        left_s = self.deadline - time.time()
        # Leave room to read the response and hand back a result.
        affordable = int((left_s - 5.0) * self._observed_tok_per_sec() * 0.8)
        if affordable < self._MIN_USEFUL_TOKENS:
            raise BudgetExhausted(
                f"{left_s:.0f}s left decodes ~{max(affordable, 0)} tokens at "
                f"{self._observed_tok_per_sec():.0f} tok/s")
        return min(max_tokens, affordable)

    @property
    def avg_call_ms(self) -> float:
        """Average observed per-call latency (0.0 before the first call).
        Feeds the refinement loop's one-iteration cost estimate."""
        if not self.call_count:
            return 0.0
        return self.total_time_ms / self.call_count

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
        max_tokens = self._budget_max_tokens(max_tokens)
        with LLMAdapter._counter_lock:
            self.call_count += 1
            call_no = self.call_count

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
        self._emit("llm_start", f"call #{call_no}",
                   call=call_no, max_tokens=max_tokens,
                   temperature=temperature)
        data = self._send(body, call_no)
        # The streaming send already emitted token events; emit a
        # closing marker with totals so the TUI can replace the live
        # row with a compact summary.
        elapsed_ms = (time.time() - start) * 1000
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0) \
            or data.get("usage", {}).get("total_tokens", 0)
        with LLMAdapter._counter_lock:
            self.total_time_ms += elapsed_ms
        self._emit("llm_end", f"{completion_tokens} tok · {elapsed_ms:.0f}ms",
                   call=call_no, tokens=completion_tokens,
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

        with LLMAdapter._counter_lock:
            self.total_tokens += tokens
        return content, tokens, elapsed_ms

    def _send(self, body: dict, call_no: int = 0) -> dict:
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
                with LLMAdapter._slots:
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
                                    # Tagged with the call it belongs to:
                                    # concurrent generations interleave in
                                    # this stream, and an untagged token
                                    # cannot be attributed to one of them.
                                    self._emit("token", delta, call=call_no)
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


class BudgetExhausted(Exception):
    """Not enough of ATLAS_V3_TIMEOUT is left to start another generation.

    Raised from LLMAdapter.__call__ rather than checked at phase boundaries.
    Boundary checks cannot hold: every phase runs its own internal loop —
    PR-CoT alone issues two calls — so a check that reserves one call is
    already wrong by the second. Measured 2026-08-03: a boundary check with
    ~50s left correctly allowed PR-CoT against a ~34s reserve, and PR-CoT
    spent 44s then started a 21s call, overrunning the cap.

    The pipeline catches this and returns its best candidate so far, which is
    the contract an anytime algorithm owes its caller.
    """


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

    `test_input` is piped to the run as standard input (the /execute
    `stdin` field) — the same stdin contract the bench sandbox adapters
    implement, so per-candidate test inputs reach the candidate under
    test.
    """

    def __init__(self, project_files: Optional[Dict[str, str]] = None):
        self.project_files = project_files or {}

    def __call__(self, code: str, test_input: str = "",
                 language: str = "python",
                 timeout: int = 15) -> Tuple[bool, str, str]:
        """Execute `code` in the sandbox.

        `language` defaults to python so every existing call site is
        unchanged. It exists because the behavioural probe must run
        JavaScript inside the sandbox rather than as a subprocess of this
        service, and a hardcoded "python" body meant the probe's request
        could never be honoured -- it raised TypeError at the call and was
        silently converted to "inconclusive", so no real browser probe ever
        produced evidence.
        """
        body = {
            "code": code,
            "language": language,
            "timeout": timeout,
        }
        if test_input:
            # Empty string keeps the executor default (inherit server
            # stdin) — every no-input call site passes "" positionally.
            body["stdin"] = test_input
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
            # Client read timeout is derived from the requested execution
            # budget plus bounded overhead, never a fixed value below it: a
            # probe asking for 60s against a hardcoded 45s client timeout
            # would have been cut off by its own caller.
            _client_timeout = max(45, int(timeout) + 30)
            with urllib.request.urlopen(req, timeout=_client_timeout) as resp:
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


# ---------------------------------------------------------------------------
# Adapter records
# ---------------------------------------------------------------------------
#
# ADAPTER KNOWLEDGE, which is why it lives here: which criteria each adapter
# can observe, and what its verifier can therefore demonstrate. contract.py
# must not learn either -- it stays generic and derives completeness, coverage
# and closure from what an adapter reports.
#
# Records are built by calling contract.build directly. Nothing here consults
# the retiring evidence.py: strength is read from the OBSERVATIONS, not from
# that module's graded string, so this file is a producer of contract records
# rather than a translator of someone else's grade.

import contract

# Adapter identities. These are the ids the pipeline records carry, declared
# here because they name THIS layer's verifiers. test_adapters.py pins them
# against the retiring module's copies for as long as that module exists.
ADAPTER_BROWSER_CANVAS_JS = "browser_canvas_js"
ADAPTER_BROWSER_INLINE_SCRIPT = "browser_inline_script"
ADAPTER_JAVASCRIPT_COMPILE = "javascript_compile"
ADAPTER_CSS_SYNTAX = "css_syntax"
ADAPTER_ALGORITHMIC_IO = "algorithmic_io"
ADAPTER_PYTHON_COMPILE = "python_compile"
ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED = "interactive_python_unsupported"
ADAPTER_UNSUPPORTED = "unsupported"

_BROWSER_ADAPTERS = (ADAPTER_BROWSER_CANVAS_JS, ADAPTER_BROWSER_INLINE_SCRIPT)

# What the browser probe can observe. Opaque ids: nothing above this layer
# interprets them, and this layer never decides what the TASK required.
BROWSER_REQUIRED = ["temporal_progress", "input_causality"]
BROWSER_OPTIONAL = ["collision_transition", "food_or_score_transition"]

# Adapters that answer the same question and cannot observe any of it. They
# declare the criteria so coverage can report them unmeasurable rather than
# silently missing.
_DECLARES_BROWSER_CRITERIA = _BROWSER_ADAPTERS + (
    ADAPTER_JAVASCRIPT_COMPILE, ADAPTER_PYTHON_COMPILE,
    ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED, ADAPTER_UNSUPPORTED)

# Identity of this producer's grading. It must change whenever the grading
# changes, or two incomparable measurements would compare equal.
LIVE_ADAPTER_VERSION = "0.1.0-prototype"


def _capabilities(adapter):
    """What this adapter can observe. Everything else contract.build reports
    not_applicable, so "we could not look" stays distinct from "absent"."""
    if adapter in _BROWSER_ADAPTERS:
        return list(BROWSER_REQUIRED) + list(BROWSER_OPTIONAL)
    return []


def _requirements(adapter):
    if adapter in _DECLARES_BROWSER_CRITERIA:
        return ([contract.requirement(c) for c in BROWSER_REQUIRED]
                + [contract.requirement(c, required=False)
                   for c in BROWSER_OPTIONAL])
    # An adapter with no declared criteria says so plainly. That is not the
    # same as an adapter whose criteria all passed.
    return []


def _observations(adapter, behavior):
    """One observation per criterion this adapter can measure.

    A criterion it can measure and did not see is UNOBSERVED, never REFUTED --
    except an OPTIONAL one on a run that did produce a behaviour trace, where
    absence is a real negative observation rather than a gap.
    """
    observations = {}
    if adapter not in _BROWSER_ADAPTERS:
        return observations
    for cid in BROWSER_REQUIRED:
        observations[cid] = contract.observation(
            contract.DEMONSTRATED if behavior.get(cid) else contract.UNOBSERVED)
    for cid in BROWSER_OPTIONAL:
        if behavior.get(cid):
            status = contract.DEMONSTRATED
        elif behavior:
            status = contract.REFUTED
        else:
            status = contract.UNOBSERVED
        observations[cid] = contract.observation(status)
    return observations


def _strength_and_execution(adapter, accepted, supported, behavior):
    """What this verifier demonstrated, and whether its run completed.

    Derived from the observations themselves. An oracle claim is made only
    where an oracle ran; a probe that executed cleanly but missed a required
    behaviour demonstrated runtime, not behaviour; and an artifact the adapter
    cannot support is unverified, never failed.
    """
    if not accepted and not (adapter in _BROWSER_ADAPTERS and behavior):
        # The verifier demonstrated nothing at all. A browser probe that DID
        # produce a trace is the exception: its own observations stand even
        # when the smoke check rejected the candidate.
        return contract.SYNTAX, contract.EXEC_ERROR, False
    if not supported:
        return contract.SYNTAX, contract.EXEC_SKIPPED, False
    if adapter == ADAPTER_ALGORITHMIC_IO:
        return contract.ORACLE, contract.EXEC_OK, True
    if adapter in _BROWSER_ADAPTERS:
        if not behavior or not behavior.get("runtime_clean", True):
            return contract.SYNTAX, contract.EXEC_OK, True
        if any(not behavior.get(cid) for cid in BROWSER_REQUIRED):
            return contract.RUNTIME, contract.EXEC_OK, True
        return contract.BEHAVIORAL, contract.EXEC_OK, True
    return contract.SYNTAX, contract.EXEC_OK, True


def contract_record(live_record, *, contract_id, contract_version,
                    artifact_scope, evaluation_context_hash,
                    candidate_content_hash,
                    minimum_closure_strength=contract.BEHAVIORAL):
    """One finalized contract record, built here and derived by contract.py."""
    adapter = live_record.get("adapter") or ADAPTER_UNSUPPORTED
    behavior = live_record.get("behavior") or {}
    accepted = bool(live_record.get("accepted", True))
    supported = bool(live_record.get("supported", True))
    if adapter in _BROWSER_ADAPTERS and behavior and not behavior.get("supported", True):
        supported = False

    strength, execution_status, supported = _strength_and_execution(
        adapter, accepted, supported, behavior)

    task = contract.task_contract(contract_id, contract_version,
                                  _requirements(adapter),
                                  minimum_closure_strength=minimum_closure_strength)
    return contract.build(
        task, adapter, LIVE_ADAPTER_VERSION, _observations(adapter, behavior),
        _capabilities(adapter), strength, execution_status=execution_status,
        supported=supported, artifact_scope=artifact_scope,
        evaluation_context_hash=evaluation_context_hash,
        candidate_content_hash=candidate_content_hash)


def evidence_envelope(result, *, contract_id, contract_version, artifact_scope,
                      evaluation_context, delivered_code, selection=None):
    """The one entry point main.py calls. None means: no evidence to send.

    None is a positive statement -- nothing was measured -- and is distinct
    from a malformed envelope, which this never produces: a record that cannot
    be serialised raises instead.
    """
    live_record = result.get("evidence")
    if not live_record:
        return None
    record = contract_record(
        live_record,
        contract_id=contract_id, contract_version=contract_version,
        artifact_scope=artifact_scope,
        evaluation_context_hash=contract.content_hash(evaluation_context),
        candidate_content_hash=contract.content_hash(result.get("code") or ""))
    return contract.envelope(record, selection,
                             contract.content_hash(delivered_code))
