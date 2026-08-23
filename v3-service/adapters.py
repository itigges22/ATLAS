"""Outbound service adapters for the V3 service: llama-server chat/embedding
clients, the sandbox client, internal-auth plumbing, and the pattern-cache
write hook."""

import json
import os
import re
import socket
import threading
import time
import contextlib
import dataclasses
import http.client
import urllib.request
import urllib.error
import urllib.parse
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


REQUEST_ID_HEADER = "X-ATLAS-Request-ID"
INVOCATION_ID_HEADER = "X-ATLAS-V3-Invocation-ID"


@dataclasses.dataclass(frozen=True)
class RequestIdentity:
    """The identity of the request an adapter is serving.

    Frozen and owned by the request thread that builds it. Worker threads
    read it off the request-scoped adapter instance, which is how it reaches
    them — never off a ContextVar, which a new thread does not inherit.
    """
    request_id: str = ""
    invocation_id: str = ""

    def headers(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if self.request_id:
            out[REQUEST_ID_HEADER] = self.request_id
        if self.invocation_id:
            out[INVOCATION_ID_HEADER] = self.invocation_id
        return out


class RequestIdentityMissing(Exception):
    """An adapter serving a request has no identity to send.

    Raised instead of sending an unattributed inference call. A permissive
    upstream would accept that call and answer it, so the omission would
    otherwise surface only as a missing correlation ID in someone else's
    logs — or, against an upstream that enforces attribution, as candidate
    scarcity with no stated cause.
    """


def _service_headers(rid: str = "", invocation_id: str = "") -> dict:
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
        headers[REQUEST_ID_HEADER] = rid
    if invocation_id:
        headers[INVOCATION_ID_HEADER] = invocation_id
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
        # Request-scoped cancellation. None for callers with no request
        # (bench, CLI): they are never cancelled and must not be affected.
        self.cancel_scope = None
        # Request-scoped identity, set by the request thread that builds this
        # adapter and read by every generation it opens — including ones
        # dispatched from PlanSearch's worker threads, which is the whole
        # reason it lives here rather than in a ContextVar. None carries the
        # same meaning as a None cancel_scope: no request (bench, CLI).
        self.request_identity: Optional[RequestIdentity] = None
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
        # No new inference once the caller is gone.
        #
        # V3 is a synchronous server: its only disconnect signal is a
        # BrokenPipeError on the next SSE write, which sets `disconnected` on
        # the progress callback, and the pipeline consulted that flag only at
        # PHASE boundaries. Nothing consulted it where the GPU is actually
        # spent. Measured on a real acquisition: a run whose agent request had
        # already returned at its 570 s work deadline STARTED a 24th
        # generation at +569.8 s and ran it 39.8 s to completion, leaving the
        # relay holding an in-flight call after the terminal.
        #
        # This is the one chokepoint every V3 generation passes through, so
        # the check belongs here rather than at each of its callers. It stops
        # a call from STARTING; a call already in flight is a separate
        # concern and is not claimed to be handled by this.
        if getattr(self._progress, "disconnected", False):
            raise ClientDisconnected(
                "client disconnected; refusing to start another generation")
        # The scope is the signal that does not depend on discovering a broken
        # output socket: the handler cancels it the moment the parent goes.
        if self.cancel_scope is not None and self.cancel_scope.cancelled:
            raise Cancelled("request cancelled; refusing to start another generation")
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

    def _inference_headers(self) -> Dict[str, str]:
        """Headers for one inference call, resolved from the adapter's own
        request identity.

        Deliberately does NOT fall back to the request-ID ContextVar. A
        generation dispatched from a PlanSearch worker thread does not
        inherit the request thread's ContextVar, so a fallback here reads as
        "no request" and silently strips attribution from exactly the calls
        that are hardest to trace back. The identity is request-scoped state
        and travels on the request-scoped adapter, like cancel_scope.

        Serving a request without an identity is a wiring error, not a
        runtime condition, so it raises rather than sending the call: an
        upstream that does not enforce attribution would answer it, and the
        defect would survive as missing correlation IDs instead of an error.
        """
        if self.request_identity is None:
            if self.cancel_scope is not None:
                raise RequestIdentityMissing(
                    "adapter is serving a request (cancel_scope set) but has "
                    "no request_identity; refusing to send an unattributed "
                    "inference call")
            # No request at all (bench, CLI): nothing to attribute.
            return {"Content-Type": "application/json"}
        headers = {"Content-Type": "application/json"}
        headers.update(self.request_identity.headers())
        return headers

    @contextlib.contextmanager
    def _open_inference(self, payload: bytes):
        """One inference connection, registered with the request's scope.

        Registration is the cancellation handle. A scope already cancelled
        refuses to open at all, so no call can slip through between the
        dispatch check and the socket.
        """
        parsed = urllib.parse.urlsplit(INFERENCE_URL)
        host, port = parsed.hostname or "127.0.0.1", parsed.port
        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=600)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=600)
        scope = self.cancel_scope
        if scope is not None and not scope.register(conn):
            conn.close()
            raise Cancelled("request cancelled before the connection opened")
        try:
            # After registration, so a cancelled scope is reported as
            # cancellation rather than as whatever the next check happens to
            # be. Registration is where cancellation is decided.
            headers = self._inference_headers()
            path = (parsed.path or "") + "/v1/chat/completions"
            conn.request("POST", path, body=payload, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read()[:200]
                raise urllib.error.HTTPError(
                    INFERENCE_URL, resp.status, detail.decode("utf-8", "replace"),
                    hdrs=None, fp=None)
            yield resp
        finally:
            if scope is not None:
                scope.unregister(conn)
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

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

        payload = json.dumps(chat_body).encode()
        for attempt in range(5):
            try:
                with LLMAdapter._slots:
                    # http.client, not urlopen: the connection object exists
                    # BEFORE the request is sent, so a cancelling thread has a
                    # handle at every point -- waiting for response headers,
                    # mid-stream, and deep inside a long generation. urlopen
                    # offers no handle until headers arrive, which is exactly
                    # the window a cancellation has to survive.
                    with self._open_inference(payload) as resp:
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


# --- connection teardown ------------------------------------------------------
#
# HONEST NOTE ON THE INTERFACE USED.
#
# Interrupting a thread already blocked in recv() requires shutdown() on the
# underlying socket. close() alone does not do it: measured over real sockets,
# a call blocked waiting for response headers took the upstream's full 5.76s
# to return, and a mid-stream cancellation never reached the upstream at all.
#
# http.client exposes that socket as HTTPConnection.sock. It is an ordinary
# instance attribute -- no leading underscore, stable across every CPython 3.x
# -- but it is NOT part of the documented http.client API. This is the one
# undocumented interface the cancellation path depends on, and it is named
# here rather than buried at the call site.
#
# The dependency is guarded, not assumed: HTTPCONNECTION_EXPOSES_SOCK records
# whether the attribute exists on this runtime, a test fails loudly if a future
# Python removes it, and teardown degrades to close() rather than raising.
# Cancellation would weaken on such a runtime, so the build must break first.

HTTPCONNECTION_EXPOSES_SOCK = "sock" in http.client.HTTPConnection("localhost").__dict__


def _abort_connection(conn) -> bool:
    """Tear a connection down hard. Returns True if the socket was shut down.

    Safe when no socket exists yet, when the connection is already closed, and
    when called twice: every failure path falls through to close(). A raise
    here would leave later connections in the same cancel() unclosed, so
    nothing is allowed to propagate.
    """
    did_shutdown = False
    try:
        sock = getattr(conn, "sock", None)
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            did_shutdown = True
    except (OSError, AttributeError):
        # Already closed, never connected, or an unexpected object. close()
        # below still runs.
        pass
    try:
        conn.close()
    except Exception:  # noqa: BLE001 - a close that fails is still cancelled
        pass
    return did_shutdown


class CancelScope:
    """Request-scoped cancellation for one V3 invocation.

    V3 is a synchronous server, so there is no task tree to cancel. What it
    does have is a set of live outbound connections, and closing those is what
    actually stops work: a blocked read returns, and the upstream sees its
    client go away.

    Measured before this existed: a generation dispatched at a parent's work
    deadline ran 39.8s to completion after the agent request had returned,
    and the inference stub recorded its client as still connected the whole
    time. Waiting for a broken SSE write to notice cannot fix that -- while a
    call is in flight nothing is being written.

    Cancellation is idempotent, scoped to one invocation, and never reaches
    another request's connections.
    """

    def __init__(self, invocation_id: str = ""):
        self.invocation_id = invocation_id
        self._lock = threading.Lock()
        self._cancelled = False
        self._live = set()
        self.closed_on_cancel = 0

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, conn) -> bool:
        """Track a live connection. Returns False if already cancelled, in
        which case the caller must not proceed."""
        with self._lock:
            if self._cancelled:
                return False
            self._live.add(conn)
            return True

    def unregister(self, conn) -> None:
        with self._lock:
            self._live.discard(conn)

    def cancel(self) -> int:
        """Close every live connection. Safe to call repeatedly."""
        with self._lock:
            self._cancelled = True
            live, self._live = list(self._live), set()
        for conn in live:
            _abort_connection(conn)
        with self._lock:
            self.closed_on_cancel += len(live)
        return len(live)


class Cancelled(Exception):
    """Raised where a cancelled scope stops work, so callers can tell an
    intentional stop from a transport failure."""


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
                 timeout: int = 15,
                 files: Optional[Dict[str, str]] = None) -> Tuple[bool, str, str]:
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
        # Per-call staging on top of project context: a self-test case's own
        # input file is this request's, and where the two name the same file
        # the case wins. Omitting `files` leaves every existing caller's body
        # byte for byte what it was.
        staged = dict(self.project_files or {})
        if files:
            staged.update(files)
        if staged:
            body["files"] = staged
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


# Criterion ids this layer declares. Opaque above it: nothing interprets them,
# and no task vocabulary appears here.
CRITERION_ORACLE_CASES = "oracle_cases_pass"
CRITERION_PARSES = "parses"

# What each adapter can observe, and therefore what it may report on.
_CAPABILITIES = {
    ADAPTER_ALGORITHMIC_IO: [CRITERION_ORACLE_CASES],
    ADAPTER_CSS_SYNTAX: [CRITERION_PARSES],
}

# The strength a TASK on this artifact class must reach before it may close.
# Declared per adapter, never a universal floor: a stylesheet has no runtime
# behaviour to demand, an I/O task is not closed by anything weaker than its
# oracle, and code whose behaviour matters but cannot be observed here stays
# open rather than closing on a compile.
_CLOSURE_FLOOR = {
    ADAPTER_ALGORITHMIC_IO: contract.ORACLE,
    ADAPTER_CSS_SYNTAX: contract.SYNTAX,
}


def closure_floor(adapter):
    return _CLOSURE_FLOOR.get(adapter, contract.BEHAVIORAL)


def _capabilities(adapter):
    """What this adapter can observe. Everything else contract.build reports
    not_applicable, so "we could not look" stays distinct from "absent"."""
    if adapter in _BROWSER_ADAPTERS:
        return list(BROWSER_REQUIRED) + list(BROWSER_OPTIONAL)
    return list(_CAPABILITIES.get(adapter, []))


def _requirements(adapter):
    if adapter in _DECLARES_BROWSER_CRITERIA:
        return ([contract.requirement(c) for c in BROWSER_REQUIRED]
                + [contract.requirement(c, required=False)
                   for c in BROWSER_OPTIONAL])
    return [contract.requirement(c) for c in _CAPABILITIES.get(adapter, [])]


def _observations(adapter, accepted, probe):
    """One observation per criterion this adapter can measure.

    A criterion it can measure and did not see is UNOBSERVED, never REFUTED --
    except an OPTIONAL one on a run that did produce a behaviour trace, where
    absence is a real negative observation rather than a gap.
    """
    observations = {}
    if adapter in _BROWSER_ADAPTERS:
        behavior = probe or {}
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
    for cid in _CAPABILITIES.get(adapter, []):
        observations[cid] = contract.observation(
            contract.DEMONSTRATED if accepted else contract.UNOBSERVED)
    return observations


def _supported(adapter, probe):
    """Whether this adapter could measure this artifact at all. Unsupported is
    unverified, never failed."""
    if adapter in (ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED, ADAPTER_UNSUPPORTED):
        return False
    if adapter in _BROWSER_ADAPTERS:
        # No probe trace means the behaviour question went unanswered.
        return bool(probe) and bool(probe.get("supported", True))
    return True


def _strength_and_execution(adapter, accepted, supported, probe):
    """What this verifier demonstrated, and whether its run completed.

    Derived from the observations themselves. An oracle claim is made only
    where an oracle ran; a probe that executed cleanly but missed a required
    behaviour demonstrated runtime, not behaviour; and an artifact the adapter
    cannot support is unverified, never failed.
    """
    behavior = probe or {}
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


def contract_record(*, adapter, accepted, probe=None, contract_id,
                    contract_version, artifact_scope, evaluation_context_hash,
                    candidate_content_hash, minimum_closure_strength=None):
    """One finalized contract record, built here and derived by contract.py.

    The caller hands over raw observation inputs -- which verifier ran, whether
    it accepted the artifact, and the probe trace if there was one. Everything
    else is this layer's declaration or the contract's derivation; no grading
    from elsewhere is translated.
    """
    supported = _supported(adapter, probe)
    strength, execution_status, supported = _strength_and_execution(
        adapter, accepted, supported, probe)
    floor = minimum_closure_strength or closure_floor(adapter)

    task = contract.task_contract(contract_id, contract_version,
                                  _requirements(adapter),
                                  minimum_closure_strength=floor)
    return contract.build(
        task, adapter, LIVE_ADAPTER_VERSION,
        _observations(adapter, accepted, probe), _capabilities(adapter),
        strength, execution_status=execution_status, supported=supported,
        artifact_scope=artifact_scope,
        evaluation_context_hash=evaluation_context_hash,
        candidate_content_hash=candidate_content_hash)


def evidence_envelope(result, *, delivered_code, selection=None):
    """The one entry point main.py calls. None means: no evidence to send.

    None is a positive statement -- nothing was measured -- and is distinct
    from a malformed envelope, which this never produces: a record that cannot
    be serialised raises instead.
    """
    record = result.get("evidence_record")
    if not record:
        return None
    return contract.envelope(record, selection or result.get("contract_selection"),
                             contract.content_hash(delivered_code))


# ---------------------------------------------------------------------------
# Adapter routing and probe mechanics
# ---------------------------------------------------------------------------
#
# Which verifier an artifact gets, and the machinery that verifier needs.
# Browser-shaped vocabulary lives here and nowhere above: contract.py stays
# generic, and the pipeline asks this layer rather than knowing about canvases
# or keydown handlers.

# --------------------------------------------------------------- adapters ---
#
# Evidence strength must come from the VERIFIER THAT RAN, never from the file
# extension. The first cut keyed off extension and mapped every .py to
# behavioral_complete, which is wrong for Pygame, Tkinter, curses and Flask:
# those receive a compile smoke and nothing more, and would have closed the
# pipeline claiming behaviour nobody demonstrated. It also sent .css through a
# JavaScript probe and treated every .js as a canvas game.
#
# Adapters carry the domain knowledge. Everything above them -- the strength
# ordering, coverage, early-return policy, ranking, and the unsupported vs
# failed distinction -- stays prompt-agnostic.


_INTERACTIVE_PY_RE = re.compile(
    r"\b(import\s+pygame|from\s+pygame|import\s+tkinter|from\s+tkinter|"
    r"import\s+curses|from\s+curses|Flask\s*\(|FastAPI\s*\(|"
    r"QApplication|import\s+PySide|import\s+PyQt)", re.I)

_INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)



def select_adapter(file_path: str, code: str, has_io_oracle: bool = False) -> str:
    """Which verifier can speak for this artifact. Capability, not keywords."""
    ext = (file_path or "").lower().rsplit(".", 1)
    ext = ("." + ext[-1]) if len(ext) == 2 else ""
    code = code or ""

    if ext in (".py",):
        if _INTERACTIVE_PY_RE.search(code):
            return ADAPTER_INTERACTIVE_PYTHON_UNSUPPORTED
        return ADAPTER_ALGORITHMIC_IO if has_io_oracle else ADAPTER_PYTHON_COMPILE
    if ext in (".js", ".mjs"):
        # A plain Node script or a module of helpers is NOT a canvas game.
        return ADAPTER_BROWSER_CANVAS_JS if js_is_instrumentable(code) else ADAPTER_JAVASCRIPT_COMPILE
    if ext in (".jsx", ".tsx", ".ts"):
        return ADAPTER_UNSUPPORTED          # needs transpilation first
    if ext in (".html", ".htm"):
        for m in _INLINE_SCRIPT_RE.finditer(code):
            if js_is_instrumentable(m.group(1)):
                return ADAPTER_BROWSER_INLINE_SCRIPT
        return ADAPTER_UNSUPPORTED
    if ext == ".css":
        return ADAPTER_CSS_SYNTAX
    return ADAPTER_UNSUPPORTED


def extract_inline_script(html: str) -> str:
    return "\n".join(m.group(1) for m in _INLINE_SCRIPT_RE.finditer(html or ""))



def extract_inline_script(html: str) -> str:
    return "\n".join(m.group(1) for m in _INLINE_SCRIPT_RE.finditer(html or ""))



# --------------------------------------------------------------- JS probe ---

# Instrumentation only. It never inspects the artifact's identifiers, so it
# does not care what the author named the snake, the loop, or the score.
_JS_HARNESS = r"""
// Fully deterministic instrumentation: a VIRTUAL clock, no wall time.
//
// The previous version used real setTimeout across two separate processes,
// so OS scheduling jitter could give the baseline and keyed runs different
// frame counts -- and a raw trace diff then read that as "input caused a
// change". An animation whose key handler does nothing could be scored
// input-causal. Correctness must not depend on machine load, so nothing here
// touches real time: callbacks go in a priority queue keyed by (due time,
// insertion order), both runs advance through identical virtual timestamps,
// input is injected at an exact virtual instant, and both runs execute the
// same bounded number of callbacks.
const MODE = process.argv[3] || 'baseline';
const INPUT_AT = 300;          // virtual ms
const MAX_VT   = 6000;         // virtual ms ceiling
const MAX_CB   = 4000;         // callback ceiling (runaway scheduling)

let __vt = 0, __seq = 0, __cbs = 0;
const __q = [];
const __push = (fn, delay) => {
  const d = Math.max(0, Number(delay) || 0);
  const id = ++__seq;
  __q.push({ due: __vt + d, seq: id, fn });
  return id;
};
const __cancel = (id) => { const i = __q.findIndex(t => t.seq === id); if (i >= 0) __q.splice(i, 1); };
global.setTimeout = (fn, d) => __push(fn, d);
global.setInterval = (fn, d) => { const self = { id: 0 };
  const tick = () => { try { fn(); } catch (e) { __err(e); } self.id = __push(tick, d); };
  self.id = __push(tick, d); return self.id; };
global.clearTimeout = __cancel; global.clearInterval = __cancel;
global.requestAnimationFrame = (fn) => __push(() => fn(__vt), 16);
global.cancelAnimationFrame = __cancel;
global.Date = class extends Date { constructor(...a){ super(...(a.length?a:[0])); }
  static now(){ return __vt; } };
global.performance = { now: () => __vt };

const __ev = { runtime_clean:true, supported:true, error:null, ended:false, textSets:0 };
const __err = (e) => { __ev.runtime_clean = false; __ev.error = String(e && e.message || e).slice(0,200); };
const __rects = [];
let __seed = 12345;
Math.random = () => { __seed = (__seed * 1103515245 + 12345) & 0x7fffffff; return __seed / 0x7fffffff; };

function __ctx() {
  return new Proxy({}, { get: (_, p) => {
    if (typeof p === 'symbol') return undefined;
    if (['fillStyle','strokeStyle','font','lineWidth','textAlign','textBaseline','globalAlpha'].includes(p)) return '';
    return (...a) => {
      // Record any positioned draw, not just rects: path and image games
      // must not read as inert.
      if (p === 'fillRect' || p === 'strokeRect' || p === 'rect' || p === 'arc' ||
          p === 'moveTo' || p === 'lineTo' || p === 'drawImage' || p === 'fillText')
        __rects.push(p + ':' + a.slice(0,2).map(v => Math.round(Number(v)||0)).join(','));
    };
  }, set: () => true });
}
const __canvas = { width:400, height:400, getContext:__ctx, addEventListener:(e,f)=>{(__L[e] ||= []).push(f);},
                   getBoundingClientRect:()=>({left:0,top:0,width:400,height:400}), style:{} };
const __L = {};
function __el(id){
  if (String(id).toLowerCase().includes('canvas')) return __canvas;
  return new Proxy({ style:{}, classList:{add(){},remove(){},toggle(){}},
                     addEventListener:(e,f)=>{(__L[e] ||= []).push(f);}, appendChild(){}, focus(){} },
    { get:(t,p)=> p in t ? t[p] : '',
      set:(t,p,v)=>{ if((p==='textContent'||p==='innerHTML'||p==='innerText') && t[p] !== undefined && String(t[p]) !== String(v)) __ev.textSets++; t[p]=v; return true; } });
}
global.document = { getElementById:__el, querySelector:(s)=>__el(String(s)), querySelectorAll:()=>[],
                    createElement:()=>__el('x'), body:{appendChild(){},style:{}},
                    addEventListener:(e,f)=>{(__L[e] ||= []).push(f);} };
global.window = { addEventListener:(e,f)=>{(__L[e] ||= []).push(f);}, innerWidth:800, innerHeight:600,
                  document: global.document, location:{ reload:()=>{ __ev.ended = true; }, href:'' } };
global.location = global.window.location;
global.alert = () => { __ev.ended = true; };
process.on('uncaughtException', __err);

const __src = require('fs').readFileSync(process.argv[2], 'utf8');
try { (0, eval)(__src); } catch (e) { __err(e);
  console.log(JSON.stringify({ ...__ev, trace:'', early:'' })); process.exit(0); }

const __fire = (k, code) => (__L['keydown']||[]).forEach(f => {
  try { f({ key:k, code:k, keyCode:code, which:code, preventDefault(){}, stopPropagation(){} }); } catch(e){ __err(e); }
});

// Deterministic drain: always the same virtual instants, same budget.
let __early = '';
let __injected = false, __drove = false;
while (__q.length && __vt <= MAX_VT && __cbs < MAX_CB) {
  __q.sort((a,b) => a.due - b.due || a.seq - b.seq);
  const t = __q.shift();
  __vt = Math.max(__vt, t.due);
  if (!__injected && __vt >= INPUT_AT) { __injected = true; if (MODE === 'input') __fire('ArrowUp', 38); }
  if (__early === '' && __vt >= 900) __early = __rects.slice(0, 60).join('|');
  if (!__drove && __vt >= 2500) { __drove = true; for (let i=0;i<6;i++) __fire('ArrowRight', 39); }
  try { t.fn(); } catch (e) { __err(e); }
  __cbs++;
}
console.log(JSON.stringify({ ...__ev, early: __early, trace: __rects.slice(0, 600).join('|'), cbs: __cbs }));
"""



def js_probe_source() -> str:
    return _JS_HARNESS


# Artifacts the shim can meaningfully instrument. Anything else reports
# supported=false — unverified, NOT failed.

def js_probe_source_inline() -> str:
    """The harness, adapted to run as ONE blob inside the sandbox.

    The sandbox executes a single code string with no argv and no artifact
    file, so mode and artifact arrive as pre-declared constants instead.
    """
    src = _JS_HARNESS
    src = src.replace("const MODE = process.argv[3] || 'baseline';",
                      "const MODE = __MODE__;")
    src = src.replace("const __src = require('fs').readFileSync(process.argv[2], 'utf8');",
                      "const __src = __ARTIFACT__;")
    return src



_CANVAS_RE = re.compile(r"getContext\s*\(|requestAnimationFrame|addEventListener\s*\(\s*['\"]keydown", re.I)
_NODE_ONLY_RE = re.compile(r"\brequire\s*\(|\bmodule\.exports\b|\bprocess\.(argv|stdin)\b")



def js_is_instrumentable(code: str) -> bool:
    if not code or not code.strip():
        return False
    if _NODE_ONLY_RE.search(code):
        return False        # a Node script, not browser code
    return bool(_CANVAS_RE.search(code))



def combine_runs(baseline: Optional[Dict], keyed: Optional[Dict]) -> Optional[Dict]:
    """Turn two controlled runs into behavioural evidence.

    Causality is a DIFFERENCE between an unkeyed and a keyed run from an
    identical deterministic start. A single run cannot tell "input changed
    the world" from "a timer moved pixels" — an animation that ignores
    input passed an earlier single-run version of this check.
    """
    if not baseline or not keyed:
        return None
    if not baseline.get("runtime_clean", True) or not keyed.get("runtime_clean", True):
        return {"supported": True, "runtime_clean": False,
                "error": baseline.get("error") or keyed.get("error"),
                "temporal_progress": False, "input_causality": False,
                "collision_transition": False, "food_or_score_transition": False}
    # Temporal progress is derived from the trace itself, not from a fixed
    # virtual timestamp: a game that dies in nine callbacks (a snake starting
    # next to a wall) never reaches a timestamp snapshot, and read as inert.
    # Splitting the recorded draws in half and comparing is timing-free.
    trace = baseline.get("trace", "")
    parts = [x for x in trace.split("|") if x]
    half = len(parts) // 2
    first, second = parts[:half], parts[half:2 * half]
    return {
        "supported": True,
        "runtime_clean": True,
        # Rendering kept changing on its own.
        "temporal_progress": half > 0 and first != second,
        # The keyed run diverged from the unkeyed one.
        "input_causality": bool(baseline.get("trace")) and baseline.get("trace") != keyed.get("trace"),
        "collision_transition": bool(baseline.get("ended") or keyed.get("ended")),
        "food_or_score_transition": (baseline.get("textSets", 0) or 0) > 0,
    }



def parse_probe_output(stdout: str) -> Optional[Dict]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# Required behaviours for an interactive game artifact. Coverage is judged
# against these; anything absent keeps the pipeline open.
