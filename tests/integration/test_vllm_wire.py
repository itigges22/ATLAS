"""Wire-level integration tests for the vLLM port.

Run a Python `http.server` in a thread that pretends to be vLLM, then
point the ATLAS code paths at it and verify they (a) shape the request
correctly and (b) parse the response correctly. No GPU, no actual model,
no httpx — just the standard library and the project's own urllib paths.

What we cover:
  - benchmark.runner.BenchmarkRunner._call_llm sends a /v1/chat/completions
    body with messages, max_tokens, and chat_template_kwargs and parses
    choices[0].message.content.
  - benchmark.v3_runner LLMAdapter sends /v1/completions with prompt +
    max_tokens (translated from n_predict), drops cache_prompt, and parses
    choices[0].text + token_logprobs.
  - benchmarks.v301_runner.LLMClient.chat sends /v1/chat/completions with
    presence_penalty + chat_template_kwargs.enable_thinking=true.
  - benchmarks.v301_runner.LLMClient.completion_nothink sends the same
    endpoint with chat_template_kwargs.enable_thinking=false.
  - geometric_lens.embedding_extractor.extract_embedding sends
    /v1/embeddings with model + input and returns the float list.

If any of these fail, the live vLLM run will fail — these are the request
shapes and response parses that the live stack depends on.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock vLLM server
# ---------------------------------------------------------------------------

class MockVllm:
    """In-process HTTP server that captures last-seen request and replies
    with canned vLLM-shaped responses."""

    def __init__(self):
        self.last_path = None
        self.last_body = None
        # Default canned responses for each endpoint.
        self.chat_completion_response = {
            "id": "chat-1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3.5-9b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": None,
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        self.completion_response = {
            "id": "cmpl-1",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "qwen3.5-9b",
            "choices": [{
                "index": 0,
                "text": "print(1)",
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2, -0.05]},
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        self.embeddings_response = {
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.01] * 4096,
            }],
            "model": "qwen3.5-9b-embed",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        self._lock = threading.Lock()

    def make_handler(mock):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a, **k):
                pass  # quiet

            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    body = b'{"status":"ok"}'
                    if self.path == "/v1/models":
                        body = b'{"data":[{"id":"qwen3.5-9b","object":"model"}]}'
                    self.wfile.write(body)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                with mock._lock:
                    mock.last_path = self.path
                    mock.last_body = json.loads(raw) if raw else {}

                if self.path == "/v1/chat/completions":
                    body = json.dumps(mock.chat_completion_response).encode()
                elif self.path == "/v1/completions":
                    body = json.dumps(mock.completion_response).encode()
                elif self.path == "/v1/embeddings":
                    body = json.dumps(mock.embeddings_response).encode()
                else:
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
        return Handler


@pytest.fixture
def mock_vllm():
    """Spin up an in-process mock vLLM on a free port for the test."""
    mock = MockVllm()
    server = HTTPServer(("127.0.0.1", 0), mock.make_handler())
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    mock.url = f"http://127.0.0.1:{port}"
    yield mock
    server.shutdown()
    thread.join(timeout=2)


# ---------------------------------------------------------------------------
# benchmark.runner._call_llm — chat completions, vLLM-shaped
# ---------------------------------------------------------------------------

def test_runner_call_llm_sends_chat_completions(mock_vllm):
    from benchmark.runner import BenchmarkRunner

    r = BenchmarkRunner(llm_url=mock_vllm.url)
    content, tokens, _ = r._call_llm("hello world", temperature=0, max_tokens=64)

    assert mock_vllm.last_path == "/v1/chat/completions"
    body = mock_vllm.last_body
    assert body["model"] == "qwen3.5-9b"
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0
    assert isinstance(body.get("messages"), list)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][1]["content"].endswith("hello world")
    # Qwen3.5: must use chat_template_kwargs, NOT a /nothink prefix in the prompt.
    assert "chat_template_kwargs" in body
    assert "/nothink" not in body["messages"][1]["content"]
    assert content == "ok"
    assert tokens == 1


# ---------------------------------------------------------------------------
# benchmark.v3_runner LLMAdapter — /v1/completions with logprobs
# ---------------------------------------------------------------------------

def test_runner_call_llm_ignores_legacy_cache_prompt_kwarg(mock_vllm):
    """benchmark/v2_runner.py still passes cache_prompt=True through to the
    runner — the kwarg now just gets silently ignored (vLLM's prefix caching
    is enabled at server level, not per-request). Verify this back-compat
    path doesn't blow up and that the cache_prompt field never leaks into
    the outgoing request body."""
    from benchmark.runner import BenchmarkRunner

    r = BenchmarkRunner(llm_url=mock_vllm.url)
    # All the kwargs v2_runner uses, including cache_prompt=True.
    content, tokens, _ = r._call_llm(
        "test prompt",
        temperature=0,
        max_tokens=32,
        cache_prompt=True,
        seed=42,
        think=False,
    )
    assert content == "ok"
    assert tokens == 1
    body = mock_vllm.last_body
    # cache_prompt must not appear in the wire request.
    assert "cache_prompt" not in body
    assert body["seed"] == 42
    assert body["max_tokens"] == 32


def test_v3_adapter_translates_n_predict_and_drops_cache_prompt(mock_vllm):
    from benchmark.runner import BenchmarkRunner
    from benchmark.v3_runner import LLMAdapter

    r = BenchmarkRunner(llm_url=mock_vllm.url)
    adapter = LLMAdapter(r)

    content, tokens, _ms = adapter("PROMPT", temperature=0.6, max_tokens=128, seed=42)

    assert mock_vllm.last_path == "/v1/completions"
    body = mock_vllm.last_body
    assert body["prompt"] == "PROMPT"
    assert body["max_tokens"] == 128
    assert "n_predict" not in body, "n_predict must be translated to max_tokens"
    assert "cache_prompt" not in body, "cache_prompt is llama.cpp-only and must be dropped"
    assert body["seed"] == 42
    assert body["logprobs"] == 1
    assert content == "print(1)"
    assert tokens == 3
    # Logprobs come from token_logprobs (not chat-completions content[].logprob).
    assert adapter.last_logprobs == [-0.1, -0.2, -0.05]


# ---------------------------------------------------------------------------
# benchmarks.v301_runner.LLMClient — chat with thinking, and completion_nothink
# ---------------------------------------------------------------------------

def test_v3_adapter_handles_missing_logprobs(mock_vllm):
    """If a future vLLM upgrade or config change drops logprobs from
    the response, the V3 adapter must not crash. last_logprobs should
    just come back empty — Phase 2 candidate scoring degrades to
    sandbox-only signal."""
    from benchmark.runner import BenchmarkRunner
    from benchmark.v3_runner import LLMAdapter

    # vLLM /v1/completions response with no logprobs field at all.
    mock_vllm.completion_response = {
        "id": "cmpl-no-lp",
        "object": "text_completion",
        "created": 0,
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "text": "print('no logprobs')",
            "finish_reason": "stop",
            # logprobs key intentionally absent
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    r = BenchmarkRunner(llm_url=mock_vllm.url)
    adapter = LLMAdapter(r)
    content, tokens, _ = adapter("PROMPT", temperature=0, max_tokens=64, seed=None)

    assert content == "print('no logprobs')"
    assert tokens == 3
    assert adapter.last_logprobs == []  # graceful empty, not a crash


def test_v301_chat_thinking_on(mock_vllm):
    from benchmarks.v301_runner import LLMClient

    c = LLMClient(url=mock_vllm.url)
    content, reasoning, tokens, _ms = c.chat(
        [{"role": "user", "content": "hi"}],
        temperature=1.0,
        max_tokens=4096,
    )

    assert mock_vllm.last_path == "/v1/chat/completions"
    body = mock_vllm.last_body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["presence_penalty"] == 1.5
    assert body["top_k"] == 20
    assert body["top_p"] == 0.95
    assert body["max_tokens"] == 4096
    assert content == "ok"
    assert tokens == 1


def test_v301_completion_nothink_disables_thinking(mock_vllm):
    from benchmarks.v301_runner import LLMClient

    c = LLMClient(url=mock_vllm.url)
    content, tokens, _ms = c.completion_nothink(
        "system text", "user question",
        temperature=0.0, max_tokens=256,
    )

    assert mock_vllm.last_path == "/v1/chat/completions"
    body = mock_vllm.last_body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # Verify NO legacy /nothink soft command leaked through.
    serialized = json.dumps(body)
    assert "/nothink" not in serialized, "Qwen3.5 dropped /nothink — should never appear"
    assert content == "ok"
    assert tokens == 1


# ---------------------------------------------------------------------------
# geometric_lens.embedding_extractor — /v1/embeddings
# ---------------------------------------------------------------------------

def test_v301_chat_handles_reasoning_split(mock_vllm):
    """When vLLM uses --reasoning-parser qwen3, thinking lands in
    reasoning_content and the answer in content. v301_runner.LLMClient.chat
    must surface both fields to the caller — historical bugs forgot to read
    reasoning_content and silently dropped the model's chain-of-thought."""
    from benchmarks.v301_runner import LLMClient

    # Override the chat response shape: reasoning + content split.
    mock_vllm.chat_completion_response = {
        "id": "chat-2",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Answer: B",
                "reasoning_content": "Let me think about this...\nThe correct answer is B.",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    }

    c = LLMClient(url=mock_vllm.url)
    content, reasoning, tokens, _ = c.chat([{"role": "user", "content": "what's the answer?"}])
    assert content == "Answer: B"
    assert "correct answer is B" in reasoning
    assert tokens == 30


def test_v301_chat_handles_empty_content(mock_vllm):
    """vLLM returns empty content when the budget runs out mid-thinking
    (all tokens went to reasoning_content, none left for the answer).
    This happened a lot with llama.cpp and we have to handle it
    gracefully — return what we have rather than crashing."""
    from benchmarks.v301_runner import LLMClient

    mock_vllm.chat_completion_response = {
        "id": "chat-budget",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",  # nothing left after thinking ate the budget
                "reasoning_content": "I'm still thinking about this very hard...",
            },
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 8000, "total_tokens": 8050},
    }

    c = LLMClient(url=mock_vllm.url)
    content, reasoning, tokens, _ = c.chat([{"role": "user", "content": "hard question"}])
    assert content == ""  # caller decides how to handle empty response
    assert "thinking" in reasoning
    assert tokens == 8000


def test_v301_chat_handles_missing_reasoning_content(mock_vllm):
    """If vLLM is started without --reasoning-parser, reasoning_content is
    absent. The runner must still extract content and not crash."""
    from benchmarks.v301_runner import LLMClient

    mock_vllm.chat_completion_response = {
        "id": "chat-3",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3.5-9b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Answer: C"},
            # No reasoning_content key at all.
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
    }

    c = LLMClient(url=mock_vllm.url)
    content, reasoning, tokens, _ = c.chat([{"role": "user", "content": "what's the answer?"}])
    assert content == "Answer: C"
    assert reasoning == "" or reasoning is None
    assert tokens == 5


def test_v301_chat_retries_on_503(mock_vllm):
    """vLLM returns 503 when warming up or all slots busy. The runner must
    back off and retry rather than failing the whole task."""
    from benchmarks.v301_runner import LLMClient

    # Patch the handler to fail twice with 503, then succeed on the third call.
    call_count = {"n": 0}
    BaseHandler = mock_vllm.make_handler()

    class Flaky(BaseHandler):
        def do_POST(self):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                self.send_response(503)
                self.end_headers()
                return
            BaseHandler.do_POST(self)

    import threading
    from http.server import HTTPServer
    server = HTTPServer(("127.0.0.1", 0), Flaky)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        c = LLMClient(url=f"http://127.0.0.1:{port}", max_retries=5)
        content, _, _, _ = c.chat([{"role": "user", "content": "hi"}], max_tokens=8)
        assert content == "ok"
        assert call_count["n"] == 3, f"expected 2 retries + 1 success, got {call_count['n']} calls"
    finally:
        server.shutdown()


def test_lens_embedding_extractor(monkeypatch, mock_vllm):
    monkeypatch.setenv("LLAMA_EMBED_URL", mock_vllm.url)
    monkeypatch.setenv("LLAMA_EMBED_MODEL", "qwen3.5-9b-embed")

    sys.path.insert(0, str(PROJECT_ROOT / "geometric-lens"))
    from geometric_lens.embedding_extractor import extract_embedding, extract_embeddings_batch

    emb = extract_embedding("def add(a,b): return a+b")
    assert mock_vllm.last_path == "/v1/embeddings"
    body = mock_vllm.last_body
    assert body["model"] == "qwen3.5-9b-embed"
    assert body["input"] == "def add(a,b): return a+b"
    assert isinstance(emb, list) and len(emb) == 4096

    # Batched form takes a list and returns one vector per input.
    mock_vllm.embeddings_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 4096},
            {"object": "embedding", "index": 1, "embedding": [0.2] * 4096},
        ],
        "model": "qwen3.5-9b-embed",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    embs = extract_embeddings_batch(["a", "b"])
    assert len(embs) == 2
    assert embs[0][0] == 0.1
    assert embs[1][0] == 0.2
    body = mock_vllm.last_body
    assert body["input"] == ["a", "b"]
