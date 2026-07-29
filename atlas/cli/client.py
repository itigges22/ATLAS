"""HTTP client for llama-server, geometric-lens, and sandbox. Pure urllib, no dependencies."""

import contextlib
import json
import os
import urllib.request
import urllib.error
from typing import Dict, Optional, List, Tuple

from atlas.cli import compose as compose_config

# Shell env wins; otherwise the Docker .env's port keys drive the URLs.
INFERENCE_URL = compose_config.service_url("llama")
RAG_API_URL = (os.environ.get("ATLAS_RAG_URL")
               or compose_config.service_url("lens"))
SANDBOX_URL = compose_config.service_url("sandbox")
MODEL_NAME = os.environ.get("ATLAS_MODEL_NAME", "local-model")


# --- Provider selection ---
# The default backend is the local llama-server (OpenAI-compatible, no auth).
# Setting ATLAS_LLM_PROVIDER=minimax routes the generation calls to the
# MiniMax remote API instead: a region-scoped OpenAI-compatible base URL, a
# MiniMax model id, and a Bearer credential. Health, embedding, lens, and
# sandbox calls are local services and stay on their own URLs regardless of
# the selected provider.
_LLAMA_PROVIDER = "llama"
_MINIMAX_PROVIDER = "minimax"

# Region -> OpenAI-compatible base URL. Each value already carries the /v1
# suffix, so request paths append "/chat/completions" and "/completions".
_MINIMAX_OPENAI_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
_MINIMAX_DEFAULT_REGION = "global_en"
_MINIMAX_MODELS = ("MiniMax-M3", "MiniMax-M2.7")
_MINIMAX_DEFAULT_MODEL = "MiniMax-M3"


def _provider() -> str:
    """Active generation provider (defaults to the local llama-server)."""
    name = (os.environ.get("ATLAS_LLM_PROVIDER") or _LLAMA_PROVIDER).strip().lower()
    return _MINIMAX_PROVIDER if name == _MINIMAX_PROVIDER else _LLAMA_PROVIDER


def _minimax_region() -> str:
    region = (os.environ.get("ATLAS_MINIMAX_REGION")
              or _MINIMAX_DEFAULT_REGION).strip().lower()
    return region if region in _MINIMAX_OPENAI_BASE_URLS else _MINIMAX_DEFAULT_REGION


def _openai_base() -> str:
    """OpenAI-compatible base URL for the active provider (no trailing slash).

    For llama-server this is the local `.../v1`, keeping the historical
    request paths byte-for-byte; for MiniMax it is the region endpoint.
    """
    if _provider() == _MINIMAX_PROVIDER:
        return _MINIMAX_OPENAI_BASE_URLS[_minimax_region()]
    return f"{INFERENCE_URL}/v1"


def _model_name() -> str:
    """Model id sent in generation requests for the active provider."""
    if _provider() == _MINIMAX_PROVIDER:
        model = (os.environ.get("ATLAS_MINIMAX_MODEL")
                 or _MINIMAX_DEFAULT_MODEL).strip()
        return model if model in _MINIMAX_MODELS else _MINIMAX_DEFAULT_MODEL
    return MODEL_NAME


def _auth_headers() -> Dict[str, str]:
    """Authorization headers for the active generation provider.

    The local llama-server needs none; MiniMax authorizes with a Bearer
    credential read from ATLAS_MINIMAX_API_KEY.
    """
    if _provider() == _MINIMAX_PROVIDER:
        key = (os.environ.get("ATLAS_MINIMAX_API_KEY") or "").strip()
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


def _post(url: str, body: dict, timeout: int = 120,
          headers: Optional[Dict[str, str]] = None) -> dict:
    """POST JSON, return parsed response."""
    data = json.dumps(body).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 10) -> dict:
    """GET JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --- Health checks ---

def check_llama() -> Tuple[bool, str]:
    """Check llama-server health. The model id comes from /v1/models —
    llama-server's /health carries no model metadata."""
    try:
        _get(f"{INFERENCE_URL}/health")
    except Exception as e:
        return False, str(e)
    # Best-effort: /v1/models failing doesn't make the server unhealthy,
    # it just leaves the model id unknown.
    with contextlib.suppress(Exception):
        d = _get(f"{INFERENCE_URL}/v1/models")
        entries = d.get("data") or d.get("models") or []
        if entries:
            raw = entries[0].get("id") or entries[0].get("name") or ""
            if raw:
                return True, os.path.basename(str(raw))
    return True, "unknown"


def check_rag_api() -> Tuple[bool, str]:
    try:
        d = _get(f"{RAG_API_URL}/health")
        return True, d.get("status", "ok")
    except Exception as e:
        return False, str(e)


def check_sandbox() -> Tuple[bool, str]:
    try:
        d = _get(f"{SANDBOX_URL}/health")
        return True, d.get("status", "ok")
    except Exception as e:
        return False, str(e)


# --- Generation ---

def generate(prompt: str, max_tokens: int = 8192,
             temperature: float = 0.6, stop: Optional[List[str]] = None,
             timeout: int = 900) -> dict:
    """Generate via the provider's OpenAI-compatible /completions endpoint
    (raw prompt, includes thinking)."""
    body = {
        "model": _model_name(),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
    }
    if stop:
        body["stop"] = stop
    return _post(f"{_openai_base()}/completions", body, timeout=timeout,
                 headers=_auth_headers())


def generate_stream(prompt: str, max_tokens: int = 8192,
                    temperature: float = 0.6, stop: Optional[List[str]] = None,
                    timeout: int = 900):
    """Stream generation via the provider's /completions endpoint with
    stream=true.

    Yields (token_text, is_done) tuples.
    """
    body = {
        "model": _model_name(),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
        "stream": True,
    }
    if stop:
        body["stop"] = stop

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{_openai_base()}/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 **_auth_headers()},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buffer = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buffer += chunk
            # Process complete lines
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    event = json.loads(payload)
                    choices = event.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        finish = choices[0].get("finish_reason")
                        yield text, finish is not None
                        if finish is not None:
                            return
                except json.JSONDecodeError:
                    continue


def chat(messages: List[Dict], max_tokens: int = 8192,
         temperature: float = 0.6, timeout: int = 900) -> dict:
    """Generate via the provider's OpenAI-compatible /chat/completions.

    llama-server applies the GGUF's own chat template (--jinja) and remote
    providers apply their own, so this stays model-agnostic — no hand-built
    ChatML markers or stop tokens.
    """
    body = {
        "model": _model_name(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return _post(f"{_openai_base()}/chat/completions", body, timeout=timeout,
                 headers=_auth_headers())


def chat_stream(messages: List[Dict], max_tokens: int = 8192,
                temperature: float = 0.6, timeout: int = 900):
    """Stream /v1/chat/completions. Yields (token_text, is_done) tuples.

    `reasoning_content` deltas (templates whose thinking llama-server
    parses out of `content`) are bridged into literal <think>…</think>
    tags so callers keep a single thinking-detection path.
    """
    body = {
        "model": _model_name(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{_openai_base()}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 **_auth_headers()},
    )

    in_reasoning = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buffer = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    if in_reasoning:
                        yield "</think>", True
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                finish = choices[0].get("finish_reason")
                reasoning = delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        yield "<think>", False
                    yield reasoning, False
                if content:
                    if in_reasoning:
                        in_reasoning = False
                        yield "</think>", False
                    yield content, finish is not None
                if finish is not None:
                    if in_reasoning:
                        yield "</think>", True
                    return


# --- Embeddings ---

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from llama-server /embedding endpoint."""
    try:
        d = _post(f"{INFERENCE_URL}/embedding", {"content": text}, timeout=30)
        return d[0]["embedding"]
    except Exception:
        return None


# --- Lens scoring ---

def score_code(code: str) -> Tuple[float, float]:
    """Score code through Geometric Lens. Returns (energy, normalized)."""
    try:
        d = _post(
            f"{RAG_API_URL}/internal/lens/score-text",
            {"text": f"SOLUTION: {code}"},
            timeout=30,
        )
        return d.get("energy", 0.0), d.get("normalized", 0.5)
    except Exception:
        return 0.0, 0.5


def score_code_combined(code: str) -> dict:
    """Score code through combined C(x) + G(x) endpoint.

    Returns dict with cx_energy, cx_normalized, gx_score, verdict, gx_available.
    """
    try:
        d = _post(
            f"{RAG_API_URL}/internal/lens/gx-score",
            {"text": f"SOLUTION: {code}"},
            timeout=30,
        )
        return {
            "cx_energy": d.get("cx_energy", 0.0),
            "cx_normalized": d.get("cx_normalized", 0.5),
            "gx_score": d.get("gx_score", 0.5),
            "verdict": d.get("verdict", "unavailable"),
            "gx_available": d.get("gx_available", False),
        }
    except Exception:
        return {
            "cx_energy": 0.0, "cx_normalized": 0.5,
            "gx_score": 0.5, "verdict": "unavailable",
            "gx_available": False,
        }


def analyze_sandbox(code: str, passed: bool, stdout: str, stderr: str,
                    expected_output: str = "") -> dict:
    """Analyze sandbox result with structured error classification and G(x) scoring."""
    try:
        return _post(
            f"{RAG_API_URL}/internal/sandbox/analyze",
            {
                "code": code,
                "passed": passed,
                "stdout": stdout,
                "stderr": stderr,
                "expected_output": expected_output,
                "include_gx": True,
            },
            timeout=30,
        )
    except Exception:
        return {"error": "analysis_unavailable", "passed": passed}


# --- Sandbox ---

def run_sandbox(code: str, test_code: str = "",
                timeout_sec: int = 30) -> Tuple[bool, str, str]:
    """Execute code in sandbox. Returns (passed, stdout, stderr).

    The executor's ExecuteResponse reports `success`; `passed` is kept
    as a fallback for older sandbox builds.
    """
    try:
        body = {
            "code": code,
            "test_code": test_code,
            "timeout": timeout_sec,
        }
        d = _post(f"{SANDBOX_URL}/execute", body, timeout=timeout_sec + 10)
        passed = d.get("success", d.get("passed", False))
        return passed, d.get("stdout", ""), d.get("stderr", "")
    except Exception as e:
        return False, "", str(e)
