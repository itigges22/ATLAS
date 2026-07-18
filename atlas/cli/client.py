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


def _post(url: str, body: dict, timeout: int = 120) -> dict:
    """POST JSON, return parsed response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
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
    """Generate via llama-server /v1/completions (raw prompt, includes thinking)."""
    body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
    }
    if stop:
        body["stop"] = stop
    return _post(f"{INFERENCE_URL}/v1/completions", body, timeout=timeout)


def generate_stream(prompt: str, max_tokens: int = 8192,
                    temperature: float = 0.6, stop: Optional[List[str]] = None,
                    timeout: int = 900):
    """Stream generation via llama-server /v1/completions with stream=true.

    Yields (token_text, is_done) tuples.
    """
    body = {
        "model": MODEL_NAME,
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
        f"{INFERENCE_URL}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
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
    """Generate via llama-server /v1/chat/completions.

    llama-server applies the GGUF's own chat template (--jinja), so this
    stays model-agnostic — no hand-built ChatML markers or stop tokens.
    """
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return _post(f"{INFERENCE_URL}/v1/chat/completions", body, timeout=timeout)


def chat_stream(messages: List[Dict], max_tokens: int = 8192,
                temperature: float = 0.6, timeout: int = 900):
    """Stream /v1/chat/completions. Yields (token_text, is_done) tuples.

    `reasoning_content` deltas (templates whose thinking llama-server
    parses out of `content`) are bridged into literal <think>…</think>
    tags so callers keep a single thinking-detection path.
    """
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{INFERENCE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
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
