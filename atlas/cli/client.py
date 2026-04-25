"""HTTP client for vLLM (gen + embed), geometric-lens, and sandbox. Pure urllib, no dependencies."""

import json
import os
import urllib.request
import urllib.error
from typing import Optional, List, Tuple

# vLLM gen instance (port 8000 by default).
INFERENCE_URL = os.environ.get(
    "LLAMA_GEN_URL",
    os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8000"),
)
# vLLM embed instance (port 8001 by default).
EMBED_URL = os.environ.get(
    "LLAMA_EMBED_URL",
    os.environ.get("ATLAS_EMBED_URL", INFERENCE_URL),
)
RAG_API_URL = os.environ.get("ATLAS_LENS_URL", os.environ.get("ATLAS_RAG_URL", "http://localhost:31144"))
SANDBOX_URL = os.environ.get("ATLAS_SANDBOX_URL", "http://localhost:30820")
MODEL_NAME = os.environ.get("LLAMA_GEN_MODEL", os.environ.get("ATLAS_MODEL_NAME", "qwen3.5-9b"))
EMBED_MODEL_NAME = os.environ.get("LLAMA_EMBED_MODEL", "qwen3.5-9b-embed")


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


def _probe(url: str, timeout: int = 10) -> int:
    """GET, return HTTP status. urllib raises HTTPError on non-2xx, so any
    successful return means 2xx. Use for endpoints whose body shape is
    not guaranteed (e.g. vLLM `/health` returns 200 with an empty body)."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


# --- Health checks ---

def check_llama() -> Tuple[bool, str]:
    """Check vLLM gen instance health and get model name."""
    try:
        # vLLM 0.17+ /health returns 200 with an empty body — don't try to
        # json.loads it (would JSONDecodeError on every healthy stack).
        _probe(f"{INFERENCE_URL}/health")
        # Model name lives on /v1/models (returns OpenAI-shape JSON).
        try:
            models = _get(f"{INFERENCE_URL}/v1/models")
            model_list = models.get("models", models.get("data", []))
            if model_list:
                m = model_list[0]
                model = m.get("id", m.get("model", m.get("name", "unknown")))
            else:
                model = "connected"
        except Exception:
            model = "connected"
        return True, model
    except Exception as e:
        return False, str(e)


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
    """Generate via vLLM /v1/completions (raw prompt, includes thinking)."""
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
    """Stream generation via vLLM /v1/completions with stream=true.

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


# --- Embeddings ---

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from vLLM /v1/embeddings (OpenAI-compatible)."""
    try:
        d = _post(
            f"{EMBED_URL}/v1/embeddings",
            {"model": EMBED_MODEL_NAME, "input": text},
            timeout=30,
        )
        return d["data"][0]["embedding"]
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
                stdin: str = "", timeout_sec: int = 30) -> Tuple[bool, str, str]:
    """Execute code in sandbox. Returns (passed, stdout, stderr)."""
    try:
        body = {
            "code": code,
            "test_code": test_code,
            "stdin": stdin,
            "timeout": timeout_sec,
        }
        d = _post(f"{SANDBOX_URL}/execute", body, timeout=timeout_sec + 10)
        return d.get("passed", False), d.get("stdout", ""), d.get("stderr", "")
    except Exception as e:
        return False, "", str(e)
