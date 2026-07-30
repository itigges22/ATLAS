"""LLM generation and response post-processing for the V3 pipeline and
the bench harness.

`chat_completion()` calls llama-server's /v1/chat/completions endpoint with
structured messages, so the model's embedded chat template is applied
(llama-server runs with `--jinja`). Reasoning is controlled with the
`enable_thinking` chat-template kwarg (default off); templates that don't define
it ignore it. If a model returns its answer in `reasoning_content`, or leaves a
`<think>` block in `content`, that output is recovered/stripped so callers always
receive plain text. Any GGUF's own prompt format is honored without per-model
handling.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


def _service_token() -> str:
    """Internal-auth token (empty = auth disabled). Resolution: explicit
    ATLAS_SERVICE_TOKEN_FILE, the container secret mount, then the repo
    checkout's secrets/ file (stages/ lives under v3-service/, so
    parents[2] is the repo root)."""
    explicit = os.environ.get("ATLAS_SERVICE_TOKEN_FILE")
    candidates = [explicit] if explicit else [
        "/run/atlas-secrets/service-token",
        str(Path(__file__).resolve().parents[2] / "secrets" / "service-token"),
    ]
    for path in candidates:
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            continue
    return ""


def _auth_headers() -> dict:
    tok = _service_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _install_auth_opener() -> None:
    """Cover every urllib call site in the bench pipeline with the
    internal-auth header (urllib merges these under explicit per-request
    headers)."""
    tok = _service_token()
    if not tok:
        return
    opener = urllib.request.build_opener()
    opener.addheaders = [("Authorization", f"Bearer {tok}")]
    urllib.request.install_opener(opener)


_install_auth_opener()

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Plain-English system prompt — no model-specific directives (no `/nothink`).
DEFAULT_SYSTEM_PROMPT = "You are an expert programmer. Respond directly and concisely."

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CHATML_TURN = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s*\n(.*?)(?:<\|im_end\|>|\Z)",
    re.DOTALL,
)


def strip_reasoning_leak(text: str) -> str:
    """Remove a leaked reasoning block from model output. With
    `enable_thinking: false` reasoning lands in a separate field, but a model
    that emits a `<think>` block into the content anyway is handled here.
    Covers all three shapes: a closed `<think>...</think>` pair, an orphan
    closing tag (`...</think>answer` — keep the answer), and an orphan opening
    tag (`answer<think>truncated...` — keep the answer, drop the unclosed
    reasoning to end-of-text)."""
    if not text:
        return text
    out = _THINK_BLOCK.sub("", text)
    # Orphan closing tag (text before the open was lost / pre-fill artifact):
    # the real content follows the close.
    if "</think>" in out and "<think>" not in out:
        out = out.split("</think>", 1)[1]
    # Orphan opening tag (reasoning truncated mid-thought, no close): the real
    # content, if any, precedes the open; everything after is reasoning.
    if "<think>" in out:
        out = out.split("<think>", 1)[0]
    return out.strip()


def extract_code(response: str) -> str:
    """
    Extract code from an LLM response.

    Handles various formats:
    - Markdown code blocks with any language label (```python, ```javascript, ...)
    - Plain code blocks (``` ... ```)
    - Raw code without blocks
    - optional <think>...</think> reasoning blocks (stripped before extraction)

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code
    """
    # Strip template-emitted thinking blocks first; they can consume tokens
    # before the actual code output
    think_pattern = r'<think>.*?</think>'
    response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()

    # Safety net: strip unclosed <think> tags (edge case where
    # thinking mode doesn't fully strip thinking)
    if '<think>' in response and '</think>' not in response:
        response = response[:response.index('<think>')].strip()

    # Try MBPP [BEGIN]...[DONE] delimiters first
    begin_done_pattern = r'\[BEGIN\]\s*\n(.*?)(?:\[DONE\]|$)'
    begin_matches = re.findall(begin_done_pattern, response, re.DOTALL)
    if begin_matches:
        # Return the last match (the model's answer, not the few-shot examples)
        return begin_matches[-1].strip()

    # Extract fenced code with an optional language label. The V3 service
    # supports multiple languages, so limiting labels to Python leaves fences
    # such as ```javascript in the returned source and causes false syntax
    # failures downstream.
    pattern = r'```[^\S\r\n]*[A-Za-z0-9_+.#-]*[^\S\r\n]*\r?\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        # Return the longest match (likely the main code block)
        return max(matches, key=len).strip()

    # No code blocks found, assume raw code
    # Strip common prefixes/suffixes
    code = response.strip()

    # Remove common LLM artifacts
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip lines that look like explanations
        if line.strip().startswith('Here') and ':' in line:
            continue
        if line.strip().startswith('This function'):
            continue
        if line.strip().startswith('The function'):
            continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip()


def chatml_to_messages(prompt: str) -> List[Dict[str, str]]:
    """Convert a ChatML-formatted string into structured chat messages. Callers
    that assemble a ChatML prompt can pass it straight to `chat_completion`. A
    string with no ChatML markers becomes a single user message."""
    turns = _CHATML_TURN.findall(prompt or "")
    if not turns:
        return [{"role": "user", "content": (prompt or "").strip()}]
    messages = []
    for role, content in turns:
        content = content.strip()
        # Drop a trailing empty `assistant` turn (the generation cue).
        if role == "assistant" and not content:
            continue
        # Scrub any lingering `/nothink` directive from migrated system prompts.
        if role == "system":
            content = content.replace("/nothink", "").strip()
        messages.append({"role": role, "content": content})
    return messages or [{"role": "user", "content": (prompt or "").strip()}]


def _parse_logprobs(data: dict) -> List[float]:
    """Parse per-token logprobs from an OpenAI-style chat-completions response
    (`choices[0].logprobs.content[].logprob`). Returns [] if absent."""
    try:
        lp = data["choices"][0].get("logprobs") or {}
        toks = lp.get("content") or []
        return [t["logprob"] for t in toks if "logprob" in t]
    except (KeyError, IndexError, TypeError):
        return []


def chat_completion(
    llm_url: str,
    user: Optional[str] = None,
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    seed: Optional[int] = None,
    enable_thinking: bool = False,
    want_logprobs: bool = False,
    timeout: float = 600.0,
    extra_body: Optional[dict] = None,
) -> Dict:
    """Generate a completion model-agnostically via /v1/chat/completions.

    Provide EITHER `user` (+ optional `system`) OR a pre-built `messages` list
    (e.g. from `chatml_to_messages()`). Returns dict:
    {content, reasoning, tokens, time_ms, logprobs, raw}. `content` is cleaned
    (reasoning-leak stripped); if `content` is empty but the model emitted
    `reasoning_content`, that is used as the content fallback.
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user or ""})

    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # The model's OWN jinja template decides how to honor this; templates
        # without the kwarg ignore it harmlessly.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if seed is not None:
        body["seed"] = seed
    if want_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    if extra_body:
        body.update(extra_body)

    endpoint = f"{llm_url}/v1/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    start = time.time()
    if HAS_HTTPX:
        resp = httpx.post(endpoint, json=body, timeout=timeout,
                          headers=_auth_headers())
        resp.raise_for_status()
        data = resp.json()
    else:
        req = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    time_ms = (time.time() - start) * 1000

    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    # Universal fallback: a reasoning model that ignored enable_thinking and put
    # its answer in reasoning_content still yields usable output.
    if not content.strip() and reasoning.strip():
        content = reasoning
    content = strip_reasoning_leak(content)

    usage = data.get("usage", {}) or {}
    tokens = usage.get("completion_tokens", 0)
    return {
        "content": content,
        "reasoning": reasoning,
        "tokens": tokens,
        "time_ms": time_ms,
        "logprobs": _parse_logprobs(data) if want_logprobs else [],
        "raw": data,
    }
