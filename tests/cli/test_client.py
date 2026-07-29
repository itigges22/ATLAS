"""Tests for atlas.cli.client seams against the real service contracts."""

from atlas.cli import client


def test_run_sandbox_reads_success_field(monkeypatch):
    """The sandbox executor's ExecuteResponse reports `success` — the
    client must read it (and must not send fields ExecuteRequest doesn't
    define, like stdin)."""
    captured = {}

    def fake_post(url, body, timeout=120):
        captured["url"] = url
        captured["body"] = body
        return {"success": True, "stdout": "hi\n", "stderr": ""}

    monkeypatch.setattr(client, "_post", fake_post)
    passed, stdout, stderr = client.run_sandbox("print('hi')", "assert True")
    assert passed is True
    assert stdout == "hi\n"
    assert "stdin" not in captured["body"]
    assert captured["body"]["code"] == "print('hi')"
    assert captured["body"]["test_code"] == "assert True"


def test_run_sandbox_falls_back_to_passed_field(monkeypatch):
    """Older sandbox builds returned `passed` — keep reading it when
    `success` is absent."""
    monkeypatch.setattr(client, "_post",
                        lambda url, body, timeout=120: {"passed": True,
                                                        "stdout": "",
                                                        "stderr": ""})
    passed, _, _ = client.run_sandbox("code")
    assert passed is True


def test_run_sandbox_returns_error_on_connection_failure(monkeypatch):
    def boom(url, body, timeout=120):
        raise OSError("connection refused")

    monkeypatch.setattr(client, "_post", boom)
    passed, stdout, stderr = client.run_sandbox("code")
    assert passed is False
    assert "connection refused" in stderr


def test_check_llama_reads_model_id_from_v1_models(monkeypatch):
    """llama-server's /health carries no model metadata; the id comes
    from /v1/models (fallback: "unknown")."""
    def fake_get(url, timeout=10):
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "/models/Qwen3.5-9B-Q6_K.gguf"}]}
        raise AssertionError(url)

    monkeypatch.setattr(client, "_get", fake_get)
    ok, model = client.check_llama()
    assert ok is True
    assert model == "Qwen3.5-9B-Q6_K.gguf"


def test_check_llama_unknown_when_models_endpoint_missing(monkeypatch):
    def fake_get(url, timeout=10):
        if url.endswith("/health"):
            return {"status": "ok"}
        raise OSError("404")

    monkeypatch.setattr(client, "_get", fake_get)
    ok, model = client.check_llama()
    assert ok is True
    assert model == "unknown"


def test_chat_stream_bridges_reasoning_content_into_think_tags(monkeypatch):
    """reasoning_content deltas surface as literal <think>…</think> so
    the solve pipeline keeps one thinking-detection path."""
    events = [
        b'data: {"choices":[{"delta":{"reasoning_content":"pondering"},'
        + b'"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{"content":"print(42)"},'
        + b'"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        b'data: [DONE]\n',
    ]

    class Response:
        def __init__(self):
            self._chunks = list(events)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n):
            return self._chunks.pop(0) if self._chunks else b""

    monkeypatch.setattr(client.urllib.request, "urlopen",
                        lambda req, timeout=900: Response())
    tokens = [text for text, _done in client.chat_stream(
        [{"role": "user", "content": "q"}])]
    assert tokens == ["<think>", "pondering", "</think>", "print(42)"]


def _capture_post(monkeypatch):
    """Replace client._post with a capturing stub; returns the capture dict."""
    captured = {}

    def fake_post(url, body, timeout=120, headers=None):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers or {}
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(client, "_post", fake_post)
    return captured


def test_chat_defaults_to_local_llama_without_auth(monkeypatch):
    """With no provider configured the chat path stays on the local
    llama-server: the historical /v1 URL, ATLAS_MODEL_NAME, and no auth."""
    monkeypatch.delenv("ATLAS_LLM_PROVIDER", raising=False)
    captured = _capture_post(monkeypatch)
    client.chat([{"role": "user", "content": "q"}])
    assert captured["url"] == f"{client.INFERENCE_URL}/v1/chat/completions"
    assert captured["body"]["model"] == client.MODEL_NAME
    assert "Authorization" not in captured["headers"]


def test_chat_minimax_global_endpoint_model_and_bearer(monkeypatch):
    """provider=minimax routes chat to the global MiniMax endpoint with the
    default MiniMax model id and a Bearer credential."""
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "minimax")
    monkeypatch.delenv("ATLAS_MINIMAX_REGION", raising=False)
    monkeypatch.delenv("ATLAS_MINIMAX_MODEL", raising=False)
    monkeypatch.setenv("ATLAS_MINIMAX_API_KEY", "sample-credential")
    captured = _capture_post(monkeypatch)
    client.chat([{"role": "user", "content": "q"}])
    assert captured["url"] == "https://api.minimax.io/v1/chat/completions"
    assert captured["body"]["model"] == "MiniMax-M3"
    assert captured["headers"]["Authorization"] == "Bearer sample-credential"


def test_chat_minimax_cn_region_and_model_selection(monkeypatch):
    """The CN region and an explicit model id are honored."""
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "minimax")
    monkeypatch.setenv("ATLAS_MINIMAX_REGION", "cn_zh")
    monkeypatch.setenv("ATLAS_MINIMAX_MODEL", "MiniMax-M2.7")
    monkeypatch.setenv("ATLAS_MINIMAX_API_KEY", "sample-credential")
    captured = _capture_post(monkeypatch)
    client.chat([{"role": "user", "content": "q"}])
    assert captured["url"] == "https://api.minimaxi.com/v1/chat/completions"
    assert captured["body"]["model"] == "MiniMax-M2.7"


def test_minimax_unknown_region_and_model_fall_back_to_defaults(monkeypatch):
    """Unrecognized region/model values fall back to the global endpoint and
    the default MiniMax model rather than emitting an invalid request."""
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "minimax")
    monkeypatch.setenv("ATLAS_MINIMAX_REGION", "mars")
    monkeypatch.setenv("ATLAS_MINIMAX_MODEL", "not-a-model")
    monkeypatch.setenv("ATLAS_MINIMAX_API_KEY", "sample-credential")
    captured = _capture_post(monkeypatch)
    client.generate("hello")
    assert captured["url"] == "https://api.minimax.io/v1/completions"
    assert captured["body"]["model"] == "MiniMax-M3"


def test_minimax_without_credential_sends_no_auth_header(monkeypatch):
    """A missing credential must not fabricate an empty Bearer header."""
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "minimax")
    monkeypatch.delenv("ATLAS_MINIMAX_API_KEY", raising=False)
    captured = _capture_post(monkeypatch)
    client.chat([{"role": "user", "content": "q"}])
    assert captured["headers"] == {}

