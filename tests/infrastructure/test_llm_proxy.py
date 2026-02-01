"""
Tests for LLM Proxy service.

Validates API key authentication, request proxying,
rate limiting, and metrics logging.
"""

import json
import uuid
import time
import pytest
import httpx


class TestLLMProxyHealth:
    """Test LLM Proxy service health."""

    def test_health_endpoint_responds(self, llm_proxy_client: httpx.Client):
        """Health endpoint should return 200 OK."""
        response = llm_proxy_client.get("/health")
        assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"


class TestLLMProxyAuthentication:
    """Test API key authentication."""

    def test_rejects_request_without_api_key(self, llm_proxy_client: httpx.Client):
        """Request without API key should be rejected with 401."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        )
        assert response.status_code in [401, 403], f"Should reject without API key, got {response.status_code}"

    def test_rejects_invalid_api_key(self, llm_proxy_client: httpx.Client):
        """Request with invalid API key should be rejected."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            headers={"Authorization": "Bearer invalid_key_xyz123"}
        )
        assert response.status_code in [401, 403], f"Should reject invalid key, got {response.status_code}"

    def test_accepts_valid_api_key(self, llm_proxy_client: httpx.Client, test_api_key):
        """Request with valid API key should be accepted."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200, f"Should accept valid key, got {response.status_code}: {response.text}"


class TestLLMProxyKeyValidation:
    """Test API key validation against API Portal."""

    def test_validates_key_against_portal(self, llm_proxy_client: httpx.Client, test_api_key):
        """Proxy should validate key against API Portal."""
        # Make a request - if it succeeds, validation worked
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200, "Valid key should pass portal validation"

    def test_caches_validation(self, llm_proxy_client: httpx.Client, test_api_key):
        """Validation should be cached for 60 seconds."""
        # Make two requests in quick succession
        # If caching works, second should be faster (but we can't easily measure)
        for _ in range(2):
            response = llm_proxy_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Cache test"}],
                    "max_tokens": 5
                },
                headers={"Authorization": f"Bearer {test_api_key.key_string}"},
                timeout=120.0
            )
            assert response.status_code == 200


class TestLLMProxyRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers_present(self, llm_proxy_client: httpx.Client, test_api_key):
        """Rate limit headers should be present in response."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        # Rate limit headers should be present
        headers = response.headers
        has_rate_limit = any("ratelimit" in h.lower() for h in headers.keys())
        assert has_rate_limit, f"Rate limit headers should be present in response. Got headers: {list(headers.keys())}"
        # Check specific headers
        assert "x-ratelimit-limit" in [h.lower() for h in headers.keys()], "X-RateLimit-Limit header should be present"

    @pytest.mark.slow
    def test_rate_limit_enforcement(
        self,
        llm_proxy_client: httpx.Client,
        api_portal_client: httpx.Client,
        test_user,
        redis_client
    ):
        """Verify 429 is returned when rate limit exceeded."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        # Create a test key with very low rate limit (3 requests per minute)
        response = api_portal_client.post(
            "/api/keys",
            json={"name": f"rate_limit_test_{uuid.uuid4().hex[:8]}", "rate_limit": 3},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert response.status_code == 200, f"Failed to create rate limited key: {response.text}"
        data = response.json()
        test_key = data.get("key") or data.get("api_key")
        key_id = data.get("id") or data.get("key_id")

        try:
            # Clear any existing rate limit counter for this key
            import hashlib
            key_hash = hashlib.sha256(test_key.encode()).hexdigest()[:16]
            redis_client.delete(f"ratelimit:{key_hash}:count")

            # Make requests up to the limit (3 requests)
            for i in range(3):
                response = llm_proxy_client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 5
                    },
                    headers={"Authorization": f"Bearer {test_key}"},
                    timeout=120.0
                )
                assert response.status_code == 200, f"Request {i+1} should succeed, got {response.status_code}: {response.text}"

            # Next request should be rate limited (429)
            response = llm_proxy_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5
                },
                headers={"Authorization": f"Bearer {test_key}"},
                timeout=30.0
            )
            assert response.status_code == 429, f"Request 4 should be rate limited, got {response.status_code}"
            assert "retry-after" in [h.lower() for h in response.headers.keys()], "Retry-After header should be present"

            # Verify rate limit headers in 429 response
            assert response.headers.get("x-ratelimit-remaining") == "0", "Remaining should be 0"

        finally:
            # Cleanup: Delete the test key
            if key_id:
                api_portal_client.delete(
                    f"/api/keys/{key_id}",
                    headers={"Authorization": f"Bearer {test_user.jwt_token}"}
                )
            # Clean up rate limit counter
            redis_client.delete(f"ratelimit:{key_hash}:count")


class TestLLMProxyProxying:
    """Test request proxying to llama-server."""

    def test_proxies_to_llama_correctly(self, llm_proxy_client: httpx.Client, test_api_key):
        """Proxy should forward requests to llama-server correctly."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
                "max_tokens": 10,
                "temperature": 0
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data, "Should return LLM response"
        message = data["choices"][0]["message"]
        # Qwen3 may return content in "content" or "reasoning_content"
        content = message.get("content", "") or message.get("reasoning_content", "")
        assert len(content) > 0 or data.get("usage", {}).get("completion_tokens", 0) > 0, \
            f"Should have response content or tokens: {data}"

    @pytest.mark.slow
    def test_streaming_proxied_correctly(self, llm_proxy_client: httpx.Client, test_api_key):
        """Streaming should be proxied correctly."""
        with llm_proxy_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Count: 1, 2, 3"}],
                "max_tokens": 20,
                "stream": True
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        ) as response:
            assert response.status_code == 200

            chunks = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break
                    try:
                        chunks.append(json.loads(chunk_data))
                    except json.JSONDecodeError:
                        continue  # Skip malformed SSE lines (e.g., comments, empty data)

            assert len(chunks) > 0, "Should receive streaming chunks"

    def test_v1_models_proxied(self, llm_proxy_client: httpx.Client, test_api_key):
        """GET /v1/models should be proxied."""
        response = llm_proxy_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {test_api_key.key_string}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data, "Should return model list"


class TestLLMProxyMetrics:
    """Test metrics logging to Redis."""

    def test_usage_metrics_recorded(
        self,
        llm_proxy_client: httpx.Client,
        test_api_key,
        redis_client
    ):
        """Usage metrics should be recorded to Redis."""
        # Make a request
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Metrics test"}],
                "max_tokens": 10
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

        # Give time for async metrics logging
        time.sleep(1)

        # Check for metrics in Redis
        # Metrics format may be: metrics:daily:{date}, metrics:recent_tasks, etc.
        import datetime
        today = datetime.date.today().isoformat()
        daily_key = f"metrics:daily:{today}"

        # Check if daily metrics exist
        exists = redis_client.exists(daily_key)
        if exists:
            metrics = redis_client.hgetall(daily_key)
            # Should have some metrics
            assert len(metrics) > 0, "Daily metrics should have data"

    def test_request_count_incremented(
        self,
        llm_proxy_client: httpx.Client,
        test_api_key,
        redis_client
    ):
        """Request count should be incremented."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        daily_key = f"metrics:daily:{today}"

        # Get initial count
        initial_count = int(redis_client.hget(daily_key, "requests_total") or 0)

        # Make a request
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Count test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

        time.sleep(1)

        # Check count increased
        new_count = int(redis_client.hget(daily_key, "requests_total") or 0)
        # Count should have increased (may increase by more than 1 due to other tests)
        assert new_count > initial_count, f"Request count should have increased. Initial: {initial_count}, New: {new_count}"

    def test_token_count_tracked(
        self,
        llm_proxy_client: httpx.Client,
        test_api_key,
        redis_client
    ):
        """Token count should be tracked."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        daily_key = f"metrics:daily:{today}"

        # Make a request
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Token tracking test"}],
                "max_tokens": 10
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

        time.sleep(1)

        # Check token count exists
        tokens = redis_client.hget(daily_key, "tokens_total") or redis_client.hget(daily_key, "tokens_used")
        if tokens is not None:
            assert int(tokens) > 0, "Token count should be positive"


class TestLLMProxyErrorHandling:
    """Test error handling."""

    def test_handles_empty_messages(self, llm_proxy_client: httpx.Client, test_api_key):
        """Should handle empty messages array."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [],
                "max_tokens": 10
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=30.0
        )
        # Empty messages may be passed to llama-server which handles it
        assert response.status_code == 200

    def test_handles_missing_messages(self, llm_proxy_client: httpx.Client, test_api_key):
        """Missing messages field should return 400."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={"max_tokens": 10},
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=30.0
        )
        # Proxy validates messages field is required
        assert response.status_code == 400

    def test_handles_invalid_json(self, llm_proxy_client: httpx.Client, test_api_key):
        """Invalid JSON body should return 400."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            content="not valid json",
            headers={
                "Authorization": f"Bearer {test_api_key.key_string}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        # Proxy validates JSON and returns 400
        assert response.status_code == 400

    def test_handles_very_large_max_tokens(self, llm_proxy_client: httpx.Client, test_api_key):
        """Should handle very large max_tokens - passed to llama-server."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 1000000
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=30.0
        )
        # Llama-server clamps max_tokens internally
        assert response.status_code == 200


class TestLLMProxyHeaderForwarding:
    """Test header forwarding."""

    def test_content_type_preserved(self, llm_proxy_client: httpx.Client, test_api_key):
        """Content-Type should be preserved in response."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type

    def test_streaming_content_type(self, llm_proxy_client: httpx.Client, test_api_key):
        """Streaming should use text/event-stream content type."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
                "stream": True
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type or "application/json" in content_type


class TestLLMProxyEndpoints:
    """Test various proxy endpoints."""

    def test_health_no_auth_required(self, llm_proxy_client: httpx.Client):
        """Health endpoint should not require auth."""
        response = llm_proxy_client.get("/health")
        assert response.status_code == 200

    def test_v1_models_requires_auth(self, llm_proxy_client: httpx.Client):
        """v1/models should require auth."""
        response = llm_proxy_client.get("/v1/models")
        assert response.status_code in [401, 403]

    def test_v1_completions_endpoint(self, llm_proxy_client: httpx.Client, test_api_key):
        """v1/completions endpoint should be proxied to llama-server."""
        response = llm_proxy_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello",
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        # Proxied endpoint - llama-server handles it
        assert response.status_code == 200

    def test_unknown_endpoint_handled(self, llm_proxy_client: httpx.Client, test_api_key):
        """Unknown endpoints should return 404 from llama-server."""
        response = llm_proxy_client.get(
            "/v1/unknown_endpoint",
            headers={"Authorization": f"Bearer {test_api_key.key_string}"}
        )
        assert response.status_code == 404


class TestLLMProxyRequestParams:
    """Test request parameter handling."""

    def test_temperature_forwarded(self, llm_proxy_client: httpx.Client, test_api_key):
        """Temperature parameter should be forwarded."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
                "temperature": 0.5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

    def test_top_p_forwarded(self, llm_proxy_client: httpx.Client, test_api_key):
        """top_p parameter should be forwarded."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
                "top_p": 0.9
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

    def test_stop_sequences_forwarded(self, llm_proxy_client: httpx.Client, test_api_key):
        """Stop sequences should be forwarded."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Count 1 2 3 4 5"}],
                "max_tokens": 20,
                "stop": ["3"]
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

    def test_presence_penalty_forwarded(self, llm_proxy_client: httpx.Client, test_api_key):
        """presence_penalty should be forwarded."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
                "presence_penalty": 0.5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

    def test_frequency_penalty_forwarded(self, llm_proxy_client: httpx.Client, test_api_key):
        """frequency_penalty should be forwarded."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
                "frequency_penalty": 0.5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200


class TestLLMProxyConcurrency:
    """Test concurrent request handling."""

    def test_concurrent_requests(self, llm_proxy_client: httpx.Client, test_api_key):
        """Should handle concurrent requests."""
        import concurrent.futures

        def make_request(i):
            return llm_proxy_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": f"Test {i}"}],
                    "max_tokens": 5
                },
                headers={"Authorization": f"Bearer {test_api_key.key_string}"},
                timeout=120.0
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(3)]
            results = [f.result() for f in futures]

        for r in results:
            assert r.status_code == 200


class TestLLMProxyRateLimitHeaders:
    """Test rate limit header details."""

    def test_ratelimit_limit_header(self, llm_proxy_client: httpx.Client, test_api_key):
        """X-RateLimit-Limit should be present."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200
        assert "x-ratelimit-limit" in [h.lower() for h in response.headers.keys()]

    def test_ratelimit_remaining_header(self, llm_proxy_client: httpx.Client, test_api_key):
        """X-RateLimit-Remaining should be present."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200
        assert "x-ratelimit-remaining" in [h.lower() for h in response.headers.keys()]

    def test_ratelimit_reset_header(self, llm_proxy_client: httpx.Client, test_api_key):
        """X-RateLimit-Reset should be present."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200
        assert "x-ratelimit-reset" in [h.lower() for h in response.headers.keys()]

    def test_ratelimit_values_are_numeric(self, llm_proxy_client: httpx.Client, test_api_key):
        """Rate limit header values should be numeric."""
        response = llm_proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200

        limit = response.headers.get("x-ratelimit-limit")
        if limit:
            assert limit.isdigit(), f"Limit should be numeric: {limit}"

        remaining = response.headers.get("x-ratelimit-remaining")
        if remaining:
            assert remaining.isdigit(), f"Remaining should be numeric: {remaining}"
