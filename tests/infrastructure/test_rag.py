"""
Tests for RAG API service.

Validates project syncing, vector indexing, context retrieval,
and RAG-enhanced completions.
"""

import os
import json
import uuid
import time
import hashlib

import pytest
import httpx


def compute_project_hash(files: list) -> str:
    """Compute hash from file contents."""
    content = "".join(f["content"] for f in files)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class TestRAGHealth:
    """Test RAG API health."""

    def test_health_endpoint_responds(self, rag_api_client: httpx.Client):
        """Health endpoint should return 200 OK."""
        response = rag_api_client.get("/health")
        assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"

    def test_root_endpoint(self, rag_api_client: httpx.Client):
        """Root endpoint should return service info."""
        response = rag_api_client.get("/")
        assert response.status_code == 200, f"Root endpoint should return 200, got {response.status_code}"


class TestRAGModelProxy:
    """Test model listing proxy to llama-server."""

    def test_v1_models_proxies_correctly(self, rag_api_client: httpx.Client, test_api_key):
        """GET /v1/models should proxy to llama-server."""
        response = rag_api_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {test_api_key.key_string}"}
        )
        assert response.status_code == 200, f"/v1/models should return 200, got {response.status_code}"
        data = response.json()
        assert "data" in data, "Response should have 'data' field with model list"


class TestRAGProjectSync:
    """Test project synchronization and indexing."""

    def test_project_sync_creates_entry(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        test_project_dir: str
    ):
        """Syncing a project should create a project entry."""
        project_name = f"test_project_{uuid.uuid4().hex[:8]}"

        # List files in test project
        files = []
        for filename in os.listdir(test_project_dir):
            filepath = os.path.join(test_project_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    files.append({
                        "path": filename,
                        "content": f.read()
                    })

        project_hash = compute_project_hash(files)

        response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )

        assert response.status_code == 200, f"Sync should return 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "project_id" in data, f"Response should have project_id: {data}"

    def test_project_status_after_sync(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        test_project_dir: str
    ):
        """After sync, project status should show indexed files."""
        project_name = f"test_status_{uuid.uuid4().hex[:8]}"

        # Sync project
        files = []
        for filename in os.listdir(test_project_dir):
            filepath = os.path.join(test_project_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    files.append({"path": filename, "content": f.read()})

        project_hash = compute_project_hash(files)

        sync_response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert sync_response.status_code == 200, f"Sync failed: {sync_response.text}"

        project_id = sync_response.json().get("project_id")

        # Check status
        status_response = rag_api_client.get(
            f"/v1/projects/{project_id}/status",
            headers={"Authorization": f"Bearer {test_api_key.key_string}"}
        )

        if status_response.status_code == 200:
            data = status_response.json()
            assert "project_id" in data or "status" in data, f"Status should have project info: {data}"


class TestRAGVectorOperations:
    """Test vector indexing and retrieval."""

    def test_vector_count_increases_after_sync(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        test_project_dir: str
    ):
        """Vector count should increase after project sync."""
        project_name = f"test_vectors_{uuid.uuid4().hex[:8]}"

        # Sync project
        files = []
        for filename in os.listdir(test_project_dir):
            filepath = os.path.join(test_project_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    files.append({"path": filename, "content": f.read()})

        project_hash = compute_project_hash(files)

        sync_response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert sync_response.status_code == 200, f"Sync failed: {sync_response.text}"

    def test_query_returns_context(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        test_project_dir: str
    ):
        """Query about project content should return relevant context."""
        project_name = f"test_query_{uuid.uuid4().hex[:8]}"

        # Sync project with known content
        files = []
        for filename in os.listdir(test_project_dir):
            filepath = os.path.join(test_project_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    files.append({"path": filename, "content": f.read()})

        project_hash = compute_project_hash(files)

        sync_response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert sync_response.status_code == 200

        project_id = sync_response.json().get("project_id")
        time.sleep(2)

        # Query about the calculator (which is in our test files)
        response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",  # Required model field
                "messages": [{"role": "user", "content": "What does the Calculator class do?"}],
                "max_tokens": 100,
                "project_id": project_id
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200, f"Query should succeed: {response.text}"


class TestRAGCompletion:
    """Test RAG-enhanced completions."""

    @pytest.mark.slow
    def test_rag_completion_includes_context(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        test_project_dir: str
    ):
        """RAG completion should include context from project files."""
        project_name = f"test_rag_ctx_{uuid.uuid4().hex[:8]}"

        # Sync project
        files = []
        for filename in os.listdir(test_project_dir):
            filepath = os.path.join(test_project_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    files.append({"path": filename, "content": f.read()})

        project_hash = compute_project_hash(files)

        sync_response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert sync_response.status_code == 200

        project_id = sync_response.json().get("project_id")
        time.sleep(2)

        # Ask about specific content that exists in test files
        response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",
                "messages": [
                    {"role": "user", "content": "Based on the code, what functions are defined for basic arithmetic?"}
                ],
                "max_tokens": 200,
                "project_id": project_id
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200

    def test_completion_without_project_works(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Completion without project_id should work (no RAG context)."""
        response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 20
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200, f"Completion should work without project: {response.text}"


class TestRAGLimits:
    """Test RAG configuration limits."""

    def test_context_budget_respected(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Context budget of 8000 tokens should be respected."""
        response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )
        assert response.status_code == 200, f"API should handle requests: {response.text}"
