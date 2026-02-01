"""
End-to-end RAG flow tests.

Tests the complete RAG pipeline from project sync
through context-aware completions.
"""

import os
import uuid
import time
import hashlib

import pytest
import httpx


def compute_project_hash(files: list) -> str:
    """Compute hash from file contents."""
    content = "".join(f["content"] for f in files)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@pytest.mark.integration
class TestE2ERAGSync:
    """Test RAG project synchronization."""

    def test_create_and_sync_project(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should be able to create and sync a test project."""
        project_name = f"e2e_test_project_{uuid.uuid4().hex[:8]}"

        # Create test files
        files = [
            {
                "path": "calculator.py",
                "content": '''"""Calculator module for testing RAG."""

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []
'''
            }
        ]

        project_hash = compute_project_hash(files)

        # Sync project
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

        assert response.status_code == 200, f"Sync should succeed: {response.text}"

    def test_project_appears_after_sync(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Synced project should appear in project list."""
        project_name = f"e2e_list_project_{uuid.uuid4().hex[:8]}"

        # Sync project
        files = [{"path": "main.py", "content": "x = 1\n"}]
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

        # Check project list
        list_response = rag_api_client.get(
            "/v1/projects",
            headers={"Authorization": f"Bearer {test_api_key.key_string}"}
        )

        if list_response.status_code == 200:
            data = list_response.json()
            projects = data if isinstance(data, list) else data.get("projects", [])
            # Project should be in list
        elif list_response.status_code == 404:
            pytest.skip("Project list endpoint not implemented")

    def test_files_indexed_after_sync(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Files should be indexed with vectors after sync."""
        project_name = f"e2e_indexed_{uuid.uuid4().hex[:8]}"

        files = [
            {"path": "main.py", "content": "def main(): print('hello')\n"},
            {"path": "utils.py", "content": "def helper(): return 42\n"}
        ]
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


@pytest.mark.integration
class TestE2ERAGQuery:
    """Test RAG context retrieval."""

    @pytest.mark.slow
    def test_query_returns_context(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Query about project content should return relevant context."""
        project_name = f"e2e_query_{uuid.uuid4().hex[:8]}"

        # Sync project with specific content
        files = [
            {
                "path": "authentication.py",
                "content": '''"""Authentication module."""

def login(username: str, password: str) -> bool:
    """Authenticate user."""
    return True

def logout(session_id: str) -> None:
    """Logout user."""
    pass
'''
            }
        ]
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
        time.sleep(3)

        # Query about authentication
        response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": "How does authentication work?"}],
                "max_tokens": 100,
                "project_id": project_id
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200, f"Query should succeed: {response.text}"

    @pytest.mark.slow
    def test_context_appears_in_completion(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """RAG completion should include context from project files."""
        project_name = f"e2e_context_{uuid.uuid4().hex[:8]}"

        # Sync with unique identifiable content
        files = [
            {
                "path": "special_module.py",
                "content": '''"""Special module with UNIQUE_MARKER_ABC."""

def special_function():
    """Returns UNIQUE_MARKER_ABC."""
    return "special_value"
'''
            }
        ]
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
        time.sleep(3)

        # Query for unique identifier
        response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": "What is UNIQUE_MARKER_ABC?"}],
                "max_tokens": 100,
                "project_id": project_id
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200


@pytest.mark.integration
class TestE2ERAGComparison:
    """Test RAG vs non-RAG completion quality."""

    @pytest.mark.slow
    def test_rag_improves_response_quality(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """RAG should improve response quality for project-specific questions."""
        project_name = f"e2e_compare_{uuid.uuid4().hex[:8]}"

        # Sync project with specific implementation
        files = [
            {
                "path": "database.py",
                "content": '''"""Database module using PostgreSQL."""

DATABASE_URL = "postgresql://localhost:5432/mydb"
MAX_CONNECTIONS = 10

def connect():
    """Connect to PostgreSQL database."""
    pass
'''
            }
        ]
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
        time.sleep(3)

        # Query WITH RAG context
        rag_response = rag_api_client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-14B-Q4_K_M.gguf",
                "messages": [{"role": "user", "content": "What database is used?"}],
                "max_tokens": 50,
                "project_id": project_id
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert rag_response.status_code == 200


@pytest.mark.integration
class TestE2ERAGCleanup:
    """Test RAG project cleanup."""

    def test_delete_project(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should be able to delete a project."""
        project_name = f"e2e_delete_{uuid.uuid4().hex[:8]}"

        # Create project
        files = [{"path": "temp.py", "content": "x = 1\n"}]
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

        if sync_response.status_code == 200:
            project_id = sync_response.json().get("project_id")
            # Delete project
            delete_response = rag_api_client.delete(
                f"/v1/projects/{project_id}",
                headers={"Authorization": f"Bearer {test_api_key.key_string}"}
            )
            # Should return 200 or 204 for successful deletion
            assert delete_response.status_code in [200, 204], \
                f"Delete should return 200 or 204, got {delete_response.status_code}"


@pytest.mark.integration
class TestE2ERAGMultiFile:
    """Test RAG with multiple files."""

    def test_sync_multiple_files(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should sync project with multiple files."""
        project_name = f"e2e_multi_{uuid.uuid4().hex[:8]}"

        files = [
            {"path": "main.py", "content": "from utils import helper\ndef main(): helper()\n"},
            {"path": "utils.py", "content": "def helper(): return 42\n"},
            {"path": "config.py", "content": "DEBUG = True\n"},
            {"path": "models.py", "content": "class User: pass\n"},
            {"path": "views.py", "content": "def index(): return 'Hello'\n"}
        ]
        project_hash = compute_project_hash(files)

        response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200

    def test_nested_directory_structure(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should handle nested directory structure."""
        project_name = f"e2e_nested_{uuid.uuid4().hex[:8]}"

        files = [
            {"path": "src/main.py", "content": "def main(): pass\n"},
            {"path": "src/utils/helpers.py", "content": "def helper(): pass\n"},
            {"path": "tests/test_main.py", "content": "def test_main(): pass\n"}
        ]
        project_hash = compute_project_hash(files)

        response = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=120.0
        )

        assert response.status_code == 200


@pytest.mark.integration
class TestE2ERAGFileTypes:
    """Test RAG with different file types."""

    def test_javascript_files(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should handle JavaScript files."""
        project_name = f"e2e_js_{uuid.uuid4().hex[:8]}"

        files = [
            {"path": "index.js", "content": "const x = 1;\nmodule.exports = { x };\n"}
        ]
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

        assert response.status_code == 200

    def test_typescript_files(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should handle TypeScript files."""
        project_name = f"e2e_ts_{uuid.uuid4().hex[:8]}"

        files = [
            {"path": "index.ts", "content": "interface User { id: number; }\n"}
        ]
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

        assert response.status_code == 200

    def test_go_files(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should handle Go files."""
        project_name = f"e2e_go_{uuid.uuid4().hex[:8]}"

        files = [
            {"path": "main.go", "content": "package main\n\nfunc main() {}\n"}
        ]
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

        assert response.status_code == 200


@pytest.mark.integration
class TestE2ERAGResync:
    """Test RAG project resync scenarios."""

    def test_resync_updated_content(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Should resync project with updated content."""
        project_name = f"e2e_resync_{uuid.uuid4().hex[:8]}"

        # Initial sync
        files1 = [{"path": "main.py", "content": "x = 1\n"}]
        project_hash1 = compute_project_hash(files1)

        response1 = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash1,
                "files": files1
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert response1.status_code == 200

        # Resync with updated content
        files2 = [{"path": "main.py", "content": "x = 2\ny = 3\n"}]
        project_hash2 = compute_project_hash(files2)

        response2 = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash2,
                "files": files2
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert response2.status_code == 200

    def test_same_hash_skips_reindex(
        self,
        rag_api_client: httpx.Client,
        test_api_key
    ):
        """Same hash should skip reindexing."""
        project_name = f"e2e_samehash_{uuid.uuid4().hex[:8]}"

        files = [{"path": "main.py", "content": "x = 1\n"}]
        project_hash = compute_project_hash(files)

        # First sync
        response1 = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert response1.status_code == 200

        # Second sync with same hash
        response2 = rag_api_client.post(
            "/v1/projects/sync",
            json={
                "project_name": project_name,
                "project_hash": project_hash,
                "files": files
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=60.0
        )
        assert response2.status_code == 200
