"""
Pytest configuration and fixtures for ATLAS infrastructure tests.

This module provides shared fixtures for testing all ATLAS services including:
- Redis client
- HTTP clients for each service
- Test user and API key management
- Test project creation
- Cleanup utilities
"""

import os
import sys
import json
import uuid
import time
import tempfile
import shutil
from typing import Generator, Optional
from dataclasses import dataclass

import pytest
import redis
import httpx

# Service endpoints - using cluster IPs when running inside cluster,
# or localhost with NodePort when running externally
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

# Determine if running inside cluster or externally
IN_CLUSTER = os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")

if IN_CLUSTER:
    API_PORTAL_URL = os.environ.get("API_PORTAL_URL", "http://api-portal:3000")
    RAG_API_URL = os.environ.get("RAG_API_URL", "http://rag-api:8001")
    LLAMA_URL = os.environ.get("LLAMA_URL", "http://llama-service:8000")
    LLM_PROXY_URL = os.environ.get("LLM_PROXY_URL", "http://llm-proxy:8000")
    SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://sandbox:8020")
    DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://atlas-dashboard:3001")
else:
    # Running locally - use NodePort or port-forward
    API_PORTAL_URL = os.environ.get("API_PORTAL_URL", "http://localhost:30000")
    RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:31144")
    LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:32735")
    LLM_PROXY_URL = os.environ.get("LLM_PROXY_URL", "http://localhost:30080")
    SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://localhost:30820")
    DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:30001")

# Default timeouts
DEFAULT_TIMEOUT = 30.0
LLM_TIMEOUT = 120.0


@dataclass
class TestUser:
    """Test user credentials and tokens."""
    username: str
    email: str
    password: str
    jwt_token: Optional[str] = None
    is_admin: bool = False


@dataclass
class TestAPIKey:
    """Test API key data."""
    key_id: str
    key_string: str
    name: str


@pytest.fixture(scope="session")
def redis_client() -> Generator[redis.Redis, None, None]:
    """
    Create a Redis client for testing.

    Uses port-forwarding or direct connection depending on environment.
    """
    # Try to connect via kubectl port-forward first, then cluster service
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        client.ping()
        yield client
    except redis.ConnectionError:
        # Try connecting via kubectl port-forward
        import subprocess
        # Start port-forward in background
        proc = subprocess.Popen(
            ["kubectl", "port-forward", "svc/redis", "6379:6379"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2)
        try:
            client = redis.Redis(host="localhost", port=6379, decode_responses=True)
            client.ping()
            yield client
        finally:
            proc.terminate()
            proc.wait()


@pytest.fixture(scope="session")
def api_portal_client() -> Generator[httpx.Client, None, None]:
    """HTTP client for API Portal service."""
    with httpx.Client(base_url=API_PORTAL_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="session")
def rag_api_client() -> Generator[httpx.Client, None, None]:
    """HTTP client for RAG API service."""
    with httpx.Client(base_url=RAG_API_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="session")
def llama_client() -> Generator[httpx.Client, None, None]:
    """HTTP client for llama-server (direct, no proxy)."""
    with httpx.Client(base_url=LLAMA_URL, timeout=LLM_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="session")
def llm_proxy_client() -> Generator[httpx.Client, None, None]:
    """HTTP client for LLM Proxy service."""
    with httpx.Client(base_url=LLM_PROXY_URL, timeout=LLM_TIMEOUT) as client:
        yield client



@pytest.fixture(scope="session")
def sandbox_client() -> Generator[httpx.Client, None, None]:
    """HTTP client for Sandbox executor service."""
    with httpx.Client(base_url=SANDBOX_URL, timeout=120.0) as client:
        yield client


@pytest.fixture(scope="session")
def dashboard_client() -> Generator[httpx.Client, None, None]:
    """HTTP client for Atlas Dashboard."""
    with httpx.Client(base_url=DASHBOARD_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="function")
def test_user(api_portal_client: httpx.Client) -> Generator[TestUser, None, None]:
    """
    Create a test user and clean up after test.

    Registers a new user with unique credentials and provides JWT token.
    """
    unique_id = str(uuid.uuid4())[:8]
    user = TestUser(
        username=f"testuser_{unique_id}",
        email=f"testuser_{unique_id}@example.com",  # Use .com domain for validation
        password=f"TestPass123_{unique_id}"  # Avoid special chars that may cause issues
    )

    # Register user
    response = api_portal_client.post(
        "/api/auth/register",
        json={
            "username": user.username,
            "email": user.email,
            "password": user.password
        }
    )

    if response.status_code == 200:
        data = response.json()
        user.jwt_token = data.get("token") or data.get("access_token")
        user.is_admin = data.get("is_admin", False)

    yield user

    # Cleanup: We can't easily delete users, but test data will be minimal


@pytest.fixture(scope="function")
def test_api_key(api_portal_client: httpx.Client, test_user: TestUser) -> Generator[TestAPIKey, None, None]:
    """
    Create a test API key for the test user.

    Requires a valid test user with JWT token.
    """
    if not test_user.jwt_token:
        pytest.skip("Test user creation failed, cannot create API key")

    unique_id = str(uuid.uuid4())[:8]
    key_name = f"test_key_{unique_id}"

    response = api_portal_client.post(
        "/api/keys",
        json={"name": key_name},
        headers={"Authorization": f"Bearer {test_user.jwt_token}"}
    )

    if response.status_code != 200:
        pytest.fail(f"Failed to create API key: {response.status_code} - {response.text}")

    data = response.json()
    api_key = TestAPIKey(
        key_id=data.get("id") or data.get("key_id"),
        key_string=data.get("key") or data.get("api_key"),
        name=key_name
    )

    yield api_key

    # Cleanup: Delete the API key
    if api_key.key_id:
        api_portal_client.delete(
            f"/api/keys/{api_key.key_id}",
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )


@pytest.fixture(scope="function")
def test_project_dir() -> Generator[str, None, None]:
    """
    Create a temporary project directory with sample code files.

    Creates Python, JavaScript, Go files for testing multi-language indexing.
    """
    project_dir = tempfile.mkdtemp(prefix="atlas_test_project_")

    # Create sample Python file
    with open(os.path.join(project_dir, "main.py"), "w") as f:
        f.write('''"""Main module for test project."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.history = []

    def calculate(self, op: str, a: int, b: int) -> int:
        """Perform calculation."""
        if op == "add":
            result = add(a, b)
        elif op == "subtract":
            result = subtract(a, b)
        else:
            raise ValueError(f"Unknown operation: {op}")
        self.history.append((op, a, b, result))
        return result

if __name__ == "__main__":
    calc = Calculator()
    print(calc.calculate("add", 1, 2))
''')

    # Create sample JavaScript file
    with open(os.path.join(project_dir, "utils.js"), "w") as f:
        f.write('''// Utility functions

function multiply(a, b) {
    return a * b;
}

function divide(a, b) {
    if (b === 0) {
        throw new Error("Division by zero");
    }
    return a / b;
}

module.exports = { multiply, divide };
''')

    # Create sample Go file
    with open(os.path.join(project_dir, "main.go"), "w") as f:
        f.write('''package main

import "fmt"

// Add returns the sum of two integers
func Add(a, b int) int {
    return a + b
}

// Multiply returns the product of two integers
func Multiply(a, b int) int {
    return a * b
}

func main() {
    fmt.Println(Add(1, 2))
    fmt.Println(Multiply(3, 4))
}
''')

    # Create sample test file
    with open(os.path.join(project_dir, "test_main.py"), "w") as f:
        f.write('''"""Tests for main module."""
import pytest
from main import add, subtract, Calculator

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0

def test_calculator():
    calc = Calculator()
    assert calc.calculate("add", 2, 3) == 5
    assert len(calc.history) == 1
''')

    yield project_dir

    # Cleanup
    shutil.rmtree(project_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def cleanup_redis_keys(redis_client: redis.Redis) -> Generator[list, None, None]:
    """
    Fixture to track and clean up Redis keys created during tests.

    Yields a list that tests can append keys to for automatic cleanup.
    """
    keys_to_cleanup = []

    yield keys_to_cleanup

    # Cleanup all tracked keys
    for key in keys_to_cleanup:
        try:
            redis_client.delete(key)
        except Exception:
            pass


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        # Add integration marker to tests in integration folder
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
