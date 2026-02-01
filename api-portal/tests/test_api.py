"""
Comprehensive tests for API Portal

Tests cover:
- User registration and login
- API key generation, listing, deletion, and validation
- Usage statistics
- Model endpoints (OpenAI-compatible)
- Admin model management
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import app components
from src.database import Base, get_db
from src.main import app
from src.config import settings


# Create test database
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override the dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def client():
    """Create a fresh test client with clean database for each test"""
    # Create tables
    Base.metadata.create_all(bind=engine)

    with TestClient(app) as test_client:
        yield test_client

    # Drop tables after test
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def registered_user(client):
    """Register a user and return credentials"""
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpassword123"
    }
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 200
    data = response.json()
    return {
        "user_data": user_data,
        "token": data["access_token"],
        "user": data["user"]
    }


@pytest.fixture
def auth_headers(registered_user):
    """Get authorization headers for authenticated requests"""
    return {"Authorization": f"Bearer {registered_user['token']}"}


# ============ Health Check Tests ============

class TestHealthCheck:
    def test_health_endpoint(self, client):
        """Test health check endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-portal"


# ============ User Registration Tests ============

class TestUserRegistration:
    def test_register_new_user(self, client):
        """Test registering a new user"""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "securepassword123"
        }
        response = client.post("/api/auth/register", json=user_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == user_data["email"]
        assert data["user"]["username"] == user_data["username"]
        assert data["user"]["is_active"] is True

    def test_first_user_is_admin(self, client):
        """Test that first registered user becomes admin"""
        user_data = {
            "email": "admin@example.com",
            "username": "adminuser",
            "password": "adminpassword123"
        }
        response = client.post("/api/auth/register", json=user_data)
        assert response.status_code == 200

        data = response.json()
        assert data["user"]["is_admin"] is True

    def test_second_user_not_admin(self, client):
        """Test that second registered user is not admin"""
        # Register first user (admin)
        client.post("/api/auth/register", json={
            "email": "first@example.com",
            "username": "firstuser",
            "password": "password123"
        })

        # Register second user
        response = client.post("/api/auth/register", json={
            "email": "second@example.com",
            "username": "seconduser",
            "password": "password123"
        })
        assert response.status_code == 200

        data = response.json()
        assert data["user"]["is_admin"] is False

    def test_register_duplicate_email(self, client):
        """Test that duplicate email registration fails"""
        user_data = {
            "email": "duplicate@example.com",
            "username": "user1",
            "password": "password123"
        }
        client.post("/api/auth/register", json=user_data)

        # Try to register with same email
        response = client.post("/api/auth/register", json={
            "email": "duplicate@example.com",
            "username": "user2",
            "password": "password123"
        })
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_register_duplicate_username(self, client):
        """Test that duplicate username registration fails"""
        user_data = {
            "email": "user1@example.com",
            "username": "sameusername",
            "password": "password123"
        }
        client.post("/api/auth/register", json=user_data)

        # Try to register with same username
        response = client.post("/api/auth/register", json={
            "email": "user2@example.com",
            "username": "sameusername",
            "password": "password123"
        })
        assert response.status_code == 400
        assert "already taken" in response.json()["detail"].lower()

    def test_register_short_password(self, client):
        """Test that short password is rejected"""
        response = client.post("/api/auth/register", json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "short"
        })
        assert response.status_code == 422  # Validation error

    def test_register_invalid_email(self, client):
        """Test that invalid email is rejected"""
        response = client.post("/api/auth/register", json={
            "email": "not-an-email",
            "username": "testuser",
            "password": "password123"
        })
        assert response.status_code == 422  # Validation error


# ============ User Login Tests ============

class TestUserLogin:
    def test_login_with_username(self, client, registered_user):
        """Test login with username"""
        response = client.post("/api/auth/login", json={
            "username": registered_user["user_data"]["username"],
            "password": registered_user["user_data"]["password"]
        })
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["user"]["username"] == registered_user["user_data"]["username"]

    def test_login_with_email(self, client, registered_user):
        """Test login with email"""
        response = client.post("/api/auth/login", json={
            "username": registered_user["user_data"]["email"],
            "password": registered_user["user_data"]["password"]
        })
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data

    def test_login_wrong_password(self, client, registered_user):
        """Test login with wrong password fails"""
        response = client.post("/api/auth/login", json={
            "username": registered_user["user_data"]["username"],
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user fails"""
        response = client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "password123"
        })
        assert response.status_code == 401


# ============ Get Current User Tests ============

class TestGetCurrentUser:
    def test_get_me_authenticated(self, client, auth_headers, registered_user):
        """Test getting current user info when authenticated"""
        response = client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["email"] == registered_user["user_data"]["email"]
        assert data["username"] == registered_user["user_data"]["username"]

    def test_get_me_unauthenticated(self, client):
        """Test getting current user info fails without auth"""
        response = client.get("/api/auth/me")
        # 401 Unauthorized is correct per HTTP spec when no auth is provided
        assert response.status_code == 401

    def test_get_me_invalid_token(self, client):
        """Test getting current user with invalid token fails"""
        response = client.get("/api/auth/me", headers={
            "Authorization": "Bearer invalid_token"
        })
        assert response.status_code == 401


# ============ API Key Creation Tests ============

class TestAPIKeyCreation:
    def test_create_api_key(self, client, auth_headers):
        """Test creating an API key"""
        response = client.post("/api/keys", json={
            "name": "Test Key"
        }, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Test Key"
        assert "key" in data  # Full key only returned on creation
        assert data["key"].startswith(settings.api_key_prefix)
        assert data["rate_limit"] == 100  # Default rate limit

    def test_create_api_key_with_rate_limit(self, client, auth_headers):
        """Test creating an API key with custom rate limit"""
        response = client.post("/api/keys", json={
            "name": "High Rate Key",
            "rate_limit": 500
        }, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["rate_limit"] == 500

    def test_create_api_key_with_expiration(self, client, auth_headers):
        """Test creating an API key with expiration"""
        response = client.post("/api/keys", json={
            "name": "Expiring Key",
            "expires_days": 30
        }, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["expires_at"] is not None

    def test_create_api_key_unauthenticated(self, client):
        """Test creating API key fails without auth"""
        response = client.post("/api/keys", json={
            "name": "Test Key"
        })
        # 401 Unauthorized is correct per HTTP spec when no auth is provided
        assert response.status_code == 401


# ============ API Key Listing Tests ============

class TestAPIKeyListing:
    def test_list_api_keys_empty(self, client, auth_headers):
        """Test listing API keys when none exist"""
        response = client.get("/api/keys", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["keys"] == []
        assert data["total"] == 0

    def test_list_api_keys(self, client, auth_headers):
        """Test listing API keys after creating some"""
        # Create two keys
        client.post("/api/keys", json={"name": "Key 1"}, headers=auth_headers)
        client.post("/api/keys", json={"name": "Key 2"}, headers=auth_headers)

        response = client.get("/api/keys", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 2
        assert len(data["keys"]) == 2
        # Keys should not include full key in listing
        for key in data["keys"]:
            assert "key" not in key or key.get("key") is None
            assert "key_prefix" in key


# ============ API Key Deletion Tests ============

class TestAPIKeyDeletion:
    def test_delete_api_key(self, client, auth_headers):
        """Test deleting an API key"""
        # Create a key
        create_response = client.post("/api/keys", json={
            "name": "To Delete"
        }, headers=auth_headers)
        key_id = create_response.json()["id"]

        # Delete it
        response = client.delete(f"/api/keys/{key_id}", headers=auth_headers)
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()

        # Verify it's gone
        list_response = client.get("/api/keys", headers=auth_headers)
        assert list_response.json()["total"] == 0

    def test_delete_nonexistent_key(self, client, auth_headers):
        """Test deleting non-existent key fails"""
        response = client.delete("/api/keys/99999", headers=auth_headers)
        assert response.status_code == 404


# ============ API Key Toggle Tests ============

class TestAPIKeyToggle:
    def test_toggle_api_key(self, client, auth_headers):
        """Test toggling API key active status"""
        # Create a key
        create_response = client.post("/api/keys", json={
            "name": "Toggle Test"
        }, headers=auth_headers)
        key_id = create_response.json()["id"]

        # Toggle it off
        response = client.patch(f"/api/keys/{key_id}/toggle", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["is_active"] is False

        # Toggle it back on
        response = client.patch(f"/api/keys/{key_id}/toggle", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["is_active"] is True


# ============ API Key Validation Tests ============

class TestAPIKeyValidation:
    def test_validate_valid_key(self, client, auth_headers, registered_user):
        """Test validating a valid API key"""
        # Create a key
        create_response = client.post("/api/keys", json={
            "name": "Valid Key"
        }, headers=auth_headers)
        raw_key = create_response.json()["key"]

        # Validate it
        response = client.post("/api/validate-key", json={
            "api_key": raw_key
        })
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert data["user"] == registered_user["user_data"]["username"]

    def test_validate_invalid_key(self, client):
        """Test validating an invalid API key"""
        response = client.post("/api/validate-key", json={
            "api_key": "sk-llm-invalid123456"
        })
        assert response.status_code == 401
        assert response.json()["valid"] is False

    def test_validate_no_key(self, client):
        """Test validation with no key provided"""
        response = client.post("/api/validate-key", json={})
        assert response.status_code == 400

    def test_validate_disabled_key(self, client, auth_headers):
        """Test validating a disabled API key"""
        # Create and disable a key
        create_response = client.post("/api/keys", json={
            "name": "Disabled Key"
        }, headers=auth_headers)
        key_id = create_response.json()["id"]
        raw_key = create_response.json()["key"]

        # Disable it
        client.patch(f"/api/keys/{key_id}/toggle", headers=auth_headers)

        # Try to validate
        response = client.post("/api/validate-key", json={
            "api_key": raw_key
        })
        assert response.status_code == 401
        assert response.json()["valid"] is False


# ============ Usage Stats Tests ============

class TestUsageStats:
    def test_get_usage_stats(self, client, auth_headers):
        """Test getting usage statistics"""
        response = client.get("/api/usage", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data
        assert "total_tokens_input" in data
        assert "total_tokens_output" in data
        assert "requests_today" in data
        assert "tokens_today" in data


# ============ Model Endpoint Tests ============

class TestModelEndpoints:
    def test_list_models_empty(self, client):
        """Test listing models when none exist"""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

    def test_get_nonexistent_model(self, client):
        """Test getting a non-existent model"""
        response = client.get("/v1/models/nonexistent-model")
        assert response.status_code == 404


# ============ Admin Model Management Tests ============

class TestAdminModelManagement:
    def test_admin_list_models(self, client, auth_headers):
        """Test admin listing models"""
        response = client.get("/api/admin/models", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total" in data

    def test_admin_create_model(self, client, auth_headers):
        """Test admin creating a model"""
        response = client.post("/api/admin/models", json={
            "model_id": "test-model",
            "name": "Test Model",
            "context_length": 4096,
            "max_output": 2048
        }, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "test-model"
        assert data["name"] == "Test Model"
        assert data["is_auto_discovered"] is False

    def test_admin_update_model(self, client, auth_headers):
        """Test admin updating a model"""
        # Create first
        client.post("/api/admin/models", json={
            "model_id": "update-test",
            "name": "Original Name",
            "context_length": 4096,
            "max_output": 2048
        }, headers=auth_headers)

        # Update
        response = client.patch("/api/admin/models/update-test", json={
            "model_id": "update-test",
            "name": "Updated Name",
            "context_length": 8192,
            "max_output": 4096
        }, headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["context_length"] == 8192

    def test_admin_delete_model(self, client, auth_headers):
        """Test admin deleting a model"""
        # Create first
        client.post("/api/admin/models", json={
            "model_id": "delete-test",
            "name": "To Delete",
            "context_length": 4096,
            "max_output": 2048
        }, headers=auth_headers)

        # Delete
        response = client.delete("/api/admin/models/delete-test", headers=auth_headers)
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()

    def test_non_admin_cannot_access_admin_endpoints(self, client):
        """Test non-admin users cannot access admin endpoints"""
        # Register first user (admin)
        client.post("/api/auth/register", json={
            "email": "admin@example.com",
            "username": "adminuser",
            "password": "password123"
        })

        # Register second user (non-admin)
        response = client.post("/api/auth/register", json={
            "email": "user@example.com",
            "username": "normaluser",
            "password": "password123"
        })
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Try to access admin endpoint
        response = client.get("/api/admin/models", headers=headers)
        assert response.status_code == 403


# ============ Web UI Route Tests ============

class TestWebUIRoutes:
    def test_home_page(self, client):
        """Test home page returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_login_page(self, client):
        """Test login page returns HTML"""
        response = client.get("/login")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_register_page(self, client):
        """Test register page returns HTML"""
        response = client.get("/register")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_dashboard_page(self, client):
        """Test dashboard page returns HTML"""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
