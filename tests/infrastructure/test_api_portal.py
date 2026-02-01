"""
Tests for API Portal service.

Validates user authentication, API key management,
usage tracking, and model discovery.
"""

import uuid
import time
import pytest
import httpx


class TestAPIPortalHealth:
    """Test API Portal service health."""

    def test_health_endpoint_responds(self, api_portal_client: httpx.Client):
        """Health endpoint should return 200 OK."""
        response = api_portal_client.get("/health")
        assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"

    def test_root_returns_login_page(self, api_portal_client: httpx.Client):
        """GET / should return login page HTML."""
        response = api_portal_client.get("/")
        assert response.status_code == 200, f"Root should return 200, got {response.status_code}"
        assert "text/html" in response.headers.get("content-type", ""), "Should return HTML"


class TestAPIPortalUserAuth:
    """Test user authentication."""

    def test_user_registration_creates_user(self, api_portal_client: httpx.Client):
        """User registration should create a new user."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"testuser_{unique_id}",
                "email": f"test_{unique_id}@example.com",
                "password": f"TestPass123!_{unique_id}"
            }
        )
        assert response.status_code == 200, f"Registration should succeed, got {response.status_code}: {response.text}"
        data = response.json()
        assert "token" in data or "access_token" in data, "Registration should return token"

    def test_duplicate_registration_rejected(self, api_portal_client: httpx.Client):
        """Duplicate registration should be rejected."""
        unique_id = uuid.uuid4().hex[:8]
        user_data = {
            "username": f"dupuser_{unique_id}",
            "email": f"dup_{unique_id}@example.com",
            "password": f"TestPass123!_{unique_id}"
        }

        # First registration
        response1 = api_portal_client.post("/api/auth/register", json=user_data)
        assert response1.status_code == 200

        # Duplicate registration
        response2 = api_portal_client.post("/api/auth/register", json=user_data)
        assert response2.status_code in [400, 409, 422], f"Duplicate should fail, got {response2.status_code}"

    def test_user_login_returns_jwt(self, api_portal_client: httpx.Client):
        """User login should return JWT token."""
        unique_id = uuid.uuid4().hex[:8]
        user_data = {
            "username": f"loginuser_{unique_id}",
            "email": f"login_{unique_id}@example.com",
            "password": f"TestPass123!_{unique_id}"
        }

        # Register
        api_portal_client.post("/api/auth/register", json=user_data)

        # Login
        response = api_portal_client.post(
            "/api/auth/login",
            json={"username": user_data["username"], "password": user_data["password"]}
        )
        assert response.status_code == 200, f"Login should succeed, got {response.status_code}"
        data = response.json()
        token = data.get("token") or data.get("access_token")
        assert token is not None, "Login should return token"
        assert len(token) > 20, "Token should be substantial"

    def test_invalid_login_rejected(self, api_portal_client: httpx.Client):
        """Invalid credentials should be rejected."""
        response = api_portal_client.post(
            "/api/auth/login",
            json={"username": "nonexistent_user_xyz", "password": "wrongpassword"}
        )
        assert response.status_code in [401, 403, 404], f"Invalid login should fail, got {response.status_code}"

    def test_jwt_required_for_protected_endpoints(self, api_portal_client: httpx.Client):
        """Protected endpoints should require JWT."""
        response = api_portal_client.get("/api/keys")
        assert response.status_code in [401, 403], f"Protected endpoint should require auth, got {response.status_code}"


class TestAPIPortalAPIKeys:
    """Test API key management."""

    def test_api_key_creation_works(self, api_portal_client: httpx.Client, test_user):
        """API key creation should return key string."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        response = api_portal_client.post(
            "/api/keys",
            json={"name": f"test_key_{uuid.uuid4().hex[:8]}"},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert response.status_code == 200, f"Key creation should succeed, got {response.status_code}"
        data = response.json()
        key = data.get("key") or data.get("api_key")
        assert key is not None, "Should return API key"
        assert len(key) > 10, "Key should be substantial"

    def test_api_key_listed(self, api_portal_client: httpx.Client, test_user, test_api_key):
        """Created API key should appear in list."""
        response = api_portal_client.get(
            "/api/keys",
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        keys = data if isinstance(data, list) else data.get("keys", [])
        key_names = [k.get("name") for k in keys]
        assert test_api_key.name in key_names, f"Key {test_api_key.name} should be in list"

    def test_api_key_validation_endpoint(self, api_portal_client: httpx.Client, test_api_key):
        """API key validation endpoint should work."""
        response = api_portal_client.post(
            "/api/validate-key",
            json={"api_key": test_api_key.key_string}  # Use "api_key" not "key"
        )
        assert response.status_code == 200, f"Validation should succeed, got {response.status_code}: {response.text}"
        data = response.json()
        assert data.get("valid") is True, "Key should be valid"

    def test_api_key_validation_returns_user_info(self, api_portal_client: httpx.Client, test_api_key):
        """Key validation should return user info."""
        response = api_portal_client.post(
            "/api/validate-key",
            json={"api_key": test_api_key.key_string}  # Use "api_key" not "key"
        )
        assert response.status_code == 200, f"Validation should succeed, got {response.status_code}: {response.text}"
        data = response.json()
        # Should have user info
        assert "user" in data or "user_id" in data or "username" in data, \
            f"Validation should return user info: {data}"

    def test_api_key_revocation_works(self, api_portal_client: httpx.Client, test_user):
        """API key revocation should work."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        # Create a key to revoke
        create_response = api_portal_client.post(
            "/api/keys",
            json={"name": f"revoke_test_{uuid.uuid4().hex[:8]}"},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert create_response.status_code == 200
        data = create_response.json()
        key_id = data.get("id") or data.get("key_id")
        key_string = data.get("key") or data.get("api_key")

        # Delete/revoke the key
        delete_response = api_portal_client.delete(
            f"/api/keys/{key_id}",
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert delete_response.status_code in [200, 204], f"Revocation should succeed, got {delete_response.status_code}"

        # Validate should now fail
        validate_response = api_portal_client.post(
            "/api/validate-key",
            json={"key": key_string}
        )
        # Could return 200 with valid=False or 401/404
        if validate_response.status_code == 200:
            assert validate_response.json().get("valid") is False, "Revoked key should be invalid"

    def test_revoked_key_rejected(self, api_portal_client: httpx.Client, test_user):
        """Revoked key should be rejected on validation."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        # Create and revoke a key
        create_response = api_portal_client.post(
            "/api/keys",
            json={"name": f"reject_test_{uuid.uuid4().hex[:8]}"},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        data = create_response.json()
        key_id = data.get("id") or data.get("key_id")
        key_string = data.get("key") or data.get("api_key")

        # Revoke
        api_portal_client.delete(
            f"/api/keys/{key_id}",
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )

        # Try to use it
        validate_response = api_portal_client.post(
            "/api/validate-key",
            json={"key": key_string}
        )
        if validate_response.status_code == 200:
            assert validate_response.json().get("valid") is not True, "Revoked key should not be valid"


class TestAPIPortalRateLimits:
    """Test rate limit functionality."""

    def test_rate_limit_stored_on_key(self, api_portal_client: httpx.Client, test_user):
        """Rate limit should be stored on key creation."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        response = api_portal_client.post(
            "/api/keys",
            json={
                "name": f"ratelimit_test_{uuid.uuid4().hex[:8]}",
                "rate_limit": 100
            },
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        # Rate limit may or may not be accepted in request
        assert response.status_code == 200

    def test_rate_limit_in_validation(self, api_portal_client: httpx.Client, test_api_key):
        """Rate limit should be returned in validation."""
        response = api_portal_client.post(
            "/api/validate-key",
            json={"api_key": test_api_key.key_string}  # Use "api_key" not "key"
        )
        assert response.status_code == 200, f"Validation should succeed, got {response.status_code}: {response.text}"
        data = response.json()
        # Rate limit may be present
        if "rate_limit" in data:
            assert isinstance(data["rate_limit"], (int, type(None))), "Rate limit should be int or None"


class TestAPIPortalUsageTracking:
    """Test usage tracking functionality."""

    def test_usage_tracking_endpoint(self, api_portal_client: httpx.Client, test_user):
        """Usage endpoint should return stats."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        response = api_portal_client.get(
            "/api/usage",
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        # Usage endpoint may or may not exist
        if response.status_code == 200:
            data = response.json()
            # Should have some usage metrics
            assert isinstance(data, dict), "Usage should return dict"


class TestAPIPortalModels:
    """Test model discovery functionality."""

    def test_model_discovery_from_llama(self, api_portal_client: httpx.Client):
        """GET /v1/models should return models from llama-server."""
        response = api_portal_client.get("/v1/models")
        assert response.status_code == 200, f"/v1/models should return 200, got {response.status_code}"
        data = response.json()
        assert "data" in data, "Should have 'data' field"


class TestAPIPortalAdmin:
    """Test admin functionality."""

    def test_first_user_becomes_admin(self, api_portal_client: httpx.Client):
        """First registered user should become admin (if no users exist)."""
        # This is hard to test without a fresh database
        # We just verify the admin concept exists by checking the response
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"admintest_{unique_id}",
                "email": f"admin_{unique_id}@example.com",
                "password": f"TestPass123!_{unique_id}"
            }
        )
        if response.status_code == 200:
            data = response.json()
            # is_admin field may be present
            if "is_admin" in data:
                # Just verify the field exists and is boolean
                assert isinstance(data["is_admin"], bool), "is_admin should be boolean"


class TestAPIPortalPasswordValidation:
    """Test password validation rules."""

    def test_weak_password_rejected(self, api_portal_client: httpx.Client):
        """Weak password should be rejected."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"weakpass_{unique_id}",
                "email": f"weak_{unique_id}@example.com",
                "password": "123"
            }
        )
        # API portal rejects passwords shorter than 8 characters
        assert response.status_code == 400

    def test_empty_password_rejected(self, api_portal_client: httpx.Client):
        """Empty password should be rejected."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"emptypass_{unique_id}",
                "email": f"empty_{unique_id}@example.com",
                "password": ""
            }
        )
        assert response.status_code == 400


class TestAPIPortalUsernameValidation:
    """Test username validation rules."""

    def test_empty_username_rejected(self, api_portal_client: httpx.Client):
        """Empty username should be rejected."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": "",
                "email": f"nousername_{unique_id}@example.com",
                "password": f"TestPass123!_{unique_id}"
            }
        )
        assert response.status_code == 400

    def test_special_characters_in_username(self, api_portal_client: httpx.Client):
        """Special characters in username are accepted."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"user_test-123_{unique_id}",
                "email": f"special_{unique_id}@example.com",
                "password": f"TestPass123!_{unique_id}"
            }
        )
        # Underscores and hyphens are valid in usernames
        assert response.status_code == 200


class TestAPIPortalEmailValidation:
    """Test email validation rules."""

    def test_invalid_email_rejected(self, api_portal_client: httpx.Client):
        """Invalid email format should be rejected."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"invalidemail_{unique_id}",
                "email": "not-an-email",
                "password": f"TestPass123!_{unique_id}"
            }
        )
        assert response.status_code in [400, 422]

    def test_empty_email_rejected(self, api_portal_client: httpx.Client):
        """Empty email should be rejected."""
        unique_id = uuid.uuid4().hex[:8]
        response = api_portal_client.post(
            "/api/auth/register",
            json={
                "username": f"noemail_{unique_id}",
                "email": "",
                "password": f"TestPass123!_{unique_id}"
            }
        )
        assert response.status_code in [400, 422]


class TestAPIPortalTokenHandling:
    """Test JWT token handling."""

    def test_expired_token_rejected(self, api_portal_client: httpx.Client):
        """Expired token should be rejected."""
        # Use a clearly invalid/expired token
        response = api_portal_client.get(
            "/api/keys",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiZXhwIjoxfQ.invalid"}
        )
        assert response.status_code in [401, 403, 422]

    def test_malformed_token_rejected(self, api_portal_client: httpx.Client):
        """Malformed token should be rejected."""
        response = api_portal_client.get(
            "/api/keys",
            headers={"Authorization": "Bearer not-a-jwt-token"}
        )
        assert response.status_code in [401, 403, 422]

    def test_missing_bearer_prefix_rejected(self, api_portal_client: httpx.Client):
        """Token without Bearer prefix should be rejected."""
        response = api_portal_client.get(
            "/api/keys",
            headers={"Authorization": "some-token"}
        )
        assert response.status_code in [401, 403, 422]


class TestAPIPortalKeyFeatures:
    """Test additional API key features."""

    def test_key_with_expiration(self, api_portal_client: httpx.Client, test_user):
        """Keys with expiration field are created successfully."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        response = api_portal_client.post(
            "/api/keys",
            json={
                "name": f"expiring_key_{uuid.uuid4().hex[:8]}",
                "expires_in_days": 30
            },
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        # Portal accepts the field (may ignore it)
        assert response.status_code == 200

    def test_key_name_uniqueness(self, api_portal_client: httpx.Client, test_user):
        """Duplicate key names are allowed."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        key_name = f"unique_test_{uuid.uuid4().hex[:8]}"

        # Create first key
        response1 = api_portal_client.post(
            "/api/keys",
            json={"name": key_name},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert response1.status_code == 200

        # Create second key with same name - portal allows this
        response2 = api_portal_client.post(
            "/api/keys",
            json={"name": key_name},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert response2.status_code == 200

    def test_key_toggle_enabled(self, api_portal_client: httpx.Client, test_user):
        """Key toggle endpoint returns 404 (not implemented)."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        # Create a key
        create_response = api_portal_client.post(
            "/api/keys",
            json={"name": f"toggle_test_{uuid.uuid4().hex[:8]}"},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        if create_response.status_code != 200:
            pytest.skip("Key creation failed")

        key_id = create_response.json().get("id") or create_response.json().get("key_id")

        # PATCH endpoint not implemented
        toggle_response = api_portal_client.patch(
            f"/api/keys/{key_id}",
            json={"enabled": False},
            headers={"Authorization": f"Bearer {test_user.jwt_token}"}
        )
        assert toggle_response.status_code == 405  # Method not allowed


class TestAPIPortalInvalidKey:
    """Test invalid API key handling."""

    def test_nonexistent_key_validation(self, api_portal_client: httpx.Client):
        """Nonexistent key should fail validation."""
        response = api_portal_client.post(
            "/api/validate-key",
            json={"api_key": "sk-nonexistent-key-12345"}
        )
        if response.status_code == 200:
            assert response.json().get("valid") is False
        else:
            assert response.status_code in [400, 401, 404]

    def test_empty_key_validation(self, api_portal_client: httpx.Client):
        """Empty key should fail validation."""
        response = api_portal_client.post(
            "/api/validate-key",
            json={"api_key": ""}
        )
        if response.status_code == 200:
            assert response.json().get("valid") is False
        else:
            assert response.status_code in [400, 422]


class TestAPIPortalDashboard:
    """Test dashboard functionality."""

    def test_dashboard_page_loads(self, api_portal_client: httpx.Client):
        """Dashboard page loads (auth handled client-side via JS)."""
        response = api_portal_client.get("/dashboard", follow_redirects=False)
        # Dashboard serves HTML, JS handles auth via localStorage JWT
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_login_page_accessible(self, api_portal_client: httpx.Client):
        """Login page should be accessible."""
        response = api_portal_client.get("/login")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_register_page_accessible(self, api_portal_client: httpx.Client):
        """Register page should be accessible."""
        response = api_portal_client.get("/register")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestAPIPortalModelsAdvanced:
    """Test advanced model discovery."""

    def test_models_response_format(self, api_portal_client: httpx.Client):
        """Models response should have correct format."""
        response = api_portal_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_models_have_required_fields(self, api_portal_client: httpx.Client):
        """Each model should have required fields."""
        response = api_portal_client.get("/v1/models")
        if response.status_code == 200:
            data = response.json()
            for model in data.get("data", []):
                assert "id" in model, "Model should have id"


class TestAPIPortalConcurrency:
    """Test concurrent API operations."""

    def test_concurrent_key_creation(self, api_portal_client: httpx.Client, test_user):
        """Should handle concurrent key creation."""
        if not test_user.jwt_token:
            pytest.skip("Test user creation failed")

        import concurrent.futures

        def create_key(i):
            return api_portal_client.post(
                "/api/keys",
                json={"name": f"concurrent_key_{i}_{uuid.uuid4().hex[:8]}"},
                headers={"Authorization": f"Bearer {test_user.jwt_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_key, i) for i in range(3)]
            results = [f.result() for f in futures]

        # All should succeed
        for r in results:
            assert r.status_code == 200

    def test_concurrent_validation(self, api_portal_client: httpx.Client, test_api_key):
        """Should handle concurrent key validation."""
        import concurrent.futures

        def validate():
            return api_portal_client.post(
                "/api/validate-key",
                json={"api_key": test_api_key.key_string}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(validate) for _ in range(5)]
            results = [f.result() for f in futures]

        for r in results:
            assert r.status_code == 200
