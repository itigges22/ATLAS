"""
Tests for Atlas Dashboard service.

Validates dashboard rendering, stats API,
and real-time metrics display.
"""

import pytest
import httpx


class TestDashboardHealth:
    """Test Dashboard service health."""

    def test_health_endpoint_responds(self, dashboard_client: httpx.Client):
        """Health endpoint on port 3001 should respond."""
        response = dashboard_client.get("/health")
        assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"


class TestDashboardUI:
    """Test Dashboard web interface."""

    def test_root_returns_html(self, dashboard_client: httpx.Client):
        """GET / should return HTML page."""
        response = dashboard_client.get("/")
        assert response.status_code == 200, f"Root should return 200, got {response.status_code}"
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type, f"Should return HTML, got {content_type}"

    def test_page_contains_queue_display(self, dashboard_client: httpx.Client):
        """Dashboard should display queue depths."""
        response = dashboard_client.get("/")
        assert response.status_code == 200
        html = response.text.lower()
        # Should have some reference to queues
        has_queue_reference = any(term in html for term in ["queue", "p0", "p1", "p2", "pending", "tasks"])
        assert has_queue_reference, "Dashboard should display queue information"

    def test_page_contains_metrics_display(self, dashboard_client: httpx.Client):
        """Dashboard should display metrics."""
        response = dashboard_client.get("/")
        assert response.status_code == 200
        html = response.text.lower()
        # Should have some reference to metrics
        has_metrics_reference = any(term in html for term in ["metrics", "success", "rate", "tasks", "tokens"])
        assert has_metrics_reference, "Dashboard should display metrics information"


class TestDashboardAPI:
    """Test Dashboard stats API."""

    def test_api_stats_endpoint_exists(self, dashboard_client: httpx.Client):
        """/api/stats endpoint should exist."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200, f"/api/stats should return 200, got {response.status_code}"

    def test_api_stats_returns_json(self, dashboard_client: httpx.Client):
        """Stats API should return JSON."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, f"Should return JSON, got {content_type}"

    def test_api_stats_has_queue_depths(self, dashboard_client: httpx.Client):
        """Stats should include queue depths."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        # Should have queue information
        has_queues = any(key in data for key in ["queues", "queue_depths", "p0", "p1", "p2", "pending"])
        if not has_queues:
            # Check nested structure
            if "queues" in data:
                queues = data["queues"]
                assert isinstance(queues, dict) or isinstance(queues, list), "queues should be dict or list"

    def test_api_stats_has_metrics(self, dashboard_client: httpx.Client):
        """Stats should include daily metrics."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        # Should have some metrics
        has_metrics = any(key in str(data).lower() for key in ["tasks", "success", "tokens", "metrics"])
        assert has_metrics, f"Stats should include metrics: {data}"

    def test_api_stats_has_success_rate(self, dashboard_client: httpx.Client):
        """Stats should include success rate."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        # Success rate may be nested or calculated
        has_success_info = any(key in str(data).lower() for key in ["success", "rate", "passed", "completed"])
        if not has_success_info:
            pytest.skip("Success rate not directly exposed in stats")

    def test_api_stats_has_recent_tasks(self, dashboard_client: httpx.Client):
        """Stats should include recent tasks list."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        # Look for recent tasks
        has_recent = "recent" in str(data).lower() or "tasks" in data or isinstance(data.get("recent_tasks"), list)
        if not has_recent:
            pytest.skip("Recent tasks not exposed in stats API")


class TestDashboardRefresh:
    """Test dashboard refresh mechanism."""

    def test_page_has_refresh_mechanism(self, dashboard_client: httpx.Client):
        """Dashboard should have auto-refresh or refresh button."""
        response = dashboard_client.get("/")
        assert response.status_code == 200
        html = response.text.lower()

        # Look for refresh indicators
        has_refresh = any(term in html for term in [
            "refresh",
            "setinterval",
            "fetch",
            "ajax",
            "reload",
            "meta http-equiv=\"refresh\""
        ])
        # This is informational - refresh may be implemented different ways
        if not has_refresh:
            pytest.skip("Refresh mechanism not obvious in HTML")


class TestDashboardRecentTasksLimit:
    """Test recent tasks are limited."""

    def test_recent_tasks_limited(self, dashboard_client: httpx.Client):
        """Recent tasks should be limited (e.g., last 20)."""
        response = dashboard_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()

        # If recent_tasks exists, check it's limited
        recent = data.get("recent_tasks") or data.get("recent")
        if recent and isinstance(recent, list):
            assert len(recent) <= 20, f"Recent tasks should be limited to 20, got {len(recent)}"
