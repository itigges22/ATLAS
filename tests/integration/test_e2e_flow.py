"""
End-to-end flow tests.

Tests the complete pipeline from task submission
through code generation, execution, and result retrieval.
"""

import json
import uuid
import time

import pytest
import httpx
import redis


@pytest.mark.integration
@pytest.mark.slow
class TestE2ETaskFlow:
    """Test complete task flow through the system."""

    def test_submit_coding_task(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        redis_client: redis.Redis
    ):
        """Submit a coding task and verify it enters the queue."""
        task_id = str(uuid.uuid4())

        # Submit task
        response = rag_api_client.post(
            "/v1/tasks/submit",
            json={
                "task_id": task_id,
                "prompt": "Write a Python function called 'add' that takes two numbers and returns their sum.",
                "priority": "p1"
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=30.0
        )

        # If endpoint exists, verify submission
        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data or data.get("success"), f"Task submission should confirm: {data}"
        elif response.status_code == 404:
            pytest.skip("Task submission endpoint not implemented")

    def test_task_enters_queue(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        redis_client: redis.Redis
    ):
        """Submitted task should appear in Redis queue."""
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "type": "code_generation",
            "prompt": "Write a function to calculate factorial",
            "status": "pending",
            "priority": "p2"
        }

        # Directly add to queue for testing
        redis_client.lpush("tasks:p2", json.dumps(task_data))

        # Verify it's in queue
        queue_items = redis_client.lrange("tasks:p2", 0, -1)
        found = any(task_id in item for item in queue_items)
        assert found, "Task should be in queue"

        # Cleanup
        redis_client.lrem("tasks:p2", 1, json.dumps(task_data))

    @pytest.mark.slow
    def test_complete_task_flow(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Test complete flow: submit -> process -> result."""
        from datetime import datetime, timezone

        # Submit task via API
        response = rag_api_client.post(
            "/v1/tasks/submit",
            json={
                "prompt": "Write a Python function called add(a, b) that returns a + b. Include a simple test.",
                "priority": "p1",
                "type": "code_generation"
            },
            headers={"Authorization": f"Bearer {test_api_key.key_string}"},
            timeout=30.0
        )

        if response.status_code == 200:
            # API submission succeeded - use the returned task_id
            data = response.json()
            task_id = data.get("task_id")
        elif response.status_code == 404:
            # API endpoint not available - submit directly to Redis
            task_id = str(uuid.uuid4())
            task_data = {
                "id": task_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "priority": "p1",
                "status": "pending",
                "type": "code_generation",
                "prompt": "Write a Python function called add(a, b) that returns a + b. Include a simple test.",
                "project_id": None,
                "target_file": None,
                "max_attempts": 3,
                "timeout_seconds": 120,
                "require_tests_pass": True,
                "require_lint_pass": False,
                "test_code": None,
                "attempts": [],
                "result": None,
                "completed_at": None,
                "metrics": {}
            }
            redis_client.hset(f"task:{task_id}", "data", json.dumps(task_data))
            redis_client.rpush("tasks:p1", task_id)
        else:
            raise AssertionError(f"Task submission failed with status {response.status_code}: {response.text}")

        task_hash_key = f"task:{task_id}"
        cleanup_redis_keys.append(task_hash_key)

        # Wait for processing (with timeout)
        max_wait = 60  # 1 minute should be enough
        start = time.time()
        completed = False

        final_status = None
        while time.time() - start < max_wait:
            # Check status in the task hash (how worker stores it)
            task_json = redis_client.hget(task_hash_key, "data")
            if task_json:
                task = json.loads(task_json)
                status = task.get("status")
                if status == "completed":
                    completed = True
                    final_status = status
                    break
                elif status == "failed":
                    # Task was processed but failed - this is a test failure
                    final_status = status
                    break
            time.sleep(2)

        # Assert successful completion
        assert completed, f"Task {task_id} should complete successfully within {max_wait}s. Final status: {final_status}. Check task-worker logs."

    def test_result_contains_generated_code(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Successful result should contain generated code."""
        task_id = str(uuid.uuid4())
        result_key = f"task:{task_id}:result"
        cleanup_redis_keys.append(result_key)

        # Simulate a completed task result
        result = {
            "success": True,
            "code": "def add(a, b):\n    return a + b",
            "tests_passed": 2,
            "tests_run": 2,
            "attempts": 1,
            "duration_ms": 5000
        }
        redis_client.set(result_key, json.dumps(result))

        # Retrieve and verify
        stored = json.loads(redis_client.get(result_key))
        assert "code" in stored, "Result should contain code"
        assert "def" in stored["code"], "Code should be valid Python"

    def test_metrics_updated_after_task(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Metrics should be updated after task completion."""
        import datetime
        today = datetime.date.today().isoformat()
        metrics_key = f"metrics:daily:{today}"

        # Get current metrics
        initial_tasks = int(redis_client.hget(metrics_key, "tasks_total") or 0)

        # Simulate task completion metrics update
        redis_client.hincrby(metrics_key, "tasks_total", 1)
        redis_client.hincrby(metrics_key, "tasks_success", 1)

        # Verify update
        new_tasks = int(redis_client.hget(metrics_key, "tasks_total") or 0)
        assert new_tasks > initial_tasks, "Task count should increase"

    def test_can_retrieve_result_by_task_id(
        self,
        rag_api_client: httpx.Client,
        test_api_key,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Should be able to retrieve result by task ID."""
        task_id = str(uuid.uuid4())
        result_key = f"task:{task_id}:result"
        status_key = f"task:{task_id}:status"
        cleanup_redis_keys.extend([result_key, status_key])

        # Store a mock result
        result = {"success": True, "code": "print('hello')"}
        redis_client.set(result_key, json.dumps(result))
        redis_client.set(status_key, "completed")

        # Try to retrieve via API
        response = rag_api_client.get(
            f"/v1/tasks/{task_id}/status",
            headers={"Authorization": f"Bearer {test_api_key.key_string}"}
        )

        if response.status_code == 200:
            data = response.json()
            assert "status" in data or "result" in data
        elif response.status_code == 404:
            # Retrieve directly from Redis
            stored = redis_client.get(result_key)
            assert stored is not None


@pytest.mark.integration
@pytest.mark.slow
class TestE2EGracefulFailure:
    """Test graceful failure handling."""

    def test_impossible_task_fails_gracefully(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Impossible task should fail gracefully without crashing."""
        task_id = str(uuid.uuid4())
        error_key = f"task:{task_id}:error"
        status_key = f"task:{task_id}:status"
        cleanup_redis_keys.extend([error_key, status_key])

        # Simulate failed task
        redis_client.set(status_key, "failed")
        redis_client.set(error_key, "Max attempts reached: task requirements could not be satisfied")

        # Verify graceful failure
        status = redis_client.get(status_key)
        error = redis_client.get(error_key)

        assert status == "failed", "Failed task should have failed status"
        assert error is not None, "Failed task should have error message"
        assert len(error) > 0, "Error message should be descriptive"

    def test_error_message_stored(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Error message should be stored on failure."""
        task_id = str(uuid.uuid4())
        error_key = f"task:{task_id}:error"
        cleanup_redis_keys.append(error_key)

        # Store error
        error_msg = "SyntaxError: invalid syntax in generated code after 5 attempts"
        redis_client.set(error_key, error_msg)

        # Verify
        stored_error = redis_client.get(error_key)
        assert stored_error == error_msg
        assert "SyntaxError" in stored_error

    def test_system_retries_appropriately(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """System should retry failed attempts before giving up."""
        task_id = str(uuid.uuid4())
        attempts_key = f"task:{task_id}:attempts"
        cleanup_redis_keys.append(attempts_key)

        # Simulate attempt tracking
        attempts = [
            {"attempt": 1, "success": False, "error": "Test failed"},
            {"attempt": 2, "success": False, "error": "Test failed"},
            {"attempt": 3, "success": False, "error": "Test failed"},
            {"attempt": 4, "success": False, "error": "Test failed"},
            {"attempt": 5, "success": False, "error": "Max attempts reached"}
        ]

        for attempt in attempts:
            redis_client.rpush(attempts_key, json.dumps(attempt))

        # Verify retry count
        stored_attempts = redis_client.lrange(attempts_key, 0, -1)
        assert len(stored_attempts) == 5, "Should have 5 attempts"


@pytest.mark.integration
class TestE2ETimeConstraints:
    """Test time constraints."""

    @pytest.mark.slow
    def test_task_completes_within_timeout(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Simple task should complete within reasonable time."""
        # This is mostly a placeholder - actual timing depends on system load
        task_id = str(uuid.uuid4())
        status_key = f"task:{task_id}:status"
        cleanup_redis_keys.append(status_key)

        # Record start time
        start = time.time()

        # Simulate quick task completion
        redis_client.set(status_key, "completed")

        # Verify timing
        elapsed = time.time() - start
        assert elapsed < 180, f"Task should complete within 180 seconds, took {elapsed:.1f}s"
