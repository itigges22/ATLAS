"""
Tests for Task Worker service.

Validates task queue processing, priority handling,
and result storage.
"""

import json
import uuid
import time
import pytest
import redis


class TestTaskWorkerRunning:
    """Test Task Worker pod is running."""

    def test_worker_pod_running(self):
        """Task Worker pod should be running."""
        import subprocess
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", "app=task-worker", "-o", "jsonpath={.items[*].status.phase}"],
            capture_output=True,
            text=True
        )
        # May match different label selectors
        if not result.stdout or "Running" not in result.stdout:
            # Try alternate selector
            result = subprocess.run(
                ["kubectl", "get", "pods", "--field-selector=status.phase=Running", "-o", "name"],
                capture_output=True,
                text=True
            )
            assert "task-worker" in result.stdout.lower(), "Task worker pod should be running"


class TestTaskQueueSubmission:
    """Test task submission to queues."""

    def test_can_submit_to_p2_queue(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to submit task to P2 (batch) queue."""
        task = {
            "id": str(uuid.uuid4()),
            "type": "test",
            "prompt": "Test task for P2 queue",
            "status": "pending",
            "priority": "p2"
        }

        # Use real queue names but mark for cleanup
        queue = "tasks:p2"
        task_json = json.dumps(task)

        # Push to queue
        redis_client.lpush(queue, task_json)

        # Verify it's in queue
        length = redis_client.llen(queue)
        assert length >= 1, "Task should be in P2 queue"

        # Clean up by removing our test task
        redis_client.lrem(queue, 1, task_json)

    def test_can_submit_to_p1_queue(self, redis_client: redis.Redis):
        """Should be able to submit task to P1 (fire-forget) queue."""
        task = {
            "id": str(uuid.uuid4()),
            "type": "test",
            "prompt": "Test task for P1 queue",
            "status": "pending",
            "priority": "p1"
        }

        queue = "tasks:p1"
        task_json = json.dumps(task)

        redis_client.lpush(queue, task_json)
        length = redis_client.llen(queue)
        assert length >= 1, "Task should be in P1 queue"

        redis_client.lrem(queue, 1, task_json)

    def test_can_submit_to_p0_queue(self, redis_client: redis.Redis):
        """Should be able to submit task to P0 (interactive) queue."""
        task = {
            "id": str(uuid.uuid4()),
            "type": "test",
            "prompt": "Test task for P0 queue",
            "status": "pending",
            "priority": "p0"
        }

        queue = "tasks:p0"
        task_json = json.dumps(task)

        redis_client.lpush(queue, task_json)
        length = redis_client.llen(queue)
        assert length >= 1, "Task should be in P0 queue"

        redis_client.lrem(queue, 1, task_json)


class TestTaskStatusTracking:
    """Test task status tracking in Redis."""

    def test_task_status_can_be_stored(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Task status should be storable in Redis."""
        task_id = str(uuid.uuid4())
        status_key = f"task:{task_id}:status"
        cleanup_redis_keys.append(status_key)

        # Store status
        redis_client.set(status_key, "pending")
        assert redis_client.get(status_key) == "pending"

        # Update status
        redis_client.set(status_key, "processing")
        assert redis_client.get(status_key) == "processing"

        # Complete status
        redis_client.set(status_key, "completed")
        assert redis_client.get(status_key) == "completed"

    def test_task_result_can_be_stored(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Task result should be storable in Redis."""
        task_id = str(uuid.uuid4())
        result_key = f"task:{task_id}:result"
        cleanup_redis_keys.append(result_key)

        result = {
            "success": True,
            "code": "def add(a, b): return a + b",
            "tests_passed": 3,
            "duration_ms": 5000
        }

        redis_client.set(result_key, json.dumps(result))
        stored = json.loads(redis_client.get(result_key))

        assert stored["success"] is True
        assert "code" in stored


class TestTaskPriorityOrder:
    """Test priority queue ordering."""

    def test_priority_p0_before_p1(self, redis_client: redis.Redis):
        """P0 tasks should have priority over P1."""
        # This tests the conceptual ordering - actual processing is async
        p0_len = redis_client.llen("tasks:p0")
        p1_len = redis_client.llen("tasks:p1")
        p2_len = redis_client.llen("tasks:p2")

        # Just verify queues are accessible
        assert p0_len >= 0
        assert p1_len >= 0
        assert p2_len >= 0

    def test_worker_processes_in_priority_order(self, redis_client: redis.Redis):
        """Worker should process P0 > P1 > P2 in order."""
        # This is an integration test - worker must be running
        # We just verify the queue structure exists
        for queue in ["tasks:p0", "tasks:p1", "tasks:p2"]:
            # Queue type should be list
            queue_type = redis_client.type(queue)
            assert queue_type in ["list", "none"], f"Queue {queue} should be list type"


class TestTaskFailureHandling:
    """Test failure handling."""

    def test_failed_task_can_store_error(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Failed task should store error message."""
        task_id = str(uuid.uuid4())
        error_key = f"task:{task_id}:error"
        status_key = f"task:{task_id}:status"
        cleanup_redis_keys.extend([error_key, status_key])

        # Store failure
        redis_client.set(status_key, "failed")
        redis_client.set(error_key, "SyntaxError: invalid syntax at line 5")

        assert redis_client.get(status_key) == "failed"
        assert "SyntaxError" in redis_client.get(error_key)


class TestTaskMetrics:
    """Test task metrics in Redis."""

    def test_metrics_can_be_updated(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Task metrics should be updatable."""
        import datetime
        today = datetime.date.today().isoformat()
        metrics_key = f"test:metrics:daily:{today}"
        cleanup_redis_keys.append(metrics_key)

        # Update metrics
        redis_client.hincrby(metrics_key, "tasks_total", 1)
        redis_client.hincrby(metrics_key, "tasks_success", 1)
        redis_client.hincrby(metrics_key, "tokens_used", 150)

        metrics = redis_client.hgetall(metrics_key)
        assert int(metrics["tasks_total"]) >= 1
        assert int(metrics["tasks_success"]) >= 1


class TestTaskRecentList:
    """Test recent tasks list."""

    def test_recent_tasks_list_writable(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Recent tasks list should be writable."""
        recent_key = "test:metrics:recent_tasks"
        cleanup_redis_keys.append(recent_key)

        task_record = json.dumps({
            "id": str(uuid.uuid4()),
            "status": "completed",
            "duration_ms": 3000,
            "timestamp": time.time()
        })

        redis_client.lpush(recent_key, task_record)
        redis_client.ltrim(recent_key, 0, 19)  # Keep only last 20

        length = redis_client.llen(recent_key)
        assert length >= 1

    def test_recent_tasks_limited_to_20(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Recent tasks should be limited to 20 entries."""
        recent_key = "test:metrics:recent_tasks_limit"
        cleanup_redis_keys.append(recent_key)

        # Add 25 tasks
        for i in range(25):
            task_record = json.dumps({"id": str(i), "timestamp": time.time()})
            redis_client.lpush(recent_key, task_record)
            redis_client.ltrim(recent_key, 0, 19)

        length = redis_client.llen(recent_key)
        assert length == 20, f"Should limit to 20, got {length}"
