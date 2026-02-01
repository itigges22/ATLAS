"""
Task queue operations using Redis.

Queue structure:
  - tasks:p0, tasks:p1, tasks:p2  (lists, priority queues)
  - task:{id}                      (hash, task data)
  - results:{id}                   (hash, completion data)
  - metrics:daily:{date}           (hash, aggregated metrics)
"""

import redis
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

class Priority(Enum):
    INTERACTIVE = "p0"  # User waiting, process immediately
    FIRE_FORGET = "p1"  # User wants it soon
    BATCH = "p2"        # Overnight/background

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    created_at: str
    priority: str
    status: str
    type: str
    prompt: str
    project_id: Optional[str] = None
    target_file: Optional[str] = None
    max_attempts: int = 5
    timeout_seconds: int = 300
    require_tests_pass: bool = True
    require_lint_pass: bool = False
    test_code: Optional[str] = None  # Optional test code to validate against
    attempts: list = None
    result: Optional[Dict] = None
    completed_at: Optional[str] = None
    metrics: Optional[Dict] = None

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []
        if self.metrics is None:
            self.metrics = {}

class TaskQueue:
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def submit(
        self,
        prompt: str,
        task_type: str = "code_generation",
        priority: Priority = Priority.FIRE_FORGET,
        project_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a new task to the queue. Returns task ID."""
        task_id = str(uuid.uuid4())

        task = Task(
            id=task_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            priority=priority.value,
            status=TaskStatus.PENDING.value,
            type=task_type,
            prompt=prompt,
            project_id=project_id,
            **kwargs
        )

        # Store task data
        self.redis.hset(f"task:{task_id}", mapping={
            "data": json.dumps(asdict(task))
        })

        # Add to priority queue
        self.redis.rpush(f"tasks:{priority.value}", task_id)

        return task_id

    def pop(self) -> Optional[Task]:
        """Pop highest priority task. Returns None if all queues empty."""
        for priority in [Priority.INTERACTIVE, Priority.FIRE_FORGET, Priority.BATCH]:
            task_id = self.redis.lpop(f"tasks:{priority.value}")
            if task_id:
                return self._get_task(task_id)
        return None

    def _get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task by ID."""
        data = self.redis.hget(f"task:{task_id}", "data")
        if data:
            return Task(**json.loads(data))
        return None

    def update(self, task: Task):
        """Update task data in Redis."""
        self.redis.hset(f"task:{task.id}", mapping={
            "data": json.dumps(asdict(task))
        })

    def get_status(self, task_id: str) -> Optional[Dict]:
        """Get task status for API response."""
        task = self._get_task(task_id)
        if task:
            return {
                "id": task.id,
                "status": task.status,
                "attempts": len(task.attempts),
                "result": task.result,
                "completed_at": task.completed_at
            }
        return None

    def get_queue_stats(self) -> Dict:
        """Get current queue statistics."""
        return {
            "p0_waiting": self.redis.llen("tasks:p0"),
            "p1_waiting": self.redis.llen("tasks:p1"),
            "p2_waiting": self.redis.llen("tasks:p2"),
            "total_waiting": sum([
                self.redis.llen("tasks:p0"),
                self.redis.llen("tasks:p1"),
                self.redis.llen("tasks:p2")
            ])
        }

    def publish_completion(self, task_id: str):
        """Notify any listeners that task is complete."""
        self.redis.publish(f"task:complete:{task_id}", "done")

    def wait_for_completion(self, task_id: str, timeout: int = 300) -> Optional[Dict]:
        """Block until task completes or timeout. For interactive use."""
        pubsub = self.redis.pubsub()
        pubsub.subscribe(f"task:complete:{task_id}")

        # Check if already complete
        task = self._get_task(task_id)
        if task and task.status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
            return self.get_status(task_id)

        # Wait for completion message
        for message in pubsub.listen():
            if message["type"] == "message":
                return self.get_status(task_id)

        return None
