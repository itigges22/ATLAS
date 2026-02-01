"""
Metrics collection for ATLAS system.
"""

import redis
import json
from datetime import datetime, date
from typing import Dict, Any

class MetricsCollector:
    """Collect and aggregate metrics for dashboard."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def record_task(self, task):
        """Record metrics from completed task."""
        today = date.today().isoformat()

        # Increment counters
        self.redis.hincrby(f"metrics:daily:{today}", "tasks_total", 1)

        if task.result and task.result.get("success"):
            self.redis.hincrby(f"metrics:daily:{today}", "tasks_success", 1)
        else:
            self.redis.hincrby(f"metrics:daily:{today}", "tasks_failed", 1)

        # Accumulate totals
        if task.metrics:
            self.redis.hincrbyfloat(
                f"metrics:daily:{today}",
                "total_tokens",
                task.metrics.get("total_tokens", 0)
            )
            self.redis.hincrbyfloat(
                f"metrics:daily:{today}",
                "total_duration_ms",
                task.metrics.get("total_duration_ms", 0)
            )
            self.redis.hincrbyfloat(
                f"metrics:daily:{today}",
                "total_attempts",
                task.metrics.get("attempts", 0)
            )

        # Record individual task for history
        self.redis.lpush("metrics:recent_tasks", json.dumps({
            "task_id": task.id,
            "type": task.type,
            "success": task.result.get("success") if task.result else False,
            "attempts": len(task.attempts),
            "duration_ms": task.metrics.get("total_duration_ms", 0) if task.metrics else 0,
            "completed_at": task.completed_at
        }))

        # Keep only last 1000 tasks
        self.redis.ltrim("metrics:recent_tasks", 0, 999)

    def get_daily_stats(self, day: str = None) -> Dict:
        """Get statistics for a specific day."""
        day = day or date.today().isoformat()
        stats = self.redis.hgetall(f"metrics:daily:{day}")

        # Convert to proper types
        return {
            "date": day,
            "tasks_total": int(stats.get("tasks_total", 0)),
            "tasks_success": int(stats.get("tasks_success", 0)),
            "tasks_failed": int(stats.get("tasks_failed", 0)),
            "total_tokens": int(float(stats.get("total_tokens", 0))),
            "total_duration_ms": int(float(stats.get("total_duration_ms", 0))),
            "total_attempts": int(float(stats.get("total_attempts", 0))),
            "success_rate": (
                int(stats.get("tasks_success", 0)) /
                max(int(stats.get("tasks_total", 0)), 1)
            ),
            "avg_attempts": (
                float(stats.get("total_attempts", 0)) /
                max(int(stats.get("tasks_total", 0)), 1)
            )
        }

    def get_recent_tasks(self, limit: int = 20) -> list:
        """Get recent task completions."""
        tasks = self.redis.lrange("metrics:recent_tasks", 0, limit - 1)
        return [json.loads(t) for t in tasks]
