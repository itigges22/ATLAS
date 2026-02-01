"""
Task worker: Pulls tasks from queue, runs ralph-loop, stores results.

This is the main entry point for the task-worker container.
"""

import os
import re
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict

from task_queue import TaskQueue, TaskStatus
from ralph_loop import RalphLoop, RalphLoopResult, GenerationResult
from executor import SandboxExecutor
from metrics import MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-api:8001")
LLAMA_URL = os.getenv("LLAMA_URL", "http://llama-service:8000")
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://sandbox:8020")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "1"))
# Use direct llama access to bypass API auth for internal worker
USE_DIRECT_LLAMA = os.getenv("USE_DIRECT_LLAMA", "true").lower() == "true"

class TaskWorker:
    def __init__(self):
        self.queue = TaskQueue(REDIS_URL)
        self.metrics = MetricsCollector(REDIS_URL)
        self.executor = SandboxExecutor(SANDBOX_URL)

    def run(self):
        """Main worker loop - runs forever."""
        logger.info("Task worker starting...")
        logger.info(f"  Redis: {REDIS_URL}")
        logger.info(f"  RAG API: {RAG_API_URL}")
        logger.info(f"  LLaMA: {LLAMA_URL}")
        logger.info(f"  Sandbox: {SANDBOX_URL}")

        while True:
            try:
                task = self.queue.pop()

                if task:
                    self.process_task(task)
                else:
                    time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(5)  # Back off on errors

    def process_task(self, task):
        """Process a single task through the ralph-loop."""
        logger.info(f"Processing task {task.id}: {task.type}")

        # Update status to running
        task.status = TaskStatus.RUNNING.value
        self.queue.update(task)

        try:
            # Get RAG context if project specified
            context = {}
            if task.project_id:
                context = self._get_rag_context(task.prompt, task.project_id)

            # Configure ralph-loop
            loop = RalphLoop(
                generator=self._generate,
                executor=self.executor.execute,
                max_attempts=task.max_attempts,
                timeout_seconds=task.timeout_seconds,
                require_tests_pass=task.require_tests_pass,
                require_lint_pass=task.require_lint_pass
            )

            # Run ralph-loop
            result = loop.run(
                prompt=task.prompt,
                context=context,
                on_attempt=lambda a: self._log_attempt(task.id, a)
            )

            # Update task with results
            task.status = TaskStatus.COMPLETED.value if result.success else TaskStatus.FAILED.value
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.result = {
                "success": result.success,
                "stop_reason": result.stop_reason.value,
                "final_code": result.final_code,
                "attempts_count": len(result.attempts),
                "total_duration_ms": result.total_duration_ms,
                "total_tokens": result.total_tokens
            }
            task.attempts = [self._serialize_attempt(a) for a in result.attempts]
            task.metrics = {
                "total_duration_ms": result.total_duration_ms,
                "total_tokens": result.total_tokens,
                "attempts": len(result.attempts)
            }

            self.queue.update(task)
            self.queue.publish_completion(task.id)

            # Record metrics
            self.metrics.record_task(task)

            # Store successful completions for training
            if result.success and result.final_code:
                self._store_training_example(task, result, context)

            logger.info(f"Task {task.id} completed: {result.stop_reason.value}")

        except Exception as e:
            logger.error(f"Task {task.id} failed with exception: {e}")
            task.status = TaskStatus.FAILED.value
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.result = {"success": False, "error": str(e)}
            self.queue.update(task)
            self.queue.publish_completion(task.id)

    def _generate(self, prompt: str, context: Dict, temperature: float) -> GenerationResult:
        """Generate code using llama-server directly or via rag-api."""
        # Build messages
        messages = []

        if context.get("rag_context"):
            messages.append({
                "role": "system",
                "content": f"You are a coding assistant. Use this context from the codebase:\n\n{context['rag_context']}"
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        # Call llama-server directly to bypass auth
        start = time.time()
        api_url = LLAMA_URL if USE_DIRECT_LLAMA else RAG_API_URL
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": "Qwen3-14B.Q6_K.gguf",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        # Extract code from response
        content = data["choices"][0]["message"]["content"]
        code = self._extract_code(content)

        return GenerationResult(
            code=code,
            tokens_in=data.get("usage", {}).get("prompt_tokens", 0),
            tokens_out=data.get("usage", {}).get("completion_tokens", 0),
            duration_ms=int((time.time() - start) * 1000),
            temperature=temperature,
            raw_response=data
        )

    def _extract_code(self, content: str) -> str:
        """Extract code blocks from LLM response."""
        # Try python-specific blocks first
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fall back to generic code blocks
        pattern = r'```\n?(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()

        # If no code blocks, return the whole content
        return content.strip()

    def _get_rag_context(self, query: str, project_id: str) -> Dict:
        """Retrieve RAG context for the query."""
        try:
            response = requests.post(
                f"{RAG_API_URL}/v1/context/retrieve",
                json={
                    "query": query,
                    "project_id": project_id,
                    "top_k": 10
                },
                timeout=30
            )
            if response.ok:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get RAG context: {e}")
        return {}

    def _log_attempt(self, task_id: str, attempt):
        """Log attempt for real-time monitoring."""
        logger.info(
            f"Task {task_id} attempt {attempt.attempt_number}: "
            f"{'SUCCESS' if attempt.success else 'FAILED'} "
            f"({attempt.execution.tests_passed}/{attempt.execution.tests_run} tests)"
        )

    def _store_training_example(self, task, result, context):
        """Store successful completion for training data."""
        import json as json_module

        # Calculate quality score based on attempts (fewer = better)
        attempts_count = len(result.attempts)
        quality_score = max(0.5, 1.0 - (attempts_count - 1) * 0.1)

        training_example = {
            "task_id": task.id,
            "prompt": task.prompt,
            "context": context.get("rag_context", ""),
            "completion": result.final_code,
            "quality_score": quality_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts_count,
            "total_tokens": result.total_tokens
        }

        # Store in Redis for later export
        self.queue.redis.rpush("training:examples", json_module.dumps(training_example))
        logger.info(f"Stored training example for task {task.id} (quality: {quality_score:.2f})")

    def _serialize_attempt(self, attempt) -> Dict:
        """Convert attempt to JSON-serializable dict."""
        return {
            "attempt_number": attempt.attempt_number,
            "started_at": attempt.started_at,
            "completed_at": attempt.completed_at,
            "generation": {
                "tokens_in": attempt.generation.tokens_in,
                "tokens_out": attempt.generation.tokens_out,
                "duration_ms": attempt.generation.duration_ms,
                "temperature": attempt.generation.temperature
            },
            "execution": {
                "compile_success": attempt.execution.compile_success,
                "tests_run": attempt.execution.tests_run,
                "tests_passed": attempt.execution.tests_passed,
                "lint_score": attempt.execution.lint_score,
                "error_type": attempt.execution.error_type,
                "error_message": attempt.execution.error_message
            },
            "success": attempt.success
        }

if __name__ == "__main__":
    worker = TaskWorker()
    worker.run()
