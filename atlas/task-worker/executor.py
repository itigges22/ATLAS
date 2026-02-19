"""
Sandbox executor client for task worker.
"""

import requests
import logging
from typing import Optional, List, Dict
from ralph_loop import ExecutionResult

logger = logging.getLogger(__name__)

class SandboxExecutor:
    """Client for the sandbox execution service."""

    def __init__(self, sandbox_url: str = "http://sandbox:8020"):
        self.sandbox_url = sandbox_url

    def execute(self, code: str, context: Dict) -> ExecutionResult:
        """
        Execute code in the sandbox.

        Args:
            code: The generated code to execute
            context: Contains test_code, requirements, etc.

        Returns:
            ExecutionResult with test outcomes
        """
        try:
            response = requests.post(
                f"{self.sandbox_url}/execute",
                json={
                    "code": code,
                    "language": "python",
                    "test_code": context.get("test_code"),
                    "requirements": context.get("requirements", []),
                    "timeout": context.get("timeout", 30)
                },
                timeout=120
            )

            if response.ok:
                data = response.json()
                return ExecutionResult(
                    success=data["success"],
                    compile_success=data["compile_success"],
                    tests_run=data["tests_run"],
                    tests_passed=data["tests_passed"],
                    lint_score=data.get("lint_score"),
                    stdout=data["stdout"],
                    stderr=data["stderr"],
                    error_type=data.get("error_type"),
                    error_message=data.get("error_message")
                )
            else:
                return ExecutionResult(
                    success=False,
                    compile_success=False,
                    tests_run=0,
                    tests_passed=0,
                    lint_score=None,
                    stdout="",
                    stderr=f"Sandbox returned {response.status_code}: {response.text}",
                    error_type="SandboxError",
                    error_message=response.text
                )

        except requests.Timeout:
            return ExecutionResult(
                success=False,
                compile_success=False,
                tests_run=0,
                tests_passed=0,
                lint_score=None,
                stdout="",
                stderr="Sandbox request timed out",
                error_type="Timeout",
                error_message="Sandbox did not respond in time"
            )
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return ExecutionResult(
                success=False,
                compile_success=False,
                tests_run=0,
                tests_passed=0,
                lint_score=None,
                stdout="",
                stderr=str(e),
                error_type=type(e).__name__,
                error_message=str(e)
            )

    def health_check(self) -> bool:
        """Check if sandbox is available."""
        try:
            response = requests.get(f"{self.sandbox_url}/health", timeout=5)
            return response.ok
        except Exception:
            return False
