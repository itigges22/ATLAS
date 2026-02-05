"""
Benchmark code execution runner.

Handles sending prompts to the LLM, extracting code from responses,
and executing code in isolated sandboxes with resource limits.
"""

"""
Benchmark code execution runner.

Handles sending prompts to the LLM, extracting code from responses,
and executing code in isolated sandboxes with resource limits.
"""

import json
import os
import re
import signal
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, List

# Try httpx first, fall back to urllib
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# resource module only available on Unix
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

from .config import config
from .models import BenchmarkTask, AttemptResult, TaskResult


class CodeExecutionError(Exception):
    """Error during code execution."""
    pass


class LLMConnectionError(Exception):
    """Error connecting to LLM service."""
    pass


def extract_code(response: str) -> str:
    """
    Extract Python code from LLM response.

    Handles various formats:
    - Markdown code blocks (```python ... ```)
    - Plain code blocks (``` ... ```)
    - Raw code without blocks
    - Qwen3 <think>...</think> blocks (stripped before extraction)

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code
    """
    # Strip Qwen3 thinking blocks first - they can consume tokens
    # before the actual code output
    think_pattern = r'<think>.*?</think>'
    response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()

    # Try to extract from markdown code blocks
    # Pattern for ```python ... ``` or ```py ... ```
    pattern = r'```(?:python|py)?\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    if matches:
        # Return the longest match (likely the main code block)
        return max(matches, key=len).strip()

    # Try generic code blocks
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return max(matches, key=len).strip()

    # No code blocks found, assume raw code
    # Strip common prefixes/suffixes
    code = response.strip()

    # Remove common LLM artifacts
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip lines that look like explanations
        if line.strip().startswith('Here') and ':' in line:
            continue
        if line.strip().startswith('This function'):
            continue
        if line.strip().startswith('The function'):
            continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip()


def set_resource_limits(memory_mb: int = 512, timeout_sec: int = 30):
    """
    Set resource limits for the subprocess.

    Args:
        memory_mb: Memory limit in megabytes
        timeout_sec: CPU time limit in seconds
    """
    # Memory limit (in bytes)
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    # CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, (timeout_sec, timeout_sec))

    # Prevent forking
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def _make_preexec_fn(memory_mb: int, timeout_sec: int):
    """
    Create a preexec_fn that sets resource limits for the subprocess.

    Args:
        memory_mb: Memory limit in megabytes
        timeout_sec: CPU time limit in seconds

    Returns:
        Function to be called in subprocess before exec
    """
    def preexec():
        if HAS_RESOURCE:
            # Memory limit (virtual address space)
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_sec, timeout_sec))

    return preexec


def execute_code(
    code: str,
    test_code: str,
    timeout_sec: int = 30,
    memory_mb: int = 512
) -> Tuple[bool, str, str, float]:
    """
    Execute code with test cases in an isolated subprocess.

    Args:
        code: The generated code to execute
        test_code: Test assertions to run
        timeout_sec: Execution timeout in seconds
        memory_mb: Memory limit in megabytes

    Returns:
        Tuple of (passed, stdout, stderr, execution_time_ms)
    """
    # Combine code and tests
    full_code = f"{code}\n\n{test_code}"

    # Write to temporary file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        start_time = time.time()

        # Execute in subprocess with resource limits via preexec_fn
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=_make_preexec_fn(memory_mb, timeout_sec),
            env={
                **os.environ,
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONUNBUFFERED': '1',
            },
        )

        execution_time_ms = (time.time() - start_time) * 1000

        passed = result.returncode == 0
        return passed, result.stdout, result.stderr, execution_time_ms

    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout_sec} seconds", timeout_sec * 1000

    except Exception as e:
        return False, "", str(e), 0.0

    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


class BenchmarkRunner:
    """
    Runs benchmark tasks against an LLM.

    Handles:
    - Sending prompts to the LLM API
    - Extracting code from responses
    - Executing code with tests
    - Recording results
    - Retry logic with error feedback (ralph-loop pattern)
    """

    def __init__(
        self,
        llm_url: str = None,
        timeout_sec: int = None,
        memory_mb: int = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the benchmark runner.

        Args:
            llm_url: URL for the LLM API (defaults to config)
            timeout_sec: Execution timeout per task
            memory_mb: Memory limit per task
            max_retries: Max retries for LLM connection failures
            retry_delay: Delay between retries in seconds
        """
        self.llm_url = llm_url or config.llama_api_url
        self.timeout_sec = timeout_sec or config.default_timeout_seconds
        self.memory_mb = memory_mb or config.default_memory_limit_mb
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # HTTP client with longer timeout for inference
        if HAS_HTTPX:
            self.client = httpx.Client(timeout=120.0)
        else:
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None:
            self.client.close()

    def close(self):
        """Close the HTTP client."""
        if self.client is not None:
            self.client.close()

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        error_context: str = None
    ) -> Tuple[str, int, float]:
        """
        Call the LLM API with retry logic.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            error_context: Previous error for ralph-loop retry

        Returns:
            Tuple of (response_text, tokens_generated, inference_time_ms)
        """
        # Build the full prompt with error context if provided
        if error_context:
            full_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt failed with error:\n{error_context}\n\n"
                f"Please fix the code and try again."
            )
        else:
            full_prompt = prompt

        messages = [
            {"role": "user", "content": full_prompt}
        ]

        request_body = {
            "model": "qwen3-14b",  # Model name doesn't matter for local llama-server
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                if HAS_HTTPX and self.client is not None:
                    response = self.client.post(
                        f"{self.llm_url}/chat/completions",
                        json=request_body
                    )
                    response.raise_for_status()
                    data = response.json()
                else:
                    # Fall back to urllib
                    req = urllib.request.Request(
                        f"{self.llm_url}/chat/completions",
                        data=json.dumps(request_body).encode('utf-8'),
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode('utf-8'))

                inference_time_ms = (time.time() - start_time) * 1000

                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("completion_tokens", 0)

                return content, tokens, inference_time_ms

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                last_error = f"URL error: {str(e)}"
            except Exception as e:
                if HAS_HTTPX:
                    import httpx as httpx_module
                    if isinstance(e, httpx_module.HTTPStatusError):
                        last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                    elif isinstance(e, httpx_module.RequestError):
                        last_error = f"Request error: {str(e)}"
                    else:
                        last_error = str(e)
                else:
                    last_error = str(e)

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        raise LLMConnectionError(f"Failed to connect to LLM after {self.max_retries} attempts: {last_error}")

    def run_task(
        self,
        task: BenchmarkTask,
        k: int = 1,
        temperature: float = None,
        use_ralph_loop: bool = False
    ) -> TaskResult:
        """
        Run a benchmark task with k attempts.

        Args:
            task: The benchmark task to run
            k: Number of attempts
            temperature: Sampling temperature (default: 0 for k=1, 0.8 otherwise)
            use_ralph_loop: Whether to feed errors back for retries

        Returns:
            TaskResult with all attempts
        """
        if temperature is None:
            temperature = config.default_temperature_pass1 if k == 1 else config.default_temperature_passk

        result = TaskResult(task_id=task.task_id)
        error_context = None

        for attempt_num in range(1, k + 1):
            try:
                # Get LLM response
                response, tokens, inference_time = self._call_llm(
                    task.prompt,
                    temperature=temperature,
                    error_context=error_context if use_ralph_loop else None
                )

                # Extract code
                generated_code = extract_code(response)

                # Execute with tests
                passed, stdout, stderr, exec_time = execute_code(
                    generated_code,
                    task.test_code,
                    timeout_sec=self.timeout_sec,
                    memory_mb=self.memory_mb
                )

                # Record attempt
                attempt = AttemptResult(
                    task_id=task.task_id,
                    attempt_number=attempt_num,
                    generated_code=generated_code,
                    passed=passed,
                    execution_time_ms=exec_time,
                    error_output=stderr if not passed else "",
                    tokens_generated=tokens,
                    inference_time_ms=inference_time,
                    stdout=stdout,
                    stderr=stderr
                )
                result.attempts.append(attempt)

                # Update totals
                result.total_tokens += tokens
                result.total_inference_time_ms += inference_time
                result.total_execution_time_ms += exec_time

                # Track best attempt
                if passed and result.best_attempt is None:
                    result.best_attempt = attempt_num

                # Update error context for ralph-loop
                if not passed and use_ralph_loop:
                    error_context = stderr or "Tests failed"

            except LLMConnectionError as e:
                # Record failed attempt due to connection error
                attempt = AttemptResult(
                    task_id=task.task_id,
                    attempt_number=attempt_num,
                    generated_code="",
                    passed=False,
                    execution_time_ms=0,
                    error_output=f"LLM connection error: {str(e)}",
                    tokens_generated=0,
                    inference_time_ms=0
                )
                result.attempts.append(attempt)

            except Exception as e:
                # Record failed attempt due to unexpected error
                attempt = AttemptResult(
                    task_id=task.task_id,
                    attempt_number=attempt_num,
                    generated_code="",
                    passed=False,
                    execution_time_ms=0,
                    error_output=f"Unexpected error: {str(e)}",
                    tokens_generated=0,
                    inference_time_ms=0
                )
                result.attempts.append(attempt)

        return result

    def run_task_dry(self, task: BenchmarkTask) -> TaskResult:
        """
        Dry run a task (validate parsing without LLM calls).

        Args:
            task: The benchmark task to validate

        Returns:
            TaskResult with validation status
        """
        result = TaskResult(task_id=task.task_id)

        # Validate task has required fields
        try:
            assert task.prompt, "Missing prompt"
            assert task.entry_point, "Missing entry_point"
            assert task.test_code, "Missing test_code"

            # Try running canonical solution with tests
            if task.canonical_solution:
                passed, stdout, stderr, exec_time = execute_code(
                    task.canonical_solution,
                    task.test_code,
                    timeout_sec=self.timeout_sec,
                    memory_mb=self.memory_mb
                )

                attempt = AttemptResult(
                    task_id=task.task_id,
                    attempt_number=0,  # 0 indicates canonical solution test
                    generated_code=task.canonical_solution,
                    passed=passed,
                    execution_time_ms=exec_time,
                    error_output=stderr if not passed else "",
                    stdout=stdout,
                    stderr=stderr
                )
                result.attempts.append(attempt)
                result.total_execution_time_ms = exec_time

                if passed:
                    result.best_attempt = 0

        except AssertionError as e:
            attempt = AttemptResult(
                task_id=task.task_id,
                attempt_number=0,
                generated_code="",
                passed=False,
                execution_time_ms=0,
                error_output=f"Validation error: {str(e)}"
            )
            result.attempts.append(attempt)

        return result


def run_benchmark_dry(
    tasks: List[BenchmarkTask],
    progress_callback=None
) -> List[TaskResult]:
    """
    Dry run all tasks (validate without LLM calls).

    Args:
        tasks: List of tasks to validate
        progress_callback: Optional callback(task_idx, task_id, passed)

    Returns:
        List of TaskResult objects
    """
    results = []

    with BenchmarkRunner() as runner:
        for idx, task in enumerate(tasks):
            result = runner.run_task_dry(task)
            results.append(result)

            if progress_callback:
                progress_callback(idx, task.task_id, result.passed)

    return results


def run_benchmark(
    tasks: List[BenchmarkTask],
    k: int = 1,
    temperature: float = None,
    use_ralph_loop: bool = False,
    progress_callback=None,
    save_callback=None
) -> List[TaskResult]:
    """
    Run benchmark on all tasks.

    Args:
        tasks: List of tasks to run
        k: Number of attempts per task
        temperature: Sampling temperature
        use_ralph_loop: Whether to use error feedback for retries
        progress_callback: Optional callback(task_idx, task_id, passed)
        save_callback: Optional callback(result) to save results incrementally

    Returns:
        List of TaskResult objects
    """
    results = []

    with BenchmarkRunner() as runner:
        for idx, task in enumerate(tasks):
            result = runner.run_task(
                task,
                k=k,
                temperature=temperature,
                use_ralph_loop=use_ralph_loop
            )
            results.append(result)

            if progress_callback:
                progress_callback(idx, task.task_id, result.passed)

            if save_callback:
                save_callback(result)

    return results
