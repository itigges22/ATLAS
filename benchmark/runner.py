"""
Benchmark code execution.

Runs generated code in isolated subprocesses with resource limits, either
against test assertions (`execute_code`) or stdin/stdout cases
(`execute_code_stdio`). LLM generation lives in stages.llm_client; the
V3 orchestration lives in v3_runner.
"""

import os
import subprocess
import tempfile
import time
from typing import Tuple, List

# resource module only available on Unix
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


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


def execute_code_stdio(
    code: str,
    test_inputs: List[str],
    test_outputs: List[str],
    timeout_sec: int = 30,
    memory_mb: int = 512
) -> Tuple[bool, str, str, float]:
    """
    Execute code with stdin/stdout test cases (for competitive-programming style problems).

    Writes code to a temp file, runs it once per test case with stdin piped in,
    and compares stdout to expected output.

    Args:
        code: The generated code to execute
        test_inputs: List of stdin input strings
        test_outputs: List of expected stdout strings
        timeout_sec: Execution timeout per test case in seconds
        memory_mb: Memory limit in megabytes

    Returns:
        Tuple of (all_passed, combined_stdout, combined_stderr, total_exec_time_ms)
    """
    if not test_inputs or not test_outputs:
        return False, "", "No test cases provided for stdio evaluation", 0.0

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    all_passed = True
    combined_stdout = []
    combined_stderr = []
    total_time_ms = 0.0

    try:
        for i, (inp, expected) in enumerate(zip(test_inputs, test_outputs)):
            try:
                start_time = time.time()

                result = subprocess.run(
                    ['python3', temp_path],
                    input=inp,
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

                exec_time_ms = (time.time() - start_time) * 1000
                total_time_ms += exec_time_ms

                actual = result.stdout.strip()
                expected_clean = expected.strip()

                if result.returncode != 0:
                    all_passed = False
                    combined_stderr.append(
                        f"Test {i+1}: runtime error (exit {result.returncode})\n{result.stderr}"
                    )
                elif actual != expected_clean:
                    all_passed = False
                    combined_stderr.append(
                        f"Test {i+1}: wrong answer\n"
                        f"  Expected: {expected_clean[:200]}\n"
                        f"  Got:      {actual[:200]}"
                    )
                combined_stdout.append(actual)

            except subprocess.TimeoutExpired:
                all_passed = False
                combined_stderr.append(f"Test {i+1}: timed out after {timeout_sec}s")
                total_time_ms += timeout_sec * 1000

            except Exception as e:
                all_passed = False
                combined_stderr.append(f"Test {i+1}: {str(e)}")

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            # best-effort: swallow on failure (caller continues)
            pass

    return (
        all_passed,
        "\n---\n".join(combined_stdout),
        "\n".join(combined_stderr),
        total_time_ms
    )
