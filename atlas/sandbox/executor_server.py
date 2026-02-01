"""
Sandbox execution server.

Provides isolated code execution with resource limits.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import resource
import logging
import re
import time
from typing import Dict, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Code Execution Sandbox")

# Configuration
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "60"))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "512"))
WORKSPACE_BASE = Path(os.getenv("WORKSPACE_BASE", "/tmp/sandbox"))

class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    test_code: Optional[str] = None
    requirements: Optional[List[str]] = None
    timeout: int = 30

class ExecuteResponse(BaseModel):
    success: bool
    compile_success: bool
    tests_run: int
    tests_passed: int
    lint_score: Optional[float]
    stdout: str
    stderr: str
    error_type: Optional[str]
    error_message: Optional[str]
    execution_time_ms: int

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/execute", response_model=ExecuteResponse)
def execute_code(request: ExecuteRequest):
    """Execute code in isolated environment."""

    if request.language != "python":
        raise HTTPException(
            status_code=400,
            detail=f"Language '{request.language}' not yet supported"
        )

    # Create isolated workspace
    workspace = tempfile.mkdtemp(dir=WORKSPACE_BASE)

    try:
        result = execute_python(
            code=request.code,
            test_code=request.test_code,
            workspace=Path(workspace),
            timeout=min(request.timeout, MAX_EXECUTION_TIME),
            requirements=request.requirements
        )
        return result
    finally:
        # Cleanup workspace
        shutil.rmtree(workspace, ignore_errors=True)

def execute_python(
    code: str,
    test_code: Optional[str],
    workspace: Path,
    timeout: int,
    requirements: Optional[List[str]]
) -> ExecuteResponse:
    """Execute Python code with tests."""

    start_time = time.time()

    # Write main code
    main_file = workspace / "solution.py"
    main_file.write_text(code)

    # First: syntax check (compile)
    compile_result = check_syntax(main_file)
    if not compile_result["success"]:
        return ExecuteResponse(
            success=False,
            compile_success=False,
            tests_run=0,
            tests_passed=0,
            lint_score=None,
            stdout="",
            stderr=compile_result["error"],
            error_type="SyntaxError",
            error_message=compile_result["error"],
            execution_time_ms=int((time.time() - start_time) * 1000)
        )

    # Install requirements if specified
    if requirements:
        install_result = install_requirements(requirements, workspace, timeout)
        if not install_result["success"]:
            return ExecuteResponse(
                success=False,
                compile_success=True,
                tests_run=0,
                tests_passed=0,
                lint_score=None,
                stdout="",
                stderr=install_result["error"],
                error_type="DependencyError",
                error_message=f"Failed to install: {requirements}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    # Run lint check
    lint_score = run_lint(main_file)

    # Run tests if provided
    if test_code:
        test_file = workspace / "test_solution.py"
        test_file.write_text(test_code)
        test_result = run_tests(workspace, timeout)
    else:
        # No tests - just try to import/run the code
        test_result = run_import_check(main_file, timeout)

    execution_time = int((time.time() - start_time) * 1000)

    return ExecuteResponse(
        success=test_result["success"],
        compile_success=True,
        tests_run=test_result.get("tests_run", 1),
        tests_passed=test_result.get("tests_passed", 1 if test_result["success"] else 0),
        lint_score=lint_score,
        stdout=test_result.get("stdout", ""),
        stderr=test_result.get("stderr", ""),
        error_type=test_result.get("error_type"),
        error_message=test_result.get("error_message"),
        execution_time_ms=execution_time
    )

def check_syntax(file_path: Path) -> Dict:
    """Check Python syntax without executing."""
    try:
        with open(file_path) as f:
            compile(f.read(), file_path, 'exec')
        return {"success": True}
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Line {e.lineno}: {e.msg}"
        }

def run_lint(file_path: Path) -> Optional[float]:
    """Run pylint and return score."""
    try:
        result = subprocess.run(
            ["python", "-m", "pylint", "--score=y", "--exit-zero", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Parse score from output like "Your code has been rated at 8.50/10"
        match = re.search(r'rated at ([\d.]+)/10', result.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        logger.warning(f"Lint failed: {e}")
    return None

def run_tests(workspace: Path, timeout: int) -> Dict:
    """Run pytest on the test file."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-v", "--tb=short", str(workspace)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace
        )

        # Parse pytest output for counts
        stdout = result.stdout
        stderr = result.stderr

        # Look for "X passed, Y failed" pattern
        passed = 0
        failed = 0

        passed_match = re.search(r'(\d+) passed', stdout)
        if passed_match:
            passed = int(passed_match.group(1))

        failed_match = re.search(r'(\d+) failed', stdout)
        if failed_match:
            failed = int(failed_match.group(1))

        total = passed + failed or 1

        return {
            "success": result.returncode == 0,
            "tests_run": total,
            "tests_passed": passed,
            "stdout": stdout[-2000:],  # Truncate
            "stderr": stderr[-1000:],
            "error_type": "TestFailure" if result.returncode != 0 else None,
            "error_message": stderr[:500] if result.returncode != 0 else None
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "tests_run": 0,
            "tests_passed": 0,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "error_type": "Timeout",
            "error_message": f"Tests did not complete within {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "tests_run": 0,
            "tests_passed": 0,
            "stdout": "",
            "stderr": str(e),
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

def run_import_check(file_path: Path, timeout: int) -> Dict:
    """Try to import the module to catch runtime errors."""
    try:
        result = subprocess.run(
            ["python", "-c", f"import sys; sys.path.insert(0, '{file_path.parent}'); import {file_path.stem}"],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return {
            "success": result.returncode == 0,
            "tests_run": 1,
            "tests_passed": 1 if result.returncode == 0 else 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error_type": "ImportError" if result.returncode != 0 else None,
            "error_message": result.stderr[:500] if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "tests_run": 0,
            "tests_passed": 0,
            "stdout": "",
            "stderr": f"Import check timed out after {timeout}s",
            "error_type": "Timeout",
            "error_message": f"Import check timed out after {timeout}s"
        }

def install_requirements(requirements: List[str], workspace: Path, timeout: int) -> Dict:
    """Install pip requirements in workspace."""
    try:
        result = subprocess.run(
            ["pip", "install", "--target", str(workspace), "--quiet"] + requirements,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {"success": result.returncode == 0, "error": result.stderr}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8020)
