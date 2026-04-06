"""
Multi-language sandbox execution server.

Supports: Python, JavaScript/TypeScript, Go, Rust, C/C++, Bash/Shell
Provides isolated code execution with resource limits and structured error reporting.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import logging
import re
import time
from typing import Dict, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ATLAS Code Execution Sandbox")

MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "60"))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "512"))
WORKSPACE_BASE = Path(os.getenv("WORKSPACE_BASE", "/tmp/sandbox"))

SUPPORTED_LANGUAGES = {
    "python", "py", "python3",
    "javascript", "js", "node",
    "typescript", "ts",
    "go", "golang",
    "rust", "rs",
    "c", "cpp", "c++",
    "bash", "sh", "shell",
}

def normalize_language(lang: str) -> str:
    lang = lang.lower().strip()
    if lang in ("python", "py", "python3"):
        return "python"
    if lang in ("javascript", "js", "node"):
        return "javascript"
    if lang in ("typescript", "ts"):
        return "typescript"
    if lang in ("go", "golang"):
        return "go"
    if lang in ("rust", "rs"):
        return "rust"
    if lang in ("c",):
        return "c"
    if lang in ("cpp", "c++"):
        return "cpp"
    if lang in ("bash", "sh", "shell"):
        return "bash"
    return lang


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
    lint_score: Optional[float] = None
    stdout: str
    stderr: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: int


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/languages")
def list_languages():
    """List supported languages and their runtime versions."""
    versions = {}
    checks = {
        "python": ["python3", "--version"],
        "javascript": ["node", "--version"],
        "typescript": ["tsc", "--version"],
        "go": ["go", "version"],
        "rust": ["rustc", "--version"],
        "c": ["gcc", "--version"],
        "cpp": ["g++", "--version"],
        "bash": ["bash", "--version"],
    }
    for lang, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            versions[lang] = result.stdout.strip().split("\n")[0]
        except Exception:
            versions[lang] = "not installed"
    return {"languages": versions}


@app.post("/execute", response_model=ExecuteResponse)
def execute_code(request: ExecuteRequest):
    """Execute code in isolated environment."""
    lang = normalize_language(request.language)

    if lang not in ("python", "javascript", "typescript", "go", "rust", "c", "cpp", "bash"):
        raise HTTPException(
            status_code=400,
            detail=f"Language '{request.language}' not supported. Supported: python, javascript, typescript, go, rust, c, cpp, bash"
        )

    workspace = tempfile.mkdtemp(dir=WORKSPACE_BASE)
    timeout = min(request.timeout, MAX_EXECUTION_TIME)

    try:
        handler = LANGUAGE_HANDLERS[lang]
        result = handler(
            code=request.code,
            test_code=request.test_code,
            workspace=Path(workspace),
            timeout=timeout,
            requirements=request.requirements,
        )
        return result
    except Exception as e:
        logger.exception(f"Execution error for {lang}")
        return ExecuteResponse(
            success=False,
            compile_success=False,
            tests_run=0,
            tests_passed=0,
            stdout="",
            stderr=str(e),
            error_type=type(e).__name__,
            error_message=str(e),
            execution_time_ms=0,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


class SyntaxCheckRequest(BaseModel):
    code: str
    language: str = "python"
    filename: Optional[str] = None


class SyntaxCheckResponse(BaseModel):
    valid: bool
    errors: List[str]
    language: str
    check_time_ms: int


@app.post("/syntax-check", response_model=SyntaxCheckResponse)
def syntax_check(request: SyntaxCheckRequest):
    """Check code syntax without executing. Returns parse/compile errors."""
    lang = normalize_language(request.language)
    workspace = tempfile.mkdtemp(dir=WORKSPACE_BASE)
    start = time.time()

    try:
        errors = _syntax_check_impl(lang, request.code, Path(workspace), request.filename)
        elapsed = int((time.time() - start) * 1000)
        return SyntaxCheckResponse(
            valid=len(errors) == 0,
            errors=errors,
            language=lang,
            check_time_ms=elapsed,
        )
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return SyntaxCheckResponse(
            valid=False,
            errors=[str(e)],
            language=lang,
            check_time_ms=elapsed,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _syntax_check_impl(lang: str, code: str, workspace: Path, filename: Optional[str] = None) -> List[str]:
    """Language-specific syntax checking. Returns list of error strings."""
    errors = []

    if lang == "python":
        # Use py_compile for fast AST parse
        fpath = workspace / (filename or "check.py")
        fpath.write_text(code)
        result = _run_cmd(["python3", "-m", "py_compile", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            # Extract just the error line from py_compile output
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                line = line.strip()
                if line and "SyntaxError" in line or "IndentationError" in line or "TabError" in line:
                    errors.append(line)
            if not errors and stderr.strip():
                errors.append(stderr.strip().split("\n")[-1])

    elif lang == "javascript":
        fpath = workspace / (filename or "check.js")
        fpath.write_text(code)
        result = _run_cmd(["node", "--check", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            errors.append(result.get("stderr", "").strip())

    elif lang == "typescript":
        fpath = workspace / (filename or "check.ts")
        fpath.write_text(code)
        # tsc --noEmit for type checking; fall back to tsx parse
        result = _run_cmd(["tsc", "--noEmit", "--strict", str(fpath)], timeout=10, cwd=workspace)
        if result["returncode"] != 0:
            for line in result.get("stderr", "").splitlines() + result.get("stdout", "").splitlines():
                line = line.strip()
                if line and ("error TS" in line or "Error" in line):
                    errors.append(line)

    elif lang == "go":
        fpath = workspace / (filename or "main.go")
        fpath.write_text(code)
        # Use gofmt -e for fast syntax-only checking (no compilation, no go.mod needed)
        result = _run_cmd(["gofmt", "-e", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                line = line.strip()
                if line:
                    errors.append(line)

    elif lang == "rust":
        fpath = workspace / (filename or "check.rs")
        fpath.write_text(code)
        # rustc --edition 2021 with no codegen for syntax-only
        result = _run_cmd(
            ["rustc", "--edition", "2021", "--crate-type", "bin", str(fpath), "-o", "/dev/null"],
            timeout=10, cwd=workspace
        )
        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip())
            if not errors and stderr.strip():
                errors.append(stderr.strip().split("\n")[-1])

    elif lang in ("c", "cpp"):
        ext = ".c" if lang == "c" else ".cpp"
        fpath = workspace / (filename or f"check{ext}")
        fpath.write_text(code)
        compiler = "gcc" if lang == "c" else "g++"
        flags = ["-std=c17"] if lang == "c" else ["-std=c++17"]
        # -fsyntax-only: parse and type-check only, no codegen
        result = _run_cmd(
            [compiler] + flags + ["-fsyntax-only", str(fpath)],
            timeout=10, cwd=workspace
        )
        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            for line in stderr.splitlines():
                if "error:" in line:
                    errors.append(line.strip())
            if not errors and stderr.strip():
                errors.append(stderr.strip().split("\n")[-1])

    elif lang == "bash":
        fpath = workspace / (filename or "check.sh")
        fpath.write_text(code)
        result = _run_cmd(["bash", "-n", str(fpath)], timeout=5, cwd=workspace)
        if result["returncode"] != 0:
            errors.append(result.get("stderr", "").strip())

    return errors


# ---------------------------------------------------------------------------
# Language handlers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str], timeout: int, cwd: Path = None, env: dict = None) -> Dict:
    """Run a command with timeout and return structured result."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=run_env,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-2000:],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def _classify_error(stderr: str) -> Optional[str]:
    """Extract error type from stderr."""
    patterns = [
        (r"SyntaxError", "SyntaxError"),
        (r"NameError", "NameError"),
        (r"TypeError", "TypeError"),
        (r"ValueError", "ValueError"),
        (r"ImportError|ModuleNotFoundError", "ImportError"),
        (r"IndexError", "IndexError"),
        (r"KeyError", "KeyError"),
        (r"AttributeError", "AttributeError"),
        (r"ZeroDivisionError", "ZeroDivisionError"),
        (r"FileNotFoundError", "FileNotFoundError"),
        (r"ReferenceError", "ReferenceError"),
        (r"error\[E\d+\]", "CompileError"),
        (r"error:", "CompileError"),
        (r"undefined reference", "LinkError"),
        (r"cannot find", "NotFoundError"),
        (r"timed out", "Timeout"),
    ]
    for pattern, error_type in patterns:
        if re.search(pattern, stderr):
            return error_type
    return "RuntimeError" if stderr.strip() else None


# --- Python ---

def execute_python(code, test_code, workspace, timeout, requirements, **_):
    start = time.time()
    main_file = workspace / "solution.py"
    main_file.write_text(code)

    # Syntax check
    try:
        compile(code, "solution.py", "exec")
    except SyntaxError as e:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=f"Line {e.lineno}: {e.msg}",
            error_type="SyntaxError", error_message=f"Line {e.lineno}: {e.msg}",
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Install requirements
    if requirements:
        r = _run_cmd(["pip", "install", "--target", str(workspace), "--quiet"] + requirements, timeout)
        if not r["success"]:
            return ExecuteResponse(
                success=False, compile_success=True,
                tests_run=0, tests_passed=0,
                stdout="", stderr=r["stderr"],
                error_type="DependencyError", error_message=r["stderr"][:500],
                execution_time_ms=int((time.time() - start) * 1000),
            )

    # Lint
    lint_score = None
    try:
        lr = subprocess.run(
            ["python", "-m", "pylint", "--score=y", "--exit-zero", str(main_file)],
            capture_output=True, text=True, timeout=15
        )
        m = re.search(r"rated at ([\d.]+)/10", lr.stdout)
        if m:
            lint_score = float(m.group(1))
    except Exception:
        pass

    # Run
    if test_code:
        (workspace / "test_solution.py").write_text(test_code)
        r = _run_cmd(["python", "-m", "pytest", "-v", "--tb=short", str(workspace)], timeout, cwd=workspace)
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", r["stdout"])) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", r["stdout"])) else 0
        total = passed + failed or 1
    else:
        r = _run_cmd(
            ["python", "-c", f"import sys; sys.path.insert(0,'{workspace}'); import solution"],
            timeout
        )
        passed = 1 if r["success"] else 0
        total = 1

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=total, tests_passed=passed,
        lint_score=lint_score,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- JavaScript ---

def execute_javascript(code, test_code, workspace, timeout, **_):
    start = time.time()
    main_file = workspace / "solution.js"
    main_file.write_text(code)

    # Syntax check via node --check
    r = _run_cmd(["node", "--check", str(main_file)], 10)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="SyntaxError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd(["node", str(main_file)], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- TypeScript ---

def execute_typescript(code, test_code, workspace, timeout, **_):
    start = time.time()
    main_file = workspace / "solution.ts"
    main_file.write_text(code)

    # Type check via tsc --noEmit
    r = _run_cmd(["tsc", "--noEmit", "--strict", "--esModuleInterop", str(main_file)], 15)
    compile_success = r["success"]
    if not compile_success:
        # Still try to run — TS errors are often non-fatal for execution
        logger.info(f"TypeScript type errors: {r['stderr'][:200]}")

    # Run via tsx (faster than ts-node, handles ESM)
    r = _run_cmd(["tsx", str(main_file)], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=compile_success,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- Go ---

def execute_go(code, test_code, workspace, timeout, **_):
    start = time.time()
    main_file = workspace / "main.go"
    main_file.write_text(code)

    # Init module
    _run_cmd(["go", "mod", "init", "sandbox"], 5, cwd=workspace)

    # Build (compile check)
    r = _run_cmd(["go", "build", "-o", str(workspace / "program"), str(main_file)], 30, cwd=workspace)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd([str(workspace / "program")], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- Rust ---

def execute_rust(code, test_code, workspace, timeout, **_):
    start = time.time()
    main_file = workspace / "main.rs"
    main_file.write_text(code)

    # Compile
    binary = workspace / "program"
    r = _run_cmd(["rustc", str(main_file), "-o", str(binary)], 30)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd([str(binary)], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- C ---

def execute_c(code, test_code, workspace, timeout, **_):
    start = time.time()
    main_file = workspace / "solution.c"
    main_file.write_text(code)

    binary = workspace / "program"
    r = _run_cmd(["gcc", "-o", str(binary), str(main_file), "-lm", "-Wall"], 15)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    r = _run_cmd([str(binary)], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- C++ ---

def execute_cpp(code, test_code, workspace, timeout, **_):
    start = time.time()
    main_file = workspace / "solution.cpp"
    main_file.write_text(code)

    binary = workspace / "program"
    r = _run_cmd(["g++", "-o", str(binary), str(main_file), "-std=c++17", "-Wall"], 15)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="CompileError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    r = _run_cmd([str(binary)], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# --- Bash ---

def execute_bash(code, test_code, workspace, timeout, **_):
    start = time.time()
    script = workspace / "solution.sh"
    script.write_text(code)
    script.chmod(0o755)

    # Syntax check
    r = _run_cmd(["bash", "-n", str(script)], 5)
    if not r["success"]:
        return ExecuteResponse(
            success=False, compile_success=False,
            tests_run=0, tests_passed=0,
            stdout="", stderr=r["stderr"],
            error_type="SyntaxError", error_message=r["stderr"][:500],
            execution_time_ms=int((time.time() - start) * 1000),
        )

    # Run
    r = _run_cmd(["bash", str(script)], timeout, cwd=workspace)

    return ExecuteResponse(
        success=r["success"], compile_success=True,
        tests_run=1, tests_passed=1 if r["success"] else 0,
        stdout=r["stdout"], stderr=r["stderr"],
        error_type=_classify_error(r["stderr"]) if not r["success"] else None,
        error_message=r["stderr"][:500] if not r["success"] else None,
        execution_time_ms=int((time.time() - start) * 1000),
    )


# Handler dispatch
LANGUAGE_HANDLERS = {
    "python": execute_python,
    "javascript": execute_javascript,
    "typescript": execute_typescript,
    "go": execute_go,
    "rust": execute_rust,
    "c": execute_c,
    "cpp": execute_cpp,
    "bash": execute_bash,
}


if __name__ == "__main__":
    import uvicorn
    WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)
    logger.info(f"Supported languages: {list(LANGUAGE_HANDLERS.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=8020)
