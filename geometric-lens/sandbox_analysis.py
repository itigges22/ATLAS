"""Structured sandbox feedback analysis.

Parses raw sandbox output (passed, stdout, stderr) into structured
error analysis with failure types, line numbers, and recoverability.

Used by the verify-repair-retry loop to make intelligent repair decisions
combining G(x) quality scores with concrete error diagnostics.
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TYPE_ERROR = "type_error"
    NAME_ERROR = "name_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    VALUE_ERROR = "value_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    ASSERTION_ERROR = "assertion_error"
    TIMEOUT = "timeout"
    OUTPUT_MISMATCH = "output_mismatch"
    MEMORY_ERROR = "memory_error"
    UNKNOWN = "unknown"
    NONE = "none"


class Severity(str, Enum):
    LOW = "low"         # Typo, missing import — easy targeted fix
    MEDIUM = "medium"   # Logic error in specific location
    HIGH = "high"       # Structural issue, likely wrong approach
    CRITICAL = "critical"  # Fundamental misunderstanding


# Map Python exception classes to FailureType
_ERROR_TYPE_MAP = {
    'SyntaxError': FailureType.SYNTAX_ERROR,
    'IndentationError': FailureType.SYNTAX_ERROR,
    'TabError': FailureType.SYNTAX_ERROR,
    'RuntimeError': FailureType.RUNTIME_ERROR,
    'RecursionError': FailureType.RUNTIME_ERROR,
    'TypeError': FailureType.TYPE_ERROR,
    'NameError': FailureType.NAME_ERROR,
    'UnboundLocalError': FailureType.NAME_ERROR,
    'IndexError': FailureType.INDEX_ERROR,
    'KeyError': FailureType.KEY_ERROR,
    'ValueError': FailureType.VALUE_ERROR,
    'AttributeError': FailureType.ATTRIBUTE_ERROR,
    'ImportError': FailureType.IMPORT_ERROR,
    'ModuleNotFoundError': FailureType.IMPORT_ERROR,
    'AssertionError': FailureType.ASSERTION_ERROR,
    'AssertionError': FailureType.ASSERTION_ERROR,
    'MemoryError': FailureType.MEMORY_ERROR,
    'TimeoutError': FailureType.TIMEOUT,
    'ZeroDivisionError': FailureType.RUNTIME_ERROR,
    'StopIteration': FailureType.RUNTIME_ERROR,
    'OverflowError': FailureType.RUNTIME_ERROR,
    'FileNotFoundError': FailureType.RUNTIME_ERROR,
}

# Types that are typically fixable with targeted repair
_RECOVERABLE_TYPES = {
    FailureType.SYNTAX_ERROR,
    FailureType.NAME_ERROR,
    FailureType.IMPORT_ERROR,
    FailureType.INDEX_ERROR,
    FailureType.KEY_ERROR,
    FailureType.TYPE_ERROR,
    FailureType.ATTRIBUTE_ERROR,
}

# Default severity per failure type
_SEVERITY_MAP = {
    FailureType.SYNTAX_ERROR: Severity.LOW,
    FailureType.IMPORT_ERROR: Severity.LOW,
    FailureType.NAME_ERROR: Severity.LOW,
    FailureType.TYPE_ERROR: Severity.MEDIUM,
    FailureType.INDEX_ERROR: Severity.MEDIUM,
    FailureType.KEY_ERROR: Severity.MEDIUM,
    FailureType.ATTRIBUTE_ERROR: Severity.MEDIUM,
    FailureType.VALUE_ERROR: Severity.MEDIUM,
    FailureType.ASSERTION_ERROR: Severity.MEDIUM,
    FailureType.RUNTIME_ERROR: Severity.HIGH,
    FailureType.TIMEOUT: Severity.HIGH,
    FailureType.MEMORY_ERROR: Severity.CRITICAL,
    FailureType.OUTPUT_MISMATCH: Severity.MEDIUM,
    FailureType.UNKNOWN: Severity.HIGH,
    FailureType.NONE: Severity.LOW,
}

# Regex patterns
_TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')
_ERROR_LINE_RE = re.compile(r'^(\w+(?:Error|Exception)): (.+)$', re.MULTILINE)
_SYNTAX_LINE_RE = re.compile(r'File "([^"]+)", line (\d+)')


@dataclass
class TracebackFrame:
    file: str
    line: int
    function: str
    code: Optional[str] = None


@dataclass
class SandboxResult:
    passed: bool
    failure_type: FailureType
    failure_line: Optional[int]
    failure_message: str
    is_recoverable: bool
    severity: Severity
    traceback_frames: List[TracebackFrame] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    raw_error_class: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['failure_type'] = self.failure_type.value
        d['severity'] = self.severity.value
        return d


def analyze_sandbox_output(
    passed: bool,
    stdout: str,
    stderr: str,
    expected_output: Optional[str] = None,
    gx_score: Optional[float] = None,
) -> SandboxResult:
    """Parse sandbox output into structured analysis.

    Args:
        passed: Whether sandbox execution passed
        stdout: Standard output from execution
        stderr: Standard error from execution
        expected_output: Expected output for mismatch detection
        gx_score: G(x) quality score (0-1) for combined analysis

    Returns:
        SandboxResult with structured error analysis
    """
    if passed:
        return SandboxResult(
            passed=True,
            failure_type=FailureType.NONE,
            failure_line=None,
            failure_message="",
            is_recoverable=True,
            severity=Severity.LOW,
            stdout=stdout,
            stderr=stderr,
        )

    # Parse traceback frames
    frames = _parse_traceback(stderr)

    # Detect error type and message
    failure_type, error_msg, error_class = _detect_error_type(stderr)

    # Check for timeout
    if 'timeout' in stderr.lower() or 'timed out' in stderr.lower():
        failure_type = FailureType.TIMEOUT
        error_msg = error_msg or "Execution timed out"

    # Check output mismatch
    if expected_output and stdout.strip() != expected_output.strip():
        if failure_type == FailureType.UNKNOWN:
            failure_type = FailureType.OUTPUT_MISMATCH
            error_msg = "Output doesn't match expected result"

    # Extract failure line
    failure_line = _extract_failure_line(frames, stderr)

    # Determine recoverability
    is_recoverable = failure_type in _RECOVERABLE_TYPES

    # Adjust recoverability based on G(x) if available
    if gx_score is not None:
        if gx_score < 0.1:
            is_recoverable = False
        elif gx_score > 0.5 and failure_type in _RECOVERABLE_TYPES:
            is_recoverable = True

    severity = _SEVERITY_MAP.get(failure_type, Severity.HIGH)

    # Downgrade severity if G(x) says code is mostly correct
    if gx_score is not None and gx_score > 0.6 and severity in (Severity.HIGH, Severity.CRITICAL):
        severity = Severity.MEDIUM

    suggestion = _generate_suggestion(failure_type, error_msg, failure_line, frames)

    return SandboxResult(
        passed=False,
        failure_type=failure_type,
        failure_line=failure_line,
        failure_message=error_msg,
        is_recoverable=is_recoverable,
        severity=severity,
        traceback_frames=frames,
        stdout=stdout,
        stderr=stderr,
        raw_error_class=error_class,
        suggestion=suggestion,
    )


def _parse_traceback(stderr: str) -> List[TracebackFrame]:
    """Extract traceback frames from stderr."""
    frames = []
    lines = stderr.split('\n')

    for i, line in enumerate(lines):
        match = _TB_FRAME_RE.search(line)
        if match:
            code_line = None
            if i + 1 < len(lines) and not lines[i + 1].startswith('  File '):
                candidate = lines[i + 1].strip()
                if candidate and not candidate.startswith('Traceback'):
                    code_line = candidate

            frames.append(TracebackFrame(
                file=match.group(1),
                line=int(match.group(2)),
                function=match.group(3) or "<module>",
                code=code_line,
            ))

    return frames


def _detect_error_type(stderr: str) -> tuple:
    """Detect the error type from stderr.

    Returns (FailureType, error_message, raw_error_class).
    """
    # Look for standard Python error pattern at end of traceback
    matches = list(_ERROR_LINE_RE.finditer(stderr))
    if matches:
        last = matches[-1]
        error_class = last.group(1)
        error_msg = last.group(2)
        failure_type = _ERROR_TYPE_MAP.get(error_class, FailureType.UNKNOWN)
        return failure_type, error_msg, error_class

    # Check for SyntaxError (different format)
    if 'SyntaxError' in stderr:
        msg = "Syntax error in code"
        syntax_match = re.search(r'SyntaxError: (.+)', stderr)
        if syntax_match:
            msg = syntax_match.group(1)
        return FailureType.SYNTAX_ERROR, msg, 'SyntaxError'

    # Fallback: look for any exception-like pattern
    exc_match = re.search(r'(\w+(?:Error|Exception)): (.+)', stderr)
    if exc_match:
        error_class = exc_match.group(1)
        error_msg = exc_match.group(2)
        failure_type = _ERROR_TYPE_MAP.get(error_class, FailureType.UNKNOWN)
        return failure_type, error_msg, error_class

    return FailureType.UNKNOWN, stderr[:200] if stderr else "Unknown error", None


def _extract_failure_line(frames: List[TracebackFrame], stderr: str) -> Optional[int]:
    """Get the most relevant failure line number."""
    # Prefer the last frame in user code (not stdlib/site-packages)
    user_frames = [f for f in frames
                   if not f.file.startswith('/usr/')
                   and 'site-packages' not in f.file]
    if user_frames:
        return user_frames[-1].line

    if frames:
        return frames[-1].line

    # Try to extract from SyntaxError format
    match = _SYNTAX_LINE_RE.search(stderr)
    if match:
        return int(match.group(2))

    return None


def _generate_suggestion(
    failure_type: FailureType,
    error_msg: str,
    failure_line: Optional[int],
    frames: List[TracebackFrame],
) -> str:
    """Generate a human-readable repair suggestion."""
    line_ref = f" on line {failure_line}" if failure_line else ""

    suggestions = {
        FailureType.SYNTAX_ERROR: f"Fix syntax error{line_ref}: {error_msg}",
        FailureType.NAME_ERROR: (
            f"Variable or function not defined{line_ref}: {error_msg}. "
            "Check spelling and scope."
        ),
        FailureType.IMPORT_ERROR: (
            f"Missing import{line_ref}: {error_msg}. "
            "Add the required import statement."
        ),
        FailureType.TYPE_ERROR: (
            f"Type mismatch{line_ref}: {error_msg}. "
            "Check argument types and counts."
        ),
        FailureType.INDEX_ERROR: (
            f"Index out of bounds{line_ref}: {error_msg}. "
            "Check array/list bounds."
        ),
        FailureType.KEY_ERROR: (
            f"Missing dictionary key{line_ref}: {error_msg}. "
            "Check key existence before access."
        ),
        FailureType.ATTRIBUTE_ERROR: (
            f"Missing attribute{line_ref}: {error_msg}. "
            "Check object type and available methods."
        ),
        FailureType.VALUE_ERROR: (
            f"Invalid value{line_ref}: {error_msg}. "
            "Check input validation."
        ),
        FailureType.ASSERTION_ERROR: (
            f"Assertion failed{line_ref}: {error_msg}. "
            "Check test expectations and logic."
        ),
        FailureType.RUNTIME_ERROR: (
            f"Runtime error{line_ref}: {error_msg}. "
            "Review control flow and edge cases."
        ),
        FailureType.TIMEOUT: (
            "Code exceeded time limit. "
            "Check for infinite loops or inefficient algorithms."
        ),
        FailureType.MEMORY_ERROR: (
            "Code exceeded memory limit. "
            "Reduce data structure sizes or use generators."
        ),
        FailureType.OUTPUT_MISMATCH: (
            "Output doesn't match expected result. "
            "Check formatting and edge cases."
        ),
    }

    return suggestions.get(failure_type, f"Execution failed{line_ref}: {error_msg}")


def build_repair_prompt(
    analysis: SandboxResult,
    original_code: str,
    original_prompt: str = "",
    gx_score: Optional[float] = None,
) -> str:
    """Build a structured repair prompt from sandbox analysis.

    Returns a repair instruction that can be injected as additional context
    for the LLM to fix the code.
    """
    if analysis.passed:
        return ""

    parts = [
        "The previous code attempt failed. Here is the structured error analysis:"
    ]
    parts.append(f"\n**Error Type**: {analysis.failure_type.value}")
    parts.append(f"**Message**: {analysis.failure_message}")

    if analysis.failure_line:
        parts.append(f"**Failure Line**: {analysis.failure_line}")

    if analysis.suggestion:
        parts.append(f"**Suggestion**: {analysis.suggestion}")

    # Add G(x) quality context
    if gx_score is not None:
        if gx_score > 0.6:
            parts.append(
                f"\n*Quality signal (G(x)={gx_score:.2f})*: The overall code structure "
                "appears sound. Focus on fixing the specific error location."
            )
        elif gx_score > 0.3:
            parts.append(
                f"\n*Quality signal (G(x)={gx_score:.2f})*: The code has mixed quality. "
                "The error may indicate a broader logic issue beyond the specific line."
            )
        else:
            parts.append(
                f"\n*Quality signal (G(x)={gx_score:.2f})*: The code likely has "
                "fundamental issues. Consider restructuring the approach rather than "
                "patching the specific error."
            )

    # Add relevant traceback context
    if analysis.traceback_frames:
        parts.append("\n**Traceback** (most recent call last):")
        for frame in analysis.traceback_frames[-3:]:
            code_ref = f"  → {frame.code}" if frame.code else ""
            parts.append(f"  Line {frame.line} in {frame.function}{code_ref}")

    # Recoverability hint
    if analysis.is_recoverable:
        parts.append(
            f"\nThis error is **recoverable** (severity: {analysis.severity.value}). "
            "Make a targeted fix at the identified location."
        )
    else:
        parts.append(
            f"\nThis error suggests a **structural issue** (severity: {analysis.severity.value}). "
            "Reconsider the overall approach."
        )

    return "\n".join(parts)
