"""Async sandbox client for geometric-lens.

Calls the sandbox service to execute code and returns structured results.
Used by the verify-repair-retry loop in the RAG pipeline.
"""

import os
import re
import logging
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

SANDBOX_URL = os.getenv("SANDBOX_URL", "http://sandbox:8020")
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "30"))


async def execute_sandbox(
    code: str,
    test_code: str = "",
    stdin: str = "",
    timeout_sec: int = SANDBOX_TIMEOUT,
) -> Tuple[bool, str, str]:
    """Execute code in the sandbox service.

    Args:
        code: Python code to execute
        test_code: Optional test code to validate output
        stdin: Optional stdin input
        timeout_sec: Execution timeout in seconds

    Returns:
        (passed, stdout, stderr) tuple
    """
    try:
        async with httpx.AsyncClient(timeout=timeout_sec + 10) as client:
            resp = await client.post(
                f"{SANDBOX_URL}/execute",
                json={
                    "code": code,
                    "test_code": test_code,
                    "stdin": stdin,
                    "timeout": timeout_sec,
                },
            )
            resp.raise_for_status()
            d = resp.json()
            return (
                d.get("passed", False),
                d.get("stdout", ""),
                d.get("stderr", ""),
            )
    except httpx.TimeoutException:
        return (False, "", "Sandbox execution timed out")
    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}")
        return (False, "", f"Sandbox unavailable: {e}")


def extract_code_from_response(response_text: str) -> Optional[str]:
    """Extract Python code from an LLM response.

    Handles both markdown code blocks and raw code.
    Returns the longest code block found, or None if no code detected.
    """
    # Try markdown code blocks first (```python ... ``` or ``` ... ```)
    blocks = re.findall(
        r'```(?:python)?\s*\n(.*?)```',
        response_text,
        re.DOTALL,
    )

    if blocks:
        # Return the longest block (most likely the main solution)
        return max(blocks, key=len).strip()

    # If no code blocks, check if the response looks like raw code
    lines = response_text.strip().split('\n')
    code_indicators = ('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', '=')
    code_lines = sum(1 for line in lines if line.strip().startswith(code_indicators))

    if code_lines > len(lines) * 0.3:
        return response_text.strip()

    return None
