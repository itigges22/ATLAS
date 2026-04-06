"""LLM-based pattern extraction from successful task completions."""

import logging
import re
import uuid
from typing import Optional

import httpx

from models.pattern import Pattern, PatternType, PatternTier, HALF_LIVES

logger = logging.getLogger(__name__)


async def extract_pattern(
    query: str,
    solution: str,
    retry_count: int,
    max_retries: int,
    error_context: Optional[str],
    source_files: list[str],
    llama_url: str,
) -> Optional[Pattern]:
    """
    Extract a reusable code pattern from a successful task completion.

    Calls llama-server with an extraction prompt, then classifies the type
    and computes surprise_score.
    """
    # Compute surprise_proxy
    surprise = min(retry_count / max(max_retries, 1), 1.0)

    # Classify pattern type (heuristic — no LLM call needed)
    pattern_type = classify_pattern_type(solution, error_context)
    half_life = HALF_LIVES.get(pattern_type, 14.0)

    # Extract pattern via LLM
    extracted = await _llm_extract(query, solution, llama_url)
    if not extracted:
        return None

    content, summary = extracted

    pattern = Pattern(
        id=str(uuid.uuid4()),
        type=pattern_type,
        tier=PatternTier.STM,
        content=content,
        summary=summary,
        context_query=query,
        error_context=error_context,
        surprise_score=surprise,
        access_count=0,
        success_count=1,
        failure_count=0,
        half_life_days=half_life,
        source_files=source_files,
    )

    return pattern


def classify_pattern_type(
    solution: str,
    error_context: Optional[str],
) -> PatternType:
    """
    Classify pattern type using heuristics.

    Rules (in priority order):
    1. Error message present → error_fix
    2. Import/dependency changes → api_pattern
    3. try/except, if/else control flow → bug_fix
    4. class/def structural → architectural
    5. Everything else → idiom
    """
    if error_context:
        return PatternType.ERROR_FIX

    # Check for import changes
    if re.search(r'\b(import |from \w+ import |require\(|pip install)', solution):
        # Only if imports are a major part
        import_lines = len(re.findall(r'^(?:import |from )', solution, re.MULTILINE))
        total_lines = solution.count('\n') + 1
        if import_lines / max(total_lines, 1) > 0.2:
            return PatternType.API_PATTERN

    # Check for error handling / control flow fixes
    if re.search(r'\b(try:|except |raise |if .+ is None|if not |\.get\()', solution):
        return PatternType.BUG_FIX

    # Check for structural/class-level changes
    if re.search(r'\bclass \w+', solution):
        return PatternType.ARCHITECTURAL

    return PatternType.IDIOM


async def _llm_extract(
    query: str,
    solution: str,
    llama_url: str,
) -> Optional[tuple[str, str]]:
    """
    Call llama-server to extract a reusable pattern.

    Returns (content, summary) or None on failure.
    """
    # Truncate long solutions
    max_chars = 2000
    if len(solution) > max_chars:
        solution = solution[:max_chars] + "\n... (truncated)"

    prompt = (
        "Given this task and solution, extract the key reusable code pattern "
        "as a snippet with a one-sentence description.\n\n"
        f"Task: {query}\n\n"
        f"Solution:\n```\n{solution}\n```\n\n"
        "Respond in this exact format:\n"
        "PATTERN:\n```\n<code pattern>\n```\n"
        "DESCRIPTION: <one sentence description>"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{llama_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a code pattern extraction assistant. "
                                "Extract the core reusable pattern from code solutions. "
                                "Be concise. Respond with only the pattern and description."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 200,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            if "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                content = msg.get("content", "")
                if not content:
                    content = msg.get("reasoning_content", "")

            if not content:
                return None

            return _parse_extraction(content, solution)

    except Exception as e:
        logger.warning(f"Pattern extraction LLM call failed: {e}")
        return None


def _parse_extraction(
    response: str,
    original_solution: str,
) -> tuple[str, str]:
    """Parse LLM extraction response into (content, summary)."""
    # Try to parse PATTERN: ... DESCRIPTION: ... format
    pattern_match = re.search(r"```\n?(.*?)```", response, re.DOTALL)
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", response, re.IGNORECASE)

    if pattern_match:
        code = pattern_match.group(1).strip()
        # Remove language specifier if present (e.g., "python\n")
        if code.startswith(("python\n", "py\n")):
            code = code.split("\n", 1)[1] if "\n" in code else code
    else:
        # Fallback: use the original solution as the pattern
        code = original_solution[:500]

    if desc_match:
        summary = desc_match.group(1).strip()
    else:
        # Fallback: extract meaningful summary from response
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        # Skip reasoning preamble and format labels
        skip_prefixes = (
            'okay', "let's", 'let me', 'hmm', 'the user', 'they',
            'pattern:', 'pattern', '```', 'description:',
            'so,', 'alright', 'i need', 'first,', 'looking',
        )
        summary_lines = [
            l for l in lines
            if not l.lower().startswith(skip_prefixes) and len(l) > 10
        ]
        summary = summary_lines[0] if summary_lines else "Extracted code pattern"

    return code, summary
