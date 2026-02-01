"""
Provenance tracking for indexed content.
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class ContentSource(Enum):
    HUMAN = "human"
    AI_VERIFIED = "ai_verified"
    AI_UNVERIFIED = "ai_unverified"


# AI markers to detect in commit messages
AI_MARKERS = ["[AI]", "[AUTO]", "[GENERATED]", "[BOT]"]


def is_ai_generated(commit_message: str) -> bool:
    """
    Check if a commit message indicates AI-generated content.

    Detects markers: [AI], [AUTO], [GENERATED], [BOT]

    Args:
        commit_message: The commit message to check

    Returns:
        True if AI markers found, False otherwise
    """
    if not commit_message:
        return False
    message_lower = commit_message.lower()
    for marker in AI_MARKERS:
        if marker.lower() in message_lower:
            return True
    return False


def detect_ai_markers(commit_message: str) -> list:
    """
    Detect which AI markers are present in a commit message.

    Args:
        commit_message: The commit message to check

    Returns:
        List of markers found (e.g., ["[AI]", "[BOT]"])
    """
    if not commit_message:
        return []
    message_lower = commit_message.lower()
    found = []
    for marker in AI_MARKERS:
        if marker.lower() in message_lower:
            found.append(marker)
    return found

def get_file_provenance(file_path: Path, repo_root: Path) -> Dict:
    """
    Extract provenance information from git history.

    Returns dict with source, author, commit info.
    """
    try:
        rel_path = file_path.relative_to(repo_root)

        # Get last commit info for file
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H|%ae|%aI|%s", "--", str(rel_path)],
            cwd=repo_root,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return _default_provenance()

        parts = result.stdout.strip().split("|")
        if len(parts) < 4:
            return _default_provenance()

        commit_sha, author, date, subject = parts[0], parts[1], parts[2], "|".join(parts[3:])

        # Determine source based on commit message conventions
        source = _classify_source(subject, author)

        # Check if file was merged via PR (check for merge commit)
        merged_via_pr = _check_merged_via_pr(commit_sha, repo_root)

        return {
            "source": source.value,
            "author": author,
            "commit_sha": commit_sha,
            "merged_at": date,
            "merged_via_pr": merged_via_pr,
            "commit_subject": subject
        }

    except Exception as e:
        logger.warning(f"Failed to get provenance for {file_path}: {e}")
        return _default_provenance()

def _classify_source(commit_subject: str, author: str) -> ContentSource:
    """
    Classify content source based on commit message.

    Convention: AI-generated commits include "[AI]" or "[AUTO]" prefix.
    """
    subject_lower = commit_subject.lower()

    # Check for AI markers
    ai_markers = ["[ai]", "[auto]", "[generated]", "[bot]", "auto-generated"]
    for marker in ai_markers:
        if marker in subject_lower:
            return ContentSource.AI_VERIFIED

    # Check for bot authors
    bot_authors = ["dependabot", "renovate", "github-actions"]
    for bot in bot_authors:
        if bot in author.lower():
            return ContentSource.AI_VERIFIED

    return ContentSource.HUMAN

def _check_merged_via_pr(commit_sha: str, repo_root: Path) -> bool:
    """Check if commit was part of a merged PR."""
    try:
        # Check if commit is reachable only through merge commits
        result = subprocess.run(
            ["git", "log", "--merges", "--ancestry-path", f"{commit_sha}..HEAD", "--oneline"],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip()) > 0
    except:
        return False

def _default_provenance() -> Dict:
    """Default provenance for files without git history."""
    return {
        "source": ContentSource.HUMAN.value,
        "author": "unknown",
        "commit_sha": None,
        "merged_at": None,
        "merged_via_pr": False
    }

def calculate_quality_score(file_path: Path) -> float:
    """
    Calculate quality score for a file.

    Combines:
    - Lint score (40%)
    - Test coverage if available (40%)
    - Code complexity (20%)
    """
    scores = []
    weights = []

    # Lint score
    lint_score = run_lint_check(file_path)
    if lint_score is not None:
        scores.append(lint_score / 10.0)  # Normalize to 0-1
        weights.append(0.4)

    # Test coverage (would need coverage data)
    # For now, assume 0.7 if file has corresponding test file
    test_file = file_path.parent / f"test_{file_path.name}"
    if test_file.exists():
        scores.append(0.7)
        weights.append(0.4)

    # Complexity (simplified: penalize very long files)
    try:
        lines = len(file_path.read_text().splitlines())
        complexity_score = max(0, 1 - (lines / 1000))  # Penalize >1000 lines
        scores.append(complexity_score)
        weights.append(0.2)
    except:
        pass

    if not scores:
        return 0.5  # Default

    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight

def run_lint_check(file_path: Path) -> Optional[float]:
    """Run pylint and return score (0-10)."""
    try:
        result = subprocess.run(
            ["python", "-m", "pylint", "--score=y", "--exit-zero", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        match = re.search(r'rated at ([\d.]+)/10', result.stdout)
        if match:
            return float(match.group(1))
    except:
        pass
    return None
