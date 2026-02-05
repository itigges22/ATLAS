"""
MBPP (Mostly Basic Python Programming) dataset loader.

Downloads and parses the MBPP-S (sanitized) benchmark using the canonical
3-shot prompt format from the original paper.
"""

import json
import urllib.request
from pathlib import Path
from typing import List, Dict

from .base import BaseDataset
from ..models import BenchmarkTask


# Canonical 3-shot prompt pool task IDs (from original MBPP paper)
PROMPT_POOL_IDS = {2, 3, 4}


class MBPPDataset(BaseDataset):
    """
    MBPP benchmark dataset (sanitized version) with 3-shot prompting.

    MBPP-S consists of ~397 crowd-sourced Python programming problems
    with task descriptions, test cases, and reference solutions.

    Uses the canonical 3-shot format: tasks 2, 3, 4 as few-shot examples
    followed by the target task description and tests.

    Source: https://github.com/google-research/google-research/tree/master/mbpp
    """

    # MBPP sanitized dataset URL
    DOWNLOAD_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    FILENAME = "mbpp.jsonl"

    @property
    def name(self) -> str:
        """Dataset name."""
        return "mbpp"

    @property
    def expected_count(self) -> int:
        """Expected number of tasks in MBPP-S."""
        return 500  # Standard test split (tasks 11-510)

    def __init__(self, cache_dir: Path = None):
        super().__init__(cache_dir=cache_dir)
        self._prompt_pool: Dict[int, dict] = {}  # Tasks 2/3/4 for 3-shot examples

    def download(self) -> Path:
        """
        Download the MBPP dataset if not already cached.

        Returns:
            Path to the downloaded file.
        """
        filepath = self.cache_dir / self.FILENAME

        if filepath.exists():
            return filepath

        print(f"Downloading MBPP dataset from {self.DOWNLOAD_URL}...")
        try:
            urllib.request.urlretrieve(self.DOWNLOAD_URL, filepath)
            print(f"Downloaded to {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to download MBPP: {e}")

        return filepath

    def load(self) -> "MBPPDataset":
        """
        Load the dataset (download if necessary and parse).

        Overridden to allow flexible task count for MBPP.

        Returns:
            Self for method chaining.
        """
        if self._loaded:
            return self

        filepath = self.download()
        self._tasks = self._parse(filepath)
        self._loaded = True

        # MBPP has variable counts, just ensure we got a reasonable number
        if len(self._tasks) < 300:
            raise ValueError(
                f"Expected at least 300 tasks in {self.name}, "
                f"got {len(self._tasks)}"
            )

        return self

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """
        Parse the MBPP dataset file.

        Captures tasks 2/3/4 for the 3-shot prompt pool, then converts
        tasks 11-510 as the evaluation set.

        Args:
            filepath: Path to mbpp.jsonl

        Returns:
            List of BenchmarkTask objects.
        """
        tasks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                task_id = data.get("task_id", 0)

                # Capture prompt pool tasks (2, 3, 4) for 3-shot examples
                if task_id in PROMPT_POOL_IDS:
                    self._prompt_pool[task_id] = data
                    continue

                # Only include test split (tasks 11-510)
                if task_id < 11 or task_id > 510:
                    continue

                task = self._convert_task(data)
                tasks.append(task)

        return tasks

    def _convert_task(self, data: dict) -> BenchmarkTask:
        """
        Convert raw MBPP data to BenchmarkTask.

        Args:
            data: Raw task data from MBPP

        Returns:
            BenchmarkTask object.
        """
        task_id = f"MBPP/{data['task_id']}"
        text = data["text"]  # Task description
        code = data["code"]  # Reference solution
        test_list = data.get("test_list", [])

        # Construct 3-shot prompt
        prompt = self._construct_prompt(text, test_list)

        # Extract entry point from the reference code
        entry_point = self._extract_entry_point(code)

        # Construct test code from test assertions
        test_code = self._construct_test_code(test_list, entry_point)

        return BenchmarkTask(
            task_id=task_id,
            prompt=prompt,
            canonical_solution=code,
            test_code=test_code,
            entry_point=entry_point,
            category="mbpp",
            difficulty=self._estimate_difficulty(text, code),
            tags=self._extract_tags(text, code)
        )

    def _format_example(self, data: dict) -> str:
        """
        Format a single 3-shot example in canonical MBPP format.

        Args:
            data: Raw task data with text, test_list, and code fields.

        Returns:
            Formatted example string with [BEGIN]/[DONE] delimiters.
        """
        desc = data["text"]
        tests = "\n".join(data.get("test_list", []))
        code = data["code"]
        return (
            f"You are an expert Python programmer, and here is your task: "
            f"{desc} Your code should pass these tests:\n\n{tests}\n"
            f"[BEGIN]\n{code}\n[DONE]"
        )

    def _construct_prompt(self, text: str, test_list: List[str]) -> str:
        """
        Construct a 3-shot prompt in the canonical MBPP format.

        Three examples (tasks 2/3/4) with [BEGIN]/[DONE] delimiters,
        followed by the actual task (without delimiters) for the model
        to complete.

        Args:
            text: Task description
            test_list: List of test assertion strings for the task

        Returns:
            Formatted 3-shot prompt string.
        """
        parts = []

        # Add 3-shot examples in order (tasks 2, 3, 4)
        for tid in sorted(PROMPT_POOL_IDS):
            if tid in self._prompt_pool:
                parts.append(self._format_example(self._prompt_pool[tid]))

        # Add the actual task (no [BEGIN]/[DONE] — model should generate the code)
        tests_str = "\n".join(test_list)
        parts.append(
            f"You are an expert Python programmer, and here is your task: "
            f"{text} Your code should pass these tests:\n\n{tests_str}\n"
            f"[BEGIN]\n"
        )

        return "\n\n".join(parts)

    def _extract_signature(self, code: str) -> str:
        """
        Extract function signature from reference code.

        Args:
            code: Reference solution code

        Returns:
            Function signature or empty string.
        """
        lines = code.strip().split('\n')
        for line in lines:
            if line.strip().startswith('def '):
                # Return just the def line
                return line.strip()
        return ""

    def _extract_entry_point(self, code: str) -> str:
        """
        Extract the main function name from reference code.

        Args:
            code: Reference solution code

        Returns:
            Function name.
        """
        lines = code.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                # Extract function name
                name_part = line[4:]  # Remove 'def '
                paren_idx = name_part.find('(')
                if paren_idx > 0:
                    return name_part[:paren_idx].strip()
        return "solution"

    def _construct_test_code(self, test_list: List[str], entry_point: str) -> str:
        """
        Construct test code from MBPP test assertions.

        Args:
            test_list: List of test assertion strings
            entry_point: Function name being tested

        Returns:
            Python test code string.
        """
        if not test_list:
            return f"# No tests provided\nassert {entry_point} is not None"

        # MBPP tests are already in assertion format
        test_code = "\n".join(test_list)
        return test_code

    def _estimate_difficulty(self, text: str, code: str) -> str:
        """
        Estimate task difficulty based on description and solution.

        Args:
            text: Task description
            code: Reference solution

        Returns:
            Difficulty string: 'easy', 'medium', or 'hard'
        """
        code_lines = len(code.strip().split('\n'))
        text_len = len(text)

        # Simple heuristic based on solution length
        if code_lines <= 5 and text_len < 100:
            return "easy"
        elif code_lines <= 15 or text_len < 200:
            return "medium"
        else:
            return "hard"

    def _extract_tags(self, text: str, code: str) -> List[str]:
        """
        Extract tags based on content analysis.

        Args:
            text: Task description
            code: Reference solution

        Returns:
            List of descriptive tags.
        """
        tags = ["python"]

        text_lower = text.lower()
        code_lower = code.lower()

        if "list" in text_lower or "array" in text_lower:
            tags.append("list")
        if "string" in text_lower:
            tags.append("string")
        if "sort" in text_lower:
            tags.append("sorting")
        if "recursive" in text_lower or "def " in code and code.count("def ") > 1:
            tags.append("recursion")
        if "dict" in text_lower or "dictionary" in text_lower:
            tags.append("dictionary")
        if any(word in text_lower for word in ["sum", "count", "average", "max", "min"]):
            tags.append("math")
        if "tuple" in text_lower:
            tags.append("tuple")

        return tags


def load_mbpp(cache_dir: Path = None) -> MBPPDataset:
    """
    Convenience function to load the MBPP dataset.

    Args:
        cache_dir: Optional cache directory override.

    Returns:
        Loaded MBPPDataset instance.
    """
    dataset = MBPPDataset(cache_dir=cache_dir)
    dataset.load()
    return dataset


if __name__ == "__main__":
    # Test loading when run directly
    dataset = load_mbpp()
    print(dataset.summary())
    print(f"\nFirst task: {dataset[0].task_id}")
    print(f"Entry point: {dataset[0].entry_point}")
    print(f"Prompt preview:\n{dataset[0].prompt[:300]}...")
