"""
HumanEval dataset loader.

Downloads and parses the HumanEval benchmark from OpenAI's release.
"""

import gzip
import json
import urllib.request
from pathlib import Path
from typing import List

from .base import BaseDataset
from ..models import BenchmarkTask


class HumanEvalDataset(BaseDataset):
    """
    HumanEval benchmark dataset.

    The HumanEval dataset consists of 164 hand-written Python programming problems
    with function signatures, docstrings, reference solutions, and unit tests.

    Source: https://github.com/openai/human-eval
    """

    DOWNLOAD_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
    FILENAME = "HumanEval.jsonl.gz"

    @property
    def name(self) -> str:
        """Dataset name."""
        return "humaneval"

    @property
    def expected_count(self) -> int:
        """Expected number of tasks."""
        return 164

    def download(self) -> Path:
        """
        Download the HumanEval dataset if not already cached.

        Returns:
            Path to the downloaded file.
        """
        filepath = self.cache_dir / self.FILENAME

        if filepath.exists():
            return filepath

        print(f"Downloading HumanEval dataset from {self.DOWNLOAD_URL}...")
        try:
            urllib.request.urlretrieve(self.DOWNLOAD_URL, filepath)
            print(f"Downloaded to {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to download HumanEval: {e}")

        return filepath

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """
        Parse the HumanEval dataset file.

        Args:
            filepath: Path to HumanEval.jsonl.gz

        Returns:
            List of BenchmarkTask objects.
        """
        tasks = []

        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                # The prompt in HumanEval is the function signature + docstring
                # We need to construct a proper completion prompt
                prompt = self._construct_prompt(data)

                # For the canonical solution, we need to combine the signature
                # (from prompt) with the solution body
                canonical_solution = self._build_full_solution(
                    data["prompt"],
                    data["canonical_solution"]
                )

                task = BenchmarkTask(
                    task_id=data["task_id"],
                    prompt=prompt,
                    canonical_solution=canonical_solution,
                    test_code=data["test"],
                    entry_point=data["entry_point"],
                    category="humaneval",
                    difficulty="medium",  # HumanEval doesn't have difficulty labels
                    tags=self._extract_tags(data)
                )
                tasks.append(task)

        return tasks

    def _construct_prompt(self, data: dict) -> str:
        """
        Construct a code completion prompt from HumanEval task data.

        The HumanEval 'prompt' field already contains the function signature
        and docstring. We format it as a clear code completion task.

        Args:
            data: Raw task data from HumanEval

        Returns:
            Formatted prompt string.
        """
        raw_prompt = data["prompt"]

        # The raw prompt is already well-formatted with function signature and docstring
        # We add a clear instruction for the model
        instruction = (
            "Complete the following Python function. "
            "Return ONLY the function implementation (no explanation, no markdown).\n\n"
        )

        return instruction + raw_prompt

    def _build_full_solution(self, prompt: str, solution_body: str) -> str:
        """
        Build the full solution by combining prompt (signature) with solution body.

        In HumanEval, the 'prompt' contains the function signature and docstring,
        while 'canonical_solution' contains just the function body.

        Args:
            prompt: Function signature and docstring
            solution_body: Function body (indented code)

        Returns:
            Complete function code
        """
        return prompt + solution_body

    def _extract_tags(self, data: dict) -> List[str]:
        """
        Extract tags from task data based on content analysis.

        Args:
            data: Raw task data

        Returns:
            List of descriptive tags.
        """
        tags = ["python", "function-completion"]

        prompt_lower = data["prompt"].lower()
        solution_lower = data.get("canonical_solution", "").lower()

        # Detect common patterns
        if "list" in prompt_lower or "array" in prompt_lower:
            tags.append("list")
        if "string" in prompt_lower:
            tags.append("string")
        if "sort" in prompt_lower or "sorted" in solution_lower:
            tags.append("sorting")
        if "recursive" in prompt_lower or "recursion" in prompt_lower:
            tags.append("recursion")
        if "dict" in prompt_lower or "dictionary" in prompt_lower:
            tags.append("dictionary")
        if "math" in prompt_lower or any(op in solution_lower for op in ["sqrt", "pow", "sin", "cos"]):
            tags.append("math")

        return tags


def load_humaneval(cache_dir: Path = None) -> HumanEvalDataset:
    """
    Convenience function to load the HumanEval dataset.

    Args:
        cache_dir: Optional cache directory override.

    Returns:
        Loaded HumanEvalDataset instance.
    """
    dataset = HumanEvalDataset(cache_dir=cache_dir)
    dataset.load()
    return dataset


if __name__ == "__main__":
    # Test loading when run directly
    dataset = load_humaneval()
    print(dataset.summary())
    print(f"\nFirst task: {dataset[0].task_id}")
    print(f"Entry point: {dataset[0].entry_point}")
    print(f"Prompt preview:\n{dataset[0].prompt[:200]}...")
