"""
HumanEval+ (EvalPlus) dataset loader.

Downloads HumanEval+ from the EvalPlus project via the HuggingFace rows API.
HumanEval+ augments the original 164 HumanEval problems with 80x more tests.

Source: https://huggingface.co/datasets/evalplus/humanevalplus
"""

import json
import urllib.request
from pathlib import Path
from typing import List

from .base import BaseDataset
from ..models import BenchmarkTask


class HumanEvalPlusDataset(BaseDataset):
    """
    HumanEval+ benchmark dataset (EvalPlus augmented version).

    Provides the same 164 tasks as HumanEval but with significantly more
    test cases for more rigorous evaluation.
    """

    ROWS_API = "https://datasets-server.huggingface.co/rows"
    DATASET_ID = "evalplus/humanevalplus"
    FILENAME = "humanevalplus.jsonl"

    @property
    def name(self) -> str:
        return "humaneval_plus"

    @property
    def expected_count(self) -> int:
        return 164

    def download(self) -> Path:
        """Download HumanEval+ via HuggingFace rows API and cache as JSONL."""
        filepath = self.cache_dir / self.FILENAME

        if filepath.exists():
            return filepath

        print(f"Downloading HumanEval+ dataset from HuggingFace...")
        rows = []

        # Fetch in batches of 100 (164 tasks = 2 requests)
        for offset in range(0, 200, 100):
            url = (
                f"{self.ROWS_API}?dataset={self.DATASET_ID}"
                f"&config=default&split=test&offset={offset}&length=100"
            )
            req = urllib.request.Request(url)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    batch = data.get("rows", [])
                    rows.extend(batch)
                    if len(batch) < 100:
                        break
            except Exception as e:
                raise RuntimeError(f"Failed to download HumanEval+ (offset={offset}): {e}")

        # Write to cache as JSONL
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row.get("row", row)) + "\n")

        print(f"Downloaded {len(rows)} tasks to {filepath}")
        return filepath

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """Parse the cached HumanEval+ JSONL file."""
        tasks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                task = self._convert_task(data)
                tasks.append(task)

        return tasks

    def _convert_task(self, data: dict) -> BenchmarkTask:
        """Convert a HumanEval+ row to BenchmarkTask."""
        task_id = data.get("task_id", "")
        prompt_code = data.get("prompt", "")
        entry_point = data.get("entry_point", "")
        canonical_solution = prompt_code + data.get("canonical_solution", "")

        # Build test code from the `test` field
        # The test field contains `def check(candidate):` with assertions
        # We need to append `check(<entry_point>)` to invoke it
        test_code = data.get("test", "")
        if test_code and entry_point:
            test_code = test_code + f"\ncheck({entry_point})\n"

        # Also include the plus tests if available
        plus_tests = data.get("plus", "")
        if plus_tests:
            # plus field is typically a list of additional test inputs
            # For EvalPlus, the `test` field already includes augmented tests
            # in the HuggingFace version, so we use `test` directly
            pass

        # Build the prompt — same style as original HumanEval
        instruction = (
            "Complete the following Python function. "
            "Return ONLY the function implementation (no explanation, no markdown).\n\n"
        )
        prompt = instruction + prompt_code

        tags = ["python", "function-completion"]
        prompt_lower = prompt_code.lower()
        if "list" in prompt_lower or "array" in prompt_lower:
            tags.append("list")
        if "string" in prompt_lower:
            tags.append("string")
        if "dict" in prompt_lower:
            tags.append("dictionary")

        return BenchmarkTask(
            task_id=task_id,
            prompt=prompt,
            canonical_solution=canonical_solution,
            test_code=test_code,
            entry_point=entry_point,
            category="humaneval_plus",
            difficulty="medium",
            tags=tags,
        )


if __name__ == "__main__":
    dataset = HumanEvalPlusDataset()
    dataset.load()
    print(dataset.summary())
    print(f"\nFirst task: {dataset[0].task_id}")
    print(f"Entry point: {dataset[0].entry_point}")
    print(f"Prompt preview:\n{dataset[0].prompt[:300]}...")
