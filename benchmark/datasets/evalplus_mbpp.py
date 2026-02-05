"""
MBPP+ (EvalPlus) dataset loader.

Downloads MBPP+ from the EvalPlus project via the HuggingFace rows API.
MBPP+ augments the original MBPP problems with more rigorous test cases.
Uses the canonical 3-shot prompt format.

Source: https://huggingface.co/datasets/evalplus/mbppplus
"""

import json
import urllib.request
from pathlib import Path
from typing import List, Dict

from .base import BaseDataset
from ..models import BenchmarkTask


# Canonical 3-shot examples from the original MBPP paper (tasks 2, 3, 4).
# Embedded here so MBPP+ doesn't need to download the original MBPP dataset.
_PROMPT_POOL = [
    {
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
        "code": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "  res = tuple(set(test_tup1) & set(test_tup2))\n"
            "  return (res)"
        ),
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "code": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    result = False\n"
            "    for i in range(2,int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n"
            "            result = True\n"
            "    return result"
        ),
    },
    {
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
        "code": (
            "import heapq as hq\n"
            "def heap_queue_largest(nums,n):\n"
            "  largest_nums = hq.nlargest(n, nums)\n"
            "  return largest_nums"
        ),
    },
]


class MBPPPlusDataset(BaseDataset):
    """
    MBPP+ benchmark dataset (EvalPlus augmented version) with 3-shot prompting.

    Provides 378 tasks from MBPP with significantly more test cases
    for more rigorous evaluation.
    """

    ROWS_API = "https://datasets-server.huggingface.co/rows"
    DATASET_ID = "evalplus/mbppplus"
    FILENAME = "mbppplus.jsonl"

    @property
    def name(self) -> str:
        return "mbpp_plus"

    @property
    def expected_count(self) -> int:
        return 378

    def load(self) -> "MBPPPlusDataset":
        """Load with flexible count validation."""
        if self._loaded:
            return self

        filepath = self.download()
        self._tasks = self._parse(filepath)
        self._loaded = True

        if len(self._tasks) < 350:
            raise ValueError(
                f"Expected at least 350 tasks in {self.name}, "
                f"got {len(self._tasks)}"
            )

        return self

    def download(self) -> Path:
        """Download MBPP+ via HuggingFace rows API and cache as JSONL."""
        filepath = self.cache_dir / self.FILENAME

        if filepath.exists():
            return filepath

        print(f"Downloading MBPP+ dataset from HuggingFace...")
        rows = []

        # Fetch in batches of 100 (378 tasks = 4 requests)
        for offset in range(0, 400, 100):
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
                raise RuntimeError(f"Failed to download MBPP+ (offset={offset}): {e}")

        with open(filepath, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row.get("row", row)) + "\n")

        print(f"Downloaded {len(rows)} tasks to {filepath}")
        return filepath

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """Parse the cached MBPP+ JSONL file."""
        tasks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                task = self._convert_task(data)
                tasks.append(task)

        return tasks

    def _format_example(self, data: dict) -> str:
        """Format a single 3-shot example in canonical MBPP format."""
        desc = data["text"]
        tests = "\n".join(data.get("test_list", []))
        code = data["code"]
        return (
            f"You are an expert Python programmer, and here is your task: "
            f"{desc} Your code should pass these tests:\n\n{tests}\n"
            f"[BEGIN]\n{code}\n[DONE]"
        )

    def _construct_prompt(self, text: str, test_list: List[str]) -> str:
        """Construct a 3-shot prompt using embedded prompt pool examples."""
        parts = []

        for example in _PROMPT_POOL:
            parts.append(self._format_example(example))

        tests_str = "\n".join(test_list)
        parts.append(
            f"You are an expert Python programmer, and here is your task: "
            f"{text} Your code should pass these tests:\n\n{tests_str}\n"
            f"[BEGIN]\n"
        )

        return "\n\n".join(parts)

    def _convert_task(self, data: dict) -> BenchmarkTask:
        """Convert an MBPP+ row to BenchmarkTask."""
        task_id = data.get("task_id", "")
        if isinstance(task_id, int):
            task_id = f"MBPPPlus/{task_id}"
        elif not task_id.startswith("MBPPPlus/"):
            task_id = f"MBPPPlus/{task_id}"

        text = data.get("prompt", data.get("text", ""))
        code = data.get("canonical_solution", data.get("code", ""))
        test_list = data.get("test_list", [])

        # Build the 3-shot prompt
        prompt = self._construct_prompt(text, test_list)

        # Extract entry point
        entry_point = self._extract_entry_point(code)

        # Build test code — MBPP+ `test` field contains a `check()` function
        test_code = data.get("test", "")
        test_imports = data.get("test_imports", [])
        if test_imports:
            imports_str = "\n".join(test_imports)
            test_code = imports_str + "\n" + test_code

        # If test field has check() function, append invocation
        if "def check(" in test_code and entry_point:
            test_code = test_code + f"\ncheck({entry_point})\n"

        # Fallback: use test_list assertions directly
        if not test_code.strip() and test_list:
            test_code = "\n".join(test_list)

        tags = ["python"]
        text_lower = text.lower()
        if "list" in text_lower or "array" in text_lower:
            tags.append("list")
        if "string" in text_lower:
            tags.append("string")
        if "sort" in text_lower:
            tags.append("sorting")
        if "dict" in text_lower:
            tags.append("dictionary")

        return BenchmarkTask(
            task_id=task_id,
            prompt=prompt,
            canonical_solution=code,
            test_code=test_code,
            entry_point=entry_point,
            category="mbpp_plus",
            difficulty=self._estimate_difficulty(text, code),
            tags=tags,
        )

    def _extract_entry_point(self, code: str) -> str:
        """Extract the main function name from reference code."""
        for line in code.strip().split('\n'):
            line = line.strip()
            if line.startswith('def '):
                name_part = line[4:]
                paren_idx = name_part.find('(')
                if paren_idx > 0:
                    return name_part[:paren_idx].strip()
        return "solution"

    def _estimate_difficulty(self, text: str, code: str) -> str:
        """Estimate difficulty based on description and solution length."""
        code_lines = len(code.strip().split('\n'))
        text_len = len(text)

        if code_lines <= 5 and text_len < 100:
            return "easy"
        elif code_lines <= 15 or text_len < 200:
            return "medium"
        else:
            return "hard"


if __name__ == "__main__":
    dataset = MBPPPlusDataset()
    dataset.load()
    print(dataset.summary())
    print(f"\nFirst task: {dataset[0].task_id}")
    print(f"Entry point: {dataset[0].entry_point}")
    print(f"Prompt preview:\n{dataset[0].prompt[:500]}...")
