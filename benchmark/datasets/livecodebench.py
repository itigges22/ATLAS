"""
LiveCodeBench dataset loader.

Downloads LiveCodeBench code generation problems via the HuggingFace rows API.
LiveCodeBench contains competitive programming problems from LeetCode,
Codeforces, and AtCoder with stdin/stdout evaluation.

Source: https://huggingface.co/datasets/livecodebench/code_generation_lite
"""

import json
import urllib.request
from pathlib import Path
from typing import List, Optional

from .base import BaseDataset
from ..models import BenchmarkTask


# Maximum number of test cases per problem (for memory/time)
MAX_TESTS_PER_PROBLEM = 50


class LiveCodeBenchDataset(BaseDataset):
    """
    LiveCodeBench benchmark dataset (release_v5).

    Contains competitive programming problems with two evaluation modes:
    - Problems with starter_code: function completion (LeetCode-style)
    - Problems without starter_code: full stdin/stdout scripts (Codeforces/AtCoder-style)

    All problems use stdio evaluation (eval_mode="stdio").
    """

    ROWS_API = "https://datasets-server.huggingface.co/rows"
    # Primary source is livecodebench/code_generation_lite, but it has known
    # loading issues. Fall back to the bzantium mirror per AA methodology.
    DATASET_IDS = [
        "livecodebench/code_generation_lite",
        "bzantium/livecodebench",
    ]
    CONFIG = "release_v5"
    FILENAME = "livecodebench_v5.jsonl"

    @property
    def name(self) -> str:
        return "livecodebench"

    @property
    def expected_count(self) -> int:
        return 880  # Approximate count for release_v5

    def load(self) -> "LiveCodeBenchDataset":
        """Load with flexible count validation for LCB."""
        if self._loaded:
            return self

        filepath = self.download()
        self._tasks = self._parse(filepath)
        self._loaded = True

        # LCB count varies by release; accept a wide range
        if len(self._tasks) < 100:
            raise ValueError(
                f"Expected at least 100 tasks in {self.name}, "
                f"got {len(self._tasks)}"
            )

        return self

    def download(self) -> Path:
        """Download LiveCodeBench via HuggingFace rows API and cache as JSONL."""
        filepath = self.cache_dir / self.FILENAME

        if filepath.exists():
            return filepath

        # Try each dataset source in order
        for dataset_id in self.DATASET_IDS:
            print(f"Downloading LiveCodeBench (release_v5) from {dataset_id}...")
            rows = []

            offset = 0
            while True:
                url = (
                    f"{self.ROWS_API}?dataset={dataset_id}"
                    f"&config={self.CONFIG}&split=test&offset={offset}&length=100"
                )
                req = urllib.request.Request(url)
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        batch = data.get("rows", [])
                        rows.extend(batch)
                        if len(batch) < 100:
                            break
                        offset += 100
                except Exception as e:
                    if rows:
                        print(f"Warning: partial download ({len(rows)} rows at offset {offset}): {e}")
                        break
                    print(f"  Failed: {e}")
                    rows = []
                    break

            if rows:
                with open(filepath, 'w', encoding='utf-8') as f:
                    for row in rows:
                        f.write(json.dumps(row.get("row", row)) + "\n")
                print(f"Downloaded {len(rows)} tasks to {filepath}")
                return filepath

        raise RuntimeError("Failed to download LiveCodeBench from any source")

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """Parse the cached LiveCodeBench JSONL file."""
        tasks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                task = self._convert_task(data)
                if task is not None:
                    tasks.append(task)

        return tasks

    def _convert_task(self, data: dict) -> Optional[BenchmarkTask]:
        """Convert a LiveCodeBench row to BenchmarkTask with stdio evaluation."""
        question_id = data.get("question_id", data.get("task_id", ""))
        task_id = f"LCB/{question_id}"

        question_content = data.get("question_content", "")
        if not question_content:
            return None

        starter_code = data.get("starter_code", "")
        has_starter = bool(starter_code and starter_code.strip())

        # Build prompt
        prompt = self._construct_prompt(question_content, starter_code, has_starter)

        # Extract test cases from private_test_cases or public_test_cases
        test_inputs, test_outputs = self._extract_test_cases(data)
        if not test_inputs:
            return None

        # For problems with starter_code, the canonical solution wraps the function
        # For stdin/stdout problems, canonical solution is the full script
        canonical_solution = data.get("canonical_solution", "")

        # Difficulty from metadata
        difficulty_raw = data.get("difficulty", "medium")
        if isinstance(difficulty_raw, str):
            difficulty_raw = difficulty_raw.lower()
        difficulty = self._normalize_difficulty(difficulty_raw)

        # Entry point — for function problems, extract from starter code
        entry_point = "solution"
        if has_starter:
            entry_point = self._extract_entry_point(starter_code)

        # Tags from platform/topic
        tags = ["python", "competitive-programming"]
        platform = data.get("platform", "")
        if platform:
            tags.append(platform.lower())

        return BenchmarkTask(
            task_id=task_id,
            prompt=prompt,
            canonical_solution=canonical_solution,
            test_code="",  # Not used for stdio mode
            entry_point=entry_point,
            category="livecodebench",
            difficulty=difficulty,
            tags=tags,
            eval_mode="stdio",
            test_inputs=test_inputs,
            test_outputs=test_outputs,
        )

    def _construct_prompt(
        self, question: str, starter_code: str, has_starter: bool
    ) -> str:
        """
        Build a LiveCodeBench prompt using Artificial Analysis methodology.

        For function-completion problems: includes starter code with format instructions.
        For stdin/stdout problems: instructs to read from stdin and write to stdout.
        """
        if has_starter:
            return (
                f"### Question:\n{question}\n\n"
                f"### Format: You will use the following starter code to write the "
                f"solution to the problem and enclose your code within delimiters.\n"
                f"```python\n{starter_code}\n```\n\n"
                f"### Answer: (use the provided format with backticks)\n"
            )
        else:
            return (
                f"### Question:\n{question}\n\n"
                f"### Format: Read the inputs from stdin solve the problem and write "
                f"the answer to stdout (do not directly test on the sample inputs). "
                f"Enclose your code within delimiters as follows. Ensure that when the "
                f"python program runs, it reads the inputs, runs the algorithm and "
                f"writes output to STDOUT.\n"
                f"```python\n# YOUR CODE HERE\n```\n\n"
                f"### Answer: (use the provided format with backticks)\n"
            )

    def _extract_test_cases(self, data: dict):
        """
        Extract stdin/stdout test cases from the problem data.

        Returns:
            Tuple of (test_inputs, test_outputs) lists.
        """
        test_inputs = []
        test_outputs = []

        # Try private_test_cases first (more comprehensive)
        for field_name in ("private_test_cases", "public_test_cases"):
            cases = data.get(field_name, None)
            if not cases:
                continue

            # Cases might be a JSON string or already parsed
            if isinstance(cases, str):
                try:
                    cases = json.loads(cases)
                except (json.JSONDecodeError, TypeError):
                    continue

            if isinstance(cases, list):
                for case in cases[:MAX_TESTS_PER_PROBLEM]:
                    inp = case.get("input", "")
                    out = case.get("output", case.get("expected_output", ""))
                    if inp is not None and out is not None:
                        test_inputs.append(str(inp))
                        test_outputs.append(str(out))

            if test_inputs:
                break

        return test_inputs, test_outputs

    def _extract_entry_point(self, starter_code: str) -> str:
        """Extract function name from starter code."""
        for line in starter_code.strip().split('\n'):
            line = line.strip()
            if line.startswith('def '):
                name_part = line[4:]
                paren_idx = name_part.find('(')
                if paren_idx > 0:
                    return name_part[:paren_idx].strip()
        return "solution"

    def _normalize_difficulty(self, raw: str) -> str:
        """Normalize difficulty strings to easy/medium/hard."""
        if raw in ("easy", "simple", "basic"):
            return "easy"
        elif raw in ("medium", "moderate", "intermediate"):
            return "medium"
        elif raw in ("hard", "difficult", "advanced"):
            return "hard"
        return "medium"


if __name__ == "__main__":
    dataset = LiveCodeBenchDataset()
    dataset.load()
    print(dataset.summary())
    print(f"\nTotal tasks: {len(dataset)}")
    if len(dataset) > 0:
        t = dataset[0]
        print(f"First task: {t.task_id}")
        print(f"Eval mode: {t.eval_mode}")
        print(f"Test cases: {len(t.test_inputs)}")
        print(f"Prompt preview:\n{t.prompt[:400]}...")
