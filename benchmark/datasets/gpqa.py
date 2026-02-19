"""
GPQA Diamond dataset loader.

Downloads GPQA Diamond (198 graduate-level multiple choice questions) via
the HuggingFace rows API. Questions cover biology, physics, and chemistry
and are designed to be "Google-proof".

Source: https://huggingface.co/datasets/Idavidrein/gpqa
"""

import csv
import io
import json
import re
import urllib.request
from pathlib import Path
from typing import List, Optional

from .base import BaseDataset
from ..models import BenchmarkTask


# Regex cascade for answer extraction (from Artificial Analysis methodology)
_ANSWER_PATTERNS = [
    # Primary: "Answer: A" with optional formatting
    r'(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])',
    # Fallbacks
    r'\\boxed\{[^}]*([A-Z])[^}]*\}',
    r'answer is ([a-zA-Z])',
    r'answer is \\\(([a-zA-Z])',
    r'([A-Z])\)\s*[^A-Z]*$',
    r'([A-Z])\s+is\s+the\s+correct\s+answer',
    r'([A-Z])\s*$',
    r'([A-Z])\s*\.',
    r'([A-Z])\s*[^\w]',
]


def extract_mcq_answer(response: str) -> Optional[str]:
    """
    Extract a single-letter answer (A/B/C/D) from an LLM response.

    Uses a regex cascade following the Artificial Analysis methodology.
    Always takes the LAST match found (accounts for self-correction).

    Args:
        response: The full LLM response text.

    Returns:
        Single uppercase letter (A/B/C/D) or None if extraction fails.
    """
    # Strip thinking blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, response)
        if matches:
            answer = matches[-1].upper()
            if answer in ('A', 'B', 'C', 'D'):
                return answer

    return None


class GPQADiamondDataset(BaseDataset):
    """
    GPQA Diamond benchmark dataset.

    198 graduate-level multiple choice questions in biology, physics, and chemistry.
    Evaluation is simple: extract answer letter, compare to correct answer.
    """

    CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    FILENAME = "gpqa_diamond.csv"

    @property
    def name(self) -> str:
        return "gpqa_diamond"

    @property
    def expected_count(self) -> int:
        return 198

    def load(self) -> "GPQADiamondDataset":
        """Load with flexible count validation."""
        if self._loaded:
            return self

        filepath = self.download()
        self._tasks = self._parse(filepath)
        self._loaded = True

        if len(self._tasks) < 150:
            raise ValueError(
                f"Expected at least 150 tasks in {self.name}, "
                f"got {len(self._tasks)}"
            )

        return self

    def download(self) -> Path:
        """Download GPQA Diamond CSV from OpenAI's public blob storage."""
        filepath = self.cache_dir / self.FILENAME

        if filepath.exists():
            return filepath

        print("Downloading GPQA Diamond from openaipublic.blob.core.windows.net...")
        req = urllib.request.Request(self.CSV_URL)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
                with open(filepath, 'wb') as f:
                    f.write(content)
        except Exception as e:
            raise RuntimeError(f"Failed to download GPQA Diamond: {e}")

        # Count rows
        reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
        count = sum(1 for _ in reader)
        print(f"Downloaded {count} questions to {filepath}")
        return filepath

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """Parse the GPQA Diamond CSV file."""
        tasks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, data in enumerate(reader):
                task = self._convert_task(data, idx)
                if task is not None:
                    tasks.append(task)

        return tasks

    def _convert_task(self, data: dict, idx: int) -> Optional[BenchmarkTask]:
        """Convert a GPQA row to BenchmarkTask."""
        question = data.get("Question", "")
        if not question:
            return None

        # Extract answer options
        # GPQA Diamond has fields: Correct Answer, Incorrect Answer 1/2/3
        correct = data.get("Correct Answer", "")
        incorrect_1 = data.get("Incorrect Answer 1", "")
        incorrect_2 = data.get("Incorrect Answer 2", "")
        incorrect_3 = data.get("Incorrect Answer 3", "")

        if not correct or not incorrect_1:
            return None

        # The correct answer position is randomized in the dataset
        # We need to map to A/B/C/D
        options = [correct, incorrect_1, incorrect_2, incorrect_3]
        # Shuffle deterministically based on question hash
        import hashlib
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
        # Simple Fisher-Yates with deterministic seed
        rng_state = seed
        for i in range(len(options) - 1, 0, -1):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
            j = rng_state % (i + 1)
            options[i], options[j] = options[j], options[i]

        correct_letter = chr(65 + options.index(correct))  # A/B/C/D

        # Build prompt using AA methodology
        prompt = (
            f"Answer the following multiple choice question. The last line of "
            f"your response should be in the following format: "
            f"'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n"
            f"{question}\n\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}\n"
        )

        task_id = f"GPQA/{idx+1:03d}"

        # Determine domain from High-level domain or Subdomain field
        high_domain = data.get("High-level domain", "").lower()
        subdomain = data.get("Subdomain", "").lower()
        domain_str = high_domain or subdomain
        if "physic" in domain_str:
            domain = "physics"
        elif "bio" in domain_str or "molecular" in domain_str or "genetics" in domain_str:
            domain = "biology"
        elif "chem" in domain_str or "organic" in domain_str:
            domain = "chemistry"
        else:
            domain = "science"

        return BenchmarkTask(
            task_id=task_id,
            prompt=prompt,
            canonical_solution=correct_letter,
            test_code="",  # MCQ — evaluated by answer extraction
            entry_point="answer",
            category="gpqa_diamond",
            difficulty="hard",  # All GPQA Diamond are hard by design
            tags=["mcq", "science", domain],
            eval_mode="mcq",
            test_inputs=[],
            test_outputs=[correct_letter],
        )


if __name__ == "__main__":
    dataset = GPQADiamondDataset()
    dataset.load()
    print(dataset.summary())
    print(f"\nTotal questions: {len(dataset)}")
    if len(dataset) > 0:
        t = dataset[0]
        print(f"First task: {t.task_id}")
        print(f"Correct answer: {t.canonical_solution}")
        print(f"Prompt preview:\n{t.prompt[:500]}...")
