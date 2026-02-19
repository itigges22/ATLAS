"""
SciCode dataset loader.

Downloads SciCode from the HuggingFace rows API. SciCode tests scientific
computing abilities by decomposing problems into sub-steps, where each step
builds on previous ground-truth solutions.

Target values for test assertions are stored in a separate HDF5 file
(test_data.h5) from the official SciCode repo. The file must be downloaded
from Google Drive and placed in the cache directory.

Source: https://huggingface.co/datasets/SciCode1/SciCode
HDF5 targets: https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseDataset
from ..models import BenchmarkTask


def _check_numpy_available() -> bool:
    """Check if numpy is available in the Python execution environment."""
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import numpy'],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _load_h5_targets(h5_path: Path) -> Dict[str, List[Any]]:
    """Load all target values from the SciCode test_data.h5 file.

    Returns a dict mapping step_id (e.g. '77.1') to a list of target values,
    one per test case.
    """
    import h5py
    import numpy as np

    targets = {}
    with h5py.File(h5_path, 'r') as f:
        for step_id in f.keys():
            step_group = f[step_id]
            step_targets = []
            # Tests are named test1, test2, ... in order
            test_keys = sorted(step_group.keys(), key=lambda k: int(k.replace('test', '')))
            for tk in test_keys:
                test_group = step_group[tk]
                var_keys = sorted(test_group.keys())
                if len(var_keys) == 1:
                    ds = test_group[var_keys[0]]
                    if isinstance(ds, h5py.Dataset):
                        val = ds[()]
                        if isinstance(val, bytes):
                            step_targets.append(val.decode('utf-8'))
                        else:
                            step_targets.append(val)
                    elif isinstance(ds, h5py.Group):
                        # Could be a list or sparse matrix — skip for now
                        step_targets.append(None)
                else:
                    # Multiple variables — return as tuple
                    var_vals = []
                    for vk in var_keys:
                        ds = test_group[vk]
                        if isinstance(ds, h5py.Dataset):
                            val = ds[()]
                            if isinstance(val, bytes):
                                var_vals.append(val.decode('utf-8'))
                            else:
                                var_vals.append(val)
                        else:
                            var_vals.append(None)
                    step_targets.append(tuple(var_vals))
            targets[step_id] = step_targets
    return targets


def _serialize_target(val) -> str:
    """Serialize a target value as a Python expression string.

    Handles numpy arrays, scalars, tuples, and strings.
    """
    import numpy as np

    if val is None:
        return "None"
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, np.ndarray):
        # Use full precision repr
        return "np.array(%s, dtype=np.%s)" % (
            np.array2string(val, separator=', ', threshold=10000, max_line_width=10000),
            val.dtype,
        )
    if isinstance(val, (np.integer, np.floating, np.complexfloating)):
        return repr(val.item())
    if isinstance(val, tuple):
        parts = [_serialize_target(v) for v in val]
        if len(parts) == 1:
            return "(%s,)" % parts[0]
        return "(%s)" % ", ".join(parts)
    # Fallback: use repr
    return repr(val)


class SciCodeDataset(BaseDataset):
    """
    SciCode benchmark dataset.

    Each problem is decomposed into sub-steps (~338 total across ~80 problems).
    For each sub-step, the prompt includes:
    - Required dependencies
    - Problem description
    - Previous steps' ground truth code
    - Current step description and function header

    Tests use numpy (np.allclose) for numerical comparison.
    """

    ROWS_API = "https://datasets-server.huggingface.co/rows"
    DATASET_ID = "SciCode1/SciCode"
    FILENAME_TEST = "scicode_test.jsonl"
    FILENAME_VALIDATION = "scicode_validation.jsonl"
    FILENAME_COMBINED = "scicode_combined.jsonl"
    H5_SUBDIR = "SciCode_test"
    H5_FILENAME = "test_data.h5"

    @property
    def name(self) -> str:
        return "scicode"

    @property
    def expected_count(self) -> int:
        return 338  # Approximate: total sub-steps across all problems

    def load(self) -> "SciCodeDataset":
        """Load with flexible count validation and numpy check."""
        if self._loaded:
            return self

        # Check numpy availability and warn
        if not _check_numpy_available():
            print(
                "WARNING: numpy is not available in the execution environment.\n"
                "SciCode tests use np.allclose() and WILL FAIL without numpy.\n"
                "Install numpy in the execution environment to run SciCode properly."
            )

        # Load target values from HDF5 (required for test assertions)
        h5_path = self.cache_dir / self.H5_SUBDIR / self.H5_FILENAME
        if h5_path.exists():
            self._targets = _load_h5_targets(h5_path)
            print(f"  Loaded {len(self._targets)} step targets from {h5_path.name}")
        else:
            print(
                f"WARNING: {h5_path} not found. SciCode tests will FAIL.\n"
                f"Download test_data.h5 from Google Drive into {h5_path.parent}/"
            )
            self._targets = {}

        filepath = self.download()
        self._tasks = self._parse(filepath)
        self._loaded = True

        # Accept a wide range since sub-step count varies
        if len(self._tasks) < 50:
            raise ValueError(
                f"Expected at least 50 tasks in {self.name}, "
                f"got {len(self._tasks)}"
            )

        return self

    def download(self) -> Path:
        """Download SciCode via HuggingFace rows API and cache as JSONL."""
        filepath = self.cache_dir / self.FILENAME_COMBINED

        if filepath.exists():
            return filepath

        print(f"Downloading SciCode dataset from HuggingFace...")
        all_rows = []

        for split in ("test", "validation"):
            offset = 0
            while True:
                url = (
                    f"{self.ROWS_API}?dataset={self.DATASET_ID}"
                    f"&config=default&split={split}&offset={offset}&length=100"
                )
                req = urllib.request.Request(url)
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        batch = data.get("rows", [])
                        all_rows.extend(batch)
                        if len(batch) < 100:
                            break
                        offset += 100
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        # Split doesn't exist, skip it
                        break
                    if all_rows:
                        print(f"Warning: partial download for split={split}: {e}")
                        break
                    raise RuntimeError(f"Failed to download SciCode split={split}: {e}")
                except Exception as e:
                    if all_rows:
                        print(f"Warning: partial download for split={split}: {e}")
                        break
                    raise RuntimeError(f"Failed to download SciCode split={split}: {e}")

        with open(filepath, 'w', encoding='utf-8') as f:
            for row in all_rows:
                f.write(json.dumps(row.get("row", row)) + "\n")

        print(f"Downloaded {len(all_rows)} problems to {filepath}")
        return filepath

    def _parse(self, filepath: Path) -> List[BenchmarkTask]:
        """
        Parse SciCode problems and decompose into sub-step tasks.

        Each problem contains multiple sub-steps. Each sub-step becomes
        a separate BenchmarkTask.
        """
        tasks = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                sub_tasks = self._decompose_problem(data)
                tasks.extend(sub_tasks)

        return tasks

    def _decompose_problem(self, data: dict) -> List[BenchmarkTask]:
        """
        Decompose a SciCode problem into sub-step BenchmarkTasks.

        For each sub-step:
        - Prompt = dependencies + problem desc + prior ground truth + step desc + function header
        - Test = imports + prior ground truth + step test_cases
        """
        tasks = []

        problem_id = data.get("problem_id", data.get("id", "unknown"))
        problem_desc = data.get("problem_description", data.get("description", ""))
        required_deps = data.get("required_dependencies", data.get("dependencies", ""))
        sub_steps = data.get("sub_steps", data.get("steps", []))

        if not sub_steps:
            # Try alternate structure — single problem without sub-steps
            task = self._convert_single_problem(data)
            if task:
                tasks.append(task)
            return tasks

        # Accumulate ground truth from prior steps
        prior_ground_truth = []

        for step_idx, step in enumerate(sub_steps):
            step_num = step_idx + 1
            task_id = f"SciCode/{problem_id}/step{step_num}"

            step_desc = step.get("step_description", step.get("description", ""))
            function_header = step.get("function_header", step.get("header", ""))
            ground_truth = step.get("ground_truth_code", step.get("ground_truth", step.get("code", "")))
            test_cases = step.get("test_cases", step.get("tests", ""))

            # Build the prompt
            prompt = self._build_step_prompt(
                problem_desc, required_deps, prior_ground_truth,
                step_desc, function_header, step_num
            )

            # Build test code (with target values from HDF5)
            step_number = step.get("step_number", f"{problem_id}.{step_num}")
            step_targets = self._targets.get(str(step_number), [])
            test_code = self._build_step_test_code(
                required_deps, prior_ground_truth, test_cases, step_targets
            )

            # Extract entry point from function header
            entry_point = self._extract_entry_point(function_header)

            # Canonical solution includes deps + prior ground truth + this step's code
            canonical_parts = []
            if required_deps:
                canonical_parts.append(required_deps)
            canonical_parts.extend(prior_ground_truth)
            if ground_truth:
                canonical_parts.append(ground_truth)
            canonical_solution = "\n\n".join(canonical_parts)

            task = BenchmarkTask(
                task_id=task_id,
                prompt=prompt,
                canonical_solution=canonical_solution,
                test_code=test_code,
                entry_point=entry_point,
                category="scicode",
                difficulty="hard",
                tags=["python", "scientific-computing"],
            )
            tasks.append(task)

            # Add this step's ground truth for subsequent steps
            if ground_truth:
                prior_ground_truth.append(ground_truth)

        return tasks

    def _build_step_prompt(
        self,
        problem_desc: str,
        required_deps: str,
        prior_ground_truth: List[str],
        step_desc: str,
        function_header: str,
        step_num: int,
    ) -> str:
        """Build the prompt for a single sub-step."""
        parts = []

        parts.append(
            "You are an expert scientific computing programmer. "
            "Implement the following function step.\n"
        )

        if required_deps:
            parts.append(f"Required imports:\n```python\n{required_deps}\n```\n")

        parts.append(f"Problem description:\n{problem_desc}\n")

        if prior_ground_truth:
            prior_code = "\n\n".join(prior_ground_truth)
            parts.append(
                f"Code from previous steps (already implemented):\n"
                f"```python\n{prior_code}\n```\n"
            )

        parts.append(f"Step {step_num}: {step_desc}\n")

        if function_header:
            parts.append(f"Function to implement:\n```python\n{function_header}\n```\n")

        parts.append(
            "Return ONLY the function implementation (no explanation, no markdown)."
        )

        return "\n".join(parts)

    def _build_step_test_code(
        self,
        required_deps: str,
        prior_ground_truth: List[str],
        test_cases,
        step_targets: List[Any] = None,
    ) -> str:
        """Build test code for a single sub-step.

        Injects pre-computed target values from HDF5 before each test case
        so that ``assert np.allclose(func(args), target)`` works.
        """
        parts = []

        if required_deps:
            parts.append(required_deps)

        parts.extend(prior_ground_truth)

        if test_cases:
            if isinstance(test_cases, list):
                # Inject target = <value> before each test case
                tc_parts = []
                for idx, tc in enumerate(test_cases):
                    if step_targets and idx < len(step_targets):
                        target_val = step_targets[idx]
                        tc_parts.append("target = %s" % _serialize_target(target_val))
                    tc_parts.append(str(tc))
                parts.append("\n".join(tc_parts))
            else:
                parts.append(str(test_cases))

        return "\n\n".join(parts)

    def _convert_single_problem(self, data: dict) -> Optional[BenchmarkTask]:
        """Convert a problem without sub-steps to a single BenchmarkTask."""
        problem_id = data.get("problem_id", data.get("id", "unknown"))
        task_id = f"SciCode/{problem_id}"

        problem_desc = data.get("problem_description", data.get("description", ""))
        required_deps = data.get("required_dependencies", "")
        code = data.get("ground_truth_code", data.get("code", ""))
        test_cases = data.get("test_cases", data.get("tests", ""))

        if not problem_desc:
            return None

        prompt_parts = [
            "You are an expert scientific computing programmer.\n",
            f"Problem:\n{problem_desc}\n",
        ]
        if required_deps:
            prompt_parts.append(f"Required imports:\n```python\n{required_deps}\n```\n")
        prompt_parts.append(
            "Return ONLY the implementation (no explanation, no markdown)."
        )
        prompt = "\n".join(prompt_parts)

        test_code_parts = []
        if required_deps:
            test_code_parts.append(required_deps)
        if test_cases:
            if isinstance(test_cases, list):
                test_code_parts.append("\n".join(str(tc) for tc in test_cases))
            else:
                test_code_parts.append(str(test_cases))
        test_code = "\n\n".join(test_code_parts)

        entry_point = self._extract_entry_point(code)

        canonical_parts = []
        if required_deps:
            canonical_parts.append(required_deps)
        if code:
            canonical_parts.append(code)
        canonical_solution = "\n\n".join(canonical_parts)

        return BenchmarkTask(
            task_id=task_id,
            prompt=prompt,
            canonical_solution=canonical_solution,
            test_code=test_code,
            entry_point=entry_point,
            category="scicode",
            difficulty="hard",
            tags=["python", "scientific-computing"],
        )

    def _extract_entry_point(self, code: str) -> str:
        """Extract the main function name from code or header."""
        for line in code.strip().split('\n'):
            line = line.strip()
            if line.startswith('def '):
                name_part = line[4:]
                paren_idx = name_part.find('(')
                if paren_idx > 0:
                    return name_part[:paren_idx].strip()
        return "solution"


if __name__ == "__main__":
    dataset = SciCodeDataset()
    dataset.load()
    print(dataset.summary())
    print(f"\nTotal sub-step tasks: {len(dataset)}")
    if len(dataset) > 0:
        t = dataset[0]
        print(f"First task: {t.task_id}")
        print(f"Entry point: {t.entry_point}")
        print(f"Prompt preview:\n{t.prompt[:500]}...")
