"""
Custom benchmark tasks for ATLAS V1.

Contains 100 carefully designed coding tasks across 6 categories.
"""

from pathlib import Path
import json
from typing import List

from ..models import BenchmarkTask


def load_custom_tasks() -> List[BenchmarkTask]:
    """
    Load custom benchmark tasks from tasks.json.

    Returns:
        List of BenchmarkTask objects
    """
    tasks_file = Path(__file__).parent / "tasks.json"

    if not tasks_file.exists():
        raise FileNotFoundError(f"Custom tasks file not found: {tasks_file}")

    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = []
    for item in data.get("tasks", []):
        task = BenchmarkTask.from_dict(item)
        tasks.append(task)

    return tasks


__all__ = ["load_custom_tasks"]
