"""
Shared dataclasses for the ATLAS bench harness.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class BenchmarkTask:
    """
    A single benchmark task.

    Attributes:
        task_id: Unique identifier for the task
        prompt: The task description/prompt given to the model
        canonical_solution: Reference implementation that passes all tests
        test_code: Python test code that validates the solution
        entry_point: The function name the model should implement
        category: Task category (algorithm, data_processing, etc.)
        difficulty: Task difficulty (easy, medium, hard)
        tags: List of descriptive tags
    """
    task_id: str
    prompt: str
    canonical_solution: str
    test_code: str
    entry_point: str
    category: str = "humaneval"
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)
    eval_mode: str = "function"  # "function" (concat code+tests) or "stdio" (stdin/stdout)
    test_inputs: List[str] = field(default_factory=list)   # stdin inputs for stdio mode
    test_outputs: List[str] = field(default_factory=list)  # expected stdout for stdio mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "test_code": self.test_code,
            "entry_point": self.entry_point,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }
        if self.eval_mode != "function":
            d["eval_mode"] = self.eval_mode
        if self.test_inputs:
            d["test_inputs"] = self.test_inputs
        if self.test_outputs:
            d["test_outputs"] = self.test_outputs
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            canonical_solution=data["canonical_solution"],
            test_code=data["test_code"],
            entry_point=data["entry_point"],
            category=data.get("category", "humaneval"),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
            eval_mode=data.get("eval_mode", "function"),
            test_inputs=data.get("test_inputs", []),
            test_outputs=data.get("test_outputs", []),
        )
