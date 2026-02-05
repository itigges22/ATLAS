"""
Pass@k metric calculations using unbiased estimators.

Implements the standard pass@k estimator from the Codex paper:
pass@k = 1 - C(n-c, k) / C(n, k)

where n = total samples, c = correct samples, k = number of attempts
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models import TaskResult, BenchmarkRun
from ..config import config


def _comb(n: int, k: int) -> float:
    """
    Calculate binomial coefficient C(n, k).

    Uses logarithms to avoid overflow for large values.

    Args:
        n: Total items
        k: Items to choose

    Returns:
        Binomial coefficient as float
    """
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    # Use log-sum for numerical stability
    log_result = 0.0
    for i in range(k):
        log_result += math.log(n - i) - math.log(i + 1)

    return math.exp(log_result)


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using the unbiased estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)

    This is the probability of at least one correct solution
    when sampling k times from n total attempts with c correct.

    Args:
        n: Total number of samples/attempts
        c: Number of correct samples
        k: Number of attempts for pass@k

    Returns:
        pass@k probability (0.0 to 1.0)
    """
    if n < k:
        # Not enough samples - return simple ratio
        return c / n if n > 0 else 0.0

    if c == 0:
        return 0.0

    if c >= n:
        return 1.0

    # Calculate 1 - C(n-c, k) / C(n, k)
    numerator = _comb(n - c, k)
    denominator = _comb(n, k)

    if denominator == 0:
        return 0.0

    return 1.0 - (numerator / denominator)


@dataclass
class PassAtKResult:
    """
    Results of pass@k calculation for a benchmark.

    Attributes:
        dataset: Dataset name
        total_tasks: Total number of tasks
        samples_per_task: Number of samples per task (n)
        pass_at_1: pass@1 score
        pass_at_5: pass@5 score
        pass_at_10: pass@10 score
        pass_at_20: pass@20 score
        per_task_results: Per-task c values (number of correct samples)
        confidence_intervals: Bootstrap confidence intervals
    """
    dataset: str
    total_tasks: int
    samples_per_task: int
    pass_at_1: float = 0.0
    pass_at_5: float = 0.0
    pass_at_10: float = 0.0
    pass_at_20: float = 0.0
    per_task_results: Dict[str, int] = field(default_factory=dict)
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset,
            "total_tasks": self.total_tasks,
            "samples_per_task": self.samples_per_task,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "pass_at_10": self.pass_at_10,
            "pass_at_20": self.pass_at_20,
            "per_task_results": self.per_task_results,
            "confidence_intervals": self.confidence_intervals
        }

    def to_markdown(self) -> str:
        """Generate Markdown table of results."""
        # Define targets based on dataset
        targets = {
            "humaneval": {"pass@1": 0.65, "pass@5": 0.95, "pass@20": 0.995},
            "mbpp": {"pass@1": 0.70, "pass@5": 0.95, "pass@20": 0.995},
            "humaneval_plus": {"pass@1": 0.60, "pass@5": 0.90, "pass@20": 0.99},
            "mbpp_plus": {"pass@1": 0.60, "pass@5": 0.90, "pass@20": 0.99},
            "livecodebench": {"pass@1": 0.20, "pass@5": 0.50},
            "scicode": {"pass@1": 0.10, "pass@5": 0.30},
            "custom": {"pass@1": 0.65, "pass@5": 0.95, "pass@20": 0.995},
        }
        dataset_targets = targets.get(self.dataset.lower(), targets["custom"])

        lines = [
            f"## Pass@k Results: {self.dataset}",
            "",
            f"- Total Tasks: {self.total_tasks}",
            f"- Samples per Task: {self.samples_per_task}",
            "",
            "| Metric | Score | Target | Status | 95% CI |",
            "|--------|-------|--------|--------|--------|",
        ]

        for k, score in [("1", self.pass_at_1), ("5", self.pass_at_5),
                         ("10", self.pass_at_10), ("20", self.pass_at_20)]:
            ci = self.confidence_intervals.get(f"pass_at_{k}", (score, score))
            ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]"
            target = dataset_targets.get(f"pass@{k}", None)
            if target:
                target_str = f"≥{target:.1%}"
                status = "✓" if score >= target else "✗"
            else:
                target_str = "—"
                status = "—"
            lines.append(f"| pass@{k} | {score:.1%} | {target_str} | {status} | {ci_str} |")

        return "\n".join(lines)


def calculate_pass_at_k(
    results: List[TaskResult],
    dataset: str = "benchmark",
    k_values: List[int] = None,
    bootstrap_samples: int = 1000
) -> PassAtKResult:
    """
    Calculate pass@k metrics from benchmark results.

    Args:
        results: List of TaskResult objects
        dataset: Dataset name for labeling
        k_values: List of k values to calculate (default: [1, 5, 10, 20])
        bootstrap_samples: Number of bootstrap samples for CI

    Returns:
        PassAtKResult with calculated metrics
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    total_tasks = len(results)
    if total_tasks == 0:
        return PassAtKResult(dataset=dataset, total_tasks=0, samples_per_task=0)

    # Count samples and correct per task
    samples_per_task = max(r.num_attempts for r in results) if results else 0
    per_task_c = {}  # task_id -> number correct

    for result in results:
        n = result.num_attempts
        c = result.num_passed
        per_task_c[result.task_id] = c

    # Calculate pass@k for each k
    pass_at_k_scores = {}
    for k in k_values:
        if k > samples_per_task:
            # Not enough samples for this k
            pass_at_k_scores[k] = sum(1 for r in results if r.passed) / total_tasks
        else:
            # Calculate using unbiased estimator
            total_score = 0.0
            for result in results:
                n = result.num_attempts
                c = result.num_passed
                if n >= k:
                    total_score += pass_at_k(n, c, k)
                elif n > 0:
                    # Fallback for insufficient samples
                    total_score += min(c, k) / k

            pass_at_k_scores[k] = total_score / total_tasks

    # Bootstrap confidence intervals
    confidence_intervals = {}
    if bootstrap_samples > 0 and total_tasks > 1:
        for k in k_values:
            bootstrap_scores = []
            for _ in range(bootstrap_samples):
                # Resample with replacement
                sample = random.choices(results, k=total_tasks)
                score = 0.0
                for result in sample:
                    n = result.num_attempts
                    c = result.num_passed
                    if n >= k:
                        score += pass_at_k(n, c, k)
                    elif n > 0:
                        score += min(c, k) / k
                bootstrap_scores.append(score / total_tasks)

            # 95% confidence interval
            bootstrap_scores.sort()
            lower_idx = int(0.025 * bootstrap_samples)
            upper_idx = int(0.975 * bootstrap_samples)
            confidence_intervals[f"pass_at_{k}"] = (
                bootstrap_scores[lower_idx],
                bootstrap_scores[upper_idx]
            )

    return PassAtKResult(
        dataset=dataset,
        total_tasks=total_tasks,
        samples_per_task=samples_per_task,
        pass_at_1=pass_at_k_scores.get(1, 0.0),
        pass_at_5=pass_at_k_scores.get(5, 0.0),
        pass_at_10=pass_at_k_scores.get(10, 0.0),
        pass_at_20=pass_at_k_scores.get(20, 0.0),
        per_task_results=per_task_c,
        confidence_intervals=confidence_intervals
    )


def calculate_pass_at_k_from_run(run: BenchmarkRun) -> PassAtKResult:
    """
    Calculate pass@k from a BenchmarkRun object.

    Args:
        run: BenchmarkRun with results

    Returns:
        PassAtKResult with calculated metrics
    """
    results = list(run.results.values())
    return calculate_pass_at_k(results, dataset=run.dataset)


def compare_with_baseline(
    result: PassAtKResult,
    baseline_pass1: float = None
) -> str:
    """
    Compare results with published baselines.

    Args:
        result: PassAtKResult to compare
        baseline_pass1: Baseline pass@1 score

    Returns:
        Markdown comparison table
    """
    if baseline_pass1 is None:
        # Use Qwen3-14B baselines from config
        baselines = config.qwen3_14b_baselines
        key = f"{result.dataset}_pass1"
        baseline_pass1 = baselines.get(key, 0.0)
        # Fallback for legacy dataset names
        if baseline_pass1 == 0.0:
            if result.dataset == "humaneval":
                baseline_pass1 = baselines.get("humaneval_pass1", 0.67)
            elif result.dataset == "mbpp":
                baseline_pass1 = baselines.get("mbpp_pass1", 0.734)

    diff = result.pass_at_1 - baseline_pass1
    diff_str = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"

    lines = [
        "## Comparison with Baseline",
        "",
        "| Metric | ATLAS V1 | Qwen3-14B Baseline | Difference |",
        "|--------|----------|-------------------|------------|",
        f"| pass@1 | {result.pass_at_1:.1%} | {baseline_pass1:.1%} | {diff_str} |",
    ]

    return "\n".join(lines)


def generate_pass_at_k_curve(
    results: List[TaskResult],
    max_k: int = 20
) -> List[tuple]:
    """
    Generate pass@k curve data points.

    Args:
        results: List of TaskResult objects
        max_k: Maximum k value

    Returns:
        List of (k, pass_at_k) tuples
    """
    curve = []
    for k in range(1, max_k + 1):
        pk_result = calculate_pass_at_k(results, k_values=[k])
        score = getattr(pk_result, f"pass_at_{k}", pk_result.pass_at_1)
        curve.append((k, score))
    return curve


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing pass@k calculations...")

    # Test the formula
    print(f"pass@1 with n=10, c=3: {pass_at_k(10, 3, 1):.4f}")  # Should be ~0.30
    print(f"pass@5 with n=10, c=3: {pass_at_k(10, 3, 5):.4f}")  # Should be ~0.74
    print(f"pass@10 with n=10, c=3: {pass_at_k(10, 3, 10):.4f}")  # Should be 1.0

    # Create synthetic results
    from ..models import AttemptResult

    test_results = []
    for i in range(100):
        # Simulate varying success rates
        n_attempts = 10
        n_passed = random.randint(0, n_attempts)

        attempts = [
            AttemptResult(
                task_id=f"test_{i}",
                attempt_number=j+1,
                generated_code="",
                passed=j < n_passed,
                execution_time_ms=100
            )
            for j in range(n_attempts)
        ]

        result = TaskResult(task_id=f"test_{i}", attempts=attempts)
        test_results.append(result)

    pk = calculate_pass_at_k(test_results, dataset="test")
    print("\nTest results:")
    print(pk.to_markdown())
