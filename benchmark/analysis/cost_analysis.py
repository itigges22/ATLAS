"""
Cost analysis for benchmark results.

Calculates cost-per-task, cloud API cost comparisons,
and novel efficiency metrics like Tokens/Watt-Hour.
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..models import BenchmarkRun, TaskResult
from ..config import config


@dataclass
class CostMetrics:
    """
    Computed cost metrics for a benchmark run.

    Attributes:
        total_tokens: Total tokens generated
        total_inference_time_s: Total inference time in seconds
        total_execution_time_s: Total execution time in seconds
        total_wall_time_s: Total wall clock time
        successful_tasks: Number of tasks with at least one passing attempt
        total_tasks: Total number of tasks
        estimated_energy_kwh: Estimated energy consumption in kWh
        cost_per_task_local: Local cost per successful task (USD)
        tokens_per_watt_hour: Novel metric: tokens generated per watt-hour
        tasks_per_watt_hour: Novel metric: successful tasks per watt-hour
        throughput_tasks_per_hour: Tasks completed per hour
        median_time_to_solution_s: Median wall-clock time to first success
        time_to_solution_values: Individual time-to-solution values for percentiles
        cloud_costs: Estimated costs for cloud APIs
        cost_ratio: Ratio of cloud cost to local cost
    """
    total_tokens: int = 0
    total_inference_time_s: float = 0.0
    total_execution_time_s: float = 0.0
    total_wall_time_s: float = 0.0
    successful_tasks: int = 0
    total_tasks: int = 0
    estimated_energy_kwh: float = 0.0
    cost_per_task_local: float = 0.0
    tokens_per_watt_hour: float = 0.0
    tasks_per_watt_hour: float = 0.0
    throughput_tasks_per_hour: float = 0.0
    median_time_to_solution_s: float = 0.0
    time_to_solution_values: List[float] = field(default_factory=list)
    cloud_costs: Dict[str, float] = field(default_factory=dict)
    cost_ratio: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "total_inference_time_s": self.total_inference_time_s,
            "total_execution_time_s": self.total_execution_time_s,
            "total_wall_time_s": self.total_wall_time_s,
            "successful_tasks": self.successful_tasks,
            "total_tasks": self.total_tasks,
            "estimated_energy_kwh": self.estimated_energy_kwh,
            "cost_per_task_local": self.cost_per_task_local,
            "tokens_per_watt_hour": self.tokens_per_watt_hour,
            "tasks_per_watt_hour": self.tasks_per_watt_hour,
            "throughput_tasks_per_hour": self.throughput_tasks_per_hour,
            "median_time_to_solution_s": self.median_time_to_solution_s,
            "cloud_costs": self.cloud_costs,
            "cost_ratio": self.cost_ratio
        }


class CostAnalyzer:
    """
    Analyzes costs for benchmark runs.

    Computes local costs, cloud API equivalent costs,
    and novel efficiency metrics.
    """

    def __init__(
        self,
        gpu_tdp_watts: float = None,
        gpu_cost_usd: float = None,
        gpu_lifetime_hours: float = None,
        electricity_cost_per_kwh: float = 0.12,
        gpu_utilization: float = 0.8
    ):
        """
        Initialize cost analyzer.

        Args:
            gpu_tdp_watts: GPU power draw in watts
            gpu_cost_usd: GPU cost in USD
            gpu_lifetime_hours: Expected GPU lifetime in hours
            electricity_cost_per_kwh: Electricity cost per kWh
            gpu_utilization: Estimated GPU utilization during inference
        """
        self.gpu_tdp_watts = gpu_tdp_watts or config.gpu_tdp_watts
        self.gpu_cost_usd = gpu_cost_usd or config.gpu_cost_usd
        self.gpu_lifetime_hours = gpu_lifetime_hours or config.gpu_lifetime_hours
        self.electricity_cost_per_kwh = electricity_cost_per_kwh
        self.gpu_utilization = gpu_utilization

        # Amortized GPU cost per hour
        self.gpu_cost_per_hour = self.gpu_cost_usd / self.gpu_lifetime_hours

    def analyze(self, run: BenchmarkRun) -> CostMetrics:
        """
        Analyze costs for a benchmark run.

        Args:
            run: BenchmarkRun with results

        Returns:
            CostMetrics with computed values
        """
        metrics = CostMetrics()
        time_to_solution_list = []

        # Aggregate metrics from results
        for result in run.results.values():
            metrics.total_tokens += result.total_tokens
            metrics.total_inference_time_s += result.total_inference_time_ms / 1000
            metrics.total_execution_time_s += result.total_execution_time_ms / 1000
            if result.passed:
                metrics.successful_tasks += 1
                # Time to first success = time of first passing attempt
                for attempt in result.attempts:
                    if attempt.passed:
                        tts = (attempt.inference_time_ms + attempt.execution_time_ms) / 1000
                        time_to_solution_list.append(tts)
                        break

        metrics.total_tasks = len(run.results)
        metrics.total_wall_time_s = metrics.total_inference_time_s + metrics.total_execution_time_s

        # Calculate time-to-solution statistics
        if time_to_solution_list:
            metrics.time_to_solution_values = sorted(time_to_solution_list)
            metrics.median_time_to_solution_s = statistics.median(time_to_solution_list)

        # Calculate throughput (tasks per hour)
        if metrics.total_wall_time_s > 0:
            metrics.throughput_tasks_per_hour = (metrics.total_tasks / metrics.total_wall_time_s) * 3600

        # Estimate energy consumption
        # Energy = Power × Time × Utilization
        inference_hours = metrics.total_inference_time_s / 3600
        power_kw = (self.gpu_tdp_watts * self.gpu_utilization) / 1000
        metrics.estimated_energy_kwh = power_kw * inference_hours

        # Calculate local cost
        # Local cost = hardware amortization + electricity
        hardware_cost = self.gpu_cost_per_hour * inference_hours
        electricity_cost = metrics.estimated_energy_kwh * self.electricity_cost_per_kwh
        total_local_cost = hardware_cost + electricity_cost

        if metrics.successful_tasks > 0:
            metrics.cost_per_task_local = total_local_cost / metrics.successful_tasks
        else:
            metrics.cost_per_task_local = float('inf')

        # Novel efficiency metrics
        watt_hours = metrics.estimated_energy_kwh * 1000
        if watt_hours > 0:
            metrics.tokens_per_watt_hour = metrics.total_tokens / watt_hours
            metrics.tasks_per_watt_hour = metrics.successful_tasks / watt_hours
        else:
            metrics.tokens_per_watt_hour = 0.0
            metrics.tasks_per_watt_hour = 0.0

        # Calculate cloud API costs
        metrics.cloud_costs = self._calculate_cloud_costs(metrics.total_tokens)

        # Calculate cost ratios
        if total_local_cost > 0:
            for provider, cloud_cost in metrics.cloud_costs.items():
                metrics.cost_ratio[provider] = cloud_cost / total_local_cost

        return metrics

    def analyze_results(self, results: List[TaskResult]) -> CostMetrics:
        """
        Analyze costs from a list of TaskResult objects.

        Args:
            results: List of TaskResult objects

        Returns:
            CostMetrics with computed values
        """
        # Create a temporary BenchmarkRun
        run = BenchmarkRun(
            run_id="temp",
            dataset="temp",
            k=1,
            temperature=0.0,
            results={r.task_id: r for r in results}
        )
        return self.analyze(run)

    def _calculate_cloud_costs(self, total_tokens: int) -> Dict[str, float]:
        """
        Calculate equivalent cloud API costs.

        Assumes similar token usage across providers.

        Args:
            total_tokens: Total tokens generated

        Returns:
            Dictionary of provider -> cost
        """
        costs = {}
        pricing = config.cloud_pricing

        # Estimate input tokens as 2x output tokens (prompts are longer)
        output_tokens = total_tokens
        input_tokens = total_tokens * 2

        for provider, rates in pricing.items():
            input_cost = (input_tokens / 1_000_000) * rates["input"]
            output_cost = (output_tokens / 1_000_000) * rates["output"]
            costs[provider] = input_cost + output_cost

        return costs

    def to_markdown(self, metrics: CostMetrics) -> str:
        """
        Generate Markdown report of cost analysis.

        Args:
            metrics: CostMetrics to report

        Returns:
            Formatted Markdown string
        """
        lines = [
            "## Cost Analysis",
            "",
            "### Summary",
            "",
            f"- Total Tasks: {metrics.total_tasks}",
            f"- Successful Tasks: {metrics.successful_tasks}",
            f"- Total Tokens: {metrics.total_tokens:,}",
            f"- Total Inference Time: {metrics.total_inference_time_s:.1f}s",
            f"- Total Wall Time: {metrics.total_wall_time_s:.1f}s",
            f"- Estimated Energy: {metrics.estimated_energy_kwh:.4f} kWh",
            "",
            "### Performance Metrics",
            "",
            "| Metric | Value | Target | Status |",
            "|--------|-------|--------|--------|",
        ]

        # Throughput
        throughput = metrics.throughput_tasks_per_hour
        throughput_status = "✓" if throughput >= 100 else "✗"
        lines.append(f"| Throughput | {throughput:.1f} tasks/hr | ≥100 tasks/hr | {throughput_status} |")

        # Time to Solution
        tts = metrics.median_time_to_solution_s
        tts_status = "✓" if tts < 60 else "✗"
        lines.append(f"| Time to Solution (median) | {tts:.1f}s | <60s | {tts_status} |")

        lines.extend([
            "",
            "### Local Cost",
            "",
            f"- Cost per Successful Task: ${metrics.cost_per_task_local:.6f}",
            "",
            "### Novel Efficiency Metrics (ATLAS Baselines)",
            "",
            f"- **Tokens per Watt-Hour:** {metrics.tokens_per_watt_hour:,.0f}",
            f"- **Tasks per Watt-Hour:** {metrics.tasks_per_watt_hour:.2f}",
            "",
            "*These are novel metrics establishing ATLAS V1 baselines.*",
            "",
            "### Cloud API Cost Comparison",
            "",
            "| Provider | Cloud Cost | Local Cost | Ratio | Target (≥30x) |",
            "|----------|------------|------------|-------|---------------|",
        ])

        total_local_cost = metrics.cost_per_task_local * metrics.successful_tasks if metrics.successful_tasks > 0 else 0
        for provider, cost in sorted(metrics.cloud_costs.items()):
            ratio = metrics.cost_ratio.get(provider, 0)
            status = "✓" if ratio >= 30 else "✗"
            lines.append(f"| {provider} | ${cost:.4f} | ${total_local_cost:.4f} | {ratio:.1f}x | {status} |")

        # Overall cost efficiency status
        min_ratio = min(metrics.cost_ratio.values()) if metrics.cost_ratio else 0
        lines.extend([
            "",
            f"**Cost Efficiency Target: ≥30x cheaper than cloud APIs**",
        ])
        if min_ratio >= 30:
            lines.append(f"**Status: ✓ TARGET MET** (minimum ratio: {min_ratio:.1f}x)")
        else:
            lines.append(f"**Status: ✗ Target not met** (minimum ratio: {min_ratio:.1f}x)")

        return "\n".join(lines)


def analyze_cost_from_run(run: BenchmarkRun) -> CostMetrics:
    """
    Convenience function to analyze costs from a benchmark run.

    Args:
        run: BenchmarkRun with results

    Returns:
        CostMetrics with computed values
    """
    analyzer = CostAnalyzer()
    return analyzer.analyze(run)


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing cost analysis...")

    from ..models import AttemptResult

    # Create synthetic results
    results = []
    for i in range(100):
        attempts = [
            AttemptResult(
                task_id=f"test_{i}",
                attempt_number=1,
                generated_code="def solution(): pass",
                passed=i < 70,  # 70% pass rate
                execution_time_ms=100,
                tokens_generated=500,
                inference_time_ms=2000
            )
        ]
        result = TaskResult(
            task_id=f"test_{i}",
            attempts=attempts,
            total_tokens=500,
            total_inference_time_ms=2000,
            total_execution_time_ms=100
        )
        results.append(result)

    analyzer = CostAnalyzer()
    metrics = analyzer.analyze_results(results)

    print(analyzer.to_markdown(metrics))
