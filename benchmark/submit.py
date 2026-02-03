"""
Results packaging and submission for ATLAS benchmarks.

Aggregates results, generates reports, and packages for publication.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import config
from .models import BenchmarkRun, HardwareInfo
from .analysis import calculate_pass_at_k, CostAnalyzer, collect_hardware_info
from .analysis.hardware_info import hardware_info_to_markdown
from .analysis.pass_at_k import compare_with_baseline


def get_git_commit() -> str:
    """
    Get current git commit hash.

    Returns:
        Git commit hash or empty string
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.project_root,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_git_branch() -> str:
    """
    Get current git branch name.

    Returns:
        Git branch name or empty string
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.project_root,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class SubmissionPackage:
    """
    Package benchmark results for submission/publication.

    Aggregates multiple runs and generates comprehensive reports.
    """

    def __init__(self, runs: List[BenchmarkRun] = None):
        """
        Initialize submission package.

        Args:
            runs: List of BenchmarkRun objects to include
        """
        self.runs = runs or []
        self.hardware_info = collect_hardware_info()
        self.git_commit = get_git_commit()
        self.git_branch = get_git_branch()
        self.timestamp = datetime.now().isoformat()

    def add_run(self, run: BenchmarkRun):
        """Add a benchmark run to the package."""
        self.runs.append(run)

    def add_run_from_file(self, filepath: str):
        """Load and add a benchmark run from a JSON file."""
        run = BenchmarkRun.load(filepath)
        self.add_run(run)

    def validate(self) -> List[str]:
        """
        Validate the submission package.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.runs:
            errors.append("No benchmark runs included")

        for run in self.runs:
            if not run.results:
                errors.append(f"Run {run.run_id} has no results")
            if not run.start_time:
                errors.append(f"Run {run.run_id} missing start_time")
            if not run.end_time:
                errors.append(f"Run {run.run_id} missing end_time")

        if not self.git_commit:
            errors.append("Git commit hash not available")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert submission to dictionary format.

        Returns:
            Complete submission data as dictionary
        """
        # Calculate metrics for each run
        pass_at_k_results = {}
        cost_results = {}
        analyzer = CostAnalyzer()

        for run in self.runs:
            results = list(run.results.values())
            pk = calculate_pass_at_k(results, dataset=run.dataset)
            pass_at_k_results[run.dataset] = pk.to_dict()

            cost = analyzer.analyze(run)
            cost_results[run.dataset] = cost.to_dict()

        return {
            "version": "1.0",
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "hardware_info": self.hardware_info.to_dict(),
            "runs": {run.run_id: run.to_dict() for run in self.runs},
            "pass_at_k_metrics": pass_at_k_results,
            "cost_analysis": cost_results,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary metrics across all runs."""
        summary = {
            "total_tasks": 0,
            "total_passed": 0,
            "datasets": []
        }

        for run in self.runs:
            summary["total_tasks"] += run.total_tasks
            summary["total_passed"] += run.passed_tasks
            summary["datasets"].append({
                "name": run.dataset,
                "tasks": run.total_tasks,
                "passed": run.passed_tasks,
                "pass_rate": run.pass_rate
            })

        if summary["total_tasks"] > 0:
            summary["overall_pass_rate"] = summary["total_passed"] / summary["total_tasks"]
        else:
            summary["overall_pass_rate"] = 0.0

        return summary

    def to_json(self) -> str:
        """Convert submission to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, filepath: str):
        """Save submission to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """
        Generate comprehensive Markdown report.

        Returns:
            Formatted Markdown report
        """
        lines = [
            "# ATLAS V1 Benchmark Results",
            "",
            f"**Generated:** {self.timestamp}",
            f"**Git Commit:** `{self.git_commit[:8]}` ({self.git_branch})" if self.git_commit else "",
            "",
            "---",
            "",
        ]

        # Hardware info
        lines.append(hardware_info_to_markdown(self.hardware_info))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Results for each dataset
        analyzer = CostAnalyzer()

        for run in self.runs:
            results = list(run.results.values())
            pk = calculate_pass_at_k(results, dataset=run.dataset)
            cost = analyzer.analyze(run)

            lines.append(f"## {run.dataset.upper()} Results")
            lines.append("")
            lines.append(f"- **Run ID:** {run.run_id}")
            lines.append(f"- **Tasks:** {run.total_tasks}")
            lines.append(f"- **Attempts per task (k):** {run.k}")
            lines.append(f"- **Temperature:** {run.temperature}")
            lines.append(f"- **Start:** {run.start_time}")
            lines.append(f"- **End:** {run.end_time}")
            lines.append("")

            # Pass@k results
            lines.append(pk.to_markdown())
            lines.append("")

            # Baseline comparison
            lines.append(compare_with_baseline(pk))
            lines.append("")

            # Cost analysis
            lines.append(analyzer.to_markdown(cost))
            lines.append("")
            lines.append("---")
            lines.append("")

        # Summary
        summary = self._generate_summary()
        lines.extend([
            "## Summary",
            "",
            f"- **Total Tasks:** {summary['total_tasks']}",
            f"- **Total Passed:** {summary['total_passed']}",
            f"- **Overall Pass Rate:** {summary['overall_pass_rate']:.1%}",
            "",
            "### By Dataset",
            "",
            "| Dataset | Tasks | Passed | Pass Rate |",
            "|---------|-------|--------|-----------|",
        ])

        for ds in summary["datasets"]:
            lines.append(f"| {ds['name']} | {ds['tasks']} | {ds['passed']} | {ds['pass_rate']:.1%} |")

        lines.extend([
            "",
            "---",
            "",
            "*Report generated by ATLAS V1 Benchmark Infrastructure*"
        ])

        return "\n".join(lines)

    def save_report(self, filepath: str):
        """Save Markdown report to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_markdown())


def create_submission(
    input_dirs: List[str],
    output_dir: str = None
) -> SubmissionPackage:
    """
    Create a submission package from benchmark result directories.

    Args:
        input_dirs: List of directories containing run.json files
        output_dir: Directory for submission output

    Returns:
        SubmissionPackage with all runs
    """
    package = SubmissionPackage()

    for input_dir in input_dirs:
        input_path = Path(input_dir)

        # Find all run.json files
        run_files = list(input_path.glob("**/run.json"))

        for run_file in run_files:
            package.add_run_from_file(str(run_file))

    # Validate
    errors = package.validate()
    if errors:
        print("Validation warnings:")
        for error in errors:
            print(f"  - {error}")

    # Save if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_file = output_path / "submission.json"
        package.save(str(json_file))
        print(f"Saved submission to {json_file}")

        # Save Markdown report
        md_file = output_path / "report.md"
        package.save_report(str(md_file))
        print(f"Saved report to {md_file}")

    return package


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m benchmark.submit <input_dir> [output_dir]")
        sys.exit(1)

    input_dirs = [sys.argv[1]]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else config.submissions_dir

    package = create_submission(input_dirs, str(output_dir))
    print("\nSubmission package created successfully")
