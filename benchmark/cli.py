#!/usr/bin/env python3
"""
ATLAS Benchmark CLI.

Main entry point for running benchmarks and analyzing results.

Usage:
    atlas benchmark --humaneval [--dry-run] [--k K] [--runs N]
    atlas benchmark --mbpp [--dry-run] [--k K] [--runs N]
    atlas benchmark --humaneval-plus [--dry-run] [--k K] [--runs N]
    atlas benchmark --mbpp-plus [--dry-run] [--k K] [--runs N]
    atlas benchmark --livecodebench [--dry-run] [--k K] [--runs N]
    atlas benchmark --scicode [--dry-run] [--k K] [--runs N]
    atlas benchmark --custom [--dry-run] [--k K] [--runs N]
    atlas benchmark --all [--dry-run] [--k K] [--runs N]
    atlas benchmark analyze --input DIR --output DIR
    atlas benchmark cost --input DIR --output DIR
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

from .config import config
from .models import BenchmarkTask, TaskResult, BenchmarkRun
from .runner import run_benchmark, run_benchmark_dry
from .datasets import (
    HumanEvalDataset, MBPPDataset,
    HumanEvalPlusDataset, MBPPPlusDataset,
    LiveCodeBenchDataset, SciCodeDataset,
)
from .analysis import calculate_pass_at_k, CostAnalyzer, collect_hardware_info
from .analysis.hardware_info import hardware_info_to_markdown
from .analysis.pass_at_k import compare_with_baseline


def atomic_write_json(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON atomically using temp file + rename to prevent corruption on crash.

    Args:
        filepath: Target file path
        data: Data to write as JSON
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix('.tmp')
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        shutil.move(str(tmp_path), str(filepath))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def find_completed_tasks(output_dir: Path) -> Set[str]:
    """
    Scan output directory for already-completed task results.

    Args:
        output_dir: Directory containing result_*.json files

    Returns:
        Set of completed task IDs
    """
    completed = set()
    for result_file in output_dir.glob("result_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if 'task_id' in data:
                    completed.add(data['task_id'])
        except (json.JSONDecodeError, IOError):
            # Corrupted file, ignore
            pass
    return completed


def setup_logging(output_dir: Path, run_id: str, resume: bool = False) -> logging.Logger:
    """
    Set up logging to both console and file.

    Args:
        output_dir: Directory for log file
        run_id: Unique run identifier
        resume: If True, append to existing log file

    Returns:
        Configured logger
    """
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates on resume
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - append mode if resuming
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "benchmark.log"
    file_mode = 'a' if resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    if resume:
        logger.info("=" * 60)
        logger.info(f"RESUMING run at {datetime.now().isoformat()}")
        logger.info("=" * 60)

    return logger


def load_custom_tasks() -> List[BenchmarkTask]:
    """
    Load custom benchmark tasks from tasks.json.

    Returns:
        List of BenchmarkTask objects
    """
    tasks_file = config.custom_dir / "tasks.json"

    if not tasks_file.exists():
        raise FileNotFoundError(f"Custom tasks file not found: {tasks_file}")

    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = []
    for item in data.get("tasks", data if isinstance(data, list) else []):
        task = BenchmarkTask.from_dict(item)
        tasks.append(task)

    return tasks


def run_benchmark_suite(
    dataset_name: str,
    dry_run: bool = False,
    k: int = 1,
    runs: int = 1,
    output_dir: Path = None,
    model_url: str = None,
    logger: logging.Logger = None,
    resume: bool = False
) -> BenchmarkRun:
    """
    Run a benchmark suite.

    Args:
        dataset_name: 'humaneval', 'mbpp', or 'custom'
        dry_run: If True, only validate without LLM calls
        k: Number of attempts per task
        runs: Number of independent runs (not implemented yet)
        output_dir: Directory for results
        model_url: LLM endpoint URL
        logger: Logger instance
        resume: If True, skip already-completed tasks

    Returns:
        BenchmarkRun with results
    """
    if logger is None:
        logger = logging.getLogger("benchmark")

    # Generate run ID
    run_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Set up output directory
    if output_dir is None:
        output_dir = config.results_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results if resuming
    completed_tasks: Set[str] = set()
    tasks_skipped = 0
    resume_metadata: Dict[str, Any] = {}

    if resume:
        completed_tasks = find_completed_tasks(output_dir)
        tasks_skipped = len(completed_tasks)
        if tasks_skipped > 0:
            resume_metadata = {
                "resumed": True,
                "resumed_at": datetime.now().isoformat(),
                "tasks_skipped": tasks_skipped,
            }
            logger.info(f"Resume mode: Found {tasks_skipped} completed task(s)")

    logger.info(f"Starting benchmark: {dataset_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"k (attempts): {k}")
    if resume:
        logger.info(f"Resume mode: {resume}")

    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == "humaneval":
        dataset = HumanEvalDataset()
        dataset.load()
        tasks = list(dataset)
    elif dataset_name == "mbpp":
        dataset = MBPPDataset()
        dataset.load()
        tasks = list(dataset)
    elif dataset_name == "humaneval_plus":
        dataset = HumanEvalPlusDataset()
        dataset.load()
        tasks = list(dataset)
    elif dataset_name == "mbpp_plus":
        dataset = MBPPPlusDataset()
        dataset.load()
        tasks = list(dataset)
    elif dataset_name == "livecodebench":
        dataset = LiveCodeBenchDataset()
        dataset.load()
        tasks = list(dataset)
    elif dataset_name == "scicode":
        dataset = SciCodeDataset()
        dataset.load()
        tasks = list(dataset)
    elif dataset_name == "custom":
        tasks = load_custom_tasks()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    total_tasks = len(tasks)
    logger.info(f"Loaded {total_tasks} tasks")

    # Filter out completed tasks if resuming
    if resume and completed_tasks:
        original_count = len(tasks)
        tasks = [t for t in tasks if t.task_id not in completed_tasks]
        remaining = len(tasks)
        logger.info(f"Resuming: {tasks_skipped}/{original_count} tasks already done, {remaining} remaining")
        resume_metadata["tasks_remaining"] = remaining

    # Collect hardware info
    hardware_info = collect_hardware_info()
    logger.info(f"Hardware: {hardware_info.gpu_model or 'Unknown GPU'}")

    # Set temperature based on k
    temperature = config.default_temperature_pass1 if k == 1 else config.default_temperature_passk

    # Initialize run with resume metadata if applicable
    run = BenchmarkRun(
        run_id=run_id,
        dataset=dataset_name,
        k=k,
        temperature=temperature,
        start_time=datetime.now().isoformat(),
        hardware_info=hardware_info.to_dict()
    )

    # Add resume metadata to run if resuming
    if resume_metadata:
        run.resume_info = resume_metadata

    # Load existing results into run if resuming
    if resume and completed_tasks:
        for result_file in output_dir.glob("result_*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'task_id' in data:
                        result = TaskResult.from_dict(data)
                        run.results[result.task_id] = result
            except (json.JSONDecodeError, IOError):
                pass
        logger.info(f"Loaded {len(run.results)} existing results from previous run(s)")

    # Progress tracking
    start_time = time.time()
    passed_count = 0
    failed_count = 0

    def progress_callback(idx: int, task_id: str, passed: bool):
        nonlocal passed_count, failed_count
        if passed:
            passed_count += 1
        else:
            failed_count += 1

        elapsed = time.time() - start_time
        total = len(tasks)
        remaining = total - idx - 1
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else 0

        status = "PASS" if passed else "FAIL"
        logger.info(
            f"[{idx+1}/{total}] {task_id}: {status} | "
            f"Pass: {passed_count} Fail: {failed_count} | "
            f"ETA: {eta:.0f}s"
        )

    def save_callback(result: TaskResult):
        # Save result incrementally for crash recovery using atomic writes
        run.results[result.task_id] = result
        result_file = output_dir / f"result_{result.task_id.replace('/', '_')}.json"
        atomic_write_json(result_file, result.to_dict())

    # Run benchmark
    if dry_run:
        logger.info("Running in dry-run mode (no LLM calls)...")
        results = run_benchmark_dry(tasks, progress_callback=progress_callback)
    else:
        logger.info(f"Running benchmark with k={k}, temperature={temperature}...")
        results = run_benchmark(
            tasks,
            k=k,
            temperature=temperature,
            use_ralph_loop=k > 1,  # Use error feedback for multi-attempt
            progress_callback=progress_callback,
            save_callback=save_callback
        )

    # Store results
    for result in results:
        run.results[result.task_id] = result

    run.end_time = datetime.now().isoformat()

    # Save complete run
    run_file = output_dir / "run.json"
    run.save(str(run_file))
    logger.info(f"Saved run to {run_file}")

    # Calculate and log metrics
    elapsed_total = time.time() - start_time
    logger.info(f"Benchmark complete in {elapsed_total:.1f}s")
    logger.info(f"Pass rate: {run.pass_rate:.1%} ({run.passed_tasks}/{run.total_tasks})")

    # Calculate pass@k metrics
    pk_result = calculate_pass_at_k(results, dataset=dataset_name)
    logger.info(f"pass@1: {pk_result.pass_at_1:.1%}")
    if k >= 5:
        logger.info(f"pass@5: {pk_result.pass_at_5:.1%}")
    if k >= 10:
        logger.info(f"pass@10: {pk_result.pass_at_10:.1%}")
    if k >= 20:
        logger.info(f"pass@20: {pk_result.pass_at_20:.1%}")

    # Save pass@k results
    pk_file = output_dir / "pass_at_k.json"
    with open(pk_file, 'w') as f:
        json.dump(pk_result.to_dict(), f, indent=2)

    # Save pass@k markdown
    pk_md_file = output_dir / "pass_at_k.md"
    with open(pk_md_file, 'w') as f:
        f.write(pk_result.to_markdown())
        f.write("\n\n")
        f.write(compare_with_baseline(pk_result))

    return run


def analyze_results(input_dir: Path, output_dir: Path):
    """
    Analyze benchmark results and generate reports.

    Args:
        input_dir: Directory containing benchmark results
        output_dir: Directory for analysis output
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all run.json files
    run_files = list(input_dir.glob("**/run.json"))

    if not run_files:
        print(f"No run.json files found in {input_dir}")
        return

    print(f"Found {len(run_files)} benchmark runs")

    for run_file in run_files:
        print(f"\nAnalyzing: {run_file}")

        run = BenchmarkRun.load(str(run_file))
        results = list(run.results.values())

        # Calculate pass@k
        pk_result = calculate_pass_at_k(results, dataset=run.dataset)
        print(pk_result.to_markdown())

        # Calculate cost
        analyzer = CostAnalyzer()
        cost_metrics = analyzer.analyze(run)
        print(analyzer.to_markdown(cost_metrics))

    # Generate combined report
    report_file = output_dir / "analysis_report.md"
    with open(report_file, 'w') as f:
        f.write("# ATLAS V1 Benchmark Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for run_file in run_files:
            run = BenchmarkRun.load(str(run_file))
            results = list(run.results.values())

            f.write(f"## {run.dataset.upper()} ({run.run_id})\n\n")

            pk_result = calculate_pass_at_k(results, dataset=run.dataset)
            f.write(pk_result.to_markdown())
            f.write("\n\n")
            f.write(compare_with_baseline(pk_result))
            f.write("\n\n")

            analyzer = CostAnalyzer()
            cost_metrics = analyzer.analyze(run)
            f.write(analyzer.to_markdown(cost_metrics))
            f.write("\n\n---\n\n")

    print(f"\nReport saved to {report_file}")


def cost_analysis(input_dir: Path, output_dir: Path):
    """
    Generate detailed cost analysis.

    Args:
        input_dir: Directory containing benchmark results
        output_dir: Directory for cost analysis output
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    run_files = list(input_dir.glob("**/run.json"))

    if not run_files:
        print(f"No run.json files found in {input_dir}")
        return

    analyzer = CostAnalyzer()

    report_file = output_dir / "cost_analysis.md"
    with open(report_file, 'w') as f:
        f.write("# ATLAS V1 Cost Analysis\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for run_file in run_files:
            run = BenchmarkRun.load(str(run_file))
            metrics = analyzer.analyze(run)

            f.write(f"## {run.dataset.upper()}\n\n")
            f.write(analyzer.to_markdown(metrics))
            f.write("\n\n---\n\n")

    print(f"Cost analysis saved to {report_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ATLAS Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run HumanEval in dry-run mode (validation only)
    atlas benchmark --humaneval --dry-run

    # Run HumanEval with pass@1
    atlas benchmark --humaneval --k 1

    # Run HumanEval with pass@20
    atlas benchmark --humaneval --k 20 --runs 3

    # Analyze results
    atlas benchmark analyze --input benchmark/results/v1/ --output benchmark/results/v1/analysis/
"""
    )

    # Dataset selection (multiple allowed)
    parser.add_argument("--humaneval", action="store_true", help="Run HumanEval benchmark")
    parser.add_argument("--mbpp", action="store_true", help="Run MBPP benchmark (3-shot)")
    parser.add_argument("--humaneval-plus", action="store_true", help="Run HumanEval+ (EvalPlus) benchmark")
    parser.add_argument("--mbpp-plus", action="store_true", help="Run MBPP+ (EvalPlus) benchmark")
    parser.add_argument("--livecodebench", action="store_true", help="Run LiveCodeBench benchmark")
    parser.add_argument("--scicode", action="store_true", help="Run SciCode benchmark")
    parser.add_argument("--custom", action="store_true", help="Run custom benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    # Run options
    parser.add_argument("--dry-run", action="store_true", help="Validate without LLM calls")
    parser.add_argument("--k", type=int, default=1, help="Number of attempts per task (default: 1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs (default: 1)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--model", type=str, help="LLM endpoint URL")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results, skipping completed tasks")

    # Analysis subcommands
    parser.add_argument("analyze", nargs="?", help="Run analysis on results")
    parser.add_argument("cost", nargs="?", help="Run cost analysis")
    parser.add_argument("--input", type=str, help="Input directory for analysis")

    args = parser.parse_args()

    # Handle analysis subcommands
    if args.analyze == "analyze":
        if not args.input:
            parser.error("--input required for analyze")
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else input_dir / "analysis"
        analyze_results(input_dir, output_dir)
        return

    if args.cost == "cost":
        if not args.input:
            parser.error("--input required for cost")
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else input_dir / "cost"
        cost_analysis(input_dir, output_dir)
        return

    # Determine which datasets to run
    datasets = []
    if args.all:
        datasets.extend([
            "humaneval", "mbpp", "humaneval_plus", "mbpp_plus",
            "livecodebench", "scicode", "custom",
        ])
    else:
        if args.humaneval:
            datasets.append("humaneval")
        if args.mbpp:
            datasets.append("mbpp")
        if args.humaneval_plus:
            datasets.append("humaneval_plus")
        if args.mbpp_plus:
            datasets.append("mbpp_plus")
        if args.livecodebench:
            datasets.append("livecodebench")
        if args.scicode:
            datasets.append("scicode")
        if args.custom:
            datasets.append("custom")

    if not datasets:
        parser.print_help()
        return

    # Set up output directory
    output_base = Path(args.output) if args.output else config.results_dir

    # Run benchmarks
    for dataset in datasets:
        output_dir = output_base / dataset if len(datasets) > 1 else output_base

        # Set up logging
        run_id = f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = setup_logging(output_dir, run_id, resume=args.resume)

        try:
            run_benchmark_suite(
                dataset_name=dataset,
                dry_run=args.dry_run,
                k=args.k,
                runs=args.runs,
                output_dir=output_dir,
                model_url=args.model,
                logger=logger,
                resume=args.resume
            )
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")
            if not args.dry_run:
                raise
        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    main()
