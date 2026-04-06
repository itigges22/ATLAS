"""Bench command — run benchmarks with live progress."""

import json
import os
import sys
import time
from pathlib import Path

from atlas.cli import display


def bench(dataset: str = "livecodebench", max_tasks: int = 0,
          selection_strategy: str = "random"):
    """Run benchmark with live progress display.

    Delegates to the V3 runner but displays progress inline.
    """
    display.phase(f"Benchmark: {dataset}")

    # Build runner command
    run_id = f"bench_{dataset}_{int(time.time())}"
    cmd = [
        sys.executable, "-m", "benchmark.v3_runner",
        "--run-id", run_id,
        "--baseline",
        "--selection-strategy", selection_strategy,
    ]
    if max_tasks > 0:
        cmd.extend(["--max-tasks", str(max_tasks)])

    # Set Fox environment
    env = os.environ.copy()
    env["ATLAS_USE_FOX"] = "1"
    env["ATLAS_MODEL_NAME"] = os.environ.get("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")
    env["LLAMA_URL"] = os.environ.get("ATLAS_INFERENCE_URL", "http://localhost:8080")
    env["ATLAS_LLM_PARALLEL"] = "1"
    env["ATLAS_PARALLEL_TASKS"] = "1"

    display.info(f"Run ID: {run_id}")
    display.info(f"Strategy: {selection_strategy}")
    if max_tasks > 0:
        display.info(f"Tasks: {max_tasks}")

    import subprocess
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    results_dir = Path(f"benchmark/results/{run_id}/v3_lcb/per_task")
    pass_count = 0
    task_count = 0

    try:
        for line in proc.stdout:
            line = line.strip()
            # Parse runner output for progress
            if line.startswith("[") and "/" in line and "LCB" in line:
                task_count += 1
                if "PASS" in line:
                    pass_count += 1
                total = max_tasks if max_tasks > 0 else 599
                display.progress_bar(task_count, total, pass_count, line.split("]")[-1].strip()[:40])
            elif "BENCHMARK COMPLETE" in line:
                display.progress_done()
                display.newline()
    except KeyboardInterrupt:
        proc.terminate()
        display.warn("Benchmark interrupted")
        return

    proc.wait()

    # Final results
    if results_dir.exists():
        results = list(results_dir.glob("*.json"))
        p = sum(1 for f in results if json.load(open(f)).get("passed"))
        total = len(results)
        rate = p / max(total, 1) * 100
        display.separator()
        display.success(f"pass@1: {p}/{total} ({rate:.1f}%)")
        display.info(f"Results: benchmark/results/{run_id}/")
        display.separator()
    else:
        display.error("No results found")
