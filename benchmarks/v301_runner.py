#!/usr/bin/env python3
"""
ATLAS V3.0.1 Benchmark Suite Runner.

General-purpose benchmark runner for the 26-benchmark suite.
Supports MCQ (raw pipeline) and coding (V3 pipeline) modes.

Usage:
    python -m benchmarks.v301_runner --benchmark gpqa_diamond
    python -m benchmarks.v301_runner --benchmark gpqa_diamond --dry-run
    python -m benchmarks.v301_runner --benchmark gpqa_diamond --limit 10
"""

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --- Configuration -----------------------------------------------------------

LLAMA_URL = os.environ.get("LLAMA_GEN_URL", os.environ.get("LLAMA_URL", "http://localhost:8000"))
LLAMA_GEN_MODEL = os.environ.get("LLAMA_GEN_MODEL", "qwen3.5-9b")
SEED = 42
MAX_TOKENS_MCQ = 12288  # High — Qwen3.5 uses thinking tokens from this budget; 8192 too short for hard GPQA
# Qwen3.5-9B published sampling for thinking-mode benchmarks:
TEMPERATURE = 1.0       # Qwen3.5 published (was 0.6 in V3.0.1-dev; now matches Qwen baseline)
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5  # Qwen3.5 published
BOOTSTRAP_N = 1000


# --- Utilities ----------------------------------------------------------------

def atomic_write_json(filepath: Path, data: Any) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        shutil.move(str(tmp), str(filepath))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def append_jsonl(filepath: Path, record: dict) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + '\n')


def bootstrap_ci(scores: List[float], n: int = BOOTSTRAP_N,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute mean and 95% CI via bootstrap resampling."""
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(SEED)
    means = []
    for _ in range(n):
        sample = [rng.choice(scores) for _ in range(len(scores))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(n * alpha / 2)]
    hi = means[int(n * (1 - alpha / 2))]
    mean = sum(scores) / len(scores)
    return mean, lo, hi


# --- LLM Client --------------------------------------------------------------

class LLMClient:
    """Thin client for vLLM gen instance chat completions."""

    def __init__(self, url: str = LLAMA_URL, max_retries: int = 3):
        self.url = url
        self.max_retries = max_retries

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = TEMPERATURE,
             max_tokens: int = MAX_TOKENS_MCQ,
             seed: int = None) -> Tuple[str, str, int, float]:
        """Send a chat completion request.

        Returns:
            (content, reasoning_content, tokens_used, latency_ms)
        """
        body = {
            "model": LLAMA_GEN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "presence_penalty": PRESENCE_PENALTY,
            "stream": False,
            # Force thinking on — vLLM with --reasoning-parser qwen3 surfaces it in reasoning_content.
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if seed is not None:
            body["seed"] = seed

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                req = urllib.request.Request(
                    f"{self.url}/v1/chat/completions",
                    data=json.dumps(body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=7200) as resp:
                    data = json.loads(resp.read().decode('utf-8'))

                latency_ms = (time.time() - start) * 1000
                choice = data['choices'][0]['message']
                content = choice.get('content', '')
                reasoning = choice.get('reasoning_content', '')
                tokens = data.get('usage', {}).get('completion_tokens', 0)
                # Strip stray </think> tags from reasoning-off mode
                content = re.sub(r'^</think>\s*', '', content).strip()
                return content, reasoning, tokens, latency_ms

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code == 503:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except urllib.error.URLError as e:
                last_error = str(e)
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def completion_nothink(self, system: str, user: str,
                           temperature: float = TEMPERATURE,
                           max_tokens: int = 1024,
                           seed: int = None) -> Tuple[str, int, float]:
        """Send a chat completion with thinking disabled.

        Much faster than chat() because thinking is off. Use for large MCQ
        benchmarks where thinking would be too slow.

        On vLLM the soft `/nothink` command was removed in Qwen3.5 — disable
        thinking via chat_template_kwargs.enable_thinking instead.

        Returns:
            (content, tokens_used, latency_ms)
        """
        body = {
            "model": LLAMA_GEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "presence_penalty": PRESENCE_PENALTY,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            body["seed"] = seed

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                req = urllib.request.Request(
                    f"{self.url}/v1/chat/completions",
                    data=json.dumps(body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read().decode('utf-8'))

                latency_ms = (time.time() - start) * 1000
                choice = data['choices'][0]['message']
                content = choice.get('content', '') or ''
                tokens = data.get('usage', {}).get('completion_tokens', 0)

                # Defensive: strip any leaked think markers.
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
                if '</think>' in content and '<think>' not in content:
                    content = content[content.index('</think>') + len('</think>'):].strip()
                if '<think>' in content:
                    content = content[content.index('<think>') + len('<think>'):].strip()
                return content.strip(), tokens, latency_ms

            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code == 503:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except urllib.error.URLError as e:
                last_error = str(e)
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = str(e)
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Completion failed after {self.max_retries} retries: {last_error}")


# --- MCQ Answer Extraction (from datasets/gpqa.py) ----------------------------

_ANSWER_PATTERNS = [
    r'(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])',
    r'\\boxed\{[^}]*([A-Z])[^}]*\}',
    r'answer is ([a-zA-Z])',
    r'answer is \\\(([a-zA-Z])',
    r'([A-Z])\)\s*[^A-Z]*$',
    r'([A-Z])\s+is\s+the\s+correct\s+answer',
    r'([A-Z])\s*$',
    r'([A-Z])\s*\.',
    r'([A-Z])\s*[^\w]',
]


def extract_mcq_answer(response: str, valid_options: str = "ABCD") -> Optional[str]:
    """Extract single-letter answer from LLM response."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, response)
        if matches:
            answer = matches[-1].upper()
            if answer in valid_options:
                return answer
    return None


# --- Benchmark Runner ---------------------------------------------------------

class BenchmarkRunner:
    """Run a benchmark and produce audit-grade artifacts."""

    def __init__(self, benchmark_name: str, section_name: str,
                 baseline_score: float, output_dir: Path = None,
                 dry_run: bool = False, limit: int = None):
        self.benchmark_name = benchmark_name
        self.section_name = section_name
        self.baseline_score = baseline_score
        self.dry_run = dry_run
        self.limit = limit

        if output_dir is None:
            output_dir = PROJECT_ROOT / "benchmarks" / section_name / benchmark_name
        self.output_dir = Path(output_dir)
        self.responses_path = self.output_dir / "responses.jsonl"
        self.traces_dir = self.output_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        self.llm = LLMClient()
        self.start_time = None
        self.results = []

    def find_completed(self) -> set:
        """Find already-completed task IDs for crash recovery."""
        completed = set()
        if self.responses_path.exists():
            with open(self.responses_path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if 'task_id' in rec:
                            completed.add(rec['task_id'])
                    except json.JSONDecodeError:
                        pass
        return completed

    def run_mcq(self, tasks: List[Dict[str, Any]],
                system_prompt: str = None,
                valid_options: str = "ABCD") -> Dict[str, Any]:
        """Run MCQ-format benchmark.

        Args:
            tasks: List of dicts with keys: task_id, prompt, correct_answer,
                   and optional metadata fields
            system_prompt: Optional system prompt override
            valid_options: Valid answer letters (default ABCD; pass "ABCDEFGHIJ" for 10-option MCQ)

        Returns:
            Aggregate results dict
        """
        self.start_time = datetime.now(timezone.utc)
        completed = self.find_completed()

        if system_prompt is None:
            system_prompt = (
                "You are an expert in science and reasoning. "
                "Think step by step. "
                "End your response with 'Answer: X' where X is the letter."
            )

        total = len(tasks)
        if self.limit:
            tasks = tasks[:self.limit]
            print(f"[limit] Running {len(tasks)}/{total} tasks")

        correct = 0
        attempted = 0
        skipped_completed = 0
        extraction_failures = 0

        pending = [(i, t) for i, t in enumerate(tasks)
                   if t['task_id'] not in completed]
        skipped_completed = len(tasks) - len(pending)

        if self.dry_run:
            for i, task in pending[:5]:
                print(f"[dry-run] {task['task_id']}")
            return {"dry_run": True, "task_count": len(pending)}

        state_lock = threading.Lock()

        def run_task(args):
            i, task = args
            task_id = task['task_id']
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task['prompt']},
            ]
            try:
                content, reasoning, tokens, latency_ms = self.llm.chat(
                    messages, seed=SEED + i
                )
                err = None
            except Exception as e:
                content = ""
                reasoning = ""
                tokens = 0
                latency_ms = 0
                err = str(e)

            if err:
                record = {
                    "task_id": task_id,
                    "error": err,
                    "pass": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with state_lock:
                    append_jsonl(self.responses_path, record)
                    self.results.append(0.0)
                return task_id, None, None, tokens, latency_ms, True

            extracted = extract_mcq_answer(content, valid_options)
            if extracted is None and not content.strip() and reasoning:
                extracted = extract_mcq_answer(reasoning, valid_options)
            is_correct = extracted == task['correct_answer']

            record = {
                "task_id": task_id,
                "content": content,
                "reasoning_content": reasoning,
                "extracted_answer": extracted,
                "correct_answer": task['correct_answer'],
                "pass": is_correct,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for k in ('domain', 'difficulty', 'category'):
                if k in task:
                    record[k] = task[k]

            with state_lock:
                append_jsonl(self.responses_path, record)
                self.results.append(1.0 if is_correct else 0.0)

            trace_path = self.traces_dir / f"{task_id.replace('/', '_')}.json"
            atomic_write_json(trace_path, record)

            return task_id, extracted, task['correct_answer'], tokens, latency_ms, False

        PARALLEL = int(os.environ.get('BENCHMARK_PARALLEL', '1'))
        print(f"[parallel={PARALLEL}] Processing {len(pending)} pending tasks")
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futures = {pool.submit(run_task, (i, t)): (i, t) for i, t in pending}
            for future in as_completed(futures):
                task_id, extracted, expected, tokens, latency_ms, errored = future.result()
                with state_lock:
                    attempted += 1
                    idx = attempted
                if errored:
                    print(f"[{idx}/{len(pending)}] {task_id} ERROR [{tokens}tok]", flush=True)
                elif extracted == expected:
                    with state_lock:
                        correct += 1
                    print(f"[{idx}/{len(pending)}] {task_id} CORRECT ({extracted}) [{latency_ms:.0f}ms, {tokens}tok]", flush=True)
                elif extracted is None:
                    with state_lock:
                        extraction_failures += 1
                    print(f"[{idx}/{len(pending)}] {task_id} EXTRACT_FAIL [{latency_ms:.0f}ms, {tokens}tok]", flush=True)
                else:
                    print(f"[{idx}/{len(pending)}] {task_id} WRONG ({extracted}!={expected}) [{latency_ms:.0f}ms, {tokens}tok]", flush=True)
                if attempted % 50 == 0:
                    with state_lock:
                        self._write_progress(correct, attempted, extraction_failures)

        if self.dry_run:
            print(f"\n[dry-run] Would run {len(tasks)} tasks")
            return {"dry_run": True, "task_count": len(tasks)}

        # Also count previously completed tasks
        if skipped_completed > 0:
            print(f"\n[resume] Loaded {skipped_completed} previously completed results")
            self._reload_completed_results()

        return self._finalize(attempted, extraction_failures)

    def _reload_completed_results(self):
        """Reload results from existing responses.jsonl."""
        self.results = []
        if self.responses_path.exists():
            with open(self.responses_path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.results.append(1.0 if rec.get('pass', False) else 0.0)
                    except json.JSONDecodeError:
                        pass

    def run_mcq_nothink(self, tasks: List[Dict[str, Any]],
                        system_prompt: str = None,
                        valid_options: str = "ABCDEFGHIJ") -> Dict[str, Any]:
        """Run MCQ benchmark with thinking disabled (fast mode).

        Uses /v1/chat/completions with chat_template_kwargs.enable_thinking=false.
        Much faster (~10-20s per question vs ~300s with thinking).
        Use for large benchmarks (1000+ questions).

        Args:
            tasks: List of dicts with task_id, prompt, correct_answer
            system_prompt: System prompt
            valid_options: Valid answer letters (default: A-J for 10-option MCQ)
        """
        self.start_time = datetime.now(timezone.utc)
        completed = self.find_completed()

        if system_prompt is None:
            system_prompt = (
                "You are an expert. Think step by step, then give your answer. "
                "End your response with 'Answer: X' where X is the letter."
            )

        total = len(tasks)
        if self.limit:
            tasks = tasks[:self.limit]
            print(f"[limit] Running {len(tasks)}/{total} tasks")

        correct = 0
        attempted = 0
        skipped_completed = 0
        extraction_failures = 0

        pending = [(i, t) for i, t in enumerate(tasks)
                   if t['task_id'] not in completed]
        skipped_completed = len(tasks) - len(pending)

        if self.dry_run:
            for i, task in pending[:5]:
                print(f"[dry-run] {task['task_id']}")
            return {"dry_run": True, "task_count": len(pending)}

        state_lock = threading.Lock()

        def run_task(args):
            i, task = args
            task_id = task['task_id']
            try:
                content, tokens, latency_ms = self.llm.completion_nothink(
                    system=system_prompt,
                    user=task['prompt'],
                    max_tokens=1024,
                    seed=SEED + i,
                )
                err = None
            except Exception as e:
                content = ""
                tokens = 0
                latency_ms = 0
                err = str(e)

            if err:
                record = {
                    "task_id": task_id, "error": err, "pass": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with state_lock:
                    append_jsonl(self.responses_path, record)
                    self.results.append(0.0)
                return task_id, None, None, tokens, latency_ms, True

            extracted = extract_mcq_answer(content, valid_options)
            is_correct = extracted == task['correct_answer']

            record = {
                "task_id": task_id,
                "content": content,
                "extracted_answer": extracted,
                "correct_answer": task['correct_answer'],
                "pass": is_correct,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for k in ('domain', 'difficulty', 'category'):
                if k in task:
                    record[k] = task[k]
            with state_lock:
                append_jsonl(self.responses_path, record)
                self.results.append(1.0 if is_correct else 0.0)

            return task_id, extracted, task['correct_answer'], tokens, latency_ms, False

        PARALLEL = int(os.environ.get('BENCHMARK_PARALLEL', '1'))
        print(f"[parallel={PARALLEL}] Processing {len(pending)} pending tasks")
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futures = {pool.submit(run_task, (i, t)): (i, t) for i, t in pending}
            for future in as_completed(futures):
                task_id, extracted, expected, tokens, latency_ms, errored = future.result()
                with state_lock:
                    attempted += 1
                    idx = attempted
                if errored:
                    print(f"[{idx}/{len(pending)}] {task_id} ERROR [{tokens}tok]", flush=True)
                elif extracted == expected:
                    with state_lock:
                        correct += 1
                    print(f"[{idx}/{len(pending)}] {task_id} CORRECT ({extracted}) [{latency_ms:.0f}ms, {tokens}tok]", flush=True)
                elif extracted is None:
                    with state_lock:
                        extraction_failures += 1
                    print(f"[{idx}/{len(pending)}] {task_id} EXTRACT_FAIL [{latency_ms:.0f}ms, {tokens}tok]", flush=True)
                else:
                    print(f"[{idx}/{len(pending)}] {task_id} WRONG ({extracted}!={expected}) [{latency_ms:.0f}ms, {tokens}tok]", flush=True)
                if attempted % 100 == 0:
                    with state_lock:
                        self._write_progress(correct, attempted, extraction_failures)

        if self.dry_run:
            return {"dry_run": True, "task_count": len(tasks)}

        if skipped_completed > 0:
            print(f"\n[resume] Loaded {skipped_completed} previously completed results")
            self._reload_completed_results()

        return self._finalize(attempted, extraction_failures)

    def _write_progress(self, correct: int, attempted: int,
                        extraction_failures: int):
        """Write a progress snapshot."""
        if attempted == 0:
            return
        acc = correct / attempted * 100
        progress = {
            "benchmark": self.benchmark_name,
            "attempted": attempted,
            "correct": correct,
            "accuracy": round(acc, 2),
            "extraction_failures": extraction_failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.output_dir / "progress.json", progress)
        print(f"\n  [snapshot] {correct}/{attempted} = {acc:.1f}%\n")

    def _finalize(self, attempted: int,
                  extraction_failures: int) -> Dict[str, Any]:
        """Compute final metrics and write all artifacts."""
        mean, ci_lo, ci_hi = bootstrap_ci(self.results)
        accuracy = mean * 100

        results = {
            "benchmark": self.benchmark_name,
            "section": self.section_name,
            "model": LLAMA_GEN_MODEL,
            "pipeline": "ATLAS V3.0.1",
            "total_tasks": len(self.results),
            "correct": sum(1 for r in self.results if r > 0),
            "accuracy": round(accuracy, 2),
            "ci_95_low": round(ci_lo * 100, 2),
            "ci_95_high": round(ci_hi * 100, 2),
            "extraction_failures": extraction_failures,
            "baseline_qwen": self.baseline_score,
            "delta": round(accuracy - self.baseline_score, 2),
            "seed": SEED,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS_MCQ,
            "bootstrap_n": BOOTSTRAP_N,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }

        # Write results.json
        atomic_write_json(self.output_dir / "results.json", results)

        # Write REPORT.md
        self._write_report(results)

        # Write sample_questions.jsonl
        self._write_sample_questions()

        # Clean up progress file
        progress_path = self.output_dir / "progress.json"
        if progress_path.exists():
            progress_path.unlink()

        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE: {self.benchmark_name}")
        print(f"{'='*60}")
        print(f"  Score:    {accuracy:.1f}% ({results['correct']}/{results['total_tasks']})")
        print(f"  95% CI:   [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print(f"  Baseline: {self.baseline_score}%")
        print(f"  Delta:    {results['delta']:+.1f}%")
        if extraction_failures > 0:
            print(f"  Extract failures: {extraction_failures}")
        print(f"{'='*60}")

        return results

    def _write_report(self, results: Dict[str, Any]):
        """Write REPORT.md with Qwen baseline comparison."""
        report = f"""# {self.benchmark_name.upper()} — ATLAS V3.0.1 Benchmark Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model:** Qwen3.5-9B-AWQ (AWQ-Q4 quantization)
**Pipeline:** ATLAS V3.0.1
**Seed:** {SEED}

## Results

| Metric | Value |
|--------|-------|
| ATLAS V3.0.1 | {results['accuracy']:.1f}% |
| 95% CI | [{results['ci_95_low']:.1f}%, {results['ci_95_high']:.1f}%] |
| Qwen3.5-9B baseline | {results['baseline_qwen']}% |
| Delta | {results['delta']:+.1f}% |
| Total tasks | {results.get('total_tasks', results.get('total_prompts', 'N/A'))} |
| Correct | {results.get('correct', results.get('strict_prompt_correct', results.get('loose_prompt_correct', 'N/A')))} |
| Extraction failures | {results.get('extraction_failures', 'N/A')} |

## Methodology

ATLAS V3.0.1 was evaluated in production-simulation mode wherever feasible.
Code, agent, and long-form tasks were routed through the ATLAS CLI (or Aider
on top of ATLAS) to match real-world usage. Raw-pipeline execution was used
only for benchmarks whose evaluation format (e.g., single-letter MCQ answers)
would be corrupted by the CLI's tool-calling behavior.

### Execution mode
**Raw pipeline** — MCQ questions sent directly to the model via
`/v1/chat/completions` endpoint.

### Sampling parameters
- Temperature: {TEMPERATURE}
- Top-P: {TOP_P}
- Max tokens: {MAX_TOKENS_MCQ}
- Seed: {SEED}

### Known caveats
- **Quantization gap:** Qwen baseline uses full bf16; ATLAS uses AWQ-Q4 quantization
- **Sampling divergence:** Qwen uses temp=1.0/top_p=0.95/top_k=20/presence_penalty=1.5;
  ATLAS uses temp={TEMPERATURE}/top_p={TOP_P}
- **Pipeline overhead:** ATLAS wraps the model with PlanSearch, Budget Forcing,
  Geometric Lens, etc. For MCQ benchmarks, these are largely bypassed.
- **Thinking mode:** Qwen baseline uses thinking mode by default; ATLAS allows
  the model to think (reasoning_content field) but does not control it.

## Artifact locations
- `responses.jsonl` — Full model responses for every task
- `traces/` — Per-task trace files
- `results.json` — Aggregate metrics
- `sample_questions.jsonl` — 20 representative problems
"""
        report_path = self.output_dir / "REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)

    def _write_sample_questions(self, n: int = 20):
        """Write stratified sample of questions from responses."""
        if not self.responses_path.exists():
            return

        records = []
        with open(self.responses_path, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        if not records:
            return

        # Stratified: take some correct, some wrong
        correct = [r for r in records if r.get('pass')]
        wrong = [r for r in records if not r.get('pass')]

        rng = random.Random(SEED)
        sample = []
        # Take up to n/2 from each
        half = min(n // 2, len(correct))
        sample.extend(rng.sample(correct, min(half, len(correct))))
        remaining = n - len(sample)
        sample.extend(rng.sample(wrong, min(remaining, len(wrong))))

        if len(sample) < n and len(correct) > half:
            extra = rng.sample(correct, min(n - len(sample), len(correct) - half))
            sample.extend(extra)

        sample_path = self.output_dir / "sample_questions.jsonl"
        with open(sample_path, 'w') as f:
            for rec in sample[:n]:
                # Slim down for sample
                slim = {
                    "task_id": rec.get("task_id"),
                    "pass": rec.get("pass"),
                    "extracted_answer": rec.get("extracted_answer"),
                    "correct_answer": rec.get("correct_answer"),
                }
                f.write(json.dumps(slim) + '\n')


    def run_ifeval(self, prompts_file: str) -> Dict[str, Any]:
        """Run IFEval benchmark using official Google checking library.

        Args:
            prompts_file: Path to IFEval prompts JSONL file

        Returns:
            Aggregate results dict with strict and loose accuracy
        """
        from benchmarks.eval_libs import evaluation_lib

        self.start_time = datetime.now(timezone.utc)
        completed = self.find_completed()

        # Load prompts
        inputs = evaluation_lib.read_prompt_list(prompts_file)
        total = len(inputs)

        if self.limit:
            inputs = inputs[:self.limit]
            print(f"[limit] Running {len(inputs)}/{total} tasks")

        # Generate responses
        prompt_to_response = {}
        system_prompt = "Follow the instructions exactly as given."

        pending = [(i, inp) for i, inp in enumerate(inputs)
                   if f"IFEval/{inp.key}" not in completed]

        if self.dry_run:
            for i, inp in pending[:5]:
                print(f"[dry-run] IFEval/{inp.key}")
            return {"dry_run": True, "task_count": len(pending)}

        write_lock = threading.Lock()
        attempted = 0

        def run_task(args):
            i, inp = args
            task_id = f"IFEval/{inp.key}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inp.prompt},
            ]
            try:
                content, reasoning, tokens, latency_ms = self.llm.chat(
                    messages, max_tokens=8192, seed=SEED + i
                )
            except Exception as e:
                content = f"[ERROR: {e}]"
                reasoning = ""
                tokens = 0
                latency_ms = 0

            record = {
                "task_id": task_id,
                "key": inp.key,
                "prompt": inp.prompt,
                "response": content,
                "reasoning_content": reasoning,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with write_lock:
                append_jsonl(self.responses_path, record)
                prompt_to_response[inp.prompt] = content
            return task_id, tokens, latency_ms

        PARALLEL = int(os.environ.get('BENCHMARK_PARALLEL', '1'))
        print(f"[parallel={PARALLEL}] Processing {len(pending)} pending tasks")
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futures = {pool.submit(run_task, (i, inp)): (i, inp) for i, inp in pending}
            for future in as_completed(futures):
                task_id, tokens, latency_ms = future.result()
                with write_lock:
                    attempted += 1
                    idx = attempted
                print(f"[{idx}/{len(pending)}] {task_id} [{latency_ms:.0f}ms, {tokens}tok]", flush=True)

        # Reload any previously completed responses
        if self.responses_path.exists():
            with open(self.responses_path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get('prompt') and rec.get('response'):
                            prompt_to_response[rec['prompt']] = rec['response']
                    except json.JSONDecodeError:
                        pass

        # Check using official IFEval library
        print("\n[checking] Running IFEval strict + loose checking...")

        strict_outputs = []
        loose_outputs = []
        for inp in inputs:
            if inp.prompt not in prompt_to_response:
                prompt_to_response[inp.prompt] = ""  # Missing = empty = fail
            strict_out = evaluation_lib.test_instruction_following_strict(
                inp, prompt_to_response)
            loose_out = evaluation_lib.test_instruction_following_loose(
                inp, prompt_to_response)
            strict_outputs.append(strict_out)
            loose_outputs.append(loose_out)

        # Compute metrics
        strict_prompt_correct = sum(1 for o in strict_outputs if o.follow_all_instructions)
        strict_inst_total = sum(len(o.instruction_id_list) for o in strict_outputs)
        strict_inst_correct = sum(sum(o.follow_instruction_list) for o in strict_outputs)

        loose_prompt_correct = sum(1 for o in loose_outputs if o.follow_all_instructions)
        loose_inst_total = sum(len(o.instruction_id_list) for o in loose_outputs)
        loose_inst_correct = sum(sum(o.follow_instruction_list) for o in loose_outputs)

        n = len(inputs)
        strict_prompt_acc = strict_prompt_correct / n * 100 if n else 0
        strict_inst_acc = strict_inst_correct / strict_inst_total * 100 if strict_inst_total else 0
        loose_prompt_acc = loose_prompt_correct / n * 100 if n else 0
        loose_inst_acc = loose_inst_correct / loose_inst_total * 100 if loose_inst_total else 0

        # IFEval standard metric is strict prompt-level accuracy
        accuracy = strict_prompt_acc

        # Bootstrap CI on strict prompt-level
        self.results = [1.0 if o.follow_all_instructions else 0.0 for o in strict_outputs]
        mean, ci_lo, ci_hi = bootstrap_ci(self.results)

        results = {
            "benchmark": self.benchmark_name,
            "section": self.section_name,
            "model": LLAMA_GEN_MODEL,
            "pipeline": "ATLAS V3.0.1",
            "total_prompts": n,
            "strict_prompt_accuracy": round(strict_prompt_acc, 2),
            "strict_prompt_correct": strict_prompt_correct,
            "strict_instruction_accuracy": round(strict_inst_acc, 2),
            "strict_instruction_correct": strict_inst_correct,
            "strict_instruction_total": strict_inst_total,
            "loose_prompt_accuracy": round(loose_prompt_acc, 2),
            "loose_prompt_correct": loose_prompt_correct,
            "loose_instruction_accuracy": round(loose_inst_acc, 2),
            "loose_instruction_correct": loose_inst_correct,
            "loose_instruction_total": loose_inst_total,
            "accuracy": round(accuracy, 2),
            "ci_95_low": round(ci_lo * 100, 2),
            "ci_95_high": round(ci_hi * 100, 2),
            "baseline_qwen": self.baseline_score,
            "delta": round(accuracy - self.baseline_score, 2),
            "seed": SEED,
            "temperature": TEMPERATURE,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }

        atomic_write_json(self.output_dir / "results.json", results)
        self._write_ifeval_report(results)
        self._write_sample_questions()

        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE: {self.benchmark_name}")
        print(f"{'='*60}")
        print(f"  Strict prompt-level: {strict_prompt_acc:.1f}% ({strict_prompt_correct}/{n})")
        print(f"  Strict inst-level:   {strict_inst_acc:.1f}% ({strict_inst_correct}/{strict_inst_total})")
        print(f"  Loose prompt-level:  {loose_prompt_acc:.1f}% ({loose_prompt_correct}/{n})")
        print(f"  Loose inst-level:    {loose_inst_acc:.1f}% ({loose_inst_correct}/{loose_inst_total})")
        print(f"  95% CI (strict):     [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print(f"  Baseline:            {self.baseline_score}%")
        print(f"  Delta:               {results['delta']:+.1f}%")
        print(f"{'='*60}")

        return results

    def _write_ifeval_progress(self, prompt_to_response, inputs):
        """Write IFEval progress snapshot."""
        from benchmarks.eval_libs import evaluation_lib
        correct = 0
        total = 0
        for inp in inputs:
            if inp.prompt in prompt_to_response:
                out = evaluation_lib.test_instruction_following_strict(
                    inp, prompt_to_response)
                if out.follow_all_instructions:
                    correct += 1
                total += 1
        if total:
            print(f"\n  [snapshot] strict prompt: {correct}/{total} = {correct/total*100:.1f}%\n")

    def _write_ifeval_report(self, results: Dict[str, Any]):
        """Write IFEval-specific REPORT.md."""
        report = f"""# IFEVAL — ATLAS V3.0.1 Benchmark Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model:** Qwen3.5-9B-AWQ (AWQ-Q4 quantization)
**Pipeline:** ATLAS V3.0.1
**Seed:** {SEED}

## Results

| Metric | Value |
|--------|-------|
| **Strict prompt-level accuracy** | **{results['strict_prompt_accuracy']:.1f}%** |
| Strict instruction-level accuracy | {results['strict_instruction_accuracy']:.1f}% |
| Loose prompt-level accuracy | {results['loose_prompt_accuracy']:.1f}% |
| Loose instruction-level accuracy | {results['loose_instruction_accuracy']:.1f}% |
| 95% CI (strict prompt) | [{results['ci_95_low']:.1f}%, {results['ci_95_high']:.1f}%] |
| Qwen3.5-9B baseline | {results['baseline_qwen']}% |
| Delta | {results['delta']:+.1f}% |
| Total prompts | {results['total_prompts']} |

## Methodology

IFEval (Instruction-Following Checking) tests whether models can follow
25 types of verifiable instructions (e.g., word count constraints, formatting
requirements, keyword inclusion). Checking uses the official Google Research
checking library from the IFEval paper (Zhou et al., 2023).

### Checking modes
- **Strict**: Response checked as-is
- **Loose**: 8 variations of the response tested (removing first/last lines,
  stripping asterisks); passes if any variation satisfies the constraint

### Execution mode
**Raw pipeline** - prompts sent directly to the model via /v1/chat/completions.

### Sampling parameters
- Temperature: {TEMPERATURE}
- Top-P: {TOP_P}
- Max tokens: 4096
- Seed: {SEED}

### Known caveats
- **Quantization gap:** Qwen baseline uses full bf16; ATLAS uses AWQ-Q4
- **Sampling divergence:** Qwen uses different sampling params
- **System prompt:** Simple "Follow the instructions exactly as given."
- **Checking library:** Official Google IFEval code (Apache 2.0)

## Artifact locations
- `responses.jsonl` - Full model responses for every prompt
- `results.json` - Aggregate metrics (strict + loose)
- `sample_questions.jsonl` - 20 representative problems
"""
        with open(self.output_dir / "REPORT.md", 'w') as f:
            f.write(report)

    def run_ifbench(self, prompts_file: str) -> Dict[str, Any]:
        """Run IFBench benchmark using official Allen AI evaluation library.

        Uses the official IFBench evaluation code (58 constraint types)
        from github.com/allenai/IFBench. Reports both strict and loose accuracy.

        Args:
            prompts_file: Path to IFBench prompts JSONL file

        Returns:
            Aggregate results dict
        """
        from benchmarks.eval_libs.ifbench import evaluation_lib as ifbench_eval

        self.start_time = datetime.now(timezone.utc)
        completed = self.find_completed()

        # Load prompts
        prompts = []
        with open(prompts_file, 'r') as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))

        total = len(prompts)
        if self.limit:
            prompts = prompts[:self.limit]
            print(f"[limit] Running {len(prompts)}/{total} tasks")

        # The system prompt for IFBench is inlined per-task at line ~1051;
        # the previous module-level `system_prompt = "/nothink\nFollow ..."`
        # was a Qwen2 holdover (the `/nothink` soft-command no longer works
        # on Qwen3.5) AND was never read by the closure below.

        attempted = 0
        attempted_lock = threading.Lock()

        # Filter out already-completed tasks
        pending = [(i, item) for i, item in enumerate(prompts)
                   if f"IFBench/{item.get('key', i)}" not in completed]

        if self.dry_run:
            for i, item in pending[:5]:
                print(f"[dry-run] IFBench/{item.get('key', i)}")
            return {"dry_run": True, "task_count": len(pending)}

        def run_task(args):
            i, item = args
            task_id = f"IFBench/{item.get('key', i)}"
            try:
                messages = [
                    {"role": "system", "content": "Follow the instructions exactly as given."},
                    {"role": "user", "content": item['prompt']},
                ]
                content, reasoning, tokens, latency_ms = self.llm.chat(
                    messages, max_tokens=8192, seed=SEED + i,
                )
            except Exception as e:
                content = f"[ERROR: {e}]"
                tokens = 0
                latency_ms = 0

            record = {
                "task_id": task_id,
                "prompt": item['prompt'],
                "response": content,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with attempted_lock:
                append_jsonl(self.responses_path, record)

            return task_id, tokens, latency_ms, len(content)

        # Parallel execution — 4 slots match vLLM gen instance --parallel 4
        PARALLEL = int(os.environ.get('BENCHMARK_PARALLEL', '1'))
        print(f"[parallel={PARALLEL}] Processing {len(pending)} pending tasks")

        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            futures = {pool.submit(run_task, (i, item)): (i, item) for i, item in pending}
            for future in as_completed(futures):
                task_id, tokens, latency_ms, content_len = future.result()
                with attempted_lock:
                    attempted += 1
                    idx = attempted
                print(f"[{idx}/{len(pending)}] {task_id}: {tokens}tok {latency_ms/1000:.0f}s content={content_len}c", flush=True)

        if self.dry_run:
            return {"dry_run": True, "task_count": len(prompts)}

        # Reload all responses for evaluation
        prompt_to_response = {}
        if self.responses_path.exists():
            with open(self.responses_path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get('prompt') and 'response' in rec:
                            prompt_to_response[rec['prompt']] = rec['response']
                    except json.JSONDecodeError:
                        pass

        # Run official IFBench evaluation (58 constraint types)
        print(f"\n[eval] Running official IFBench evaluation ({len(prompt_to_response)} responses)...")

        inputs = ifbench_eval.read_prompt_list(prompts_file)
        if self.limit:
            inputs = inputs[:self.limit]

        strict_outputs = []
        loose_outputs = []
        for inp in inputs:
            if inp.prompt not in prompt_to_response:
                prompt_to_response[inp.prompt] = ""
            strict_out = ifbench_eval.test_instruction_following_strict(
                inp, prompt_to_response)
            loose_out = ifbench_eval.test_instruction_following_loose(
                inp, prompt_to_response)
            strict_outputs.append(strict_out)
            loose_outputs.append(loose_out)

        # Compute metrics
        n = len(inputs)
        strict_prompt_correct = sum(1 for o in strict_outputs if o.follow_all_instructions)
        strict_inst_total = sum(len(o.instruction_id_list) for o in strict_outputs)
        strict_inst_correct = sum(sum(o.follow_instruction_list) for o in strict_outputs)
        loose_prompt_correct = sum(1 for o in loose_outputs if o.follow_all_instructions)
        loose_inst_total = sum(len(o.instruction_id_list) for o in loose_outputs)
        loose_inst_correct = sum(sum(o.follow_instruction_list) for o in loose_outputs)

        strict_prompt_acc = strict_prompt_correct / n * 100 if n else 0
        strict_inst_acc = strict_inst_correct / strict_inst_total * 100 if strict_inst_total else 0
        loose_prompt_acc = loose_prompt_correct / n * 100 if n else 0
        loose_inst_acc = loose_inst_correct / loose_inst_total * 100 if loose_inst_total else 0

        # IFBench paper reports loose prompt-level accuracy
        accuracy = loose_prompt_acc

        self.results = [1.0 if o.follow_all_instructions else 0.0 for o in loose_outputs]
        mean, ci_lo, ci_hi = bootstrap_ci(self.results)

        results = {
            "benchmark": self.benchmark_name,
            "section": self.section_name,
            "model": LLAMA_GEN_MODEL,
            "pipeline": "ATLAS V3.0.1",
            "total_prompts": n,
            "strict_prompt_accuracy": round(strict_prompt_acc, 2),
            "strict_prompt_correct": strict_prompt_correct,
            "strict_instruction_accuracy": round(strict_inst_acc, 2),
            "loose_prompt_accuracy": round(loose_prompt_acc, 2),
            "loose_prompt_correct": loose_prompt_correct,
            "loose_instruction_accuracy": round(loose_inst_acc, 2),
            "accuracy": round(accuracy, 2),
            "ci_95_low": round(ci_lo * 100, 2),
            "ci_95_high": round(ci_hi * 100, 2),
            "baseline_qwen": self.baseline_score,
            "delta": round(accuracy - self.baseline_score, 2),
            "seed": SEED,
            "temperature": TEMPERATURE,
            "evaluator": "official allenai/IFBench (58 constraints)",
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }

        atomic_write_json(self.output_dir / "results.json", results)
        self._write_report(results)
        self._write_sample_questions()

        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE: {self.benchmark_name}")
        print(f"{'='*60}")
        print(f"  Strict prompt-level: {strict_prompt_acc:.1f}% ({strict_prompt_correct}/{n})")
        print(f"  Strict inst-level:   {strict_inst_acc:.1f}%")
        print(f"  Loose prompt-level:  {loose_prompt_acc:.1f}% ({loose_prompt_correct}/{n})")
        print(f"  Loose inst-level:    {loose_inst_acc:.1f}%")
        print(f"  95% CI (loose):      [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print(f"  Baseline:            {self.baseline_score}%")
        print(f"  Delta:               {results['delta']:+.1f}%")
        print(f"{'='*60}")

        return results


# --- GPQA Diamond Benchmark --------------------------------------------------

def load_gpqa_diamond_tasks() -> List[Dict[str, Any]]:
    """Load GPQA Diamond tasks using the existing dataset loader."""
    from benchmark.datasets.gpqa import GPQADiamondDataset
    ds = GPQADiamondDataset()
    ds.load()

    tasks = []
    for t in ds.tasks:
        tasks.append({
            "task_id": t.task_id,
            "prompt": t.prompt,
            "correct_answer": t.canonical_solution,
            "domain": next((tag for tag in t.tags if tag in
                          ('physics', 'biology', 'chemistry', 'science')), 'science'),
            "difficulty": t.difficulty,
        })
    return tasks


# --- IFEval Benchmark ---------------------------------------------------------

def load_ifeval_prompts_file() -> str:
    """Return path to IFEval prompts JSONL file."""
    cache_dir = PROJECT_ROOT / "benchmark" / "datasets" / ".cache"
    filepath = cache_dir / "ifeval.jsonl"
    if not filepath.exists():
        raise FileNotFoundError(
            f"IFEval dataset not found at {filepath}. "
            "Download it first from HuggingFace google/IFEval."
        )
    return str(filepath)


# --- IFBench Benchmark --------------------------------------------------------

def load_ifbench_prompts_file() -> str:
    """Return path to IFBench prompts.

    Uses the existing IFBench dataset loader to download and cache.
    Returns path to the cached JSONL file.
    """
    from benchmark.datasets.ifbench import IFBenchDataset
    ds = IFBenchDataset()
    filepath = ds.download()
    return str(filepath)


# --- MMLU-Pro Benchmark -------------------------------------------------------

# MMLU-Pro full test set is 12032 questions; running all under Qwen methodology
# (thinking mode, 12288 max_tokens, parallel 4) projects to ~100 days on our hardware.
# We sample deterministically with seed 42. Both the baseline and the ATLAS V3
# pipeline read from this same loader, so the 1000-task subset is identical across
# runs — keeping the baseline vs ATLAS comparison apples-to-apples.
MMLU_PRO_SAMPLE = 1000
MMLU_PRO_SEED = 42


def load_mmlu_pro_tasks() -> List[Dict[str, Any]]:
    """Load MMLU-Pro from HuggingFace (12K test questions, 10-option MCQ).

    Returns a seeded sample of MMLU_PRO_SAMPLE tasks for runtime budget;
    deterministic across baseline and ATLAS V3 pipeline runs.
    """
    cache_dir = PROJECT_ROOT / "benchmark" / "datasets" / ".cache"
    filepath = cache_dir / "mmlu_pro.jsonl"

    if not filepath.exists():
        print("Downloading MMLU-Pro from HuggingFace...")
        all_rows = []
        offset = 0
        batch = 100
        max_retries = 5
        while True:
            url = (
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset=TIGER-Lab%2FMMLU-Pro&config=default"
                f"&split=test&offset={offset}&length={batch}"
            )
            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data_resp = json.loads(resp.read().decode('utf-8'))
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        wait = 2 ** (attempt + 1)
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            else:
                print(f"  Failed after {max_retries} retries at offset {offset}")
                break

            rows = data_resp.get('rows', [])
            if not rows:
                break
            for r in rows:
                all_rows.append(r['row'])
            offset += len(rows)
            if offset % 1000 == 0:
                print(f"  Downloaded {offset} rows...")
            if len(rows) < batch:
                break

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for row in all_rows:
                f.write(json.dumps(row) + '\n')
        print(f"Saved {len(all_rows)} questions to {filepath}")

    tasks = []
    letters = "ABCDEFGHIJ"
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            question = row.get('question', '')
            options = row.get('options', [])
            answer = row.get('answer', '')
            category = row.get('category', '')

            # Build MCQ prompt
            option_lines = []
            for j, opt in enumerate(options):
                if j < len(letters):
                    option_lines.append(f"{letters[j]}) {opt}")
            options_text = "\n".join(option_lines)

            prompt = (
                f"Answer the following multiple choice question. "
                f"End your response with 'Answer: X' where X is the letter.\n\n"
                f"{question}\n\n{options_text}\n"
            )

            valid_letters = letters[:len(options)]
            tasks.append({
                "task_id": f"MMLU-Pro/{idx+1:05d}",
                "prompt": prompt,
                "correct_answer": answer,
                "category": category,
            })

    # Deterministic sample (seed 42) so baseline and ATLAS runs see the same 1000 tasks.
    if len(tasks) > MMLU_PRO_SAMPLE:
        rng = random.Random(MMLU_PRO_SEED)
        tasks = rng.sample(tasks, MMLU_PRO_SAMPLE)
        tasks.sort(key=lambda t: t['task_id'])  # stable task ordering for resume logic
    return tasks


# --- SuperGPQA Benchmark -----------------------------------------------------

def load_supergpqa_tasks() -> List[Dict[str, Any]]:
    """Load SuperGPQA from cached JSONL (26K questions, 10-option MCQ)."""
    cache_dir = PROJECT_ROOT / "benchmark" / "datasets" / ".cache"
    filepath = cache_dir / "supergpqa.jsonl"

    if not filepath.exists():
        print("Downloading SuperGPQA...")
        url = "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl"
        urllib.request.urlretrieve(url, str(filepath))
        count = sum(1 for _ in open(filepath))
        print(f"Downloaded {count} questions to {filepath}")

    tasks = []
    letters = "ABCDEFGHIJ"
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            question = row.get('question', '')
            options = row.get('options', [])
            answer_letter = row.get('answer_letter', '')
            discipline = row.get('discipline', '')
            subfield = row.get('subfield', '')

            option_lines = []
            for j, opt in enumerate(options):
                if j < len(letters):
                    option_lines.append(f"{letters[j]}) {opt}")
            options_text = "\n".join(option_lines)

            prompt = (
                f"Answer the following multiple choice question. "
                f"End your response with 'Answer: X' where X is the letter.\n\n"
                f"{question}\n\n{options_text}\n"
            )

            tasks.append({
                "task_id": f"SuperGPQA/{idx+1:05d}",
                "prompt": prompt,
                "correct_answer": answer_letter,
                "category": f"{discipline}/{subfield}",
            })

    return tasks


# --- MMLU-Redux Benchmark -----------------------------------------------------

def load_mmlu_redux_tasks() -> List[Dict[str, Any]]:
    """Load MMLU-Redux 2.0 from cached JSONL (5600 questions, 4-option MCQ)."""
    cache_dir = PROJECT_ROOT / "benchmark" / "datasets" / ".cache"
    filepath = cache_dir / "mmlu_redux.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"MMLU-Redux not found at {filepath}. "
            "Download it first via the HuggingFace API."
        )

    tasks = []
    letters = "ABCD"
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            question = row.get('question', '')
            choices = row.get('choices', [])
            answer_idx = row.get('answer', 0)
            subject = row.get('subject', '')

            option_lines = []
            for j, opt in enumerate(choices):
                if j < len(letters):
                    option_lines.append(f"{letters[j]}) {opt}")
            options_text = "\n".join(option_lines)

            correct_letter = letters[answer_idx] if answer_idx < len(letters) else 'A'

            prompt = (
                f"Answer the following multiple choice question. "
                f"End your response with 'Answer: X' where X is the letter.\n\n"
                f"{question}\n\n{options_text}\n"
            )

            tasks.append({
                "task_id": f"MMLU-Redux/{idx+1:05d}",
                "prompt": prompt,
                "correct_answer": correct_letter,
                "category": subject,
            })

    return tasks


# --- C-Eval Benchmark --------------------------------------------------------

def load_ceval_tasks() -> List[Dict[str, Any]]:
    """Load C-Eval val split from cached JSONL (1346 Chinese MCQ questions)."""
    cache_dir = PROJECT_ROOT / "benchmark" / "datasets" / ".cache"
    filepath = cache_dir / "ceval.jsonl"

    if not filepath.exists():
        raise FileNotFoundError(
            f"C-Eval not found at {filepath}. "
            "Download it via the benchmark setup scripts."
        )

    tasks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            question = row.get('question', '')
            a = row.get('A', '')
            b = row.get('B', '')
            c = row.get('C', '')
            d = row.get('D', '')
            answer = row.get('answer', '')
            subject = row.get('subject', '')

            # Chinese prompt with standard MCQ format
            prompt = (
                "请回答以下选择题。在回答的最后一行写上'Answer: X'，其中X是选项字母。\n\n"
                f"{question}\n\n"
                f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n"
            )

            tasks.append({
                "task_id": f"C-Eval/{idx+1:04d}",
                "prompt": prompt,
                "correct_answer": answer,
                "category": subject,
            })

    return tasks


# --- CLI Entry Point ----------------------------------------------------------

BENCHMARKS = {
    "gpqa_diamond": {
        "section": "section_a_knowledge_stem",
        "baseline": 81.7,
        "mode": "mcq",
        "loader": load_gpqa_diamond_tasks,
    },
    "ifeval": {
        "section": "section_b_instruction_following",
        "baseline": 91.5,
        "mode": "ifeval",
        "loader": load_ifeval_prompts_file,
    },
    "ifbench": {
        "section": "section_b_instruction_following",
        "baseline": 64.5,
        "mode": "ifbench",
        "loader": load_ifbench_prompts_file,
    },
    "mmlu_pro": {
        "section": "section_a_knowledge_stem",
        "baseline": 82.5,
        "mode": "mcq",  # thinking mode per Qwen methodology (was mcq_nothink — 1024 token cap starved reasoning)
        "valid_options": "ABCDEFGHIJ",  # 10-option MCQ
        "loader": load_mmlu_pro_tasks,
    },
    "supergpqa": {
        "section": "section_a_knowledge_stem",
        "baseline": 58.2,
        "mode": "mcq_nothink",
        "loader": load_supergpqa_tasks,
    },
    "mmlu_redux": {
        "section": "section_a_knowledge_stem",
        "baseline": 91.1,
        "mode": "mcq_nothink",
        "loader": load_mmlu_redux_tasks,
    },
    "c_eval": {
        "section": "section_a_knowledge_stem",
        "baseline": 88.2,
        "mode": "mcq_nothink",
        "loader": load_ceval_tasks,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS V3.0.1 Benchmark Suite")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()),
                        help="Benchmark to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load tasks but don't run inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to run")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    bench = BENCHMARKS[args.benchmark]
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"ATLAS V3.0.1 Benchmark Suite")
    print(f"Benchmark: {args.benchmark}")
    print(f"Mode: {bench['mode']}")
    print(f"Baseline: {bench['baseline']}%")
    print(f"LLM URL: {LLAMA_URL}")
    print()

    runner = BenchmarkRunner(
        benchmark_name=args.benchmark,
        section_name=bench['section'],
        baseline_score=bench['baseline'],
        output_dir=output_dir,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    data = bench['loader']()

    if bench['mode'] == 'mcq':
        print(f"Loaded {len(data)} tasks\n")
        # MMLU-Pro has 10 options (A-J); GPQA and other 4-option MCQs default to A-D.
        valid_options = bench.get('valid_options', 'ABCD')
        results = runner.run_mcq(data, valid_options=valid_options)
    elif bench['mode'] == 'mcq_nothink':
        print(f"Loaded {len(data)} tasks (nothink mode)\n")
        results = runner.run_mcq_nothink(data)
    elif bench['mode'] == 'ifeval':
        print(f"IFEval prompts file: {data}\n")
        results = runner.run_ifeval(data)
    elif bench['mode'] == 'ifbench':
        print(f"IFBench prompts file: {data}\n")
        results = runner.run_ifbench(data)
    else:
        raise NotImplementedError(f"Mode {bench['mode']} not yet implemented")

    if not args.dry_run:
        print(f"\nArtifacts written to: {runner.output_dir}")


if __name__ == "__main__":
    main()
