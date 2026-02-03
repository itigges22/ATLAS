# ATLAS: Pre-Benchmark Hardening — 3 Tasks

You're working on the ATLAS project. The benchmark infrastructure is built and verified, but three issues need fixing before production runs. The codebase is at the project root, benchmarks in `benchmark/`.

---

## Task 1: Fix 9 Escaped Mutants in Custom Tasks

During mutation testing (replacing `return X` with `return None`), these 9 tasks' test suites DID NOT catch the mutant — meaning bad code could pass their tests and inflate pass@k scores:

**Escaped mutants:**
- `ALGO_003`, `ALGO_010`
- `DATA_010`, `DATA_016`
- `API_008`, `API_010`, `API_012`
- `TEST_002`
- `BUG_002`

For each of these 9 tasks in `benchmark/custom/tasks.json`:

1. Read the current `test_code` and `canonical_solution`
2. Identify WHY the mutation (`return None` replacing every `return`) was not caught — common reasons:
   - Tests only check truthiness, not specific values (e.g., `assert result` passes for any non-None)
   - Tests don't cover the return value at all (side-effect only tests)
   - Tests use loose comparisons that happen to pass with None
   - Tests only check one code path, and the mutated path isn't tested
3. Add or strengthen assertions so that `return None` ALWAYS causes at least one test to fail
4. Ensure the canonical_solution still passes all tests after your changes
5. Do NOT change the `prompt`, `entry_point`, `task_id`, or `canonical_solution` — only `test_code`

**Validation after fixing:**

```bash
# Run canonical solutions against updated tests (all 100 must pass)
python benchmark/custom/validate.py

# Run mutation test on just the 9 fixed tasks
python -c "
import json, subprocess, tempfile, os

with open('benchmark/custom/tasks.json') as f:
    data = json.load(f)
    tasks = data['tasks'] if isinstance(data, dict) and 'tasks' in data else data

targets = ['ALGO_003','ALGO_010','DATA_010','DATA_016','API_008','API_010','API_012','TEST_002','BUG_002']
caught = 0
for task in tasks:
    if not isinstance(task, dict) or task.get('task_id') not in targets:
        continue
    mutant = task['canonical_solution'].replace('return ', 'return None  # ')
    code = mutant + '\n' + task['test_code']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        result = subprocess.run(['python', f.name], capture_output=True, timeout=10)
        if result.returncode != 0:
            caught += 1
            print(f'  {task[\"task_id\"]}: CAUGHT ✓')
        else:
            print(f'  {task[\"task_id\"]}: ESCAPED ✗')
        os.unlink(f.name)
print(f'\nMutation detection: {caught}/9 targets caught')
assert caught == 9, f'FAIL: Only {caught}/9 mutants caught'
"
```

All 9 must say CAUGHT. Do not proceed until this passes.

---

## Task 2: Add Crash Recovery/Resume to run_full_benchmarks.sh

The benchmark script `scripts/run_full_benchmarks.sh` must be bulletproof for overnight runs. Currently if it crashes, everything restarts from zero. Fix this.

### Requirements:

**2a. Per-phase checkpoint files:** After each benchmark phase completes successfully, write a checkpoint file (e.g., `benchmark/results/full_run_<timestamp>/.checkpoint_humaneval_pass1`). On restart, skip phases that have checkpoint files.

**2b. Per-task resume within a phase:** The benchmark CLI already saves results per-task as JSON. On resume, the CLI should detect existing result files in the output directory and skip tasks that already have results. If this isn't already implemented in `benchmark/cli.py`, add a `--resume` flag (see Task 3).

**2c. Crash detection and retry with 3-strike rule:**

```bash
run_phase() {
    local phase_name="$1"
    local command="$2"
    local checkpoint="$3"
    local max_retries=3
    local retry_count=0
    local crash_log="${RESULTS_DIR}/crash_log.json"

    if [[ -f "$checkpoint" ]]; then
        echo "⏭ $phase_name already complete, skipping"
        return 0
    fi

    while [[ $retry_count -lt $max_retries ]]; do
        echo "▶ Running $phase_name (attempt $((retry_count + 1))/$max_retries)"
        local start_time=$(date +%s)

        # Run with resume flag
        if eval "$command --resume"; then
            touch "$checkpoint"
            echo "✓ $phase_name complete"
            # Log success to crash_log
            return 0
        else
            retry_count=$((retry_count + 1))
            local end_time=$(date +%s)
            # Log crash event as JSON to crash_log:
            # {timestamp, phase, attempt, exit_code, duration, last_stderr}
            echo "⚠ $phase_name crashed (attempt $retry_count/$max_retries)"

            if [[ $retry_count -lt $max_retries ]]; then
                echo "  Waiting 30s before retry..."
                sleep 30
            fi
        fi
    done

    # 3 strikes — log fatal error and continue to next phase
    echo "✗ $phase_name FAILED after $max_retries attempts"
    # Log detailed error to crash_log with stderr from last attempt
    # DO NOT exit — continue to next phase so partial results are still collected
    return 1
}
```

**2d. Crash log format** — `crash_log.json` should be an array of events:

```json
[
  {
    "timestamp": "2026-02-03T01:23:45Z",
    "phase": "humaneval_pass1",
    "attempt": 1,
    "exit_code": 137,
    "duration_seconds": 3421,
    "stderr_tail": "last 500 chars of stderr",
    "tasks_completed_before_crash": 87,
    "tasks_remaining": 77
  }
]
```

**2e. Final report includes crash data:** The benchmark report (generated in Phase 3) should include a "Run Health" section that shows:
- Total crashes across all phases
- Which phases required retries
- Whether any phases hit the 3-strike limit
- Total wall-clock time including retries

**2f. Signal handling:** Trap SIGINT/SIGTERM so that a manual Ctrl+C during a run still writes partial results cleanly before exiting rather than corrupting JSON files.

---

## Task 3: Add --resume Flag to benchmark/cli.py

If `--resume` isn't already supported in the CLI, add it:

1. When `--resume` is passed, before starting a benchmark phase:
   - Scan the output directory for existing result files
   - Build a set of already-completed task IDs
   - Filter the task list to exclude completed tasks
   - Log: `"Resuming: {completed}/{total} tasks already done, {remaining} remaining"`
2. When saving results, use atomic writes (write to `.tmp` file, then rename) to prevent corruption on crash
3. Append to existing `benchmark.log` instead of overwriting when resuming
4. Track resume events in the results metadata:

```json
{
  "resumed": true,
  "resumed_at": "2026-02-03T01:24:15Z",
  "tasks_skipped": 87,
  "tasks_remaining": 77,
  "previous_crash_exit_code": 137
}
```

---

## Validation

After all three tasks are done:

1. **All 100 canonical solutions pass:** `python benchmark/custom/validate.py`

2. **All 9 previously-escaped mutants now caught** (run the mutation script from Task 1 above)

3. **Resume logic works:**

```bash
# Simulate: run 5 tasks of custom pass@1, kill it, resume
timeout 120 python -m benchmark.cli --custom --k 1 --runs 1 --output /tmp/test_resume/ || true
# Count results
ls /tmp/test_resume/submissions/*.json 2>/dev/null | wc -l
# Resume — should pick up where it left off
python -m benchmark.cli --custom --k 1 --runs 1 --output /tmp/test_resume/ --resume
```

4. **Crash log is created when a run fails**

5. **`scripts/run_full_benchmarks.sh` respects checkpoints** (run it, kill it, re-run — should skip completed phases)
