#!/usr/bin/env python3
"""
Validate custom benchmark tasks.

Runs all canonical solutions against their test cases to ensure correctness.
"""

import hashlib
import json
import sys
import tempfile
import subprocess
from pathlib import Path


def check_lock(tasks_file: Path) -> bool:
    """Verify tasks.json matches the approved hash in tasks.json.lock.

    The lock records the SHA-256 of the task set that was human-reviewed;
    a drifted hash means the tasks changed without re-approval, which
    would silently change what the benchmark measures. Missing lock file
    is a warning (older checkouts), mismatch is a failure.
    """
    lock_file = tasks_file.with_suffix('.json.lock')
    if not lock_file.exists():
        print(f"Warning: {lock_file.name} not found — task set is unapproved")
        return True
    locked = lock_file.read_text().split()[0]
    actual = hashlib.sha256(tasks_file.read_bytes()).hexdigest()
    if actual != locked:
        print(f"Error: {tasks_file.name} does not match its approved hash")
        print(f"  locked:  {locked}")
        print(f"  actual:  {actual}")
        print("If the change is intentional, re-approve: update "
              f"{lock_file.name} with the new hash, approver, and date.")
        return False
    print(f"Lock OK: {tasks_file.name} matches approved hash\n")
    return True


def validate_task(task: dict) -> dict:
    """
    Validate a single task by running its canonical solution against tests.

    Args:
        task: Task dictionary with canonical_solution and test_code

    Returns:
        Dict with task_id, passed, error
    """
    task_id = task['task_id']

    # Combine solution and tests
    code = task['canonical_solution'] + '\n\n' + task['test_code']

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {'task_id': task_id, 'passed': True, 'error': None}
        else:
            return {'task_id': task_id, 'passed': False, 'error': result.stderr}

    except subprocess.TimeoutExpired:
        return {'task_id': task_id, 'passed': False, 'error': 'Timeout'}
    except Exception as e:
        return {'task_id': task_id, 'passed': False, 'error': str(e)}
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    """Validate all custom tasks."""
    tasks_file = Path(__file__).parent / 'tasks.json'

    if not tasks_file.exists():
        print(f"Error: {tasks_file} not found")
        sys.exit(1)

    if not check_lock(tasks_file):
        sys.exit(1)

    with open(tasks_file, 'r') as f:
        data = json.load(f)

    tasks = data.get('tasks', [])
    print(f"Validating {len(tasks)} tasks...\n")

    passed = 0
    failed = 0
    failures = []

    for task in tasks:
        result = validate_task(task)

        if result['passed']:
            passed += 1
            print(f"  [PASS] {result['task_id']}")
        else:
            failed += 1
            failures.append(result)
            print(f"  [FAIL] {result['task_id']}")
            if result['error']:
                # Print first few lines of error
                error_lines = result['error'].strip().split('\n')
                for line in error_lines[:3]:
                    print(f"         {line}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    if failed > 0:
        print("\nFailed tasks:")
        for f in failures:
            print(f"  - {f['task_id']}")
        sys.exit(1)
    else:
        print("\nAll tasks validated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
