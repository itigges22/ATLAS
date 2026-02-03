#!/usr/bin/env python3
"""
Validate custom benchmark tasks.

Runs all canonical solutions against their test cases to ensure correctness.
"""

import json
import sys
import tempfile
import subprocess
from pathlib import Path


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
