#!/usr/bin/env python3
"""Validate a trained LoRA adapter before deployment."""

import json
import os
import sys
import requests

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://llama-service:8000")
VALIDATION_PROMPTS = [
    {
        "messages": [
            {"role": "user", "content": "Write a Python function to reverse a string"}
        ],
        "expected_contains": ["def", "return", "reverse"]
    },
    {
        "messages": [
            {"role": "user", "content": "Fix this bug: def add(a, b): return a - b"}
        ],
        "expected_contains": ["return", "+", "a + b"]
    },
    {
        "messages": [
            {"role": "user", "content": "Write a simple hello world in JavaScript"}
        ],
        "expected_contains": ["console", "log", "hello"]
    }
]

def validate_adapter(adapter_path: str = None) -> bool:
    """Run validation prompts against the model."""
    passed = 0
    failed = 0

    for test in VALIDATION_PROMPTS:
        try:
            response = requests.post(
                f"{LLM_ENDPOINT}/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": test["messages"],
                    "max_tokens": 200,
                    "temperature": 0.1
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].lower()

            # Check if response contains expected keywords
            all_found = all(kw.lower() in content for kw in test["expected_contains"])
            if all_found:
                passed += 1
                print(f"PASS: {test['messages'][0]['content'][:50]}...")
            else:
                failed += 1
                print(f"FAIL: {test['messages'][0]['content'][:50]}...")
                print(f"  Missing: {[kw for kw in test['expected_contains'] if kw.lower() not in content]}")

        except Exception as e:
            failed += 1
            print(f"ERROR: {test['messages'][0]['content'][:50]}... - {e}")

    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0

    print(f"\nValidation Results: {passed}/{total} passed ({success_rate:.1f}%)")

    # Require at least 66% pass rate
    return success_rate >= 66

if __name__ == "__main__":
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = validate_adapter(adapter_path)
    sys.exit(0 if success else 1)
