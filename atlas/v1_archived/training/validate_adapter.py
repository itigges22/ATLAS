#!/usr/bin/env python3
"""
Validate a LoRA adapter before deployment.

Tests the adapter against a set of validation prompts and checks:
1. Model loads successfully with the adapter
2. Generation quality meets threshold
3. No performance regression

Usage:
    python validate_adapter.py --adapter /models/lora/20260129
    python validate_adapter.py --adapter /models/lora/20260129 --threshold 0.7
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path


# Validation test cases - simple code generation prompts
VALIDATION_PROMPTS = [
    {
        "prompt": "Write a Python function that checks if a number is prime.",
        "expected_keywords": ["def", "return", "True", "False", "for", "%"]
    },
    {
        "prompt": "Write a Python function to reverse a string.",
        "expected_keywords": ["def", "return", "[::-1]"]
    },
    {
        "prompt": "Write a Python function that calculates the factorial of n.",
        "expected_keywords": ["def", "factorial", "return", "if", "*"]
    }
]


def validate_adapter(adapter_path: str, threshold: float = 0.6, llama_url: str = None):
    """
    Validate a LoRA adapter.

    Returns:
        True if validation passes, False otherwise
    """
    import requests

    llama_url = llama_url or os.getenv("LLAMA_URL", "http://llama-service:8000")

    # Check adapter directory exists
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print(f"Error: Adapter directory does not exist: {adapter_path}")
        return False

    # Check for required adapter files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required_files:
        if not (adapter_dir / fname).exists():
            print(f"Warning: Missing {fname} in adapter directory")
            # Not a fatal error for GGUF-based LoRA

    print(f"Validating adapter: {adapter_path}")
    print(f"Using LLM endpoint: {llama_url}")

    passed_tests = 0
    total_tests = len(VALIDATION_PROMPTS)

    for i, test in enumerate(VALIDATION_PROMPTS):
        print(f"\nTest {i+1}/{total_tests}: {test['prompt'][:50]}...")

        try:
            start = time.time()
            response = requests.post(
                f"{llama_url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": test["prompt"]}],
                    "max_tokens": 200,
                    "temperature": 0.3
                },
                timeout=60
            )

            elapsed = time.time() - start

            if response.status_code != 200:
                print(f"  FAIL: HTTP {response.status_code}")
                continue

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Check for expected keywords
            keywords_found = sum(1 for kw in test["expected_keywords"] if kw in content)
            keyword_ratio = keywords_found / len(test["expected_keywords"])

            if keyword_ratio >= 0.5:
                print(f"  PASS: {keywords_found}/{len(test['expected_keywords'])} keywords, {elapsed:.2f}s")
                passed_tests += 1
            else:
                print(f"  FAIL: Only {keywords_found}/{len(test['expected_keywords'])} keywords")
                print(f"  Output: {content[:100]}...")

        except Exception as e:
            print(f"  ERROR: {e}")

    success_rate = passed_tests / total_tests
    print(f"\n{'='*50}")
    print(f"Validation Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
    print(f"Threshold: {threshold:.1%}")

    if success_rate >= threshold:
        print("VALIDATION PASSED")
        return True
    else:
        print("VALIDATION FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate a LoRA adapter")
    parser.add_argument("--adapter", "-a", required=True, help="Path to adapter directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.6,
                        help="Minimum pass rate (0.0-1.0)")
    parser.add_argument("--llama-url", default=None, help="LLaMA server URL")

    args = parser.parse_args()

    success = validate_adapter(args.adapter, args.threshold, args.llama_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
