#!/usr/bin/env python3
"""
Export successful task completions from Redis to JSONL for training.

Usage:
    python export_training_data.py --output /data/training/20260129.jsonl
    python export_training_data.py --output /tmp/test.jsonl --limit 100
"""

import os
import sys
import json
import argparse
import redis
from datetime import datetime, timedelta

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")


def export_training_data(output_path: str, limit: int = None, min_quality: float = 0.5):
    """Export successful completions from Redis to JSONL format."""
    r = redis.from_url(REDIS_URL, decode_responses=True)

    # Get training examples from Redis
    examples = r.lrange("training:examples", 0, limit - 1 if limit else -1)

    if not examples:
        print("No training examples found in Redis")
        print("Make sure task-worker is storing successful completions")
        return 0

    exported = 0
    with open(output_path, 'w') as f:
        for example_json in examples:
            try:
                example = json.loads(example_json)

                # Filter by quality score if available
                quality = example.get("quality_score", 1.0)
                if quality < min_quality:
                    continue

                # Format for training (instruction-tuning format)
                training_item = {
                    "instruction": example.get("prompt", ""),
                    "input": example.get("context", ""),
                    "output": example.get("completion", ""),
                    "metadata": {
                        "quality_score": quality,
                        "timestamp": example.get("timestamp", ""),
                        "task_id": example.get("task_id", "")
                    }
                }

                f.write(json.dumps(training_item) + "\n")
                exported += 1

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON entry")
                continue

    print(f"Exported {exported} training examples to {output_path}")
    return exported


def main():
    global REDIS_URL
    parser = argparse.ArgumentParser(description="Export training data from Redis")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max examples to export")
    parser.add_argument("--min-quality", type=float, default=0.5, help="Minimum quality score")
    parser.add_argument("--redis-url", default=None, help="Redis connection URL")

    args = parser.parse_args()

    if args.redis_url:
        REDIS_URL = args.redis_url

    count = export_training_data(args.output, args.limit, args.min_quality)

    if count == 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
