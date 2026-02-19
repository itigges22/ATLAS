#!/usr/bin/env python3
"""Export training data from Redis for nightly fine-tuning."""

import argparse
import json
import os
import redis
from datetime import datetime
from urllib.parse import urlparse

# Use REDIS_URL if available (recommended), otherwise fall back to host/port
# Note: K8s auto-injects REDIS_PORT as "tcp://IP:PORT" when a redis service exists,
# so we use ATLAS_REDIS_PORT to avoid conflicts
REDIS_URL = os.getenv("REDIS_URL", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/training")

# Default minimum quality/rating threshold
DEFAULT_MIN_RATING = int(os.getenv("TRAINING_MIN_RATING", "4"))

def get_redis_connection():
    """Create Redis connection, handling both URL and host/port formats."""
    if REDIS_URL:
        # Parse redis://host:port format
        parsed = urlparse(REDIS_URL)
        host = parsed.hostname or "redis"
        port = parsed.port or 6379
        return redis.Redis(host=host, port=port, decode_responses=True)
    else:
        # Fall back to explicit host/port (use ATLAS_ prefix to avoid K8s conflicts)
        host = os.getenv("ATLAS_REDIS_HOST", "redis")
        port = int(os.getenv("ATLAS_REDIS_PORT", "6379"))
        return redis.Redis(host=host, port=port, decode_responses=True)

def export_training_data(output_file=None, min_quality=None):
    """Export successful task completions as training examples.

    Args:
        output_file: Path to output JSONL file. If None, generates timestamped file.
        min_quality: Minimum rating (0.0-1.0 scale or 1-5 scale) to include.
    """
    r = get_redis_connection()

    # Convert min_quality to rating threshold (handle both 0.0-1.0 and 1-5 scales)
    if min_quality is None:
        min_rating = DEFAULT_MIN_RATING
    elif min_quality <= 1.0:
        # Scale from 0.0-1.0 to 1-5
        min_rating = int(min_quality * 5)
    else:
        min_rating = int(min_quality)

    # Get all completed tasks with positive feedback
    training_examples = []

    # Scan for training data keys
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="training:*", count=100)
        for key in keys:
            try:
                # Check key type - training data could be stored as string or hash
                key_type = r.type(key)

                if key_type == "string":
                    data = r.get(key)
                    if data:
                        example = json.loads(data)
                elif key_type == "hash":
                    # Convert hash to dict
                    example = r.hgetall(key)
                    # Parse any JSON fields
                    if "messages" in example and isinstance(example["messages"], str):
                        example["messages"] = json.loads(example["messages"])
                    if "rating" in example:
                        example["rating"] = int(example["rating"]) if example["rating"] else 0
                else:
                    # Skip unsupported types (lists, sets, etc.)
                    continue

                if example and example.get("rating", 0) >= min_rating:
                    training_examples.append({
                        "messages": example.get("messages", []),
                        "metadata": {
                            "task_id": example.get("task_id"),
                            "timestamp": example.get("timestamp"),
                            "rating": example.get("rating")
                        }
                    })
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Skipping key {key}: {e}")
                continue
        if cursor == 0:
            break

    # Determine output file path
    if output_file is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"training_data_{timestamp}.jsonl")
    else:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Exported {len(training_examples)} training examples to {output_file}")
    return output_file, len(training_examples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training data from Redis")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--min-quality", type=float, default=0.6,
                        help="Minimum quality threshold (0.0-1.0 scale)")
    args = parser.parse_args()

    export_training_data(output_file=args.output, min_quality=args.min_quality)
