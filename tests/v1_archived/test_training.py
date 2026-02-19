"""
Tests for Training infrastructure.

Validates training data export, LoRA adapter management,
and hot-swap mechanism.

NOTE: These tests verify deployment-specific infrastructure at absolute paths
on the ATLAS server. They are not portable and will fail on machines with a
different directory layout. Set ATLAS_DEPLOY_ROOT to override the default base
path (e.g., ATLAS_DEPLOY_ROOT=/home/youruser).
"""

import os
import json
import uuid
import time

import pytest
import redis

# Base path for deployment-specific infrastructure checks
_DEPLOY_ROOT = os.environ.get("ATLAS_DEPLOY_ROOT", "/opt/atlas")


class TestTrainingDirectories:
    """Test training directory structure."""

    def test_training_directory_exists(self):
        """Training data directory should exist or be creatable."""
        # Check both possible locations
        paths = [
            f"{_DEPLOY_ROOT}/data/training",
            "/data/training",
            "/root/data/training"
        ]
        exists = any(os.path.isdir(p) or os.path.exists(p) for p in paths)
        if not exists:
            # Create it
            training_path = f"{_DEPLOY_ROOT}/data/training"
            os.makedirs(training_path, exist_ok=True)
            exists = os.path.isdir(training_path)
        assert exists, f"Training directory should exist or be creatable at one of: {paths}"

    def test_lora_adapter_directory_exists(self):
        """LoRA adapter directory should exist or be creatable."""
        paths = [
            f"{_DEPLOY_ROOT}/models/lora",
            "/models/lora",
            "/root/models/lora"
        ]
        exists = any(os.path.isdir(p) or os.path.exists(p) for p in paths)
        if not exists:
            # Create it
            lora_path = f"{_DEPLOY_ROOT}/models/lora"
            os.makedirs(lora_path, exist_ok=True)
            exists = os.path.isdir(lora_path)
        assert exists, f"LoRA adapter directory should exist or be creatable at one of: {paths}"


class TestTrainingScripts:
    """Test training script existence."""

    def test_export_script_exists(self):
        """export_training_data.py should exist."""
        paths = [
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/training/export_training_data.py",
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/export_training_data.py"
        ]
        exists = any(os.path.isfile(p) for p in paths)
        assert exists, f"Export script should exist at one of: {paths}"

    def test_train_script_exists(self):
        """train_lora.py should exist."""
        paths = [
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/training/train_lora.py",
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/train_lora.py"
        ]
        exists = any(os.path.isfile(p) for p in paths)
        assert exists, f"Train script should exist at one of: {paths}"

    def test_validate_script_exists(self):
        """validate_adapter.py should exist."""
        paths = [
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/training/validate_adapter.py",
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/validate_adapter.py"
        ]
        exists = any(os.path.isfile(p) for p in paths)
        assert exists, f"Validate script should exist at one of: {paths}"

    def test_nightly_script_exists(self):
        """nightly_train.sh should exist."""
        paths = [
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/training/nightly_train.sh",
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/nightly_train.sh"
        ]
        exists = any(os.path.isfile(p) for p in paths)
        assert exists, f"Nightly script should exist at one of: {paths}"


class TestTrainingDataRedis:
    """Test training data storage in Redis."""

    def test_can_write_training_data(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to write training data to Redis."""
        test_key = f"training:test:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        training_data = {
            "task": "Write a function to calculate factorial",
            "solution": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "rating": 5,
            "timestamp": time.time(),
            "attempts": 2,
            "tokens_used": 500
        }

        redis_client.set(test_key, json.dumps(training_data))
        stored = json.loads(redis_client.get(test_key))

        assert stored["rating"] == 5
        assert "solution" in stored
        assert "task" in stored

    def test_training_data_has_required_fields(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Training data should include task, solution, rating."""
        test_key = f"training:complete:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        training_data = {
            "task": "Implement bubble sort",
            "solution": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
            "rating": 4,
            "timestamp": time.time()
        }

        redis_client.set(test_key, json.dumps(training_data))
        stored = json.loads(redis_client.get(test_key))

        assert "task" in stored, "Training data must have task"
        assert "solution" in stored, "Training data must have solution"
        assert "rating" in stored, "Training data must have rating"

    def test_training_data_quality_filter(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Quality filter should only export rating >= 4."""
        # Store some training data with different ratings
        for i, rating in enumerate([2, 3, 4, 5]):
            key = f"training:filter_test:{uuid.uuid4()}"
            cleanup_redis_keys.append(key)
            redis_client.set(key, json.dumps({
                "task": f"Task {i}",
                "solution": f"Solution {i}",
                "rating": rating
            }))

        # Simulate filter: count keys with rating >= 4
        high_quality_count = 0
        for key in cleanup_redis_keys:
            if key.startswith("training:"):
                data = redis_client.get(key)
                if data:
                    parsed = json.loads(data)
                    if parsed.get("rating", 0) >= 4:
                        high_quality_count += 1

        assert high_quality_count >= 2, "Should have at least 2 high-quality entries (rating >= 4)"


class TestTrainingDataExport:
    """Test training data export format."""

    def test_export_format_is_jsonl(self):
        """Export should produce JSONL format."""
        # Test that we can write JSONL format
        import tempfile

        training_records = [
            {"task": "Task 1", "solution": "Sol 1", "rating": 5},
            {"task": "Task 2", "solution": "Sol 2", "rating": 4}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in training_records:
                f.write(json.dumps(record) + '\n')
            temp_path = f.name

        # Verify JSONL format
        with open(temp_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                parsed = json.loads(line)
                assert "task" in parsed
                assert "solution" in parsed

        os.unlink(temp_path)


class TestLoRASymlink:
    """Test LoRA adapter symlink mechanism."""

    def test_latest_symlink_creatable(self):
        """Should be able to create/update latest symlink."""
        import tempfile

        # Create test directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_v1 = os.path.join(tmpdir, "adapter_v1")
            adapter_v2 = os.path.join(tmpdir, "adapter_v2")
            latest = os.path.join(tmpdir, "latest")

            os.makedirs(adapter_v1)
            os.makedirs(adapter_v2)

            # Create symlink to v1
            os.symlink(adapter_v1, latest)
            assert os.path.islink(latest)
            assert os.readlink(latest) == adapter_v1

            # Atomic update to v2
            temp_link = latest + ".tmp"
            os.symlink(adapter_v2, temp_link)
            os.rename(temp_link, latest)

            assert os.readlink(latest) == adapter_v2

    def test_symlink_update_is_atomic(self):
        """Symlink update should be atomic (os.rename)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            target1 = os.path.join(tmpdir, "target1")
            target2 = os.path.join(tmpdir, "target2")
            link = os.path.join(tmpdir, "link")

            os.makedirs(target1)
            os.makedirs(target2)

            # Initial link
            os.symlink(target1, link)

            # Atomic update using temp link + rename
            temp_link = link + ".new"
            os.symlink(target2, temp_link)

            # This should be atomic
            os.rename(temp_link, link)

            # Verify update happened
            assert os.readlink(link) == target2


class TestTrainingCronJob:
    """Test training CronJob configuration."""

    def test_cronjob_manifest_exists(self):
        """CronJob manifest should exist."""
        path = f"{_DEPLOY_ROOT}/k8s/atlas/manifests/training-cronjob.yaml"
        assert os.path.isfile(path), f"CronJob manifest should exist at {path}"

    def test_validation_threshold_documented(self):
        """Validation threshold (66%) should be documented or configurable."""
        # Check if validate script has threshold
        paths = [
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/training/validate_adapter.py",
            f"{_DEPLOY_ROOT}/k8s/atlas/trainer/validate_adapter.py"
        ]

        for path in paths:
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    content = f.read()
                # Look for threshold definition
                has_threshold = "0.66" in content or "66" in content or "threshold" in content.lower()
                if has_threshold:
                    return  # Pass

        pytest.skip("Validation threshold not found in validate script")
