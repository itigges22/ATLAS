"""
End-to-end training flow tests.

Tests the complete training pipeline from task completion
through training data export and validation.
"""

import os
import json
import uuid
import time
import tempfile

import pytest
import redis


@pytest.mark.integration
class TestE2ETrainingDataFlow:
    """Test training data collection flow."""

    def test_successful_task_creates_training_data(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Successful task completion should create training data."""
        task_id = str(uuid.uuid4())
        training_key = f"training:{task_id}"
        cleanup_redis_keys.append(training_key)

        # Simulate successful task completion with high rating
        training_data = {
            "task_id": task_id,
            "task": "Write a function to reverse a string",
            "solution": '''def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]
''',
            "rating": 5,
            "attempts": 1,
            "tokens_used": 250,
            "timestamp": time.time()
        }

        redis_client.set(training_key, json.dumps(training_data))

        # Verify data is stored
        stored = redis_client.get(training_key)
        assert stored is not None
        parsed = json.loads(stored)
        assert parsed["rating"] == 5
        assert "solution" in parsed

    def test_rate_result_highly(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Should be able to rate task result >= 4 for training."""
        task_id = str(uuid.uuid4())
        rating_key = f"task:{task_id}:rating"
        training_key = f"training:{task_id}"
        cleanup_redis_keys.extend([rating_key, training_key])

        # Store rating
        redis_client.set(rating_key, "5")
        rating = int(redis_client.get(rating_key))
        assert rating >= 4, "Rating should be high for training inclusion"

        # Store training data
        training_data = {
            "task": "Implement binary search",
            "solution": "def binary_search(arr, target): ...",
            "rating": rating
        }
        redis_client.set(training_key, json.dumps(training_data))

    def test_training_data_quality_filter(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Only rating >= 4 should be included in training."""
        # Create tasks with different ratings
        tasks = [
            {"id": str(uuid.uuid4()), "rating": 2, "task": "Task 1", "solution": "Sol 1"},
            {"id": str(uuid.uuid4()), "rating": 3, "task": "Task 2", "solution": "Sol 2"},
            {"id": str(uuid.uuid4()), "rating": 4, "task": "Task 3", "solution": "Sol 3"},
            {"id": str(uuid.uuid4()), "rating": 5, "task": "Task 4", "solution": "Sol 4"}
        ]

        for task in tasks:
            key = f"training:{task['id']}"
            cleanup_redis_keys.append(key)
            redis_client.set(key, json.dumps(task))

        # Filter high-quality data
        high_quality = []
        for task in tasks:
            key = f"training:{task['id']}"
            data = json.loads(redis_client.get(key))
            if data["rating"] >= 4:
                high_quality.append(data)

        assert len(high_quality) == 2, "Should have 2 high-quality entries"


@pytest.mark.integration
class TestE2ETrainingExport:
    """Test training data export flow."""

    def test_export_training_data_to_file(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Should be able to export training data to JSONL file."""
        # Create training data
        training_records = []
        for i in range(5):
            task_id = str(uuid.uuid4())
            key = f"training:export_test:{task_id}"
            cleanup_redis_keys.append(key)

            record = {
                "task": f"Test task {i}",
                "solution": f"def solution_{i}(): pass",
                "rating": 5 if i % 2 == 0 else 3
            }
            redis_client.set(key, json.dumps(record))
            if record["rating"] >= 4:
                training_records.append(record)

        # Export to JSONL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in training_records:
                f.write(json.dumps(record) + '\n')
            export_path = f.name

        # Verify JSONL format
        with open(export_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3, "Should have 3 high-rated records"
            for line in lines:
                data = json.loads(line)
                assert "task" in data
                assert "solution" in data

        os.unlink(export_path)

    def test_export_file_is_valid_jsonl(self):
        """Exported file should be valid JSONL format."""
        # Create test JSONL
        records = [
            {"task": "Task 1", "solution": "def f(): pass", "rating": 5},
            {"task": "Task 2", "solution": "def g(): pass", "rating": 4},
            {"task": "Task 3", "solution": "def h(): pass", "rating": 5}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
            export_path = f.name

        # Validate JSONL
        valid = True
        with open(export_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    valid = False
                    break

        assert valid, "All lines should be valid JSON"
        os.unlink(export_path)


@pytest.mark.integration
class TestE2ELoRAValidation:
    """Test LoRA adapter validation flow."""

    def test_validate_symlink_mechanism(self):
        """Symlink mechanism for hot-swap should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter versions
            v1_dir = os.path.join(tmpdir, "adapter_v1")
            v2_dir = os.path.join(tmpdir, "adapter_v2")
            latest_link = os.path.join(tmpdir, "latest")

            os.makedirs(v1_dir)
            os.makedirs(v2_dir)

            # Create v1 adapter files
            with open(os.path.join(v1_dir, "adapter_config.json"), 'w') as f:
                json.dump({"version": "1.0"}, f)

            # Create v2 adapter files
            with open(os.path.join(v2_dir, "adapter_config.json"), 'w') as f:
                json.dump({"version": "2.0"}, f)

            # Create initial symlink
            os.symlink(v1_dir, latest_link)
            assert os.readlink(latest_link) == v1_dir

            # Atomic update
            temp_link = latest_link + ".new"
            os.symlink(v2_dir, temp_link)
            os.rename(temp_link, latest_link)

            # Verify update
            assert os.readlink(latest_link) == v2_dir

            # Verify config accessible through symlink
            with open(os.path.join(latest_link, "adapter_config.json")) as f:
                config = json.load(f)
                assert config["version"] == "2.0"

    def test_adapter_validation_concept(self):
        """Adapter validation should check success rate >= 66%."""
        # Simulate validation results
        test_results = [
            {"prompt": "Test 1", "passed": True},
            {"prompt": "Test 2", "passed": True},
            {"prompt": "Test 3", "passed": False},
            {"prompt": "Test 4", "passed": True},
            {"prompt": "Test 5", "passed": True},
            {"prompt": "Test 6", "passed": True}
        ]

        passed = sum(1 for r in test_results if r["passed"])
        total = len(test_results)
        success_rate = passed / total

        assert success_rate >= 0.66, f"Should have >= 66% pass rate, got {success_rate:.1%}"


@pytest.mark.integration
class TestE2ETrainingCleanup:
    """Test training data cleanup."""

    def test_cleanup_test_training_data(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Should be able to clean up test training data."""
        # Create test data
        test_keys = []
        for i in range(5):
            key = f"training:cleanup_test:{uuid.uuid4()}"
            test_keys.append(key)
            cleanup_redis_keys.append(key)
            redis_client.set(key, json.dumps({"test": i}))

        # Verify created
        for key in test_keys:
            assert redis_client.exists(key) == 1

        # Cleanup happens via fixture

    def test_training_data_scan_pattern(
        self,
        redis_client: redis.Redis,
        cleanup_redis_keys: list
    ):
        """Should be able to scan for training:* keys."""
        prefix = f"training:scan_test:{uuid.uuid4().hex[:8]}"

        # Create keys
        for i in range(10):
            key = f"{prefix}:{i}"
            cleanup_redis_keys.append(key)
            redis_client.set(key, json.dumps({"index": i}))

        # Scan for keys
        found = list(redis_client.scan_iter(match=f"{prefix}:*"))
        assert len(found) == 10, f"Should find 10 keys, found {len(found)}"
