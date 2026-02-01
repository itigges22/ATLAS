"""
Tests for Redis infrastructure.

Validates Redis is properly configured for ATLAS task queues,
metrics storage, and training data caching.
"""

import time
import json
import uuid
import threading

import pytest
import redis


class TestRedisBasicOperations:
    """Test basic Redis operations."""

    def test_ping_succeeds(self, redis_client: redis.Redis):
        """Redis PING command should return True."""
        result = redis_client.ping()
        assert result is True, "Redis PING should return True"

    def test_set_and_get(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """SET and GET operations should work correctly."""
        test_key = f"test:basic:{uuid.uuid4()}"
        test_value = "hello_world"
        cleanup_redis_keys.append(test_key)

        redis_client.set(test_key, test_value)
        result = redis_client.get(test_key)

        assert result == test_value, f"GET should return '{test_value}', got '{result}'"

    def test_lpush_and_rpop(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """LPUSH and RPOP should implement FIFO queue behavior."""
        test_key = f"test:list:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        # Push items to left (most recent at head)
        redis_client.lpush(test_key, "item1")
        redis_client.lpush(test_key, "item2")
        redis_client.lpush(test_key, "item3")

        # Pop from right (oldest first - FIFO)
        first = redis_client.rpop(test_key)
        second = redis_client.rpop(test_key)
        third = redis_client.rpop(test_key)

        assert first == "item1", "First popped should be 'item1'"
        assert second == "item2", "Second popped should be 'item2'"
        assert third == "item3", "Third popped should be 'item3'"

    def test_ttl_expiration(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """TTL expiration should work correctly."""
        test_key = f"test:ttl:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        # Set with 2 second TTL
        redis_client.setex(test_key, 2, "expires_soon")

        # Should exist immediately
        assert redis_client.exists(test_key) == 1, "Key should exist immediately after set"

        # Wait for expiration
        time.sleep(3)

        # Should be gone
        assert redis_client.exists(test_key) == 0, "Key should be expired after TTL"

    def test_incr(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """INCR should atomically increment counters."""
        test_key = f"test:counter:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        # Increment non-existent key starts at 1
        result1 = redis_client.incr(test_key)
        assert result1 == 1, "First INCR should return 1"

        result2 = redis_client.incr(test_key)
        assert result2 == 2, "Second INCR should return 2"

        result3 = redis_client.incrby(test_key, 10)
        assert result3 == 12, "INCRBY 10 should return 12"

    def test_hset_and_hget(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """HSET and HGET should work for hash operations."""
        test_key = f"test:hash:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        # Set hash fields
        redis_client.hset(test_key, "field1", "value1")
        redis_client.hset(test_key, "field2", "value2")

        # Get individual fields
        assert redis_client.hget(test_key, "field1") == "value1", "HGET field1 should return 'value1'"
        assert redis_client.hget(test_key, "field2") == "value2", "HGET field2 should return 'value2'"

        # Get all fields
        all_fields = redis_client.hgetall(test_key)
        assert all_fields == {"field1": "value1", "field2": "value2"}, "HGETALL should return all fields"


class TestRedisTaskQueues:
    """Test Redis task queue functionality for ATLAS."""

    def test_priority_queues_accessible(self, redis_client: redis.Redis):
        """Priority queues (tasks:p0, tasks:p1, tasks:p2) should be accessible."""
        # These queues should be accessible (may be empty)
        for queue in ["tasks:p0", "tasks:p1", "tasks:p2"]:
            # Just check we can query the length without error
            length = redis_client.llen(queue)
            assert length >= 0, f"Queue {queue} should be accessible"

    def test_can_push_to_priority_queues(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to push tasks to priority queues."""
        # Use test-specific queues to avoid interfering with real tasks
        test_queues = [
            f"test:tasks:p0:{uuid.uuid4()}",
            f"test:tasks:p1:{uuid.uuid4()}",
            f"test:tasks:p2:{uuid.uuid4()}"
        ]
        cleanup_redis_keys.extend(test_queues)

        task_data = json.dumps({
            "id": str(uuid.uuid4()),
            "type": "test",
            "prompt": "Test task"
        })

        for queue in test_queues:
            redis_client.lpush(queue, task_data)
            length = redis_client.llen(queue)
            assert length == 1, f"Queue {queue} should have 1 item after push"

    def test_priority_queue_order(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Tasks should be processed in FIFO order within each priority."""
        test_queue = f"test:queue:order:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_queue)

        # Push tasks in order
        for i in range(5):
            redis_client.lpush(test_queue, f"task_{i}")

        # Pop should return in reverse order (FIFO with LPUSH/RPOP)
        for i in range(5):
            task = redis_client.rpop(test_queue)
            assert task == f"task_{i}", f"Expected task_{i}, got {task}"


class TestRedisPubSub:
    """Test Redis pub/sub functionality."""

    def test_pubsub_works(self, redis_client: redis.Redis):
        """Pub/sub should deliver messages to subscribers."""
        channel = f"test:channel:{uuid.uuid4()}"
        received_messages = []

        def subscriber_thread():
            sub_client = redis.Redis(
                host=redis_client.connection_pool.connection_kwargs.get('host', 'localhost'),
                port=redis_client.connection_pool.connection_kwargs.get('port', 6379),
                decode_responses=True
            )
            pubsub = sub_client.pubsub()
            pubsub.subscribe(channel)

            # Wait for subscription confirmation and one message
            for message in pubsub.listen():
                if message['type'] == 'message':
                    received_messages.append(message['data'])
                    break

            pubsub.close()

        # Start subscriber in background
        thread = threading.Thread(target=subscriber_thread)
        thread.start()

        # Give subscriber time to connect
        time.sleep(0.5)

        # Publish message
        test_message = "hello_pubsub"
        redis_client.publish(channel, test_message)

        # Wait for subscriber to receive
        thread.join(timeout=5)

        assert test_message in received_messages, f"Message '{test_message}' should be received via pub/sub"


class TestRedisMetrics:
    """Test Redis metrics storage for ATLAS."""

    def test_metrics_daily_key_writable(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to write to metrics:daily:* keys."""
        import datetime
        today = datetime.date.today().isoformat()
        test_key = f"test:metrics:daily:{today}:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        # Store metrics as hash
        redis_client.hset(test_key, mapping={
            "tasks_total": "10",
            "tasks_success": "8",
            "tokens_used": "5000",
            "duration_ms": "30000"
        })

        metrics = redis_client.hgetall(test_key)
        assert metrics["tasks_total"] == "10", "tasks_total should be stored"
        assert metrics["tasks_success"] == "8", "tasks_success should be stored"

    def test_metrics_recent_tasks_writable(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to write to metrics:recent_tasks list."""
        test_key = f"test:metrics:recent_tasks:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        task_record = json.dumps({
            "id": str(uuid.uuid4()),
            "status": "completed",
            "timestamp": time.time()
        })

        redis_client.lpush(test_key, task_record)
        # Trim to last 20
        redis_client.ltrim(test_key, 0, 19)

        length = redis_client.llen(test_key)
        assert length == 1, "Recent tasks list should have 1 entry"


class TestRedisTraining:
    """Test Redis training data storage for ATLAS."""

    def test_training_key_writable(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to write training data to training:* keys."""
        test_key = f"test:training:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        training_data = json.dumps({
            "task": "Write a function to add two numbers",
            "solution": "def add(a, b): return a + b",
            "rating": 5,
            "timestamp": time.time()
        })

        redis_client.set(test_key, training_data)
        stored = redis_client.get(test_key)
        parsed = json.loads(stored)

        assert parsed["rating"] == 5, "Training data should preserve rating"
        assert "solution" in parsed, "Training data should contain solution"

    def test_training_key_scannable(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Should be able to scan for training:* keys."""
        # Create multiple test training keys
        test_prefix = f"test:training:scan:{uuid.uuid4()}"
        for i in range(5):
            key = f"{test_prefix}:{i}"
            cleanup_redis_keys.append(key)
            redis_client.set(key, json.dumps({"index": i}))

        # Scan for keys
        found_keys = list(redis_client.scan_iter(match=f"{test_prefix}:*"))
        assert len(found_keys) == 5, f"Should find 5 training keys, found {len(found_keys)}"


class TestRedisConnectionErrors:
    """Test Redis connection error handling."""

    def test_get_nonexistent_key_returns_none(self, redis_client: redis.Redis):
        """GET on nonexistent key should return None."""
        result = redis_client.get(f"nonexistent:key:{uuid.uuid4()}")
        assert result is None, "GET on nonexistent key should return None"

    def test_delete_existing_key_succeeds(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """DELETE existing key should succeed."""
        test_key = f"test:delete:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.set(test_key, "value")
        result = redis_client.delete(test_key)
        assert result == 1, "DELETE should return 1 for existing key"
        assert redis_client.exists(test_key) == 0, "Key should not exist after delete"

    def test_delete_nonexistent_key_idempotent(self, redis_client: redis.Redis):
        """DELETE nonexistent key should return 0 (idempotent)."""
        result = redis_client.delete(f"nonexistent:key:{uuid.uuid4()}")
        assert result == 0, "DELETE should return 0 for nonexistent key"

    def test_keys_pattern_matching(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """KEYS pattern matching should work correctly."""
        prefix = f"test:pattern:{uuid.uuid4()}"
        for i in range(3):
            key = f"{prefix}:key{i}"
            cleanup_redis_keys.append(key)
            redis_client.set(key, f"value{i}")

        keys = redis_client.keys(f"{prefix}:*")
        assert len(keys) == 3, f"Should find 3 keys with pattern, found {len(keys)}"

    def test_keys_no_matches_returns_empty(self, redis_client: redis.Redis):
        """KEYS with no matches should return empty list."""
        keys = redis_client.keys(f"nonexistent:pattern:{uuid.uuid4()}:*")
        assert keys == [], "KEYS with no matches should return empty list"


class TestRedisQueueOperations:
    """Test Redis queue operations in detail."""

    def test_lpush_creates_list(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """LPUSH to new list should create the list."""
        test_key = f"test:newlist:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.lpush(test_key, "first")
        assert redis_client.type(test_key) == "list", "LPUSH should create list type"

    def test_lpush_multiple_values_order(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """LPUSH multiple values should maintain correct order."""
        test_key = f"test:multilist:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.lpush(test_key, "a", "b", "c")
        result = redis_client.lrange(test_key, 0, -1)
        # LPUSH pushes left-to-right, so last arg ends up at head
        assert result == ["c", "b", "a"], f"Order should be ['c', 'b', 'a'], got {result}"

    def test_rpop_empty_list_returns_none(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """RPOP from empty list should return None."""
        test_key = f"test:emptylist:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.lpush(test_key, "only")
        redis_client.rpop(test_key)  # Remove the only item
        result = redis_client.rpop(test_key)
        assert result is None, "RPOP from empty list should return None"

    def test_llen_correct_count(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """LLEN should return correct count."""
        test_key = f"test:llen:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        for i in range(7):
            redis_client.lpush(test_key, f"item{i}")
        assert redis_client.llen(test_key) == 7, "LLEN should return 7"

    def test_lrange_returns_correct_slice(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """LRANGE should return correct slice."""
        test_key = f"test:lrange:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        for i in range(5):
            redis_client.rpush(test_key, f"item{i}")
        result = redis_client.lrange(test_key, 1, 3)
        assert result == ["item1", "item2", "item3"], f"LRANGE 1-3 incorrect, got {result}"


class TestRedisHashOperations:
    """Test Redis hash operations in detail."""

    def test_hset_creates_hash(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """HSET should create hash."""
        test_key = f"test:newhash:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.hset(test_key, "field", "value")
        assert redis_client.type(test_key) == "hash", "HSET should create hash type"

    def test_hset_overwrites_existing_field(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """HSET should overwrite existing field."""
        test_key = f"test:hashoverwrite:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.hset(test_key, "field", "value1")
        redis_client.hset(test_key, "field", "value2")
        assert redis_client.hget(test_key, "field") == "value2", "HSET should overwrite"

    def test_hget_nonexistent_hash_returns_none(self, redis_client: redis.Redis):
        """HGET on nonexistent hash should return None."""
        result = redis_client.hget(f"nonexistent:hash:{uuid.uuid4()}", "field")
        assert result is None, "HGET nonexistent hash should return None"

    def test_hget_nonexistent_field_returns_none(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """HGET on nonexistent field should return None."""
        test_key = f"test:hashfield:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.hset(test_key, "exists", "value")
        result = redis_client.hget(test_key, "nonexistent")
        assert result is None, "HGET nonexistent field should return None"

    def test_hdel_removes_field(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """HDEL should remove field."""
        test_key = f"test:hdel:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.hset(test_key, "field", "value")
        redis_client.hdel(test_key, "field")
        assert redis_client.hget(test_key, "field") is None, "HDEL should remove field"

    def test_hincrby_increments(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """HINCRBY should increment numeric field."""
        test_key = f"test:hincrby:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.hset(test_key, "counter", "5")
        result = redis_client.hincrby(test_key, "counter", 3)
        assert result == 8, "HINCRBY should increment to 8"


class TestRedisPubSubAdvanced:
    """Test Redis pub/sub advanced features."""

    def test_subscribe_to_pattern(self, redis_client: redis.Redis):
        """Subscribe to pattern should receive matching messages."""
        unique_id = uuid.uuid4()
        pattern = f"test:pattern:{unique_id}:*"
        channel = f"test:pattern:{unique_id}:specific"
        received = []

        def subscriber():
            sub_client = redis.Redis(
                host=redis_client.connection_pool.connection_kwargs.get('host', 'localhost'),
                port=redis_client.connection_pool.connection_kwargs.get('port', 6379),
                decode_responses=True
            )
            pubsub = sub_client.pubsub()
            pubsub.psubscribe(pattern)
            for msg in pubsub.listen():
                if msg['type'] == 'pmessage':
                    received.append(msg['data'])
                    break
            pubsub.close()

        thread = threading.Thread(target=subscriber)
        thread.start()
        time.sleep(0.5)
        redis_client.publish(channel, "pattern_message")
        thread.join(timeout=5)
        assert "pattern_message" in received, "Pattern subscription should receive message"


class TestRedisTransactions:
    """Test Redis transactions and pipelines."""

    def test_pipeline_batches_commands(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Pipeline should batch commands efficiently."""
        test_key = f"test:pipeline:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)

        pipe = redis_client.pipeline()
        pipe.set(test_key, "initial")
        pipe.incr(test_key)  # This will fail since value is not numeric
        pipe.set(test_key, "10")
        pipe.incr(test_key)
        results = pipe.execute(raise_on_error=False)

        # Pipeline executes all commands, some may error
        final_value = redis_client.get(test_key)
        assert final_value == "11", f"Pipeline should result in '11', got {final_value}"


class TestRedisEdgeCases:
    """Test Redis edge cases."""

    def test_unicode_in_key_and_value(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Unicode should work in keys and values."""
        test_key = f"test:unicode:{uuid.uuid4()}:키"
        cleanup_redis_keys.append(test_key)
        test_value = "값: 한글 테스트 🎉"
        redis_client.set(test_key, test_value)
        result = redis_client.get(test_key)
        assert result == test_value, f"Unicode value should be preserved, got {result}"

    def test_binary_data_in_value(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Binary data should work in values."""
        test_key = f"test:binary:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        # Use bytes directly through a non-decoded client
        binary_client = redis.Redis(
            host=redis_client.connection_pool.connection_kwargs.get('host', 'localhost'),
            port=redis_client.connection_pool.connection_kwargs.get('port', 6379),
            decode_responses=False
        )
        binary_value = b'\x00\x01\x02\xff\xfe'
        binary_client.set(test_key, binary_value)
        result = binary_client.get(test_key)
        assert result == binary_value, "Binary data should be preserved"

    def test_rapid_sequential_writes(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Rapid sequential writes should all succeed."""
        test_key = f"test:rapid:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        for i in range(100):
            redis_client.set(test_key, f"value_{i}")
        final = redis_client.get(test_key)
        assert final == "value_99", f"Last write should be 'value_99', got {final}"

    def test_concurrent_incr(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """Concurrent INCR should be atomic."""
        test_key = f"test:concurrent:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.set(test_key, "0")

        def increment():
            for _ in range(50):
                redis_client.incr(test_key)

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = int(redis_client.get(test_key))
        assert final == 200, f"Concurrent INCR should result in 200, got {final}"

    def test_set_with_ttl(self, redis_client: redis.Redis, cleanup_redis_keys: list):
        """SET with TTL should set expiration."""
        test_key = f"test:setttl:{uuid.uuid4()}"
        cleanup_redis_keys.append(test_key)
        redis_client.setex(test_key, 60, "value")
        ttl = redis_client.ttl(test_key)
        assert 55 <= ttl <= 60, f"TTL should be around 60s, got {ttl}"
