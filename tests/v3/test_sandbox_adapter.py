# tests/v3/test_sandbox_adapter.py
"""Tests for V3 SandboxAdapter dual mode (real tests vs self-verify)."""

import pytest

from benchmark.v3.self_test_gen import GeneratedTestCase


# We can't import SandboxAdapter directly because it depends on BenchmarkTask
# and execute_code internals. Test the _run_self_tests logic in isolation.

class TestSelfVerifyLogic:
    """Test the majority-vote self-verification logic."""

    def test_all_pass_returns_true(self):
        """When all self-tests pass, result is True."""
        from benchmark.v3_runner import self_verify_execute
        results = [(True, "6", ""), (True, "0", ""), (True, "2000000", "")]
        passed, stdout, stderr = self_verify_execute(results, threshold=0.6)
        assert passed is True

    def test_all_fail_returns_false(self):
        from benchmark.v3_runner import self_verify_execute
        results = [(False, "", "error1"), (False, "", "error2"), (False, "", "error3")]
        passed, stdout, stderr = self_verify_execute(results, threshold=0.6)
        assert passed is False
        assert "error1" in stderr

    def test_majority_pass_returns_true(self):
        """2/3 pass with threshold 0.6 → True."""
        from benchmark.v3_runner import self_verify_execute
        results = [(True, "6", ""), (True, "0", ""), (False, "", "wrong")]
        passed, stdout, stderr = self_verify_execute(results, threshold=0.6)
        assert passed is True

    def test_majority_fail_returns_false(self):
        """1/3 pass with threshold 0.6 → False."""
        from benchmark.v3_runner import self_verify_execute
        results = [(True, "6", ""), (False, "", "err1"), (False, "", "err2")]
        passed, stdout, stderr = self_verify_execute(results, threshold=0.6)
        assert passed is False

    def test_empty_results_returns_false(self):
        from benchmark.v3_runner import self_verify_execute
        passed, stdout, stderr = self_verify_execute([], threshold=0.6)
        assert passed is False
