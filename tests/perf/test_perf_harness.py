"""Performance harness: schema stability + budget gate."""

from benchmark.perf import harness


def test_measure_has_stable_schema():
    r = harness.measure(stamp="2026-07-05T00:00:00", git_commit="abc1234")
    assert r["schema_version"] == harness.SCHEMA_VERSION
    assert "deterministic" in r and "hardware" in r
    # hardware fields present but nullable (imported later)
    assert set(r["hardware"]) >= {"model", "first_token_ms", "tokens_per_sec"}


def test_check_passes_within_budget():
    result = {"deterministic": {"cli_import_time_s": 1.0,
                                "proxy_binary_bytes": 1000}}
    budgets = {"deterministic_max": {"cli_import_time_s": 3.0,
                                     "proxy_binary_bytes": 60_000_000}}
    v = harness.check(result, budgets)
    assert v["passed"] and not v["violations"]


def test_check_flags_regression():
    result = {"deterministic": {"cli_import_time_s": 9.9}}
    budgets = {"deterministic_max": {"cli_import_time_s": 3.0}}
    v = harness.check(result, budgets)
    assert not v["passed"]
    assert any("cli_import_time_s" in x for x in v["violations"])


def test_missing_metric_is_not_a_regression():
    result = {"deterministic": {}}
    budgets = {"deterministic_max": {"cli_import_time_s": 3.0}}
    assert harness.check(result, budgets)["passed"]


def test_real_budgets_file_loads():
    b = harness.load_budgets()
    assert "deterministic_max" in b
