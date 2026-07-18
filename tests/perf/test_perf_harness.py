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


def test_single_missing_metric_is_not_a_regression():
    # One metric couldn't be measured (None/absent) but another matched:
    # not a regression.
    result = {"deterministic": {"cli_import_time_s": 1.0,
                                "proxy_binary_bytes": None}}
    budgets = {"deterministic_max": {"cli_import_time_s": 3.0,
                                     "proxy_binary_bytes": 60_000_000}}
    assert harness.check(result, budgets)["passed"]


def test_no_matching_metrics_fails_not_passes_vacuously():
    # A result that matches zero budgeted metrics (renamed keys, empty
    # file) must FAIL — a vacuous pass disarms the gate silently.
    v = harness.check({}, {"deterministic_max": {"cli_import_time_s": 3.0}})
    assert not v["passed"]


def test_schema_version_mismatch_fails():
    result = {"schema_version": harness.SCHEMA_VERSION + 1,
              "deterministic": {"cli_import_time_s": 1.0}}
    budgets = {"deterministic_max": {"cli_import_time_s": 3.0}}
    v = harness.check(result, budgets)
    assert not v["passed"]
    assert any("schema_version" in x for x in v["violations"])


def test_real_budgets_file_loads():
    b = harness.load_budgets()
    assert "deterministic_max" in b
