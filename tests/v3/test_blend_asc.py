"""Tests for V3 Blend-ASC (Feature 2A) — Adaptive Sampling Budget."""

import json
import math
from pathlib import Path

import pytest

from benchmark.v3.blend_asc import (
    DEFAULT_K_TABLE,
    BlendASC,
    BlendASCConfig,
    BlendASCEvent,
    KAllocation,
    compute_k,
    lookup_k,
)
from benchmark.v3.budget_forcing import normalize_energy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def asc_enabled(tmp_telemetry):
    cfg = BlendASCConfig(enabled=True)
    return BlendASC(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def asc_disabled(tmp_telemetry):
    cfg = BlendASCConfig(enabled=False, default_k=5)
    return BlendASC(cfg, telemetry_dir=tmp_telemetry)


# ---------------------------------------------------------------------------
# Test: Disabled noop
# ---------------------------------------------------------------------------

class TestDisabledNoop:

    def test_returns_default_k(self, asc_disabled):
        k, tier = asc_disabled.allocate(raw_energy=5.0, task_id="t1")
        assert k == 5
        assert tier == "standard"

    def test_no_telemetry_when_disabled(self, asc_disabled, tmp_telemetry):
        asc_disabled.allocate(raw_energy=5.0, task_id="t1")
        events_file = tmp_telemetry / "blend_asc_events.jsonl"
        assert not events_file.exists()

    def test_ignores_energy_when_disabled(self, asc_disabled):
        k1, _ = asc_disabled.allocate(raw_energy=1.0)
        k2, _ = asc_disabled.allocate(raw_energy=20.0)
        assert k1 == k2  # Both return default_k


# ---------------------------------------------------------------------------
# Test: K Allocation Table Structure
# ---------------------------------------------------------------------------

class TestKAllocationTable:

    def test_table_has_4_entries(self):
        assert len(DEFAULT_K_TABLE) == 4

    def test_ranges_are_contiguous(self):
        for i in range(len(DEFAULT_K_TABLE) - 1):
            assert DEFAULT_K_TABLE[i].energy_high == DEFAULT_K_TABLE[i + 1].energy_low

    def test_covers_full_range(self):
        assert DEFAULT_K_TABLE[0].energy_low == 0.0
        assert DEFAULT_K_TABLE[-1].energy_high > 1.0  # includes 1.0

    def test_k_values_increase_with_energy(self):
        for i in range(len(DEFAULT_K_TABLE) - 1):
            assert DEFAULT_K_TABLE[i].min_k <= DEFAULT_K_TABLE[i + 1].min_k
            assert DEFAULT_K_TABLE[i].max_k <= DEFAULT_K_TABLE[i + 1].max_k

    def test_budget_tiers_match_energy(self):
        assert DEFAULT_K_TABLE[0].budget_tier == "nothink"
        assert DEFAULT_K_TABLE[1].budget_tier == "standard"
        assert DEFAULT_K_TABLE[2].budget_tier == "hard"
        assert DEFAULT_K_TABLE[3].budget_tier == "extreme"

    def test_allocation_contains(self):
        alloc = KAllocation(0.10, 0.20, 3, 4, "standard")
        assert alloc.contains(0.10)
        assert alloc.contains(0.15)
        assert not alloc.contains(0.20)  # exclusive upper bound
        assert not alloc.contains(0.09)

    def test_allocation_to_dict(self):
        alloc = KAllocation(0.10, 0.20, 3, 4, "standard")
        d = alloc.to_dict()
        assert d["energy_range"] == [0.10, 0.20]
        assert d["min_k"] == 3
        assert d["max_k"] == 4
        assert d["budget_tier"] == "standard"


# ---------------------------------------------------------------------------
# Test: lookup_k function
# ---------------------------------------------------------------------------

class TestLookupK:

    def test_easy_energy_returns_nothink(self):
        alloc = lookup_k(0.05)
        assert alloc.budget_tier == "nothink"
        assert alloc.min_k == 1
        assert alloc.max_k == 2

    def test_medium_energy_returns_standard(self):
        alloc = lookup_k(0.15)
        assert alloc.budget_tier == "standard"
        assert alloc.min_k == 3
        assert alloc.max_k == 4

    def test_hard_energy_returns_hard(self):
        alloc = lookup_k(0.25)
        assert alloc.budget_tier == "hard"
        assert alloc.min_k == 5
        assert alloc.max_k == 7

    def test_extreme_energy_returns_extreme(self):
        alloc = lookup_k(0.50)
        assert alloc.budget_tier == "extreme"
        assert alloc.min_k == 8
        assert alloc.max_k == 12

    def test_boundary_0_10(self):
        below = lookup_k(0.099)
        at = lookup_k(0.10)
        assert below.budget_tier == "nothink"
        assert at.budget_tier == "standard"

    def test_boundary_0_20(self):
        below = lookup_k(0.199)
        at = lookup_k(0.20)
        assert below.budget_tier == "standard"
        assert at.budget_tier == "hard"

    def test_boundary_0_30(self):
        below = lookup_k(0.299)
        at = lookup_k(0.30)
        assert below.budget_tier == "hard"
        assert at.budget_tier == "extreme"

    def test_energy_1_0(self):
        alloc = lookup_k(1.0)
        assert alloc.budget_tier == "extreme"

    def test_energy_0_0(self):
        alloc = lookup_k(0.0)
        assert alloc.budget_tier == "nothink"

    def test_custom_table(self):
        custom = [KAllocation(0.0, 0.5, 2, 2, "light"),
                  KAllocation(0.5, 1.01, 10, 10, "extreme")]
        assert lookup_k(0.3, custom).min_k == 2
        assert lookup_k(0.7, custom).min_k == 10

    def test_empty_table_uses_default(self):
        # Empty list is falsy, so `[] or DEFAULT_K_TABLE` uses defaults
        alloc = lookup_k(0.5, [])
        assert alloc.budget_tier == "extreme"  # 0.5 in default table


# ---------------------------------------------------------------------------
# Test: compute_k function
# ---------------------------------------------------------------------------

class TestComputeK:

    def test_returns_min_k(self):
        k, tier = compute_k(0.05)
        assert k == 1
        assert tier == "nothink"

    def test_standard_range(self):
        k, tier = compute_k(0.15)
        assert k == 3
        assert tier == "standard"

    def test_hard_range(self):
        k, tier = compute_k(0.25)
        assert k == 5
        assert tier == "hard"

    def test_extreme_range(self):
        k, tier = compute_k(0.50)
        assert k == 8
        assert tier == "extreme"


# ---------------------------------------------------------------------------
# Test: AC-2A-1 — Easy problems get k <= 2
# ---------------------------------------------------------------------------

class TestAC2A1EasyProblems:

    def test_easy_raw_energy_gets_low_k(self, asc_enabled):
        """Easy problems (normalized energy < 0.10) should get k <= 2."""
        # Raw energy 3.0 -> normalized ~0.037 (well below 0.10)
        k, tier = asc_enabled.allocate(raw_energy=3.0, task_id="t1")
        assert k <= 2, f"Easy problem got k={k}, expected <= 2"
        assert tier == "nothink"

    def test_boundary_easy(self, asc_enabled):
        """Energy just below 0.10 normalized threshold."""
        # Find raw energy that gives ~0.09 normalized
        # sigmoid((E - 9.5) * 0.5) = 0.09 → E ≈ 4.94
        k, _ = asc_enabled.allocate(raw_energy=4.9, task_id="t2")
        assert k <= 2

    def test_very_easy(self, asc_enabled):
        """Very low energy (near 0)."""
        k, tier = asc_enabled.allocate(raw_energy=1.0, task_id="t3")
        assert k <= 2
        assert tier == "nothink"

    def test_multiple_easy_tasks(self, asc_enabled):
        """90%+ of easy tasks should get k <= 2."""
        easy_energies = [1.0, 2.0, 3.0, 3.5, 4.0, 4.2, 4.5, 4.7, 4.8, 4.9]
        low_k_count = 0
        for e in easy_energies:
            k, _ = asc_enabled.allocate(raw_energy=e)
            if k <= 2:
                low_k_count += 1
        assert low_k_count >= 9, f"Only {low_k_count}/10 easy tasks got k<=2"


# ---------------------------------------------------------------------------
# Test: AC-2A-2 — Hard problems get k >= 7
# ---------------------------------------------------------------------------

class TestAC2A2HardProblems:

    def test_hard_raw_energy_gets_high_k(self, asc_enabled):
        """Hard problems (normalized energy > 0.30) should get k >= 7."""
        # Raw energy 11.0 -> normalized ~0.645
        k, tier = asc_enabled.allocate(raw_energy=11.0, task_id="t1")
        assert k >= 7, f"Hard problem got k={k}, expected >= 7"

    def test_very_hard(self, asc_enabled):
        """Very high energy."""
        k, tier = asc_enabled.allocate(raw_energy=15.0, task_id="t2")
        assert k >= 7
        assert tier == "extreme"

    def test_boundary_hard(self, asc_enabled):
        """Energy just above 0.30 normalized threshold."""
        # sigmoid((E - 9.5) * 0.5) = 0.30 → E ≈ 7.80
        k, _ = asc_enabled.allocate(raw_energy=8.0, task_id="t3")
        norm = normalize_energy(8.0)
        assert norm >= 0.30, f"Energy 8.0 normalized to {norm}, expected >= 0.30"
        assert k >= 7 or k >= 5, f"Near-boundary got k={k}"

    def test_multiple_hard_tasks(self, asc_enabled):
        """90%+ of hard tasks should get k >= 7."""
        hard_energies = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0]
        high_k_count = 0
        for e in hard_energies:
            k, _ = asc_enabled.allocate(raw_energy=e)
            if k >= 7:
                high_k_count += 1
        assert high_k_count >= 9, f"Only {high_k_count}/10 hard tasks got k>=7"


# ---------------------------------------------------------------------------
# Test: AC-2A-3/4 — Structural tests for compute efficiency
# ---------------------------------------------------------------------------

class TestAC2A3ComputeEfficiency:

    def test_easy_uses_less_compute_than_hard(self, asc_enabled):
        """Easy tasks should use strictly fewer candidates than hard tasks."""
        k_easy, _ = asc_enabled.allocate(raw_energy=3.0)
        k_hard, _ = asc_enabled.allocate(raw_energy=14.0)
        assert k_easy < k_hard

    def test_progressive_k_increase(self, asc_enabled):
        """K should increase monotonically with energy."""
        energies = [2.0, 6.5, 7.5, 12.0]
        ks = [asc_enabled.allocate(raw_energy=e)[0] for e in energies]
        for i in range(len(ks) - 1):
            assert ks[i] <= ks[i + 1], f"k did not increase: {ks}"

    def test_total_budget_correlation(self, asc_enabled):
        """Average k across mixed tasks should be less than uniform max."""
        # Mix of easy/medium/hard tasks
        energies = [2.0, 3.0, 5.0, 6.5, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        ks = [asc_enabled.allocate(raw_energy=e)[0] for e in energies]
        avg_k = sum(ks) / len(ks)
        # If uniform k=12, adaptive should average much less
        assert avg_k < 12, f"Average k={avg_k:.1f}, not saving compute"


# ---------------------------------------------------------------------------
# Test: BlendASC class
# ---------------------------------------------------------------------------

class TestBlendASCClass:

    def test_allocation_with_task_id(self, asc_enabled, tmp_telemetry):
        k, tier = asc_enabled.allocate(raw_energy=5.0, task_id="LCB_001",
                                       probe_tokens=100, probe_time_ms=50.0)
        assert isinstance(k, int)
        assert isinstance(tier, str)
        events_file = tmp_telemetry / "blend_asc_events.jsonl"
        assert events_file.exists()

    def test_get_allocation_for_energy(self, asc_enabled):
        alloc = asc_enabled.get_allocation_for_energy(0.05)
        assert alloc.budget_tier == "nothink"
        alloc = asc_enabled.get_allocation_for_energy(0.50)
        assert alloc.budget_tier == "extreme"

    def test_get_table_summary(self, asc_enabled):
        summary = asc_enabled.get_table_summary()
        assert len(summary) == 4
        assert summary[0]["budget_tier"] == "nothink"
        assert summary[3]["budget_tier"] == "extreme"

    def test_custom_k_table(self, tmp_telemetry):
        custom = [KAllocation(0.0, 1.01, 5, 5, "standard")]
        cfg = BlendASCConfig(enabled=True)
        asc = BlendASC(cfg, k_table=custom, telemetry_dir=tmp_telemetry)
        k, tier = asc.allocate(raw_energy=3.0)
        assert k == 5
        assert tier == "standard"

    def test_no_crash_without_telemetry_dir(self):
        cfg = BlendASCConfig(enabled=True)
        asc = BlendASC(cfg, telemetry_dir=None)
        k, tier = asc.allocate(raw_energy=5.0, task_id="t1")
        assert isinstance(k, int)


# ---------------------------------------------------------------------------
# Test: Energy normalization integration
# ---------------------------------------------------------------------------

class TestEnergyNormalization:

    def test_pass_energy_is_easy(self):
        """V2 measured PASS mean = 5.00 → should normalize below 0.10."""
        norm = normalize_energy(5.0)
        assert norm < 0.10, f"PASS energy normalized to {norm}, expected < 0.10"

    def test_fail_energy_is_hard(self):
        """V2 measured FAIL mean = 14.04 → should normalize above 0.30."""
        norm = normalize_energy(14.04)
        assert norm > 0.30, f"FAIL energy normalized to {norm}, expected > 0.30"

    def test_midpoint_is_half(self):
        """Energy at midpoint (9.5) should normalize to 0.5."""
        norm = normalize_energy(9.5)
        assert abs(norm - 0.5) < 0.001


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_logged_on_allocate(self, asc_enabled, tmp_telemetry):
        asc_enabled.allocate(raw_energy=5.0, task_id="LCB_001",
                            probe_tokens=150, probe_time_ms=75.0)
        events_file = tmp_telemetry / "blend_asc_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["raw_energy"] == 5.0
        assert data["probe_tokens"] == 150
        assert data["probe_time_ms"] == 75.0
        assert "normalized_energy" in data
        assert "allocated_k" in data
        assert "budget_tier" in data
        assert "timestamp" in data

    def test_multiple_events(self, asc_enabled, tmp_telemetry):
        for i in range(5):
            asc_enabled.allocate(raw_energy=float(i + 3), task_id=f"t{i}")
        events_file = tmp_telemetry / "blend_asc_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_no_telemetry_without_task_id(self, asc_enabled, tmp_telemetry):
        asc_enabled.allocate(raw_energy=5.0)  # No task_id
        events_file = tmp_telemetry / "blend_asc_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_event_to_dict(self):
        e = BlendASCEvent(
            task_id="t1", raw_energy=5.0, normalized_energy=0.07,
            allocated_k=2, budget_tier="nothink",
            probe_tokens=100, probe_time_ms=50.0,
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["raw_energy"] == 5.0
        assert d["allocated_k"] == 2
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = BlendASCConfig()
        assert cfg.enabled is False
        assert cfg.default_k == 3
        assert cfg.energy_midpoint == 9.5
        assert cfg.energy_steepness == 0.5
