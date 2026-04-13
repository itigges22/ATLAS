"""Tests for V3 DivSampling (Feature 1B)."""

import json
from pathlib import Path

import pytest

from benchmark.v3.div_sampling import (
    DEFAULT_PERTURBATIONS,
    DivSampling,
    DivSamplingConfig,
    DivSamplingEvent,
    Perturbation,
    apply_perturbation,
    get_perturbation_library,
    select_perturbation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def ds_enabled(tmp_telemetry):
    cfg = DivSamplingConfig(enabled=True)
    return DivSampling(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def ds_disabled(tmp_telemetry):
    cfg = DivSamplingConfig(enabled=False)
    return DivSampling(cfg, telemetry_dir=tmp_telemetry)


# ---------------------------------------------------------------------------
# Test: Disabled noop
# ---------------------------------------------------------------------------

class TestDisabledNoop:

    def test_apply_returns_unmodified(self, ds_disabled):
        prompt = "Solve this problem"
        result = ds_disabled.apply(prompt, candidate_index=0, task_id="t1")
        assert result == prompt

    def test_get_perturbation_returns_disabled(self, ds_disabled):
        p = ds_disabled.get_perturbation(0)
        assert p.label == "disabled"
        assert p.text == ""

    def test_no_telemetry_when_disabled(self, ds_disabled, tmp_telemetry):
        ds_disabled.apply("prompt", 0, task_id="t1")
        events_file = tmp_telemetry / "div_sampling_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: AC-1B-1 — Library contains >=10 perturbations with category coverage
# ---------------------------------------------------------------------------

class TestAC1B1PerturbationLibrary:

    def test_library_has_at_least_10(self):
        assert len(DEFAULT_PERTURBATIONS) >= 10, (
            f"Expected >=10 perturbations, got {len(DEFAULT_PERTURBATIONS)}"
        )

    def test_role_category_has_at_least_4(self):
        roles = [p for p in DEFAULT_PERTURBATIONS if p.category == "role"]
        assert len(roles) >= 4, f"Expected >=4 role perturbations, got {len(roles)}"

    def test_instruction_category_has_at_least_4(self):
        instr = [p for p in DEFAULT_PERTURBATIONS if p.category == "instruction"]
        assert len(instr) >= 4, (
            f"Expected >=4 instruction perturbations, got {len(instr)}"
        )

    def test_style_category_has_at_least_4(self):
        styles = [p for p in DEFAULT_PERTURBATIONS if p.category == "style"]
        assert len(styles) >= 4, (
            f"Expected >=4 style perturbations, got {len(styles)}"
        )

    def test_all_perturbations_have_text(self):
        for p in DEFAULT_PERTURBATIONS:
            assert len(p.text) > 10, f"Perturbation {p.label} has too-short text"

    def test_all_labels_unique(self):
        labels = [p.label for p in DEFAULT_PERTURBATIONS]
        assert len(labels) == len(set(labels)), "Duplicate labels found"

    def test_class_reports_correct_counts(self, ds_enabled):
        counts = ds_enabled.get_category_counts()
        assert counts.get("role", 0) >= 4
        assert counts.get("instruction", 0) >= 4
        assert counts.get("style", 0) >= 4


# ---------------------------------------------------------------------------
# Test: AC-1B-2 — Cosine similarity decrease (structural; runtime needs embeddings)
# ---------------------------------------------------------------------------

class TestAC1B2DiversityIncrease:

    def test_different_candidates_get_different_perturbations(self, ds_enabled):
        """Each candidate index should map to a different perturbation."""
        perturbations = [
            ds_enabled.get_perturbation(i) for i in range(ds_enabled.library_size)
        ]
        labels = [p.label for p in perturbations]
        # All should be unique within one cycle
        assert len(labels) == len(set(labels))

    def test_perturbations_modify_prompt(self, ds_enabled):
        """Applying different perturbations should produce different prompts."""
        base = "Solve this problem: find two sum"
        prompts = [
            ds_enabled.apply(base, i, task_id="t1")
            for i in range(3)
        ]
        # All should be different from each other
        assert len(set(prompts)) == 3
        # All should contain the base prompt
        for p in prompts:
            assert base in p

    def test_perturbation_does_not_modify_core_prompt(self, ds_enabled):
        """Core task description must remain intact."""
        base = "Given an array of integers, return the maximum sum subarray."
        result = ds_enabled.apply(base, 0, task_id="t1")
        assert base in result
        # Perturbation should be BEFORE the base prompt
        assert result.index(base) > 0


# ---------------------------------------------------------------------------
# Test: AC-1B-3 — No single perturbation causes >3% regression (structural)
# ---------------------------------------------------------------------------

class TestAC1B3NoRegression:

    def test_perturbations_are_positive_guidance(self):
        """All perturbations should be constructive, not destructive."""
        # These patterns indicate purely negative/destructive guidance
        destructive_patterns = [
            "don't write", "never use", "avoid all", "ignore the",
            "skip testing", "bad code", "terrible",
        ]
        for p in DEFAULT_PERTURBATIONS:
            text_lower = p.text.lower()
            for neg in destructive_patterns:
                assert neg not in text_lower, (
                    f"Perturbation {p.label} contains destructive pattern: {neg}"
                )

    def test_perturbation_does_not_inject_code(self):
        """Perturbations should not contain executable Python code."""
        for p in DEFAULT_PERTURBATIONS:
            # Check for actual code patterns, not natural language mentions
            assert "\ndef " not in p.text and not p.text.startswith("def ")
            assert "\nimport " not in p.text and not p.text.startswith("import ")
            assert "```" not in p.text


# ---------------------------------------------------------------------------
# Test: AC-1B-4 — >=3% improvement (runtime benchmark; structural test here)
# ---------------------------------------------------------------------------

class TestAC1B4Improvement:

    def test_library_covers_diverse_strategies(self, ds_enabled):
        """Library should cover multiple algorithmic reasoning strategies."""
        categories = ds_enabled.get_category_counts()
        assert len(categories) >= 3, "Need at least 3 categories"
        for cat, count in categories.items():
            assert count >= 4, f"Category {cat} has only {count} perturbations"


# ---------------------------------------------------------------------------
# Test: Perturbation application
# ---------------------------------------------------------------------------

class TestApplyPerturbation:

    def test_basic_application(self):
        p = Perturbation(text="Be careful.", category="instruction", label="careful")
        result = apply_perturbation("Solve X", p)
        assert result == "Be careful.\n\nSolve X"

    def test_empty_perturbation(self):
        p = Perturbation(text="", category="none", label="empty")
        result = apply_perturbation("Solve X", p)
        assert result == "Solve X"

    def test_multiline_prompt(self):
        p = Perturbation(text="Think hard.", category="instruction", label="think")
        prompt = "Line 1\nLine 2\nLine 3"
        result = apply_perturbation(prompt, p)
        assert result.startswith("Think hard.")
        assert prompt in result


# ---------------------------------------------------------------------------
# Test: Perturbation selection
# ---------------------------------------------------------------------------

class TestSelectPerturbation:

    def test_modular_cycling(self):
        library = DEFAULT_PERTURBATIONS
        n = len(library)
        # Index n should wrap to same as index 0
        assert select_perturbation(0, library).label == select_perturbation(n, library).label
        assert select_perturbation(1, library).label == select_perturbation(n + 1, library).label

    def test_empty_library(self):
        p = select_perturbation(0, [])
        assert p.label == "empty"
        assert p.text == ""

    def test_all_perturbations_reachable(self):
        library = DEFAULT_PERTURBATIONS
        reached = set()
        for i in range(len(library)):
            reached.add(select_perturbation(i, library).label)
        assert len(reached) == len(library)


# ---------------------------------------------------------------------------
# Test: Custom perturbations
# ---------------------------------------------------------------------------

class TestCustomPerturbations:

    def test_custom_appended(self):
        cfg = DivSamplingConfig(
            enabled=True,
            custom_perturbations=["Use recursion.", "Avoid global state."],
        )
        library = get_perturbation_library(cfg)
        assert len(library) == len(DEFAULT_PERTURBATIONS) + 2
        custom = [p for p in library if p.category == "custom"]
        assert len(custom) == 2
        assert custom[0].text == "Use recursion."
        assert custom[1].text == "Avoid global state."

    def test_custom_labels_indexed(self):
        cfg = DivSamplingConfig(
            enabled=True,
            custom_perturbations=["A", "B"],
        )
        library = get_perturbation_library(cfg)
        custom = [p for p in library if p.category == "custom"]
        assert custom[0].label == "custom_0"
        assert custom[1].label == "custom_1"

    def test_no_custom_returns_defaults(self):
        cfg = DivSamplingConfig(enabled=True)
        library = get_perturbation_library(cfg)
        assert len(library) == len(DEFAULT_PERTURBATIONS)


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_event_logged_on_apply(self, ds_enabled, tmp_telemetry):
        ds_enabled.apply("Solve X", 0, task_id="LCB_001")
        events_file = tmp_telemetry / "div_sampling_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["candidate_index"] == 0
        assert data["perturbation_label"] == DEFAULT_PERTURBATIONS[0].label
        assert data["perturbation_category"] == DEFAULT_PERTURBATIONS[0].category

    def test_multiple_events(self, ds_enabled, tmp_telemetry):
        for i in range(5):
            ds_enabled.apply("prompt", i, task_id="t1")
        events_file = tmp_telemetry / "div_sampling_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_no_telemetry_without_task_id(self, ds_enabled, tmp_telemetry):
        ds_enabled.apply("prompt", 0)  # No task_id
        events_file = tmp_telemetry / "div_sampling_events.jsonl"
        assert not events_file.exists()

    def test_no_crash_without_telemetry_dir(self):
        cfg = DivSamplingConfig(enabled=True)
        ds = DivSampling(cfg, telemetry_dir=None)
        result = ds.apply("prompt", 0, task_id="t1")
        assert "prompt" in result


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_perturbation_to_dict(self):
        p = Perturbation(text="Be fast.", category="style", label="fast")
        d = p.to_dict()
        assert d == {"text": "Be fast.", "category": "style", "label": "fast"}

    def test_event_to_dict(self):
        e = DivSamplingEvent(
            task_id="t1", candidate_index=2,
            perturbation_label="mathematician",
            perturbation_category="role",
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["candidate_index"] == 2
        assert "timestamp" in d
