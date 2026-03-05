"""Tests for V3 ACE Pipeline (Feature 3G)."""

import json
from pathlib import Path
from typing import Optional, Tuple

import pytest

from benchmark.v3.ace_pipeline import (
    ACEConfig,
    ACEEvent,
    ACEPipeline,
    PlaybookEntry,
)


# ---------------------------------------------------------------------------
# Mock callables
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns a predefined response."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list = []

    def __call__(self, prompt: str, temperature: float,
                 max_tokens: int, seed: Optional[int]) -> Tuple[str, int, float]:
        self.calls.append({
            "prompt": prompt, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed,
        })
        return self.response, 50, 25.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_telemetry(tmp_path):
    """Provide a temporary telemetry directory."""
    d = tmp_path / "telemetry"
    d.mkdir()
    return d


@pytest.fixture
def ace_enabled(tmp_telemetry):
    """ACEPipeline instance with enabled=True."""
    cfg = ACEConfig(enabled=True)
    return ACEPipeline(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def ace_disabled(tmp_telemetry):
    """ACEPipeline instance with enabled=False."""
    cfg = ACEConfig(enabled=False)
    return ACEPipeline(cfg, telemetry_dir=tmp_telemetry)


@pytest.fixture
def ace_with_entries(tmp_telemetry):
    """ACEPipeline with pre-loaded playbook entries."""
    cfg = ACEConfig(enabled=True, max_playbook_tokens=2000)
    ace = ACEPipeline(cfg, telemetry_dir=tmp_telemetry)
    ace._playbook = [
        PlaybookEntry(
            principle="Always validate input boundaries before processing",
            category="sorting",
            confidence=0.9,
            uses=5,
            successes=4,
        ),
        PlaybookEntry(
            principle="Use dynamic programming for overlapping subproblems",
            category="dp",
            confidence=0.8,
            uses=10,
            successes=7,
        ),
        PlaybookEntry(
            principle="Check for off-by-one errors in loop bounds",
            category="",  # Applies to all categories
            confidence=0.7,
            uses=8,
            successes=5,
        ),
        PlaybookEntry(
            principle="Low confidence entry that should be filtered",
            category="sorting",
            confidence=0.2,
            uses=2,
            successes=0,
        ),
    ]
    return ace


# ---------------------------------------------------------------------------
# Test: Disabled noop behavior
# ---------------------------------------------------------------------------

class TestDisabledNoop:
    """When enabled=False, ACEPipeline should be a complete noop."""

    def test_get_context_returns_empty(self, ace_disabled):
        context = ace_disabled.get_context(["sorting"], task_id="t1")
        assert context == ""

    def test_learn_returns_entry_but_does_not_add(self, ace_disabled):
        entry = ace_disabled.learn("test principle", category="cat", task_id="t1")
        assert entry.principle == "test principle"
        assert ace_disabled.playbook_size == 0

    def test_no_telemetry_when_disabled(self, ace_disabled, tmp_telemetry):
        ace_disabled.get_context(["sorting"], task_id="t1")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert not events_file.exists()

    def test_decay_confidence_noop(self, ace_disabled):
        pruned = ace_disabled.decay_confidence()
        assert pruned == 0

    def test_prune_noop(self, ace_disabled):
        pruned = ace_disabled.prune(task_id="t1")
        assert pruned == 0

    def test_playbook_empty(self, ace_disabled):
        assert ace_disabled.playbook_size == 0
        assert ace_disabled.playbook == []


# ---------------------------------------------------------------------------
# Test: PlaybookEntry
# ---------------------------------------------------------------------------

class TestPlaybookEntry:

    def test_estimate_tokens(self):
        entry = PlaybookEntry(principle="one two three four five")
        tokens = entry.estimate_tokens()
        assert tokens == int(5 * 1.3)

    def test_estimate_tokens_single_word(self):
        entry = PlaybookEntry(principle="word")
        assert entry.estimate_tokens() == int(1 * 1.3)

    def test_estimate_tokens_empty(self):
        entry = PlaybookEntry(principle="")
        assert entry.estimate_tokens() == 0

    def test_effectiveness_with_uses(self):
        entry = PlaybookEntry(principle="test", uses=10, successes=7)
        assert entry.effectiveness == pytest.approx(0.7)

    def test_effectiveness_zero_uses(self):
        entry = PlaybookEntry(principle="test", uses=0, successes=0)
        assert entry.effectiveness == 0.0

    def test_effectiveness_all_success(self):
        entry = PlaybookEntry(principle="test", uses=5, successes=5)
        assert entry.effectiveness == pytest.approx(1.0)

    def test_to_dict(self):
        entry = PlaybookEntry(
            principle="Test principle",
            category="sorting",
            confidence=0.85,
            uses=10,
            successes=7,
            created_at="2026-02-24T00:00:00Z",
            last_used="2026-02-24T12:00:00Z",
            version=2,
        )
        d = entry.to_dict()
        assert d["principle"] == "Test principle"
        assert d["category"] == "sorting"
        assert d["confidence"] == 0.85
        assert d["uses"] == 10
        assert d["successes"] == 7
        assert d["created_at"] == "2026-02-24T00:00:00Z"
        assert d["last_used"] == "2026-02-24T12:00:00Z"
        assert d["version"] == 2

    def test_to_dict_auto_timestamp(self):
        entry = PlaybookEntry(principle="test")
        d = entry.to_dict()
        assert len(d["created_at"]) > 0

    def test_from_dict(self):
        d = {
            "principle": "Test principle",
            "category": "dp",
            "confidence": 0.9,
            "uses": 5,
            "successes": 4,
            "created_at": "2026-01-01T00:00:00Z",
            "last_used": "2026-02-01T00:00:00Z",
            "version": 3,
        }
        entry = PlaybookEntry.from_dict(d)
        assert entry.principle == "Test principle"
        assert entry.category == "dp"
        assert entry.confidence == 0.9
        assert entry.uses == 5
        assert entry.successes == 4
        assert entry.version == 3

    def test_from_dict_defaults(self):
        entry = PlaybookEntry.from_dict({})
        assert entry.principle == ""
        assert entry.category == ""
        assert entry.confidence == 1.0
        assert entry.uses == 0
        assert entry.successes == 0
        assert entry.version == 1


# ---------------------------------------------------------------------------
# Test: ACEPipeline.get_context
# ---------------------------------------------------------------------------

class TestGetContext:

    def test_returns_matching_entries(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        assert "validate input" in context.lower()

    def test_includes_universal_entries(self, ace_with_entries):
        """Entries with empty category apply to all problem types."""
        context = ace_with_entries.get_context(["sorting"])
        assert "off-by-one" in context.lower()

    def test_excludes_low_confidence(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        assert "Low confidence entry" not in context

    def test_returns_empty_for_no_matches(self, ace_with_entries):
        context = ace_with_entries.get_context(["graph_theory"])
        # Still gets the universal entry (empty category)
        assert "off-by-one" in context.lower()

    def test_returns_empty_when_no_playbook(self, ace_enabled):
        context = ace_enabled.get_context(["sorting"])
        assert context == ""

    def test_ranked_by_confidence(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        lines = context.split("\n")
        # First non-header line should contain highest-confidence entry
        content_lines = [l for l in lines if l.startswith("- ")]
        assert "validate input" in content_lines[0].lower()

    def test_respects_token_budget(self, tmp_telemetry):
        cfg = ACEConfig(enabled=True, max_playbook_tokens=5)
        ace = ACEPipeline(cfg, telemetry_dir=tmp_telemetry)
        ace._playbook = [
            PlaybookEntry(
                principle="This is a very long principle with many many words to test token budget enforcement thoroughly",
                category="",
                confidence=0.9,
            ),
            PlaybookEntry(
                principle="Second entry should not appear",
                category="",
                confidence=0.8,
            ),
        ]
        context = ace.get_context(["any"])
        # With budget of 5 tokens, may fit only the first (or none)
        # The first entry has ~15 words * 1.3 = ~19 tokens, which exceeds 5
        assert context == ""

    def test_fits_multiple_within_budget(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        # With 2000 token budget, should fit multiple entries
        content_lines = [l for l in context.split("\n") if l.startswith("- ")]
        assert len(content_lines) >= 2

    def test_header_present(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        assert "Known principles" in context

    def test_empty_categories_gets_universal(self, ace_with_entries):
        context = ace_with_entries.get_context([])
        # Universal entries (empty category) should match any list
        # Actually, no categories means no match unless entry.category is empty
        assert "off-by-one" in context.lower()


# ---------------------------------------------------------------------------
# Test: ACEPipeline.learn
# ---------------------------------------------------------------------------

class TestLearn:

    def test_adds_entry_to_playbook(self, ace_enabled):
        assert ace_enabled.playbook_size == 0
        ace_enabled.learn("New principle", category="sorting")
        assert ace_enabled.playbook_size == 1

    def test_returns_created_entry(self, ace_enabled):
        entry = ace_enabled.learn("Test principle", category="dp")
        assert entry.principle == "Test principle"
        assert entry.category == "dp"
        assert entry.confidence == 1.0
        assert len(entry.created_at) > 0

    def test_multiple_learns(self, ace_enabled):
        ace_enabled.learn("Principle 1", category="cat1")
        ace_enabled.learn("Principle 2", category="cat2")
        ace_enabled.learn("Principle 3", category="cat3")
        assert ace_enabled.playbook_size == 3

    def test_learn_with_task_id_logs_telemetry(self, ace_enabled, tmp_telemetry):
        ace_enabled.learn("Test principle", task_id="t1")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text().strip())
        assert data["operation"] == "learn"
        assert data["num_entries_learned"] == 1


# ---------------------------------------------------------------------------
# Test: ACEPipeline.record_usage
# ---------------------------------------------------------------------------

class TestRecordUsage:

    def test_records_success(self, ace_with_entries):
        ace_with_entries.record_usage(
            "Always validate input boundaries before processing",
            succeeded=True,
        )
        entry = ace_with_entries._playbook[0]
        assert entry.uses == 6
        assert entry.successes == 5

    def test_records_failure(self, ace_with_entries):
        ace_with_entries.record_usage(
            "Always validate input boundaries before processing",
            succeeded=False,
        )
        entry = ace_with_entries._playbook[0]
        assert entry.uses == 6
        assert entry.successes == 4

    def test_updates_last_used(self, ace_with_entries):
        ace_with_entries.record_usage(
            "Always validate input boundaries before processing",
            succeeded=True,
        )
        entry = ace_with_entries._playbook[0]
        assert len(entry.last_used) > 0

    def test_no_error_for_unknown_principle(self, ace_with_entries):
        # Should not raise
        ace_with_entries.record_usage("Nonexistent principle", succeeded=True)

    def test_only_first_match_updated(self, ace_with_entries):
        original_uses = [e.uses for e in ace_with_entries._playbook]
        ace_with_entries.record_usage(
            "Always validate input boundaries before processing",
            succeeded=True,
        )
        # Only first entry should change
        assert ace_with_entries._playbook[0].uses == original_uses[0] + 1
        assert ace_with_entries._playbook[1].uses == original_uses[1]


# ---------------------------------------------------------------------------
# Test: ACEPipeline.decay_confidence
# ---------------------------------------------------------------------------

class TestDecayConfidence:

    def test_decays_all_entries(self, ace_with_entries):
        original = [e.confidence for e in ace_with_entries._playbook]
        ace_with_entries.decay_confidence()
        for i, entry in enumerate(ace_with_entries._playbook):
            if entry.confidence >= ace_with_entries.config.min_confidence:
                expected = original[i] * (1.0 - ace_with_entries.config.confidence_decay_rate)
                assert entry.confidence == pytest.approx(expected)

    def test_prunes_below_threshold(self, ace_with_entries):
        # The entry with confidence=0.2 is below min_confidence=0.3
        # After decay: 0.2 * 0.9 = 0.18, which is still below 0.3
        original_size = ace_with_entries.playbook_size
        pruned = ace_with_entries.decay_confidence()
        assert pruned == 1
        assert ace_with_entries.playbook_size == original_size - 1

    def test_returns_prune_count(self, ace_with_entries):
        pruned = ace_with_entries.decay_confidence()
        assert pruned >= 1

    def test_repeated_decay_prunes_more(self, tmp_telemetry):
        cfg = ACEConfig(enabled=True, confidence_decay_rate=0.5, min_confidence=0.3)
        ace = ACEPipeline(cfg, telemetry_dir=tmp_telemetry)
        ace._playbook = [
            PlaybookEntry(principle="entry1", confidence=0.9),
            PlaybookEntry(principle="entry2", confidence=0.5),
            PlaybookEntry(principle="entry3", confidence=0.35),
        ]
        # First decay: 0.9->0.45, 0.5->0.25(pruned), 0.35->0.175(pruned)
        pruned1 = ace.decay_confidence()
        assert pruned1 == 2
        assert ace.playbook_size == 1

    def test_noop_when_disabled(self, ace_disabled):
        pruned = ace_disabled.decay_confidence()
        assert pruned == 0


# ---------------------------------------------------------------------------
# Test: ACEPipeline.prune
# ---------------------------------------------------------------------------

class TestPrune:

    def test_prune_calls_decay(self, ace_with_entries):
        pruned = ace_with_entries.prune(task_id="t1")
        assert pruned >= 1

    def test_prune_logs_telemetry(self, ace_with_entries, tmp_telemetry):
        ace_with_entries.prune(task_id="prune_run")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text().strip())
        assert data["operation"] == "prune"
        assert data["num_entries_pruned"] >= 1
        assert data["task_id"] == "prune_run"

    def test_prune_no_telemetry_when_zero_pruned(self, ace_enabled, tmp_telemetry):
        ace_enabled.prune(task_id="t1")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert not events_file.exists()


# ---------------------------------------------------------------------------
# Test: Playbook persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load(self, tmp_path):
        playbook_path = str(tmp_path / "playbook.json")
        cfg = ACEConfig(enabled=True, playbook_path=playbook_path)

        # Create and populate
        ace1 = ACEPipeline(cfg)
        ace1._playbook = [
            PlaybookEntry(
                principle="Test principle",
                category="sorting",
                confidence=0.85,
                uses=5,
                successes=3,
                version=2,
            ),
        ]
        ace1._save(playbook_path)

        # Load in new instance
        ace2 = ACEPipeline(cfg)
        assert ace2.playbook_size == 1
        entry = ace2.playbook[0]
        assert entry.principle == "Test principle"
        assert entry.category == "sorting"
        assert entry.confidence == 0.85
        assert entry.uses == 5
        assert entry.successes == 3
        assert entry.version == 2

    def test_load_missing_file(self, tmp_path):
        playbook_path = str(tmp_path / "nonexistent.json")
        cfg = ACEConfig(enabled=True, playbook_path=playbook_path)
        ace = ACEPipeline(cfg)
        assert ace.playbook_size == 0

    def test_load_invalid_json(self, tmp_path):
        playbook_path = str(tmp_path / "bad.json")
        with open(playbook_path, "w") as f:
            f.write("not valid json")
        cfg = ACEConfig(enabled=True, playbook_path=playbook_path)
        ace = ACEPipeline(cfg)
        assert ace.playbook_size == 0

    def test_learn_auto_saves(self, tmp_path):
        playbook_path = str(tmp_path / "auto_save.json")
        cfg = ACEConfig(enabled=True, playbook_path=playbook_path)
        ace = ACEPipeline(cfg)
        ace.learn("Auto-saved principle", category="test")
        assert Path(playbook_path).exists()
        with open(playbook_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["principle"] == "Auto-saved principle"

    def test_load_non_list_ignored(self, tmp_path):
        playbook_path = str(tmp_path / "dict.json")
        with open(playbook_path, "w") as f:
            json.dump({"not": "a list"}, f)
        cfg = ACEConfig(enabled=True, playbook_path=playbook_path)
        ace = ACEPipeline(cfg)
        assert ace.playbook_size == 0


# ---------------------------------------------------------------------------
# Test: Properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_playbook_returns_copy(self, ace_with_entries):
        p = ace_with_entries.playbook
        p.append(PlaybookEntry(principle="extra"))
        assert ace_with_entries.playbook_size == 4

    def test_playbook_size(self, ace_with_entries):
        assert ace_with_entries.playbook_size == 4

    def test_empty_playbook(self, ace_enabled):
        assert ace_enabled.playbook_size == 0
        assert ace_enabled.playbook == []

    def test_to_dict(self, ace_with_entries):
        d = ace_with_entries.to_dict()
        assert len(d) == 4
        for entry_dict in d:
            assert "principle" in entry_dict
            assert "confidence" in entry_dict


# ---------------------------------------------------------------------------
# Test: AC-3G-1 — Maintains evolving playbook
# ---------------------------------------------------------------------------

class TestAC3G1EvolvingPlaybook:
    """AC-3G-1: System maintains an evolving playbook of principles."""

    def test_learn_adds_entries(self, ace_enabled):
        ace_enabled.learn("Principle A", category="cat1")
        ace_enabled.learn("Principle B", category="cat2")
        assert ace_enabled.playbook_size == 2

    def test_entries_are_versioned(self, ace_enabled):
        entry = ace_enabled.learn("Versioned principle")
        assert entry.version == 1

    def test_entries_have_timestamps(self, ace_enabled):
        entry = ace_enabled.learn("Timestamped principle")
        assert len(entry.created_at) > 0


# ---------------------------------------------------------------------------
# Test: AC-3G-2 — Injects relevant entries into prompts
# ---------------------------------------------------------------------------

class TestAC3G2Injection:
    """AC-3G-2: System injects relevant playbook entries within token budget."""

    def test_context_includes_relevant_entries(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        assert "validate input" in context.lower()

    def test_context_within_token_budget(self, ace_with_entries):
        context = ace_with_entries.get_context(["sorting"])
        # Rough token count: words * 1.3
        words = len(context.split())
        tokens = int(words * 1.3)
        assert tokens <= ace_with_entries.config.max_playbook_tokens + 50  # header margin


# ---------------------------------------------------------------------------
# Test: AC-3G-3 — Prunes by confidence decay
# ---------------------------------------------------------------------------

class TestAC3G3ConfidenceDecay:
    """AC-3G-3: System prunes entries below confidence threshold via decay."""

    def test_decay_reduces_confidence(self, ace_with_entries):
        original = ace_with_entries._playbook[0].confidence
        ace_with_entries.decay_confidence()
        # Entry should still exist (0.9 * 0.9 = 0.81 > 0.3)
        assert ace_with_entries._playbook[0].confidence < original

    def test_low_entries_pruned(self, ace_with_entries):
        pruned = ace_with_entries.decay_confidence()
        assert pruned >= 1
        for entry in ace_with_entries._playbook:
            assert entry.confidence >= ace_with_entries.config.min_confidence


# ---------------------------------------------------------------------------
# Test: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:

    def test_inject_event_written(self, ace_with_entries, tmp_telemetry):
        ace_with_entries.get_context(["sorting"], task_id="LCB_001")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "LCB_001"
        assert data["operation"] == "inject"
        assert data["num_entries_injected"] > 0
        assert "playbook_size" in data
        assert "timestamp" in data

    def test_learn_event_written(self, ace_enabled, tmp_telemetry):
        ace_enabled.learn("test", task_id="t1")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text().strip())
        assert data["operation"] == "learn"

    def test_prune_event_written(self, ace_with_entries, tmp_telemetry):
        ace_with_entries.prune(task_id="prune1")
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert events_file.exists()
        data = json.loads(events_file.read_text().strip())
        assert data["operation"] == "prune"

    def test_multiple_events_appended(self, ace_with_entries, tmp_telemetry):
        ace_with_entries.get_context(["sorting"], task_id="t1")
        ace_with_entries.get_context(["dp"], task_id="t2")
        ace_with_entries.get_context(["sorting"], task_id="t3")
        events_file = tmp_telemetry / "ace_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_no_telemetry_without_task_id(self, ace_with_entries, tmp_telemetry):
        ace_with_entries.get_context(["sorting"])
        events_file = tmp_telemetry / "ace_events.jsonl"
        assert not events_file.exists()

    def test_no_crash_without_telemetry_dir(self):
        cfg = ACEConfig(enabled=True)
        ace = ACEPipeline(cfg, telemetry_dir=None)
        ace._playbook = [
            PlaybookEntry(principle="test", category="cat", confidence=0.9),
        ]
        context = ace.get_context(["cat"], task_id="t1")
        assert "test" in context


# ---------------------------------------------------------------------------
# Test: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:

    def test_ace_event_to_dict(self):
        e = ACEEvent(
            task_id="t1", operation="inject",
            num_entries_injected=3, num_entries_learned=0,
            num_entries_pruned=0, playbook_size=10,
        )
        d = e.to_dict()
        assert d["task_id"] == "t1"
        assert d["operation"] == "inject"
        assert d["num_entries_injected"] == 3
        assert d["num_entries_learned"] == 0
        assert d["num_entries_pruned"] == 0
        assert d["playbook_size"] == 10
        assert "timestamp" in d

    def test_config_defaults(self):
        cfg = ACEConfig()
        assert cfg.enabled is False
        assert cfg.max_playbook_tokens == 2000
        assert cfg.confidence_decay_rate == 0.1
        assert cfg.min_confidence == 0.3
        assert cfg.playbook_path == ""

    def test_pipeline_to_dict(self, ace_with_entries):
        d = ace_with_entries.to_dict()
        assert isinstance(d, list)
        assert len(d) == 4


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_learn_empty_principle(self, ace_enabled):
        entry = ace_enabled.learn("")
        assert entry.principle == ""
        assert ace_enabled.playbook_size == 1

    def test_get_context_no_categories(self, ace_with_entries):
        context = ace_with_entries.get_context([])
        # Only universal entries (empty category) should match
        assert "off-by-one" in context.lower()
        assert "dynamic programming" not in context.lower()

    def test_record_usage_empty_playbook(self, ace_enabled):
        # Should not raise
        ace_enabled.record_usage("nonexistent", succeeded=True)

    def test_decay_empty_playbook(self, ace_enabled):
        pruned = ace_enabled.decay_confidence()
        assert pruned == 0

    def test_prune_saves_when_path_set(self, tmp_path):
        playbook_path = str(tmp_path / "prune_save.json")
        cfg = ACEConfig(enabled=True, playbook_path=playbook_path)
        ace = ACEPipeline(cfg)
        ace._playbook = [
            PlaybookEntry(principle="keeper", confidence=0.9),
            PlaybookEntry(principle="pruned", confidence=0.1),
        ]
        ace.prune()
        assert Path(playbook_path).exists()
        with open(playbook_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["principle"] == "keeper"


class TestDerivationFields:
    """Test new derivation tracking fields on PlaybookEntry."""

    def test_entry_has_derivation_fields(self):
        entry = PlaybookEntry(
            principle="test",
            entry_id="abc-123",
            depth=1,
            parent_ids=["parent-1", "parent-2"],
            provenance="derived",
            source_task_id="task_50",
        )
        assert entry.entry_id == "abc-123"
        assert entry.depth == 1
        assert entry.parent_ids == ["parent-1", "parent-2"]
        assert entry.provenance == "derived"
        assert entry.source_task_id == "task_50"

    def test_default_derivation_fields(self):
        entry = PlaybookEntry(principle="test")
        assert entry.entry_id == ""
        assert entry.depth == 0
        assert entry.parent_ids == []
        assert entry.provenance == "observed"
        assert entry.source_task_id == ""

    def test_to_dict_includes_derivation(self):
        entry = PlaybookEntry(
            principle="test", entry_id="x", depth=2,
            parent_ids=["a", "b"], provenance="derived",
        )
        d = entry.to_dict()
        assert d["entry_id"] == "x"
        assert d["depth"] == 2
        assert d["parent_ids"] == ["a", "b"]
        assert d["provenance"] == "derived"

    def test_from_dict_round_trip(self):
        entry = PlaybookEntry(
            principle="test", entry_id="x", depth=1,
            parent_ids=["p1"], provenance="derived",
            source_task_id="t1",
        )
        d = entry.to_dict()
        restored = PlaybookEntry.from_dict(d)
        assert restored.entry_id == "x"
        assert restored.depth == 1
        assert restored.parent_ids == ["p1"]
        assert restored.provenance == "derived"
        assert restored.source_task_id == "t1"


class TestDerive:
    """Test knowledge derivation with depth cap."""

    def test_derive_from_two_parents(self, ace_enabled, tmp_telemetry):
        # Create two observed principles
        a = ace_enabled.learn("sorting is O(n log n)", category="sorting", task_id="t1")
        b = ace_enabled.learn("problem needs sorted input", category="sorting", task_id="t2")

        derived = ace_enabled.derive(
            parent_ids=[a.entry_id, b.entry_id],
            new_principle="use O(n log n) preprocessing",
            category="sorting", task_id="t3",
        )
        assert derived is not None
        assert derived.depth == 1
        assert derived.provenance == "derived"
        assert set(derived.parent_ids) == {a.entry_id, b.entry_id}
        assert derived.confidence < 1.0  # derivation discount

    def test_depth_cap_at_3(self, ace_enabled):
        # Build chain: depth 0 -> 1 -> 2 -> 3
        d0 = ace_enabled.learn("base fact", task_id="t0")
        d1 = ace_enabled.derive([d0.entry_id], "derived 1", task_id="t1")
        d2 = ace_enabled.derive([d1.entry_id], "derived 2", task_id="t2")
        d3 = ace_enabled.derive([d2.entry_id], "derived 3", task_id="t3")
        assert d3.depth == 3
        # Depth 4 should be refused
        d4 = ace_enabled.derive([d3.entry_id], "derived 4", task_id="t4")
        assert d4 is None

    def test_derive_confidence_discount(self, ace_enabled):
        a = ace_enabled.learn("fact a", task_id="t1")
        b = ace_enabled.learn("fact b", task_id="t2")
        # Lower one parent's confidence
        a.confidence = 0.8
        derived = ace_enabled.derive([a.entry_id, b.entry_id], "new", task_id="t3")
        # Should be min(0.8, 1.0) * 0.7 = 0.56
        assert abs(derived.confidence - 0.56) < 0.01

    def test_derive_nonexistent_parents(self, ace_enabled):
        result = ace_enabled.derive(["nonexistent"], "new", task_id="t1")
        assert result is None

    def test_disabled_derive(self, ace_disabled):
        result = ace_disabled.derive(["x"], "new", task_id="t1")
        assert result is None


class TestKillPrinciple:
    """Test cascade kill of principles and descendants."""

    def test_kill_single(self, ace_enabled):
        a = ace_enabled.learn("principle a", task_id="t1")
        killed = ace_enabled.kill_principle(a.entry_id)
        assert killed == 1
        assert ace_enabled.playbook_size == 0

    def test_cascade_kill_children(self, ace_enabled):
        parent = ace_enabled.learn("parent", task_id="t1")
        child = ace_enabled.derive([parent.entry_id], "child", task_id="t2")
        grandchild = ace_enabled.derive([child.entry_id], "grandchild", task_id="t3")
        # Kill parent should cascade to child and grandchild
        killed = ace_enabled.kill_principle(parent.entry_id)
        assert killed == 3
        assert ace_enabled.playbook_size == 0

    def test_kill_preserves_unrelated(self, ace_enabled):
        a = ace_enabled.learn("principle a", task_id="t1")
        b = ace_enabled.learn("principle b", task_id="t2")
        child_a = ace_enabled.derive([a.entry_id], "child of a", task_id="t3")
        killed = ace_enabled.kill_principle(a.entry_id)
        assert killed == 2  # a + child_a
        assert ace_enabled.playbook_size == 1  # b survives
        assert ace_enabled.playbook[0].entry_id == b.entry_id


class TestRecordFailure:
    """Test failure tracking and auto-kill threshold."""

    def test_record_failure_increments_uses(self, ace_enabled):
        a = ace_enabled.learn("fact", task_id="t1")
        ace_enabled.record_failure(a.entry_id)
        entry = ace_enabled.playbook[0]
        assert entry.uses == 1
        assert entry.successes == 0

    def test_auto_kill_after_3_failures(self, ace_enabled):
        a = ace_enabled.learn("bad fact", task_id="t1")
        ace_enabled.record_failure(a.entry_id)
        ace_enabled.record_failure(a.entry_id)
        killed = ace_enabled.record_failure(a.entry_id)
        assert killed == 1
        assert ace_enabled.playbook_size == 0

    def test_auto_kill_cascades(self, ace_enabled):
        parent = ace_enabled.learn("bad parent", task_id="t1")
        child = ace_enabled.derive([parent.entry_id], "child", task_id="t2")
        ace_enabled.record_failure(parent.entry_id)
        ace_enabled.record_failure(parent.entry_id)
        killed = ace_enabled.record_failure(parent.entry_id)
        assert killed == 2  # parent + child
        assert ace_enabled.playbook_size == 0
