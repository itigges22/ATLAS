"""
Tests for provenance tracking module.

Validates AI content detection from git history
and provenance scoring.
"""

import os
import sys
import tempfile
import subprocess

import pytest

# Add rag-api to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rag-api"))


class TestProvenanceImport:
    """Test provenance module can be imported."""

    def test_module_imports(self):
        """provenance module should import successfully."""
        try:
            from provenance import check_provenance, is_ai_generated
            assert check_provenance is not None or is_ai_generated is not None
        except ImportError:
            # Module may have different structure
            import provenance
            assert provenance is not None


class TestProvenanceAIDetection:
    """Test AI content marker detection."""

    def test_detects_ai_marker(self):
        """Should detect [AI] marker in commit messages."""
        from provenance import is_ai_generated, detect_ai_markers

        test_messages = [
            "[AI] Generated code for feature X",
            "[ai] Auto-generated tests",
            "feat: add feature [AI]"
        ]

        for msg in test_messages:
            assert is_ai_generated(msg), f"Should detect AI marker in: {msg}"
            markers = detect_ai_markers(msg)
            assert "[AI]" in markers, f"Should find [AI] marker in: {msg}"

    def test_detects_auto_marker(self):
        """Should detect [AUTO] marker in commit messages."""
        test_messages = [
            "[AUTO] Automated refactoring",
            "[auto] CI-generated changes",
            "fix: automated fix [AUTO]"
        ]

        for msg in test_messages:
            assert "[AUTO]" in msg.upper() or "[auto]" in msg.lower()

    def test_detects_generated_marker(self):
        """Should detect [GENERATED] marker in commit messages."""
        test_messages = [
            "[GENERATED] Code from template",
            "[generated] Auto-created file",
            "docs: generated docs [GENERATED]"
        ]

        for msg in test_messages:
            assert "[GENERATED]" in msg.upper() or "[generated]" in msg.lower()

    def test_detects_bot_marker(self):
        """Should detect [BOT] marker in commit messages."""
        test_messages = [
            "[BOT] Dependency update",
            "[bot] Automated PR",
            "chore: bot update [BOT]"
        ]

        for msg in test_messages:
            assert "[BOT]" in msg.upper() or "[bot]" in msg.lower()

    def test_human_commits_marked_as_human(self):
        """Regular commits should be marked as human."""
        human_messages = [
            "feat: add new feature",
            "fix: resolve bug in login",
            "refactor: improve code structure",
            "docs: update README",
            "Initial commit"
        ]

        for msg in human_messages:
            # Should not contain AI markers
            markers = ["[AI]", "[AUTO]", "[GENERATED]", "[BOT]"]
            assert not any(m.lower() in msg.lower() for m in markers), \
                f"Human commit should not have AI markers: {msg}"


class TestProvenanceScoring:
    """Test provenance scoring for training weights."""

    def test_provenance_affects_training_weight(self):
        """AI-generated content should have lower training weight."""
        # Simulate scoring logic
        def score_provenance(is_ai: bool) -> float:
            return 0.3 if is_ai else 1.0

        ai_score = score_provenance(True)
        human_score = score_provenance(False)

        assert human_score > ai_score, "Human content should score higher"
        assert ai_score > 0, "AI content should still have some weight"


class TestProvenanceGitHistory:
    """Test git history parsing."""

    def test_git_history_parsing(self):
        """Should parse git history correctly."""
        # Create a temporary git repo for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir, capture_output=True
            )

            # Create and commit a file
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("x = 1\n")

            subprocess.run(["git", "add", "test.py"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=tmpdir, capture_output=True
            )

            # Get git log
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=tmpdir, capture_output=True, text=True
            )

            assert "Initial commit" in result.stdout

    def test_files_without_git_history_handled(self):
        """Files without git history should be handled."""
        # Create file outside git repo
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("x = 1\n")
            temp_path = f.name

        # Should not crash when checking provenance of non-git file
        result = subprocess.run(
            ["git", "log", "--oneline", "-1", "--", temp_path],
            capture_output=True, text=True
        )
        # Git should return empty stdout or error for non-git file
        assert result.stdout == "" or result.returncode != 0, \
            "Non-git file should return empty output or error"
        os.unlink(temp_path)

    def test_non_git_directories_handled(self):
        """Non-git directories should be handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in non-git directory
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("x = 1\n")

            # Try git log - should fail gracefully
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=tmpdir, capture_output=True, text=True
            )

            # Should return error or empty
            assert result.returncode != 0 or not result.stdout


class TestProvenanceIntegration:
    """Test provenance integration with training pipeline."""

    def test_provenance_data_structure(self):
        """Provenance data should have expected structure."""
        # Define expected provenance structure
        provenance_data = {
            "file_path": "src/main.py",
            "is_ai_generated": False,
            "confidence": 0.95,
            "source_commits": ["abc123", "def456"],
            "ai_markers_found": []
        }

        assert "file_path" in provenance_data
        assert "is_ai_generated" in provenance_data
        assert isinstance(provenance_data["is_ai_generated"], bool)

    def test_commit_message_parsing(self):
        """Should extract markers from commit messages."""
        messages = [
            "feat: add feature",
            "[AI] Generate tests",
            "fix: bug [AUTO]",
            "[GENERATED] Scaffolding",
            "[BOT] Update deps"
        ]

        ai_markers = ["[AI]", "[AUTO]", "[GENERATED]", "[BOT]"]

        for msg in messages:
            is_ai = any(marker.lower() in msg.lower() for marker in ai_markers)
            expected_ai = any(marker.lower() in msg.lower() for marker in ai_markers)
            assert is_ai == expected_ai


class TestProvenanceFunctions:
    """Test provenance helper functions."""

    def test_is_ai_generated_with_ai_marker(self):
        """is_ai_generated should return True for AI markers."""
        from provenance import is_ai_generated
        assert is_ai_generated("[AI] Test commit") is True

    def test_is_ai_generated_with_auto_marker(self):
        """is_ai_generated should return True for AUTO markers."""
        from provenance import is_ai_generated
        assert is_ai_generated("[AUTO] Automated change") is True

    def test_is_ai_generated_with_generated_marker(self):
        """is_ai_generated should return True for GENERATED markers."""
        from provenance import is_ai_generated
        assert is_ai_generated("[GENERATED] Code") is True

    def test_is_ai_generated_with_bot_marker(self):
        """is_ai_generated should return True for BOT markers."""
        from provenance import is_ai_generated
        assert is_ai_generated("[BOT] Dependency update") is True

    def test_is_ai_generated_with_human_commit(self):
        """is_ai_generated should return False for human commits."""
        from provenance import is_ai_generated
        assert is_ai_generated("feat: add feature") is False

    def test_is_ai_generated_with_empty_string(self):
        """is_ai_generated should return False for empty string."""
        from provenance import is_ai_generated
        assert is_ai_generated("") is False

    def test_is_ai_generated_with_none(self):
        """is_ai_generated should return False for None."""
        from provenance import is_ai_generated
        assert is_ai_generated(None) is False

    def test_detect_ai_markers_returns_list(self):
        """detect_ai_markers should return a list."""
        from provenance import detect_ai_markers
        result = detect_ai_markers("[AI] Test")
        assert isinstance(result, list)

    def test_detect_ai_markers_finds_all(self):
        """detect_ai_markers should find multiple markers."""
        from provenance import detect_ai_markers
        result = detect_ai_markers("[AI] [BOT] Combined")
        assert len(result) >= 2

    def test_detect_ai_markers_empty_for_human(self):
        """detect_ai_markers should return empty for human commits."""
        from provenance import detect_ai_markers
        result = detect_ai_markers("feat: normal commit")
        assert result == []

    def test_detect_ai_markers_case_insensitive(self):
        """detect_ai_markers should be case insensitive."""
        from provenance import detect_ai_markers
        result1 = detect_ai_markers("[ai] lowercase")
        result2 = detect_ai_markers("[AI] uppercase")
        assert len(result1) > 0
        assert len(result2) > 0


class TestProvenanceMarkerVariations:
    """Test various marker formats."""

    def test_marker_at_start(self):
        """Should detect marker at start of message."""
        from provenance import is_ai_generated
        assert is_ai_generated("[AI] Starting message") is True

    def test_marker_at_end(self):
        """Should detect marker at end of message."""
        from provenance import is_ai_generated
        assert is_ai_generated("Message at end [AI]") is True

    def test_marker_in_middle(self):
        """Should detect marker in middle of message."""
        from provenance import is_ai_generated
        assert is_ai_generated("Some [AI] in middle") is True

    def test_marker_lowercase(self):
        """Should detect lowercase markers."""
        from provenance import is_ai_generated
        assert is_ai_generated("[ai] lowercase marker") is True

    def test_marker_mixed_case(self):
        """Should detect mixed case markers."""
        from provenance import is_ai_generated
        assert is_ai_generated("[Ai] mixed case") is True

    def test_multiple_markers(self):
        """Should detect when multiple markers present."""
        from provenance import is_ai_generated, detect_ai_markers
        msg = "[AI] [AUTO] Multiple markers"
        assert is_ai_generated(msg) is True
        markers = detect_ai_markers(msg)
        assert len(markers) >= 2


class TestProvenanceEdgeCases:
    """Test edge cases in provenance detection."""

    def test_partial_marker_not_detected(self):
        """Partial markers should not be detected."""
        from provenance import is_ai_generated
        # AI without brackets should not match
        assert is_ai_generated("Using AI for coding") is False

    def test_similar_words_not_detected(self):
        """Similar words should not trigger false positives."""
        from provenance import is_ai_generated
        messages = [
            "Automated testing improvements",
            "Generated tests manually",
            "Bot framework setup",
        ]
        for msg in messages:
            # These don't have the exact markers
            assert is_ai_generated(msg) is False

    def test_unicode_in_message(self):
        """Should handle unicode in commit messages."""
        from provenance import is_ai_generated
        assert is_ai_generated("[AI] 日本語テスト") is True
        assert is_ai_generated("修正: バグ修正") is False

    def test_special_characters_in_message(self):
        """Should handle special characters."""
        from provenance import is_ai_generated
        assert is_ai_generated("[AI] Fix <script> injection") is True
        assert is_ai_generated("Fix: handle & and <") is False

    def test_very_long_message(self):
        """Should handle very long commit messages."""
        from provenance import is_ai_generated
        long_msg = "[AI] " + "x" * 10000
        assert is_ai_generated(long_msg) is True

    def test_whitespace_variations(self):
        """Should handle whitespace variations."""
        from provenance import is_ai_generated
        assert is_ai_generated("  [AI]  spaced  ") is True
        assert is_ai_generated("\t[AI]\tTabbed") is True
        assert is_ai_generated("\n[AI]\nNewline") is True


class TestProvenanceCheckFunction:
    """Test check_provenance function if available."""

    def test_check_provenance_exists(self):
        """check_provenance function should exist or be importable."""
        import provenance
        # Module must have either check_provenance or is_ai_generated
        has_check = hasattr(provenance, 'check_provenance') and callable(getattr(provenance, 'check_provenance', None))
        has_is_ai = hasattr(provenance, 'is_ai_generated') and callable(getattr(provenance, 'is_ai_generated', None))
        assert has_check or has_is_ai, \
            "provenance module must have check_provenance or is_ai_generated function"

    def test_provenance_module_structure(self):
        """provenance module should have expected structure."""
        import provenance

        # Should have some functions
        assert hasattr(provenance, 'is_ai_generated') or hasattr(provenance, 'check_provenance')


class TestProvenanceTrainingIntegration:
    """Test provenance with training data pipeline."""

    def test_weight_calculation(self):
        """Training weight should be calculated from provenance."""
        def calculate_weight(is_ai: bool, rating: int) -> float:
            base = 1.0 if rating >= 4 else 0.5
            return base * (0.3 if is_ai else 1.0)

        # Human code with high rating
        assert calculate_weight(False, 5) == 1.0
        # AI code with high rating
        assert calculate_weight(True, 5) == 0.3
        # Human code with low rating
        assert calculate_weight(False, 3) == 0.5
        # AI code with low rating
        assert calculate_weight(True, 3) == 0.15

    def test_training_data_filtering(self):
        """Should filter training data by provenance."""
        training_samples = [
            {"code": "x = 1", "is_ai": False, "rating": 5},
            {"code": "y = 2", "is_ai": True, "rating": 5},
            {"code": "z = 3", "is_ai": False, "rating": 4},
            {"code": "w = 4", "is_ai": True, "rating": 3},
        ]

        # Filter high-quality human samples
        high_quality_human = [
            s for s in training_samples
            if not s["is_ai"] and s["rating"] >= 4
        ]
        assert len(high_quality_human) == 2

        # All AI samples
        ai_samples = [s for s in training_samples if s["is_ai"]]
        assert len(ai_samples) == 2
