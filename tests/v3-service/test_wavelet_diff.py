"""Conformance suite for the peak-profile diff port (drift detector).

Translated from wavescope-mcp `src/diff.test.ts`, plus a golden-fixture test
asserting parity with the actual upstream `diffPeaks` (captured via `npx tsx`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from wavelet.context import ImportantPosition  # noqa: E402
from wavelet.diff import diff_peaks, diff_contents  # noqa: E402


def _p(position, coefficient, scale=2, label=""):
    return ImportantPosition(position=position, coefficient=coefficient, scale=scale, label=label)


class TestDiffPeaks:
    def test_identical_all_unchanged(self):
        peaks = [_p(5, 1.0), _p(20, 0.8)]
        d = diff_peaks(list(peaks), list(peaks))
        assert d.summary["unchanged"] == 2
        assert d.summary["added"] == 0
        assert d.summary["removed"] == 0

    def test_added(self):
        d = diff_peaks([_p(5, 1.0)], [_p(5, 1.0), _p(50, 0.7)])
        assert d.summary["added"] == 1
        assert any(c.kind == "added" and c.after.position == 50 for c in d.changes)

    def test_removed(self):
        d = diff_peaks([_p(5, 1.0), _p(50, 0.7)], [_p(5, 1.0)])
        assert d.summary["removed"] == 1
        assert any(c.kind == "removed" and c.before.position == 50 for c in d.changes)

    def test_shifted_within_window(self):
        d = diff_peaks([_p(20, 0.8)], [_p(21, 0.8)], window=2)
        assert d.summary["shifted"] == 1

    def test_shift_beyond_window_is_add_remove(self):
        d = diff_peaks([_p(20, 0.8)], [_p(30, 0.8)], window=2)
        assert d.summary["shifted"] == 0
        assert d.summary["added"] == 1
        assert d.summary["removed"] == 1

    def test_magnitude_changed(self):
        d = diff_peaks([_p(10, 0.5)], [_p(10, 1.5)])
        assert d.summary["magnitudeChanged"] == 1

    def test_golden_matches_upstream(self):
        # Captured from upstream src/diff.ts via `npx tsx`.
        before = [_p(5, 1.0, 2), _p(20, 0.8, 4), _p(40, 0.5, 8)]
        after = [_p(5, 1.0, 2), _p(21, 0.8, 4), _p(60, 0.9, 8)]
        d = diff_peaks(before, after, 2)
        assert d.summary == {
            "added": 1, "removed": 1, "shifted": 1, "magnitudeChanged": 0, "unchanged": 1,
        }
        assert [c.kind for c in d.changes] == ["unchanged", "shifted", "removed", "added"]


class TestDiffContents:
    def test_added_function_shows_up(self):
        before = "def a():\n    return 1\n"
        after = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        d = diff_contents(before, after, "m.py")
        assert d.diff.summary["added"] >= 1
        assert d.after_line_count > d.before_line_count

    def test_identical_contents_no_churn(self):
        src = "class Foo:\n    def bar(self):\n        return 1\n"
        d = diff_contents(src, src, "m.py")
        assert d.diff.summary["added"] == 0
        assert d.diff.summary["removed"] == 0

    def test_empty_to_content_all_added(self):
        d = diff_contents("", "def a():\n    return 1\n", "m.py")
        assert d.diff.summary["removed"] == 0
        assert d.before_line_count == 0
