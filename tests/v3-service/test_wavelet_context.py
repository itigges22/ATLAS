"""Conformance suite for FileContext band assembly + important positions.

Translated/adapted from wavescope-mcp `src/context.test.ts`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v3-service"))

from wavelet.context import FileContext  # noqa: E402

SAMPLE_PY = """import os
import sys


class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        cleaned = self._clean(data)
        return cleaned

    def _clean(self, data):
        return [d for d in data if d]


def main():
    processor = DataProcessor({})
    print(processor.process([1, 2, 3]))


if __name__ == "__main__":
    main()
"""


class TestConstruction:
    def test_empty_content(self):
        ctx = FileContext("empty.py", "")
        assert ctx.line_count == 0
        assert ctx.get_important_positions() == []

    def test_trailing_newline_dropped(self):
        ctx = FileContext("a.py", "x = 1\n")
        assert ctx.line_count == 1

    def test_no_trailing_newline(self):
        ctx = FileContext("a.py", "x = 1\ny = 2")
        assert ctx.line_count == 2


class TestImportantPositions:
    def test_finds_class_and_defs(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        positions = ctx.get_important_positions(min_coefficient=0.3, limit=20)
        assert len(positions) > 0
        labels = " ".join(p.label for p in positions)
        assert "class DataProcessor" in labels
        # All returned positions clear the threshold and sort descending.
        for i in range(1, len(positions)):
            assert abs(positions[i - 1].coefficient) >= abs(positions[i].coefficient)

    def test_limit_respected(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        assert len(ctx.get_important_positions(0.0, 3)) <= 3


class TestQueryWaveletContext:
    def test_three_bands_present(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        res = ctx.query_wavelet_context(center=8, radius=300)
        assert set(res.bands.keys()) == {"fine", "medium", "coarse"}
        assert res.bands["fine"].content  # raw lines around center
        # Fine band is raw source — the center line should appear verbatim.
        assert "def process" in res.bands["fine"].content

    def test_center_clamped_when_out_of_range(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        res = ctx.query_wavelet_context(center=9999, radius=100)
        assert res.clamped is True
        assert res.clamped_from == 9999
        assert res.center == ctx.line_count - 1

    def test_empty_file_query(self):
        ctx = FileContext("empty.py", "")
        res = ctx.query_wavelet_context(center=5, radius=100)
        assert res.clamped is True
        assert res.wavelet_peaks == []
        assert res.bands["fine"].content == ""

    def test_fine_band_radius_minimum(self):
        # radius//5 would be 2, but fine radius floors at 10 lines each side.
        ctx = FileContext("sample.py", SAMPLE_PY)
        res = ctx.query_wavelet_context(center=10, radius=10)
        start, end = res.bands["fine"].range
        assert end - start >= 10


class TestSummaryAtScale:
    def test_auto_scale_by_region_size(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        assert ctx.auto_scale(0, 40) in ctx.coefficients.scales
        # Small region snaps to a small scale, large region to a large one.
        assert ctx.auto_scale(0, 10) <= ctx.auto_scale(0, 1000)

    def test_summary_non_empty(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        summary = ctx.get_summary_at_scale(0, ctx.line_count - 1)
        assert summary

    def test_out_of_range_summary_empty(self):
        ctx = FileContext("sample.py", SAMPLE_PY)
        assert ctx.get_summary_at_scale(9000, 9999) == ""
