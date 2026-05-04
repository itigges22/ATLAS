package main

import (
	"math"
	"testing"
)

func TestLineOverlapRatio(t *testing.T) {
	tests := []struct {
		name string
		a    string
		b    string
		want float64
	}{
		{
			name: "identical files",
			a:    "alpha\nbeta\ngamma",
			b:    "alpha\nbeta\ngamma",
			want: 1.0,
		},
		{
			name: "one line different out of three",
			a:    "alpha\nbeta\ngamma",
			b:    "alpha\nbeta\nDELTA",
			// 2/3 overlap — below 0.70 threshold, so a 3-line file with
			// one changed line stays UNDER the gate (file too small to
			// gate on anyway, but this asserts the math).
			want: 2.0 / 3.0,
		},
		{
			name: "near rewrite — 99 of 100 lines kept",
			a:    repeatLines("x", 100),
			b:    repeatLines("x", 99) + "\ny",
			// Multiset: existing has 100 'x' + 0 'y'; new has 99 'x' + 1 'y'.
			// matched = 99 (the x lines); maxLen = 100 → 0.99.
			// This case MUST trip the surgical-edit gate.
			want: 0.99,
		},
		{
			name: "complete rewrite — no shared lines",
			a:    "old1\nold2\nold3",
			b:    "new1\nnew2\nnew3",
			want: 0.0,
		},
		{
			name: "legitimate refactor — 50% changed",
			a:    "L1\nL2\nL3\nL4",
			b:    "L1\nL2\nNEW3\nNEW4",
			// matched = 2, max = 4 → 0.5. Below 0.70 → NOT gated, which
			// is what we want: legitimate restructuring should pass.
			want: 0.5,
		},
		{
			name: "growing a small file — additive, all old kept",
			a:    "a\nb",
			b:    "a\nb\nc\nd\ne",
			// matched = 2, max = 5 → 0.4. Big additions are not rewrites;
			// shouldn't gate.
			want: 2.0 / 5.0,
		},
		{
			name: "empty old, non-empty new — no overlap",
			a:    "",
			b:    "line1\nline2",
			// strings.Split("", "\n") returns [""] — one empty "line".
			// It doesn't match either non-empty line in b, so matched=0,
			// max=2 → 0.0. We never reach this code path in practice
			// (file doesn't exist → ReadFile errors out before
			// lineOverlapRatio is called); asserting it can't spuriously
			// fire the gate on a file that's about to be created.
			want: 0.0,
		},
		{
			name: "both empty",
			a:    "",
			b:    "",
			want: 1.0, // matched=1, max=1 (both have a single "" element)
		},
		{
			name: "duplicate lines in source — multiset semantics",
			a:    "x\nx\nx\ny",
			b:    "x\nx\ny",
			// Multiset intersection respects counts: matched = 2 'x' + 1 'y' = 3.
			// max = 4. → 0.75. Without multiset semantics we'd over-count.
			want: 0.75,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lineOverlapRatio(tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("lineOverlapRatio() = %v, want %v", got, tt.want)
			}
		})
	}
}

// repeatLines returns line\nline\n... repeated n times (no trailing newline).
func repeatLines(line string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		if i > 0 {
			out += "\n"
		}
		out += line
	}
	return out
}
