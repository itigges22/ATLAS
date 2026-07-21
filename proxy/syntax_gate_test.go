package main

import (
	"strings"
	"testing"
)

// A genuine syntax bug in complete content must NOT be blamed on truncation,
// must quote the offending line, and must forbid an identical resend
// (TB2 2026-07-20 pytorch-model-recovery: an f-string resent verbatim 5×).
func TestFallbackSyntaxRejectionSyntaxBug(t *testing.T) {
	content := "import torch\nx = 1\ny = f\"{d[\"k[\"]}\"\n"
	msg := fallbackSyntaxRejection("a.py", content, "SyntaxError: f-string: unmatched '[' (a.py, line 3)")
	if strings.Contains(msg, "cut off") || strings.Contains(msg, "COMPLETE file content") {
		t.Errorf("must not give truncation advice for a real syntax bug: %q", msg)
	}
	if !strings.Contains(msg, "Do NOT resend") {
		t.Errorf("must forbid identical resend: %q", msg)
	}
	if !strings.Contains(msg, "line 3") || !strings.Contains(msg, `y = f`) {
		t.Errorf("must quote the offending line 3: %q", msg)
	}
}

// A genuinely truncated write keeps the "resend complete content" advice.
func TestFallbackSyntaxRejectionTruncation(t *testing.T) {
	msg := fallbackSyntaxRejection("a.py", "def f(", "SyntaxError: '(' was never closed (a.py, line 1)")
	if !strings.Contains(msg, "COMPLETE") {
		t.Errorf("truncation shape should advise resending complete content: %q", msg)
	}
}
