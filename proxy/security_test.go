package main

import (
	"strings"
	"testing"
)

func TestSafeLogFieldEscapesRecordSeparators(t *testing.T) {
	got := safeLogField("first\nforged\r\x00entry", 200)
	if strings.ContainsAny(got, "\r\n\x00") {
		t.Fatalf("safeLogField emitted a raw record separator: %q", got)
	}
	for _, escaped := range []string{`\n`, `\r`, `\x00`} {
		if !strings.Contains(got, escaped) {
			t.Fatalf("safeLogField(%q) missing %q", got, escaped)
		}
	}
}

func TestSafeLogFieldBoundsUntrustedText(t *testing.T) {
	got := safeLogField(strings.Repeat("x", 100), 12)
	if len(got) > 20 {
		t.Fatalf("bounded log field remained unexpectedly large: %q", got)
	}
}
