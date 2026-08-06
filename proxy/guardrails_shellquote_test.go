package main

import (
	"strings"
	"testing"
)

// The exact command measured on multiturn_stats, which the model re-sent 12
// times: the f-string opens on a single quote and closes on a double, so it is
// invalid Python as well as unparseable shell. Nothing the harness could
// re-quote would save it; the shape has to change.
const brokenOneLiner = `python3 -c "from stats import mean, median; print(f'mean([1, 2, 3]): {mean([1, 2, 3])}"); print(f'median([1, 2, 3]): {median([1, 2, 3])}")"`

const bashParseErr = "bash: -c: line 1: syntax error near unexpected token `)'"

func TestAnUnparseableOneLinerGetsAnActionableReason(t *testing.T) {
	hint := shellQuotingHint(brokenOneLiner, bashParseErr)
	if hint == "" {
		t.Fatal("the measured failure must produce a hint")
	}
	for _, want := range []string{"write_file", "nothing ran", "fails identically"} {
		if !strings.Contains(hint, want) {
			t.Errorf("hint should contain %q: %s", want, hint)
		}
	}
}

func TestACommandThatRanAndFailedIsNotAQuotingProblem(t *testing.T) {
	// pytest exiting non-zero is a real result, not a parse failure.
	if h := shellQuotingHint("pytest -q", "1 failed, 3 passed"); h != "" {
		t.Errorf("a real test failure must not be relabelled as quoting: %s", h)
	}
}

func TestAParseErrorInAScriptFileIsNotRedirected(t *testing.T) {
	// Already running from a file — the advice would be to do what it did.
	if h := shellQuotingHint("python3 check.py", bashParseErr); h != "" {
		t.Errorf("no redirect when the program is already a file: %s", h)
	}
}

func TestOtherInlineInterpretersAreCovered(t *testing.T) {
	for _, cmd := range []string{`node -e "console.log('x)"`, `perl -e 'print "x'`} {
		if shellQuotingHint(cmd, bashParseErr) == "" {
			t.Errorf("no hint for %s", cmd)
		}
	}
}

func TestEmptyInputsAreSafe(t *testing.T) {
	if shellQuotingHint("", bashParseErr) != "" || shellQuotingHint(brokenOneLiner, "") != "" {
		t.Error("empty command or error must not produce a hint")
	}
}
