package main

import (
	"strings"
	"testing"
)

// A program run as `prog < data` is verified as a stdin filter, which is not
// how its caller will run it.
//
// Measured on the AoC tasks, whose prompt says the program must read
// input.txt: 7 of 10 failures wrote a stdin-reading program, ran it as
// `python3 solve.py < input.txt`, and got a SUCCESSFUL result — so the model
// had every reason to believe it was finished. The checker then ran
// `python solve.py` with no redirect and got 0. Not one session that
// verified this way passed.
//
// The same model with no shell never produces this shape; it writes code that
// opens the file, because piping is not available to it. The tool is what
// makes the wrong contract reachable, so the harness has to notice.

func TestARedirectIsDetected(t *testing.T) {
	for _, cmd := range []string{
		"python3 solve.py < input.txt",
		"python solve.py <input.txt",
		"./run < data/in.txt",
	} {
		if got := stdinRedirectSource(cmd); got == "" {
			t.Errorf("no redirect detected in %q", cmd)
		}
	}
}

func TestTheRedirectedFileIsNamed(t *testing.T) {
	if got := stdinRedirectSource("python3 solve.py < input.txt"); got != "input.txt" {
		t.Errorf("want input.txt, got %q", got)
	}
}

func TestOrdinaryCommandsAreNotRedirects(t *testing.T) {
	for _, cmd := range []string{
		"python3 solve.py",
		"pytest -q",
		"go build ./...",
		"echo hi > out.txt",
		"python3 - <<'EOF'\nprint(1)\nEOF",
		"diff <(sort a) <(sort b)",
		"",
	} {
		if got := stdinRedirectSource(cmd); got != "" {
			t.Errorf("%q should not read as a stdin redirect, got %q", cmd, got)
		}
	}
}

func TestTheRejectionSaysWhatToDoInstead(t *testing.T) {
	msg := redirectOnlyVerificationMessage("input.txt")
	for _, want := range []string{"input.txt", "standalone", "no `<`", "open"} {
		if !strings.Contains(msg, want) {
			t.Errorf("rejection should mention %q: %s", want, msg)
		}
	}
}
