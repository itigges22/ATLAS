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

// Widened contract detection: a redirect anywhere in the segment and the
// cat-pipe idiom are the same stdin contract the trailing-only rule caught.
func TestStdinRedirectSourceWiderShapes(t *testing.T) {
	cases := []struct {
		cmd  string
		want string
	}{
		{"python3 solve.py < input.txt > out.txt", "input.txt"},
		{"python3 solve.py <input.txt 2>err.log", "input.txt"},
		{"cat input.txt | python3 solve.py", "input.txt"},
		{"cd /w && python3 solve.py < data.txt", "data.txt"},
		// Not stdin contracts:
		{"python3 solve.py", ""},
		{"python3 solve.py <<EOF\n1 2\nEOF", ""},
		{"diff <(sort a) <(sort b)", ""},
		{"cat notes.txt || echo missing", ""},
	}
	for _, c := range cases {
		if got := stdinRedirectSource(c.cmd); got != c.want {
			t.Errorf("stdinRedirectSource(%q) = %q, want %q", c.cmd, got, c.want)
		}
	}
}
